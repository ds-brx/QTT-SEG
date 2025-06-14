import os
import time
from pathlib import Path
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts, 
    ReduceLROnPlateau, OneCycleLR, StepLR, PolynomialLR
)
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from torchmetrics.classification import JaccardIndex

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.data.custom_dataloader import CustomDataset
from src.sam2_process.sam2_test import test
from src.utils.utils import get_parser

# Constants
SAM2_CHECKPOINT = "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        
        smooth = 1e-5
        intersection = (inputs * targets).sum(1)
        dice = (2. * intersection + smooth) / (inputs.sum(1) + targets.sum(1) + smooth)
        dice_loss = 1 - dice.mean()
        
        return bce_loss + dice_loss

def numpy_collate(batch):
    return batch

def setup_model(args, device):
    """Initialize and configure the SAM2 model with optional LoRA."""
    model = build_sam2(MODEL_CFG, SAM2_CHECKPOINT, device=device)
    
    if args.lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=2 * args.lora_rank,
            target_modules=["attn.qkv", "mlp.layers.0", "mlp.layers.1"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION
        )
        model = get_peft_model(model, lora_config)
    
    # Freeze/unfreeze parameters
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if "mask_decoder" in name or "prompt_encoder" in name:
            param.requires_grad = True
    
    return model

def get_scheduler(optimizer, args, steps_per_epoch):
    """Configure learning rate scheduler based on args."""
    schedulers = {
        "cosine_warm": lambda opt: CosineAnnealingWarmRestarts(
            opt, T_0=args.cosine_t0, T_mult=args.cosine_t_mult, eta_min=args.lr / 10
        ),
        "cosine": lambda opt: CosineAnnealingLR(
            opt, T_max=args.num_train_epochs, eta_min=args.lr / 10
        ),
        "plateau": lambda opt: ReduceLROnPlateau(
            opt, mode="max", patience=args.patience_epochs, factor=args.decay_rate
        ),
        "onecycle": lambda opt: OneCycleLR(
            opt,
            max_lr=args.lr,
            steps_per_epoch=steps_per_epoch,
            epochs=args.num_train_epochs,
            pct_start=args.onecycle_pct_start,
            div_factor=args.onecycle_div_factor,
            final_div_factor=args.onecycle_final_div_factor,
            anneal_strategy='cos'
        ),
        "step": lambda opt: StepLR(
            opt, step_size=args.step_size, gamma=args.decay_rate
        ),
        "poly": lambda opt: PolynomialLR(
            opt, total_iters=args.num_train_epochs * steps_per_epoch, power=args.poly_power
        )
    }
    return schedulers.get(args.sched, None)

def train_epoch(predictor, train_loader, optimizer, loss_fn, jaccard, device, scheduler=None):
    """Run one training epoch with proper device handling."""
    predictor.model.train()
    total_loss = total_iou = 0
    
    for batch in tqdm(train_loader, desc="Training", leave=False):
        sample = batch[0]
        if not sample:
            continue
            
        # Keep image as numpy for predictor
        image = sample["pixel_values"]  # numpy array
        mask = torch.as_tensor(sample["ground_truth_mask"], device=device).float()
        input_box = torch.as_tensor(sample["input_box"], device=device)
        
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        # Set image (requires numpy)
        predictor.set_image(image)
        
        # Prepare prompts (uses device tensors internally)
        _, _, _, unnorm_box = predictor._prep_prompts(
            point_coords=None, 
            point_labels=None, 
            box=input_box.cpu().numpy(),  # Convert to numpy for processing
            mask_logits=None, 
            normalize_coords=True
        )
        unnorm_box = torch.as_tensor(unnorm_box, device=device)
        
        # Model forward pass
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=None, 
            boxes=unnorm_box, 
            masks=None
        )

        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, _, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=unnorm_box.shape[0] > 1,
            high_res_features=high_res_features,
        )

        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])
        
        # Calculate loss and metrics
        loss = loss_fn(prd_masks[:, 0], mask)
        total_loss += loss.item()
        iou = jaccard((prd_masks[:, 0] > 0.5).int(), mask.int())
        total_iou += iou.mean().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), 1.0)
        optimizer.step()

        if scheduler and isinstance(scheduler, (OneCycleLR, PolynomialLR)):
            scheduler.step()

    return total_loss / len(train_loader.dataset), total_iou / len(train_loader.dataset)


def main(args, max_time=10**18):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    model = setup_model(args, device)
    predictor = SAM2ImagePredictor(model)
    jaccard = JaccardIndex(task="binary").to(device)
    loss_fn = BCEDiceLoss()
    
    # Data loading
    train_dataset = CustomDataset(dataset_name=args.dataset_name, split="train", args=args)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=numpy_collate)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = get_scheduler(optimizer, args, len(train_loader))
    
    # Training loop
    best_score = -float("inf")
    metrics = {"train_loss": [], "train_iou": [], "val_iou": []}
    start_time = time.time()
    
    for epoch in range(args.num_train_epochs):
        if time.time() - start_time > max_time:
            print(f"Time limit of {max_time} seconds reached.")
            break
            
        # Training
        epoch_loss, epoch_iou = train_epoch(
            predictor, train_loader, optimizer, loss_fn, jaccard, device, scheduler
        )
        metrics["train_loss"].append(epoch_loss)
        metrics["train_iou"].append(epoch_iou)
        
        # Validation
        val_score = test(split="val", predicted_model=predictor.model, args=args, device=device)
        metrics["val_iou"].append(val_score)
        print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Train IoU={epoch_iou:.4f}, Val IoU={val_score:.4f}")
        
        # Save best model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), os.path.join(args.output_dir, "sam2model.torch"))
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_score)
            elif isinstance(scheduler, (StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts)):
                scheduler.step()
    
    # Final report
    report = {
        "dataset": args.dataset_name,
        "score": best_score,
        "cost": time.time() - start_time
    }
    
    if args.return_scores_per_epoch:
        return report, {f"epoch_{i}_iou": val for i, val in enumerate(metrics["val_iou"])}
    return report

if __name__ == "__main__":
    args = get_parser().parse_args()
    report = main(args)
    print(report)