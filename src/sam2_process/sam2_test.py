import os
import torch
from tqdm import tqdm
import pandas as pd
from torchmetrics.classification import JaccardIndex
import torchvision.transforms as transforms
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from src.data.custom_dataloader import CustomDataset
from src.utils.utils import get_parser
from peft import get_peft_model, LoraConfig, TaskType

# Constants
SAM2_CHECKPOINT = "third_party/sam2/checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

def setup_predictor(args, device, model_path=None):
    """Initialize and configure the predictor with optional LoRA and model weights."""
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
    
    if model_path:
        model.load_state_dict(torch.load(model_path))
    
    predictor = SAM2ImagePredictor(model)
    predictor.model.eval()
    return predictor

def save_sample_images(image, gt_mask, prd_mask, zero_shot_mask, output_dir, idx):
    """Save sample images, ground truth, predicted masks and zero-shot masks."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert tensors to PIL images
    image_pil = transforms.ToPILImage()(image)
    
    # Process masks
    if gt_mask.ndim > 2:
        gt_mask = (torch.argmax(gt_mask, dim=0)).byte() * 255
        prd_mask = (torch.argmax(prd_mask, dim=0)).byte() * 255
        zero_shot_mask = (torch.argmax(zero_shot_mask, dim=0)).byte() * 255
    else:
        gt_mask = gt_mask.byte() * 255
        prd_mask = prd_mask.byte() * 255
        zero_shot_mask = zero_shot_mask.byte() * 255
    
    # Save all images
    image_pil.save(output_dir / f"image_{idx}.png")
    transforms.ToPILImage()(gt_mask).save(output_dir / f"gt_mask_{idx}.png")
    transforms.ToPILImage()(prd_mask).save(output_dir / f"prd_mask_{idx}.png")
    transforms.ToPILImage()(zero_shot_mask).save(output_dir / f"zero_shot_mask_{idx}.png")

def test(split="test", predicted_model=None, predicted_model_path=None, 
        args=None, save_images=False, max_samples=5, device="cuda"):
    """Run evaluation and save both predicted and zero-shot masks."""
    device = device if torch.cuda.is_available() else "cpu"
    jaccard = JaccardIndex(task="binary").to(device)
    
    # Setup both fine-tuned and zero-shot predictors
    pred_predictor = setup_predictor(args, device, predicted_model_path)
    zero_shot_predictor = setup_predictor(args, device)
    
    if predicted_model:
        pred_predictor.model = predicted_model.to(device)
    
    # Prepare dataset
    test_dataset = CustomDataset(dataset_name=args.dataset_name, split=split, args=args)
    
    # Initialize output directories
    if save_images:
        output_dir = Path(args.output_dir) / "prediction_masks" / args.dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None
    
    # Evaluation loop
    mean_iou = 0
    saved_count = 0
    
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(test_dataset), total=len(test_dataset), desc="Testing"):
            if not batch:
                continue
                
            # Move data to device
            image = batch["pixel_values"]  # numpy array
            gt_mask = torch.as_tensor(batch["ground_truth_mask"], device=device).float()
            input_box = torch.as_tensor(batch["input_box"], device=device)
            # Get predictions from both models
            for predictor, is_zero_shot in [(pred_predictor, False), (zero_shot_predictor, True)]:
                predictor.set_image(image)
                prd_masks, _, _ = predictor.predict(box=input_box)
                
                # Process masks
                prd_mask = torch.sigmoid(torch.from_numpy(
                    prd_masks[0] if gt_mask.ndim == 2 else prd_masks[:, 0]
                ).to(device))
                
                # Only calculate IoU for the fine-tuned model
                if not is_zero_shot:
                    iou = jaccard((prd_mask > 0.5).int(), gt_mask.int())
                    mean_iou += iou.item()
                
                # Save sample images if enabled
                if save_images and saved_count < max_samples and output_dir:
                    if is_zero_shot:
                        zero_shot_mask = prd_mask
                    else:
                        fine_tuned_mask = prd_mask
            
            # Save both masks after both predictions are done
            if save_images and saved_count < max_samples and output_dir:
                save_sample_images(
                    image, gt_mask, fine_tuned_mask, zero_shot_mask, 
                    output_dir, saved_count
                )
                saved_count += 1
    
    return mean_iou / len(test_dataset)

if __name__ == "__main__":
    results_csv_path = "model_comparison_results.csv"
    parser = get_parser()
    args = parser.parse_args()
    
    results = []
    for dataset in ['human_parsing', 'US', 'golf', 'terrain', 'cholec']:
        args.dataset_name = dataset
        mean_score = test(args=args, save_images=True)
        
        results.append({
            "DATASET": dataset,
            "SEED": args.seed,
            "MEAN_IOU": mean_score,
        })
    
    # Save results
    df = pd.DataFrame(results)
    write_header = not Path(results_csv_path).exists()
    df.to_csv(results_csv_path, mode='a', index=False, header=write_header)