import numpy as np
import cv2
import pandas as pd
import albumentations as A
from sklearn.model_selection import train_test_split

def get_bounding_box(mask):
    """Get bounding box [x_min, y_min, x_max, y_max] for non-zero area."""
    y_indices, x_indices = np.where(mask > 0)
    if not x_indices.size or not y_indices.size:
        return None

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    # Add random padding (up to 20 pixels)
    H, W = mask.shape
    pad = lambda val, lim, is_min: max(0, val - np.random.randint(0, 20)) if is_min else min(lim, val + np.random.randint(0, 20))
    return [pad(x_min, W, True), pad(y_min, H, True), pad(x_max, W, False), pad(y_max, H, False)]


class CustomDataset:
    def __init__(self, dataset_name, split="test", args=None):
        self.args = args or type('Args', (object,), {})()
        self.split = split
        self.dataset_name = dataset_name

        df_folder = "dataframes"
        train_df = pd.read_csv(f"{df_folder}/{dataset_name}_train.csv").sample(n=min(100, len(pd.read_csv(f"{df_folder}/{dataset_name}_train.csv"))), random_state=getattr(self.args, 'seed', 42))
        test_df = pd.read_csv(f"{df_folder}/{dataset_name}_test.csv").sample(n=min(100, len(pd.read_csv(f"{df_folder}/{dataset_name}_test.csv"))), random_state=getattr(self.args, 'seed', 42))

        if split == "train":
            self.df = train_test_split(train_df, test_size=0.2, random_state=0)[0]
        elif split == "val":
            self.df = train_test_split(train_df, test_size=0.2, random_state=0)[1]
        else:
            self.df = test_df

        self.image_col = "image_paths"
        self.label_col = "mask_paths"
        self.transform = self._build_transforms()

    def _build_transforms(self):
        if self.split != "train":
            return None
        transforms = [
            A.HorizontalFlip(p=0.5) if getattr(self.args, "horizontal_flip", False) else None,
            A.VerticalFlip(p=0.5) if getattr(self.args, "vertical_flip", False) else None,
            A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT_101, p=0.5) if getattr(self.args, "random_rotate", False) else None,
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ]
        return A.Compose([t for t in transforms if t], additional_targets={'mask': 'mask'})

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)
        max_attempts = len(self)

        for attempt in range(max_attempts):
            row = self.df.iloc[idx]
            image_path, mask_path = row[self.image_col], row[self.label_col]

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None or image.size == 0:
                idx = (idx + 1) % len(self)
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.size == 0 or np.sum(mask) == 0:
                idx = (idx + 1) % len(self)
                continue
            break
        else:
            raise RuntimeError("No valid image/mask pair found.")

        # Resize
        r = min(1024 / image.shape[1], 1024 / image.shape[0])
        new_size = (int(image.shape[1] * r), int(image.shape[0] * r))
        image = cv2.resize(image, new_size)
        mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)

        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        image = image.astype(np.float32) / 255.0
        # Process mask and bounding boxes
        H, W = mask.shape
        unique_classes = np.unique(mask)[1:]  # ignore background (0)

        binary_masks, prompts = [], []
        if len(unique_classes) > 0:
            for c in unique_classes:
                class_mask = (mask == c)
                box = get_bounding_box(class_mask)
                if box:
                    prompts.append(np.array(box).reshape(1, 4))
                    binary_masks.append(class_mask)
        else:
            binary_masks.append((mask > 0))
            prompts.append(np.array([0, 0, W, H]).reshape(1, 4))

        binary_masks = [m.astype(np.float32) for m in binary_masks]

        inputs = {
            "pixel_values": np.array(image),
            "ground_truth_mask": np.stack(binary_masks) if len(binary_masks) > 1 else binary_masks[0],
            "input_box": np.concatenate(prompts, axis=0) if len(prompts) > 1 else prompts[0]
        }
        return inputs


if __name__ == "__main__":
    class Args:
        seed = 42
        horizontal_flip = True
        vertical_flip = True
        random_rotate = True

    dataset = CustomDataset("leaf", split="train", args=Args())
    sample = dataset[1]
    for k, v in sample.items():
        print(f"{k}: shape={v.shape}, dtype={v.dtype}")
