import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import kagglehub

def download_dataset(dataset_name: str) -> str:
    """
    Downloads a dataset from Kaggle using kagglehub and returns the local path.
    """
    print(f"Downloading dataset: {dataset_name}")
    dataset_path = kagglehub.dataset_download(dataset_name)
    print(f"Dataset downloaded to: {dataset_path}")
    return dataset_path

def create_image_mask_dataframe(image_dir: str, mask_dir: str, mask_prefix: str) -> pd.DataFrame:
    """
    Creates a DataFrame with 'image_paths' and 'mask_paths' for valid image-mask pairs.
    """
    image_files = []
    mask_files = []

    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            mask_path = os.path.join(mask_dir, f"{mask_prefix}{filename}")
            if os.path.exists(mask_path):
                image_files.append(img_path)
                mask_files.append(mask_path)

    df = pd.DataFrame({'image_paths': image_files, 'mask_paths': mask_files})
    print(f"Found {len(df)} valid image-mask pairs.")
    return df

def save_splits(df: pd.DataFrame, output_dir: str, name: str, test_size: float):
    """
    Saves the train/test split as CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    train_path = os.path.join(output_dir, f"{name}_train.csv")
    test_path = os.path.join(output_dir, f"{name}_test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train CSV saved to: {train_path}")
    print(f"Test CSV saved to: {test_path}")

def main():
    parser = argparse.ArgumentParser(description="Download Kaggle dataset and prepare image-mask dataframe.")
    
    parser.add_argument("--dataset", required=True, help="Kaggle dataset name (e.g., 'aysendegerli/qatacov19-dataset')")
    parser.add_argument("--kaggle-username", help="Kaggle username (optional if in environment)")
    parser.add_argument("--kaggle-key", help="Kaggle API key (optional if in environment)")
    parser.add_argument("--image-dir", default="QaTa-COV19/QaTa-COV19-v1/Images", help="Relative path to images in dataset")
    parser.add_argument("--mask-dir", default="QaTa-COV19/QaTa-COV19-v1/Ground-truths", help="Relative path to masks in dataset")
    parser.add_argument("--mask-prefix", default="mask_", help="Prefix used in mask filenames (e.g., 'mask_')")
    parser.add_argument("--output-dir", default="dataframes", help="Directory to save CSV outputs")
    parser.add_argument("--name", default="dataset", help="Base name for output CSV files")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size (e.g., 0.2 for 20%%)")

    args = parser.parse_args()

    # Set environment variables if provided
    if args.kaggle_username and args.kaggle_key:
        os.environ["KAGGLE_USERNAME"] = args.kaggle_username
        os.environ["KAGGLE_KEY"] = args.kaggle_key

    dataset_root = download_dataset(args.dataset)

    image_dir = os.path.join(dataset_root, args.image_dir)
    mask_dir = os.path.join(dataset_root, args.mask_dir)

    df = create_image_mask_dataframe(image_dir, mask_dir, args.mask_prefix)
    save_splits(df, args.output_dir, args.name, args.test_size)

if __name__ == "__main__":
    main()
