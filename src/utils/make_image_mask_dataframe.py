import os
import argparse
import pandas as pd
from tqdm import tqdm
import re

import os
import re
import pandas as pd
from tqdm import tqdm

def extract_id(filename):
    """
    Extracts the first numeric sequence from a filename.
    
    Example:
        'image_870705_sat_13.jpg' -> '870705'
        '848728_mask_88.png' -> '848728'
    
    This numeric ID is used as a key to match images and masks.
    """
    match = re.findall(r'\d+', filename)
    return match[0] if match else None

def create_image_mask_dataframe_dynamic(image_dir, mask_dir):
    """
    Creates a DataFrame that pairs image and mask files by matching numeric IDs extracted from filenames.

    Assumes:
        - Both image and mask filenames contain at least one numeric ID.
        - The first numeric ID in each filename is used to establish correspondence.
        - For example: 'img_870.jpg' matches 'mask_870.png'.

    Args:
        image_dir (str): Directory containing image files.
        mask_dir (str): Directory containing mask files.

    Returns:
        pd.DataFrame: DataFrame with 'image_path' and 'mask_path' columns.
    """
    # Create a dictionary: {extracted_id: full_image_path}
    image_files = {
        extract_id(f): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f))
    }

    # Create a dictionary: {extracted_id: full_mask_path}
    mask_files = {
        extract_id(f): os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir)
        if os.path.isfile(os.path.join(mask_dir, f))
    }

    data = []

    print("Matching image and mask files by extracted IDs...")
    for id_ in tqdm(image_files.keys()):
        # If a matching mask with the same ID exists, pair them
        if id_ and id_ in mask_files:
            data.append({
                "image_paths": image_files[id_],
                "mask_paths": mask_files[id_]
            })

    return pd.DataFrame(data)

def create_image_mask_dataframe(image_dir, mask_dir):
    """
    Creates a DataFrame pairing image and mask files by matching filenames (ignoring extensions).
    
    Assumption:
        - image_dir and mask_dir contain files with matching base names (e.g. '123.jpg' ↔ '123.png').
        - Filenames must match exactly (excluding extensions).
    
    Args:
        image_dir (str): Directory containing image files.
        mask_dir (str): Directory containing mask files.
    
    Returns:
        pd.DataFrame: DataFrame with 'image_path' and 'mask_path'.
    """
    image_files = {os.path.splitext(f)[0]: os.path.join(image_dir, f)
                   for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))}
    mask_files = {os.path.splitext(f)[0]: os.path.join(mask_dir, f)
                  for f in os.listdir(mask_dir) if os.path.isfile(os.path.join(mask_dir, f))}

    data = []

    print("Matching image and mask files by base filename...")
    for base_name in tqdm(image_files.keys()):
        if base_name in mask_files:
            data.append({
                "image_paths": image_files[base_name],
                "mask_paths": mask_files[base_name]
            })

    return pd.DataFrame(data)


from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser(
        description="Pairs images and masks by matching filenames.\n"
                    "Assumes one-to-one filename match between --images and --masks directories."
    )
    parser.add_argument("--images", required=True, help="Path to image directory")
    parser.add_argument("--masks", required=True, help="Path to mask directory")
    parser.add_argument("--output", help="Path to save CSVs (optional directory)")
    parser.add_argument("--name", help="Base name for output CSVs", default="paired_data")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of data to use for testing (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    df = create_image_mask_dataframe_dynamic(args.images, args.masks)
    print(f"Found {len(df)} image–mask pairs.")

    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed)
    print(f"Split into {len(train_df)} train and {len(test_df)} test pairs.")

    # Define output directory
    output_dir = args.output or "dataframes"
    os.makedirs(output_dir, exist_ok=True)

    # Format filenames
    train_csv = os.path.join(output_dir, f"{args.name}_train.csv")
    test_csv = os.path.join(output_dir, f"{args.name}_test.csv")

    # Save CSVs
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Saved train CSV to {train_csv}")
    print(f"Saved test CSV to {test_csv}")


if __name__ == "__main__":
    main()
