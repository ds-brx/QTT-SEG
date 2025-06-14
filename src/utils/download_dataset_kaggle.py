import os
import argparse
import kagglehub

def download_dataset_with_kagglehub(dataset_slug, cache_dir=None):
    """
    Download a Kaggle dataset using kagglehub, optionally specifying cache directory.

    Args:
        dataset_slug (str): Kaggle dataset slug (e.g. 'username/dataset-name')
        cache_dir (str or None): Path to cache directory for kagglehub downloads.
    """
    if cache_dir:
        os.environ["KAGGLEHUB_CACHE"] = cache_dir
        print(f"Using custom cache directory: {cache_dir}")

    print(f"Downloading dataset: {dataset_slug}")
    path = kagglehub.dataset_download(dataset_slug)
    print("Download complete.")
    print("Dataset is available at:", path)

def main():
    parser = argparse.ArgumentParser(description="Download Kaggle dataset using kagglehub.")
    parser.add_argument("--dataset_slug", type=str, help="Kaggle dataset slug (username/dataset-name)")
    parser.add_argument(
        "--cache_dir", "-c",
        default=None,
        help="Custom cache directory for kagglehub downloads (default: ~/.cache/kagglehub)"
    )
    args = parser.parse_args()

    download_dataset_with_kagglehub(args.dataset_slug, args.cache_dir)

if __name__ == "__main__":
    main()
