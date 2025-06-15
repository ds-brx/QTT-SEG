import os
import argparse
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
from src.utils.utils import ALL_SEEDS

def filter_missing_files(df: pd.DataFrame) -> pd.DataFrame:
    """Remove entries with non-existent image or mask files."""
    df = df[df["image_paths"].apply(os.path.exists)]
    df = df[df["mask_paths"].apply(os.path.exists)]
    return df.reset_index(drop=True)


def load_dataset(name: str, seed: int, max_samples: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads, filters, samples, and renames the dataset for training and testing."""
    train_df = pd.read_csv(f"dataframes/{name}_train.csv")
    test_df = pd.read_csv(f"dataframes/{name}_test.csv")

    train_df = filter_missing_files(train_df)
    test_df = filter_missing_files(test_df)

    train_sample = train_df.sample(n=min(len(train_df), max_samples), random_state=seed)
    test_sample = test_df.sample(n=min(len(test_df), max_samples), random_state=seed)

    return (
        train_sample.rename(columns={'image_paths': 'image', 'mask_paths': 'label'}),
        test_sample.rename(columns={'image_paths': 'image', 'mask_paths': 'label'})
    )

def run_experiment(dataset_name: str, seed: int, time_budget: int, model_dir: str, benchmark_file: str):
    """Trains and evaluates a segmentation model and logs results."""
    print(f"\nRunning experiment: dataset={dataset_name}, seed={seed}, time={time_budget}s")

    train_data, test_data = load_dataset(dataset_name, seed)

    if train_data.empty or test_data.empty:
        print("Skipped: Empty train/test data after filtering.")
        return

    model_path = os.path.join(model_dir, f"{dataset_name}_{time_budget}_{seed}")
    
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
        label="label",
        hyperparameters={"model.sam.checkpoint_name": "facebook/sam-vit-base"},
        path=model_path,
    )

    predictor.fit(train_data=train_data, time_limit=time_budget)

    scores = predictor.evaluate(test_data, metrics=["iou"])
    print("IOU Score:", scores["iou"])

    result = {
        "dataset_name": dataset_name,
        "seed": seed,
        "time_budget": time_budget,
        "score": scores["iou"]
    }

    pd.DataFrame([result]).to_csv(benchmark_file, mode='a', index=False, header=not os.path.exists(benchmark_file))

def main():
    parser = argparse.ArgumentParser(description="QuickTune Semantic Segmentation Benchmark")
    parser.add_argument("--dataset_name", type=str, default="leaf", help="Dataset name")
    parser.add_argument("--time_budget", type=int, default=30, help="Time budget in seconds")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples per split")
    parser.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS, help="List of random seeds")
    parser.add_argument("--model_dir", type=str, default=".",help="Base directory where models will be saved")

    args = parser.parse_args()
    benchmark_file = f"benchmarks/ag_results/{args.dataset_name}_{args.time_budget}_results.csv"
    os.makedirs(os.path.dirname(benchmark_file), exist_ok=True)

    for seed in args.seeds:
        try:
            run_experiment(args.dataset_name, seed, args.time_budget, args.model_dir, benchmark_file)
        except Exception as e:
            print(f"Error in seed {seed}: {str(e)}")

if __name__ == "__main__":
    main()
