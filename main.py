import pandas as pd
import os
import argparse
from pathlib import Path
import json
from typing import Dict, Any
import numpy as np
from qtt import QuickTuner, QuickOptimizer
from qtt.predictors import PerfPredictor, CostPredictor
from src.finetune_wrapper.finetune_script import finetune_script, get_meta_data
from src.sam2_process.sam2_test import test
from src.utils.utils import get_config_space, get_parser, ALL_SEEDS

# Constants
DEFAULT_OUTPUT_DIR = Path(".")
DEFAULT_META_FILE = "src/finetune_wrapper/finetuning_results_sam.csv"
DEFAULT_TIME_BUDGET = 30  # seconds
DEFAULT_SETUP_CONFIGS = 128

def setup_paths(args: argparse.Namespace) -> Dict[str, str]:
    """Initialize and create all required directories."""
    paths = {
        'base_output': args.output_dir,
        'perf_predictor': args.output_dir / "PerfPredictor",
        'cost_predictor': args.output_dir / "CostPredictor",
        'log_dir': args.output_dir / "logs" / f"{args.dataset_name}_{args.time_budget}",
    }
    
    paths['history_dir'] = paths['log_dir'] / "qtt_history_logs"
    paths['config_checkpoints'] = paths['log_dir'] / "config_checkpoints"
    paths['opt_dir'] = paths['log_dir'] / "opt"
    paths['tuner_dir'] = paths['log_dir'] / "tuner"
    
    # Create directories
    for path in paths.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    paths['results_csv'] = args.output_dir / "results.csv"
    return paths

def load_and_prepare_meta_data(meta_file_path: str, dataset_name: str) -> tuple:
    """Load and prepare meta data for predictors."""
    meta_df = pd.read_csv(meta_file_path)
    meta_df = meta_df[meta_df["dataset"] != dataset_name]
    
    cost = meta_df["cost"]
    curve = meta_df.filter(regex=r'^epoch_\d{1,2}_iou$')
    config = meta_df.drop(columns=["cost", "dataset"] + curve.columns.tolist())
    
    return config, curve.values, cost.values.reshape(-1, 1)

def train_or_load_predictors(
    train_predictors: bool,
    X: pd.DataFrame,
    y_perf: np.ndarray,
    y_cost: np.ndarray,
    perf_path: str,
    cost_path: str,
    fit_params: Dict[str, Any]
) -> tuple:
    """Either train new predictors or load existing ones."""
    if train_predictors:
        perf_predictor = PerfPredictor(fit_params).fit(X, y_perf)
        perf_predictor.save(perf_path)
        
        cost_predictor = CostPredictor(fit_params).fit(X, y_cost)
        cost_predictor.save(cost_path)
    else:
        perf_predictor = PerfPredictor().load(perf_path)
        cost_predictor = CostPredictor().load(cost_path)
    
    return perf_predictor, cost_predictor

def run_optimization(
    optimizer: QuickOptimizer,
    dataset_name: str,
    output_dir: Path,
    seed: int,
    time_budget: int,
    setup_configs: int
) -> tuple:
    """Run the optimization process for a single seed.
    Returns:
        tuple: (traj, runtime, history, config, info)
    """
    task_info = {
        "dataset_name": dataset_name,
        "output_path": output_dir / "config_checkpoints",
        "seed": seed
    }

    metafeat = get_meta_data(dataset_name)
    optimizer.setup(setup_configs, metafeat)

    tuner = QuickTuner(
        optimizer=optimizer,
        f=finetune_script,
        path=str(output_dir / "tuner")
    )
    
    # Run optimization
    traj, runtime, history = tuner.run(fevals=100, trial_info=task_info, time_budget=time_budget)
    
    # Get best configuration
    config_id, config, score, budget, cost, info = tuner.get_incumbent()
    
    # Return all needed variables
    return traj, runtime, history, config, info

def save_results(
    results_csv_path: Path,
    dataset_name: str,
    seed: int,
    time_budget: int,
    test_score: float,
    config: Dict[str, Any],
    history: list,
    traj: list,
    runtime: list,
    history_dir: Path
):
    """Save all results to appropriate files."""
    # Save history
    history_df = pd.DataFrame([{
        'dataset': entry['dataset'],
        'config_id': entry['config-id'],
        'score': score,
        'step': step
    } for entry, score, step in zip(history, traj, runtime)])
    
    history_df.to_csv(history_dir / f"{seed}.csv", index=False)

    # Save main results
    result = {
        "DATASET": dataset_name,
        "SEED": seed,
        "TIME_BUDGET": time_budget,
        "QTT_TUNING_SCORE": test_score,
        "QTT_CONFIG": json.dumps(config)  # Serialize config dict
    }

    write_header = not results_csv_path.exists()
    pd.DataFrame([result]).to_csv(
        str(results_csv_path),
        mode='a',
        index=False,
        header=write_header
    )

def main():
    parser = argparse.ArgumentParser(description="Quicktune Segmentation Arguments")
    parser.add_argument("--dataset_name", type=str, default="leaf", help="Dataset Name")
    parser.add_argument("--time_budget", type=int, default=DEFAULT_TIME_BUDGET, help="Time Budget in Seconds")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output Directory")
    parser.add_argument("--train_predictors", action="store_true", help="Train Cost and Performance Predictors")
    parser.add_argument("--setup_configs", type=int, default=DEFAULT_SETUP_CONFIGS, help="Number of initial configs")
    
    args = parser.parse_args()
    
    # Setup paths and directories
    paths = setup_paths(args)
    
    # Prepare meta data
    X, y_perf, y_cost = load_and_prepare_meta_data(DEFAULT_META_FILE, args.dataset_name)
    
    # Initialize predictors
    fit_params = {"batch_size": 4}
    perf_predictor, cost_predictor = train_or_load_predictors(
        args.train_predictors,
        X, y_perf, y_cost,
        str(paths['perf_predictor']),
        str(paths['cost_predictor']),
        fit_params
    )

    # Run optimization for each seed
    for seed in ALL_SEEDS:
        cs = get_config_space()
        optimizer = QuickOptimizer(
            cs,
            max_fidelity=10,
            perf_predictor=perf_predictor,
            cost_predictor=cost_predictor,
            cost_aware=True,
            cost_factor=1.0,
            acq_fn="ei",
            patience=5,
            tol=0.001,
            refit=False,
            path=str(paths['opt_dir']),
            verbosity=-1 
        )

        # Run optimization
        optimization_result = run_optimization(
            optimizer,
            args.dataset_name,
            paths['log_dir'],
            seed,
            args.time_budget,
            args.setup_configs
        )
        if optimization_result is None or any(v is None for v in optimization_result):
            print(f"Skipping seed {seed} due to failed optimization")
            continue  # Skip to next seed
            
        traj, runtime, history, config, info = optimization_result
        
        # Prepare arguments for testing
        args_list = [f"--{k}={v}" for k, v in config.items()]
        args_list.extend([f"--dataset_name={args.dataset_name}"])
        tune_args = get_parser().parse_args(args_list)
        tune_args.output_dir = paths["base_output"]
        
        # Evaluate best model
        test_score = test(
            split="test",
            predicted_model=None, 
            predicted_model_path=str(Path(info["path"]) / "sam2model.torch"),
            args=tune_args,
            save_images=True
        )

        # Save results
        save_results(
            paths['results_csv'],
            args.dataset_name,
            seed,
            args.time_budget,
            test_score,
            config,
            history,
            traj,
            runtime,
            paths['history_dir']
        )

if __name__ == "__main__":
    main()
