import pandas as pd
import os
import argparse
from pathlib import Path

from qtt import QuickTuner, QuickOptimizer
from qtt.predictors import PerfPredictor, CostPredictor
from src.finetune_wrapper.finetune_script import finetune_script, get_meta_data
from src.sam2_process.sam2_test import test
from src.utils.utils import get_config_space, get_parser, ALL_SEEDS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quicktune Segmentation Arguments")
    parser.add_argument("--dataset_name", type=str, default="leaf", help="Dataset Name")
    parser.add_argument("--time_budget", type=int, default=30, help="Time Budget in Seconds")
    parser.add_argument("--output_dir", type=Path, default = Path("."), help = "Output Directory")
    parser.add_argument("--train_predictors", action="store_true", help="Train Cost and Performance Predictors")
    parser.add_argument("--setup_configs", type=int, default=128, help="Time Budget in Seconds")
    
    args = parser.parse_args()
    
    results_csv_path = os.path.join(args.output_dir, "results", f"{args.dataset_name}_{args.time_budget}_results.csv")
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    meta_file_path = "src/finetune_wrapper/finetuning_results_sam.csv"
    perf_path = os.path.join(args.output_dir, "PerfPredictor")
    cost_path = os.path.join(args.output_dir, "CostPredictor")   
    output_dir = os.path.join(args.output_dir, "logs", f"{args.dataset_name}_{args.time_budget}")
    
    meta_df = pd.read_csv(meta_file_path)
    meta_df = meta_df[meta_df["dataset"] != args.dataset_name]

    cost = meta_df["cost"]
    curve = meta_df.filter(regex=r'^epoch_\d{1,2}_iou$')
    config = meta_df.drop(columns=["cost", "dataset"] + curve.columns.tolist())
    X = config
    y = curve.values

    fit_params = {
        "batch_size": 4,
    }

    if args.train_predictors:
        perf_predictor = PerfPredictor(fit_params).fit(X, y)
        perf_predictor.save(perf_path)
        
        y = cost.values.reshape(-1,1)
        cost_predictor = CostPredictor(fit_params).fit(X, y)
        cost_predictor.save(cost_path)

    else:
        perf_predictor = PerfPredictor().load(perf_path)
        cost_predictor = CostPredictor().load(cost_path)

    seeds = ALL_SEEDS
    for s in range(len(seeds)):
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
            path= os.path.join(output_dir, "opt"),
            verbosity = -1 
        )

        task_info = {
            "dataset_name": args.dataset_name,
            "output_path": os.path.join(output_dir,"config_checkpoints"),
            "seed" : seeds[s]
        }

        metafeat = get_meta_data(args.dataset_name)

        print("Optimiser Setup")
        optimizer.setup(args.setup_configs, metafeat)

        tuner = QuickTuner(
            optimizer = optimizer,
            f = finetune_script,
            path = os.path.join(output_dir, "tuner")
        )

        traj, runtime, history = tuner.run(fevals=100, trial_info=task_info, time_budget=args.time_budget)
        config_id, config, score, budget, cost, info = tuner.get_incumbent()

        args_list = []
        for key, value in config.items():
            args_list.extend([f"--{key}", str(value)])
        args_list.extend([f"--dataset_name", args.dataset_name])
        parser = get_parser()
        tune_args, _ = parser.parse_known_args(args_list)
        
        test_score = test(
            split = "test",
            predicted_model=None, 
            predicted_model_path = os.path.join(info["path"],"sam2model.torch"),
            args=tune_args,
            save_images = True)
        
        df = pd.DataFrame([
            {
                'dataset': entry['dataset'],
                'config_id': entry['config-id'],
            }
            for entry in history
        ])
        df["score"] = traj
        df["step"] = runtime
        history_dir = os.path.join(output_dir, "qtt_history_logs")
        os.makedirs(history_dir, exist_ok=True)
        df.to_csv(os.path.join(history_dir, f"{seeds[s]}.csv"))

        result = {
            "DATASET": args.dataset_name,
            "SEED" : seeds[s],
            "TIME_BUDGET": args.time_budget,
            "QTT_TUNING_SCORE": test_score,
            "QTT_CONFIG": config
        }
        if os.path.exists(results_csv_path):
            pd.DataFrame([result]).to_csv(results_csv_path, mode='a', index=False, header=False)
        else:
            pd.DataFrame([result]).to_csv(results_csv_path, index=False)