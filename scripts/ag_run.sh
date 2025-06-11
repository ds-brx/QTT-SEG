#!/bin/bash
#SBATCH -p mlhiwidlc_gpu-rtx2080
#SBATCH -t 0-01:00
#SBATCH --gpus 1
#SBATCH -o logs/%x.%N.%A.%a.out
#SBATCH -e logs/%x.%N.%A.%a.errors
#SBATCH -J QTT-SEG
#SBATCH --mail-type=END,FAIL
#SBATCH -a 1-3

echo "Workingdir: $PWD"
echo "Started at $(date)"
echo "Running job $SLURM_JOB_NAME with task ID $SLURM_ARRAY_TASK_ID"

export PYTHONUNBUFFERED=1

BUDGET=(60 120 180)
DATASET='eyes'  # single value for all

BUDGET=${BUDGET[$SLURM_ARRAY_TASK_ID-1]}

python -m benchmarks.ag_test_binary --dataset_name "$DATASET" --time_budget "$BUDGET"

echo "DONE with dataset $DATASET"
echo "Finished at $(date)"
