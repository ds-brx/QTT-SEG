# 🧪 QuickTune Tool for Image Segmentation (QTT-SEG)

QTT-SEG adapts the **[Quicktune Tool (QTT)](https://github.com/automl/quicktunetool)** for efficient hyperparameter optimization in **image segmentation**. It uses meta-learned predictors to estimate model performance and fine-tuning cost, enabling faster and smarter AutoML.

Foundation models, such as SAM (Segment-Anything-Model), have demonstrated strong zero-shot image segmentation capabilities; however, they often underperform on domain-specific tasks. Fine-tuning these foundation models typically demands extensive manual effort and expert knowledge. In this work, we explore the use of Quick-Tune, a meta-learning-based hyperparameter optimization framework, to automate and accelerate the fine-tuning of SAM for image segmentation. (Quick-Tune for segmentation) QTT-SEG predicts performant configurations using meta-learned cost and performance models, efficiently navigating a vast search space of over 200 million configurations. We evaluate QTT-SEG on eight binary and five multiclass segmentation datasets under constrained time budgets. Our results show that QTT-SEG significantly improves over SAM’s zero-shot performance and outperforms AutoGluon Multimodal, a strong AutoML baseline, on most binary tasks under three minutes. On multiclass datasets, QTT-SEG delivers consistent gains over zero-shot SAM. These results demonstrate the potential of meta-learning to automate fine-tuning for segmentation models across diverse and specialized domains.

---

## 📦 Installation & Setup

```bash
# Download the ZIP from the anonymous link and extract it:
# https://anonymous.4open.science/r/QTT-SEG-30BE/

cd QTT-SEG-30BE

# Create and activate environment
conda create -n qtt_seg python=3.11 -y
conda activate qtt_seg

# Install required packages
pip install -r requirements.txt

# Install SAM2 from source
mkdir third_party && cd third_party
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
cd ../..
```

---

## 🧾 Step 1: Download and Prepare Dataset Splits

Before running the tuning pipeline, you must prepare `train.csv` and `test.csv` containing image-mask pairs for the target dataset. Use the helper script:

```bash
python src/utils/setup_configs.py \
  --dataset aysendegerli/qatacov19-dataset \
  --image-dir QaTa-COV19/QaTa-COV19-v1/Images \
  --mask-dir QaTa-COV19/QaTa-COV19-v1/Ground-truths \
  --mask-prefix mask_ \
  --output-dir dataframes \
  --name covid \
  --test-size 0.2
```

### Optional: Set Kaggle Credentials

If your system isn't already configured for Kaggle, provide credentials inline:

```bash
  --kaggle-username YOUR_USERNAME \
  --kaggle-key YOUR_KEY
```

This will:
- Download the dataset via [KaggleHub](https://github.com/KaggleHub/kagglehub)
- Match each image to its mask (by prefix)
- Split into training/testing sets
- Save as:  
  - `dataframes/covid_train.csv`  
  - `dataframes/covid_test.csv`

---

## 🚀 Step 2: Run QuickTune for Segmentation

```bash
python main.py \
  --dataset_name covid \
  --time_budget 60 \
  --output_dir ./results \
  --train_predictors \
  --setup_configs 128
```

### 🔧 Arguments

| Argument            | Description                                      | Default    |
|---------------------|--------------------------------------------------|------------|
| `--dataset_name`     | Name of the segmentation dataset                | `"leaf"`   |
| `--time_budget`      | Time budget in seconds for tuning               | `30`       |
| `--output_dir`       | Output folder for logs and results              | `"."`      |
| `--train_predictors` | Train cost/performance predictors               | `False`    |
| `--setup_configs`    | Number of random configs to initialize tuner    | `128`      |

---

## 📊 Outputs

After running, you will find:
- 📂 `results/`: performance CSV logs (`*_results.csv`)
- 📂 `logs/`: tuning history and checkpoints
- 📂 `CostPredictor/` and `PerfPredictor/`: trained predictors

Each experiment records:
- Best configuration found
- Final test IOU score
- Tuning trajectory and runtime

---

## 📈 Visual Results

_You can include visualizations or example masks here._

---

## 🤝 Contributing
Currently, this repository is under restricted access and contributions via pull requests are not yet enabled.

If you are interested in contributing or collaborating on this project, please consider the following options:

- **Contact the maintainers**: Reach out via email or other communication channels (to be specified) to discuss potential collaboration.
- **Watch this space**: The repository will be made public soon, and contribution guidelines will be added at that time.
- **Feature requests and issues**: If you have ideas or encounter problems, please open an issue once the repository is public, or contact the maintainers directly for feedback.

Thank you for your interest and support!
