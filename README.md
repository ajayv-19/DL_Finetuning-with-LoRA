# DL_Finetuning-with-LoRA
This is the project repository for deep learning assignment 2

---

```markdown
# Fine-Tuning RoBERTa with Custom LoRA for AGNEWS Classification

This repository contains a custom Low-Rank Adaptation (LoRA) implementation for fine-tuning the RoBERTa-base model on the AGNEWS text classification task. The goal was to stay within a strict limit of **less than 1 million trainable parameters** while achieving competitive performance.

>This is the final submission for Deep Learning Assignment 2 at NYU Tandon.

## Project Structure

```
├── checkpoints/                  # Saved model checkpoints
├── results/                      # Accuracy & Loss plots for each LoRA config
│   ├── r12_alpha36/
│   ├── r12_alpha48/
│   ├── r12_alpha60/
│   └── ...
├── custom6.ipynb                 # Main training & evaluation notebook
├── custom_lora_sweep.ipynb       # Sweep notebook for alpha/r
├── Lightweight_RoBERTa_PEFT_LORA_FineTuning.ipynb  # Experimental sweep
├── test_unlabelled.pkl           # Provided test set for prediction
├── README.md                     # This file
```

## Key Highlights

- Model: RoBERTa-base (pretrained, frozen)
- LoRA applied to: Query, Value, Output Dense (last 4 encoder layers)
- Dataset: AGNEWS (via HuggingFace + competition-provided test set)
- Trainable Parameters: 934,660 (for r=12, α=36)
- Training Epochs: 4
- Optimizer: AdamW
- Scheduler: Linear Decay
- Final Kaggle Score (Private Leaderboard): **83.45%**

## Dataset

We used the AG News dataset for training and a provided competition test set named `test_unlabelled.pkl` (5.25 MB) for final predictions and submission.  
- Training data was loaded using `datasets.load_dataset('ag_news')`  
- Test file format: Pickle (`.pkl`), contains unlabeled news article samples  
- Labels (during training) correspond to:  
  - `0`: World  
  - `1`: Sports  
  - `2`: Business  
  - `3`: Sci/Tech

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/ajayv-19/DL_Finetuning-with-LoRA.git
cd DL_Finetuning-with-LoRA
git checkout rucith
```

### 2. Install Required Packages

```bash
pip install torch transformers datasets matplotlib
```

For Mac M1/M2 (MPS) users:

```bash
conda install pytorch torchvision torchaudio -c pytorch-nightly
```

or

```bash
pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

Ensure you are on macOS 12.3+ with Xcode CLT installed.

## Running the Code

### Main Training (Single Config):

```bash
Open and run: custom6.ipynb
```

This notebook runs training for a specific LoRA configuration and saves plots in the `results/` folder.

### Parameter Sweep:

Use the `custom_lora_sweep.ipynb` or `Lightweight_RoBERTa_PEFT_LORA_FineTuning.ipynb` notebooks to try multiple `r` and `alpha` combinations.

## Results

Each subfolder inside `results/` contains:

- `accuracy_plot_r12_alpha36.png`
- `loss_plot_r12_alpha36.png`

These show model accuracy and loss (train vs. test) during training.

##  Known Limitations

- No validation set was used — model evaluated only on test set
- No early stopping — likely overfitting
- Padding was fixed at 512 tokens; dynamic padding could improve efficiency


## References

- Hu et al., [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- AGNEWS Dataset: [HuggingFace Datasets](https://huggingface.co/datasets/fancyzhx/ag_news)
- Microsoft LoRA: [GitHub](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)

## Authors

- Ruchit Jathania  
- Ajay Venkatesh  
- Navid Rohan  

Department of Electrical and Computer Engineering  
Tandon School of Engineering, New York University
```

