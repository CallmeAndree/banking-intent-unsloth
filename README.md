# Banking Intent Classification (Unsloth)

🎥 **Video Demonstration**: [Watch on Google Drive](https://drive.google.com/file/d/1-P8XpWR6N6alfPPs2oWvD6VtqtQHFUxK/view)

Fine-tuning and inference pipeline for banking intent classification. This project leverages the **Unsloth** framework with **Qwen2.5** to efficiently classify customer banking queries into 77 specific intents.

## Features

- **Efficient Fine-Tuning**: Optimized VRAM footprint and faster training cycles using Unsloth.
- **Batched Generation**: Performant batched GPU inference for evaluating large datasets.
- **Config-Driven**: Easily configure datasets, hyperparameters, and models through YAML (`configs/train.yml`, `configs/inference.yml`).
- **Plug-and-Play Pipelines**: Pre-configured bash scripts for effortless execution on local servers, Colab, or Kaggle.

---

## 1. Setup Environment

A CUDA-enabled device (Nvidia GPU) is strictly recommended for training and inference.

**Clone the repository:**
```bash
git clone https://github.com/your-username/banking-intent-unsloth.git
cd banking-intent-unsloth
```

**Install dependencies:**
This project requires Unsloth, HuggingFace Datasets, TRL, Pandas, and PyYAML.
```bash
pip install -r requirements.txt
```
*(Optional)* You can also rely on the bash scripts (`train.sh`, `inference.sh`) which will automatically attempt to install these dependencies prior to execution.

---

## 2. Download and Prepare Data

The scripts are hardcoded to read datasets from the `sample_data/` directory by default.
1. Download your dataset (e.g., Banking77) as a CSV format.
2. Put the train and test CSVs into the `sample_data/` folder:
    - Training data: `sample_data/train.csv`
    - Testing data: `sample_data/test.csv`

If you are using raw, unprocessed data, you can parse it by running the preprocessing script:
```bash
python scripts/preprocess_data.py --input ./raw_data/raw_banking77.csv --output ./sample_data
```
Alternatively, uncomment the Phase 2 block inside `train.sh` to run this step automatically during the training pipeline.

---

## 3. Train the Model

Training parameters (batch size, max tokens, learning rate, LoRA configuration, etc.) are dictated by `configs/train.yml`.

**Start the Training Pipeline:**
```bash
bash train.sh
```

**Customizing Input/Output Paths via Environment Variables:**
If you want to train on a custom dataset or save checkpoints to an external drive (like in Kaggle `/kaggle/working`), inject variables before running:
```bash
DATA_DIR=/path/to/custom_data OUTPUT_DIR=/kaggle/working/checkpoints bash train.sh
```
Upon completion, model checkpoints and standard LoRA adapters will be outputted into the `./checkpoints/` directory.

---

## 4. Run the Model (Inference)

You can run the model against an entire benchmark file (Batch Inference) or use the Python module to infer single messages on the fly. 

### Method A: Batch Inference (Evaluate Dataset)
Modify `configs/inference.yml` to reflect your checkpoint locations and hyperparameter choices (temperature, sample mechanics).

**Execute the Batch Script:**
```bash
bash inference.sh
```

**Override Environments (Example for Kaggle):**
```bash
MODEL_CHECKPOINT=/kaggle/working/checkpoints/final_checkpoint \
TEST_DATA_PATH=./sample_data/test.csv \
OUTPUT_PATH=./results/predictions.csv \
bash inference.sh
```
Predictions will be written to a new CSV file (`./results/predictions.csv`). If your `test.csv` includes ground-truth labels, the script will automatically calculate accuracy metrics.

### Method B: Single Message Python API
If you are integrating this model into a larger system (like an API or UI client), utilize the `IntentClassification` wrapper for real-time predictions.

```python
from scripts.inference import IntentClassification

# Initialize the classifier (Loads model & tokenizer automatically using configs)
clf = IntentClassification("configs/inference.yml")

# Run inference
message = "I lost my credit card while travelling, how do I freeze my account?"
predicted_intent = clf(message) 

print(f"Message: {message}")
print(f"Predicted Intent: {predicted_intent}")
```
