# ============================================================
# Fine-tuning Qwen/Qwen2.5-3B with Unsloth
# Platform: Google Colab / Kaggle
# Task: Banking Intent Classification (banking77)
# ============================================================
#
# HYPERPARAMETERS:
# - max_seq_length     : 256     (max token length per sample)
# - per_device_train_batch_size : 16
# - gradient_accumulation_steps: 4  (effective batch = 64)
# - learning_rate      : 2e-4
# - num_train_epochs   : 3
# - optimizer          : adamw_8bit (memory-efficient)
# - lr_scheduler_type  : cosine
# - warmup_ratio       : 0.05
# - weight_decay       : 0.01    (L2 regularization)
# - LoRA r             : 16
# - LoRA alpha         : 16
# - LoRA dropout       : 0.05    (regularization)
# ============================================================

# ------------------------------------------------------------------
# Step 0: Install dependencies (run this cell first on Colab/Kaggle)
# ------------------------------------------------------------------
# !pip install unsloth datasets pandas -q

import os
import pandas as pd
import torch
from datasets import Dataset

# ------------------------------------------------------------------
# Step 1: Configuration — Edit paths and hyperparameters here
# ------------------------------------------------------------------

# Resolve sample_data relative to this script: scripts/ → .. → sample_data/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_DIR = os.path.join(_SCRIPT_DIR, "..", "sample_data")
_DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "..", "checkpoints", "qwen2.5-banking77")

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────
    # Automatically resolved to <repo_root>/sample_data on any platform.
    # Override here if running on Colab/Kaggle with a different upload path.
    "data_dir": os.environ.get("DATA_DIR", _DEFAULT_DATA_DIR),
    "output_dir": os.environ.get("OUTPUT_DIR", _DEFAULT_OUTPUT_DIR),

    # ── Model ──────────────────────────────────────────────────────
    "model_name": "Qwen/Qwen2.5-3B",
    "max_seq_length": 256,

    # ── LoRA ───────────────────────────────────────────────────────
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "lora_target_modules": [
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj",
    ],

    # ── Training ───────────────────────────────────────────────────
    "per_device_train_batch_size": 16,
    "gradient_accumulation_steps": 4,   # effective batch = 64
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,              # used to compute warmup_steps at runtime
    "weight_decay": 0.01,               # L2 regularization
    "fp16": False,                        # overridden at runtime inside main()
    "bf16": False,                        # overridden at runtime inside main()
    "logging_steps": 50,
    "save_strategy": "epoch",
    "seed": 42,
}

# ------------------------------------------------------------------
# Step 2: Label mapping (77 banking intents)
# ------------------------------------------------------------------
id2label = {
    0: "activate_my_card", 1: "age_limit", 2: "apple_pay_or_google_pay", 3: "atm_support",
    4: "automatic_top_up", 5: "balance_not_updated_after_bank_transfer",
    6: "balance_not_updated_after_cheque_or_cash_deposit", 7: "beneficiary_not_allowed",
    8: "cancel_transfer", 9: "card_about_to_expire", 10: "card_acceptance",
    11: "card_arrival", 12: "card_delivery_estimate", 13: "card_linking",
    14: "card_not_working", 15: "card_payment_fee_charged", 16: "card_payment_not_recognised",
    17: "card_payment_wrong_exchange_rate", 18: "card_swallowed", 19: "cash_withdrawal_charge",
    20: "cash_withdrawal_not_recognised", 21: "change_pin", 22: "compromised_card",
    23: "contactless_not_working", 24: "country_support", 25: "declined_card_payment",
    26: "declined_cash_withdrawal", 27: "declined_transfer", 28: "direct_debit_payment_not_recognised",
    29: "disposable_card_limits", 30: "edit_personal_details", 31: "exchange_charge",
    32: "exchange_rate", 33: "exchange_via_app", 34: "extra_charge_on_statement",
    35: "failed_transfer", 36: "fiat_currency_support", 37: "get_disposable_virtual_card",
    38: "get_physical_card", 39: "getting_spare_card", 40: "getting_virtual_card",
    41: "lost_or_stolen_card", 42: "lost_or_stolen_phone", 43: "order_physical_card",
    44: "passcode_forgotten", 45: "pending_card_payment", 46: "pending_cash_withdrawal",
    47: "pending_top_up", 48: "pending_transfer", 49: "pin_blocked", 50: "receiving_money",
    51: "Refund_not_showing_up", 52: "request_refund", 53: "reverted_card_payment?",
    54: "supported_cards_and_currencies", 55: "terminate_account",
    56: "top_up_by_bank_transfer_charge", 57: "top_up_by_card_charge",
    58: "top_up_by_cash_or_cheque", 59: "top_up_failed", 60: "top_up_limits",
    61: "top_up_reverted", 62: "topping_up_by_card", 63: "transaction_charged_twice",
    64: "transfer_fee_charged", 65: "transfer_into_account",
    66: "transfer_not_received_by_recipient", 67: "transfer_timing",
    68: "unable_to_verify_identity", 69: "verify_my_identity", 70: "verify_source_of_funds",
    71: "verify_top_up", 72: "virtual_card_not_working", 73: "visa_or_mastercard",
    74: "why_verify_identity", 75: "wrong_amount_of_cash_received",
    76: "wrong_exchange_rate_for_cash_withdrawal",
}
label2id = {v: k for k, v in id2label.items()}
NUM_LABELS = len(id2label)

# ------------------------------------------------------------------
# Step 3: Load and prepare data
# ------------------------------------------------------------------
def load_data(data_dir: str):
    train_path = os.path.join(data_dir, "train.csv")
    test_path  = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    # Ensure label_name column exists
    if "label_name" not in train_df.columns:
        train_df["label_name"] = train_df["label"].map(id2label)
    if "label_name" not in test_df.columns:
        test_df["label_name"] = test_df["label"].map(id2label)

    return train_df, test_df


def format_prompt(text: str, label_name: str | None = None) -> str:
    """
    Format each sample into an instruction-following prompt.
    During training, the response (label) is included.
    During inference, omit label_name.
    """
    instruction = (
        "You are a banking assistant. "
        "Classify the following customer query into one of the 77 banking intent categories.\n\n"
        f"Query: {text}\n\n"
        "Intent:"
    )
    if label_name is not None:
        return instruction + f" {label_name}"
    return instruction


def prepare_dataset(df: pd.DataFrame) -> Dataset:
    records = []
    for _, row in df.iterrows():
        records.append({
            "text": format_prompt(str(row["text"]), str(row["label_name"])),
        })
    return Dataset.from_list(records)


# ------------------------------------------------------------------
# Step 4: Main training entry point
# ------------------------------------------------------------------
def main():
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    # Runtime detection: must run here (not at import time) so it reads the
    # actual GPU on the execution platform (Kaggle T4, Colab A100, etc.)
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_ok = torch.cuda.is_available() and not bf16_ok
    CONFIG["bf16"] = bf16_ok
    CONFIG["fp16"] = fp16_ok
    print(f"Precision: {'bf16' if bf16_ok else 'fp16' if fp16_ok else 'cpu'}")

    # 4-A: Load model with Unsloth 4-bit quantization
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        dtype=None,         # auto-detect (bf16 if supported)
        load_in_4bit=True,  # 4-bit quantization to reduce VRAM usage
    )

    # 4-B: Apply LoRA adapters (PEFT)
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        target_modules=CONFIG["lora_target_modules"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # saves memory on long contexts
        random_state=CONFIG["seed"],
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()

    # 4-C: Prepare datasets
    print("Loading data...")
    train_df, test_df = load_data(CONFIG["data_dir"])
    train_dataset = prepare_dataset(train_df)
    eval_dataset  = prepare_dataset(test_df)
    print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    # 4-D: SFTConfig = TrainingArguments + SFT-specific params (TRL >= 0.12)
    training_args = SFTConfig(
        output_dir=CONFIG["output_dir"],

        # ── SFT-specific params (must be in SFTConfig, not SFTTrainer) ──
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        dataset_num_proc=2,
        packing=False,                  # disable packing for classification prompts

        # ── Training ────────────────────────────────────────────────
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        warmup_steps=int(
            len(train_dataset)
            / CONFIG["per_device_train_batch_size"]
            / CONFIG["gradient_accumulation_steps"]
            * CONFIG["num_train_epochs"]
            * CONFIG["warmup_ratio"]
        ),
        weight_decay=CONFIG["weight_decay"],
        fp16=CONFIG["fp16"],
        bf16=CONFIG["bf16"],
        logging_steps=CONFIG["logging_steps"],
        save_strategy=CONFIG["save_strategy"],
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        optim="adamw_8bit",
        seed=CONFIG["seed"],
        report_to="none",
        dataloader_num_workers=2,
    )

    # 4-E: SFTTrainer — SFT params are now in SFTConfig, not here
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # 4-F: Train
    print("Starting training...")
    trainer_stats = trainer.train()
    print(f"Training complete. Stats: {trainer_stats}")

    # 4-G: Save final checkpoint (LoRA adapters + tokenizer)
    final_checkpoint = os.path.join(CONFIG["output_dir"], "final_checkpoint")
    model.save_pretrained(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)
    print(f"Model checkpoint saved to: {final_checkpoint}")

    # 4-H: (Optional) Save merged 16-bit model for inference
    merged_dir = os.path.join(CONFIG["output_dir"], "merged_16bit")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged 16-bit model saved to: {merged_dir}")

    return trainer_stats


# ------------------------------------------------------------------
# Step 5: Quick inference helper
# ------------------------------------------------------------------
def predict(model, tokenizer, query: str) -> str:
    """Run inference on a single customer query."""
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)

    prompt = format_prompt(query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.1,
        do_sample=False,
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract predicted intent from the response
    intent = decoded.split("Intent:")[-1].strip().split()[0]
    return intent


# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
