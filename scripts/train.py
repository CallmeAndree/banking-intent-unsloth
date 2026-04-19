# ============================================================
# Fine-tuning Qwen/Qwen2.5-3B with Unsloth
# Platform: Google Colab / Kaggle
# Task: Banking Intent Classification (banking77)
#
# Usage:
#   python scripts/train.py                          # uses configs/train.yml
#   python scripts/train.py --config /path/to/cfg.yml
#   DATA_DIR=/data OUTPUT_DIR=/out python scripts/train.py
# ============================================================

import argparse
import os

import pandas as pd
import torch
import yaml
from datasets import Dataset


# ------------------------------------------------------------------
# Step 0: CLI argument parsing
# ------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_config = os.path.normpath(
        os.path.join(_script_dir, "..", "configs", "train.yml")
    )
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2.5-3B for banking intent classification."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_default_config,
        help="Path to YAML config file (default: configs/train.yml)",
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Step 1: Load configuration from YAML
# ------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """Load training config from a YAML file.

    Environment variables DATA_DIR and OUTPUT_DIR override the
    values defined under paths: in the YAML, allowing easy
    override from train.sh or the shell.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative paths against the project root (one level up from scripts/)
    _project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )

    def _resolve(path: str) -> str:
        """Make relative paths absolute from the project root."""
        if not os.path.isabs(path):
            return os.path.normpath(os.path.join(_project_root, path))
        return path

    cfg["paths"]["data_dir"] = os.environ.get(
        "DATA_DIR", _resolve(cfg["paths"]["data_dir"])
    )
    cfg["paths"]["output_dir"] = os.environ.get(
        "OUTPUT_DIR", _resolve(cfg["paths"]["output_dir"])
    )

    return cfg


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
    test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Ensure label_name column exists
    if "label_name" not in train_df.columns:
        train_df["label_name"] = train_df["label"].map(id2label)
    if "label_name" not in test_df.columns:
        test_df["label_name"] = test_df["label"].map(id2label)

    return train_df, test_df


def format_prompt(text: str, label_name: str | None = None) -> str:
    """Format each sample into an instruction-following prompt.

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
    records = [
        {"text": format_prompt(str(row["text"]), str(row["label_name"]))}
        for _, row in df.iterrows()
    ]
    return Dataset.from_list(records)


# ------------------------------------------------------------------
# Step 4: Main training entry point
# ------------------------------------------------------------------
def main(cfg: dict) -> None:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    paths = cfg["paths"]
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    sft_cfg = cfg["sft"]
    log_cfg = cfg["logging"]
    out_cfg = cfg["output"]

    # Runtime detection: must run here so it reads the actual GPU
    # on the execution platform (Kaggle T4, Colab A100, etc.)
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16_ok = torch.cuda.is_available() and not bf16_ok
    print(f"Precision: {'bf16' if bf16_ok else 'fp16' if fp16_ok else 'cpu'}")

    # 4-A: Load model with Unsloth 4-bit quantization
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],           # null in YAML → None in Python
        load_in_4bit=model_cfg["load_in_4bit"],
    )

    # 4-B: Apply LoRA adapters (PEFT)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        use_gradient_checkpointing=lora_cfg["use_gradient_checkpointing"],
        random_state=train_cfg["seed"],
        use_rslora=lora_cfg["use_rslora"],
        loftq_config=None,
    )
    model.print_trainable_parameters()

    # 4-C: Prepare datasets
    print("Loading data...")
    train_df, test_df = load_data(paths["data_dir"])
    train_dataset = prepare_dataset(train_df)
    eval_dataset = prepare_dataset(test_df)
    print(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    # 4-D: SFTConfig = TrainingArguments + SFT-specific params (TRL >= 0.12)
    warmup_steps = int(
        len(train_dataset)
        / train_cfg["per_device_train_batch_size"]
        / train_cfg["gradient_accumulation_steps"]
        * train_cfg["num_train_epochs"]
        * train_cfg["warmup_ratio"]
    )

    training_args = SFTConfig(
        output_dir=paths["output_dir"],

        # ── SFT-specific params ──────────────────────────────────────
        dataset_text_field=sft_cfg["dataset_text_field"],
        max_seq_length=model_cfg["max_seq_length"],
        dataset_num_proc=train_cfg["dataloader_num_workers"],
        packing=sft_cfg["packing"],

        # ── Training ────────────────────────────────────────────────
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        weight_decay=train_cfg["weight_decay"],
        fp16=fp16_ok,
        bf16=bf16_ok,
        optim=train_cfg["optimizer"],
        seed=train_cfg["seed"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],

        # ── Logging & Checkpointing ─────────────────────────────────
        logging_steps=log_cfg["logging_steps"],
        save_strategy=log_cfg["save_strategy"],
        eval_strategy=log_cfg["eval_strategy"],
        load_best_model_at_end=log_cfg["load_best_model_at_end"],
        metric_for_best_model=log_cfg["metric_for_best_model"],
        report_to=log_cfg["report_to"],
    )

    # 4-E: SFTTrainer
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
    final_checkpoint = os.path.join(
        paths["output_dir"], out_cfg["final_checkpoint_subdir"]
    )
    model.save_pretrained(final_checkpoint)
    tokenizer.save_pretrained(final_checkpoint)
    print(f"Model checkpoint saved to: {final_checkpoint}")

    # 4-H: Save merged 16-bit model for inference
    merged_dir = os.path.join(paths["output_dir"], out_cfg["merged_16bit_subdir"])
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged 16-bit model saved to: {merged_dir}")


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
    args = parse_args()
    print(f"Loading config from: {args.config}")
    cfg = load_config(args.config)
    main(cfg)
