# ============================================================
# Inference — Banking Intent Classification
# Platform: Google Colab / Kaggle
# Framework: Unsloth
#
# Usage (standalone class):
#   from scripts.inference import IntentClassification
#   clf = IntentClassification("configs/inference.yml")
#   label = clf("I need to activate my card")
#
# Usage (batch inference on test.csv):
#   python scripts/inference.py
#   python scripts/inference.py --config configs/inference.yml
#   MODEL_CHECKPOINT=/kaggle/working/checkpoints/final_checkpoint \
#     python scripts/inference.py
# ============================================================

from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
import yaml


# ------------------------------------------------------------------
# Helper: resolve paths relative to project root
# ------------------------------------------------------------------
def _project_root() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    )


def _resolve(path: str) -> str:
    if not os.path.isabs(path):
        return os.path.normpath(os.path.join(_project_root(), path))
    return path


# ------------------------------------------------------------------
# IntentClassification — required interface per assignment spec
# ------------------------------------------------------------------
class IntentClassification:
    """Standalone inference class for banking intent classification.

    Args:
        model_path: Path to a YAML config file that contains at least
                    ``model_checkpoint_path`` pointing to the saved
                    LoRA checkpoint directory.

    Example:
        clf = IntentClassification("configs/inference.yml")
        label = clf("I lost my card and need a replacement.")
        print(label)   # -> "lost_or_stolen_card"
    """

    def __init__(self, model_path: str) -> None:  # noqa: D107
        from unsloth import FastLanguageModel

        # ── Load config ──────────────────────────────────────────────
        with open(model_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        checkpoint = os.environ.get(
            "MODEL_CHECKPOINT",
            _resolve(cfg["model_checkpoint_path"]),
        )
        max_seq_length: int = cfg.get("max_seq_length", 256)

        self._max_new_tokens: int = cfg.get("max_new_tokens", 20)
        self._temperature: float = cfg.get("temperature", 0.1)
        self._do_sample: bool = cfg.get("do_sample", False)

        # ── Load tokenizer & model ───────────────────────────────────
        print(f"Loading checkpoint from: {checkpoint}")
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint,
            max_seq_length=max_seq_length,
            dtype=None,          # auto-detect (bf16 / fp16)
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self._model)

        # Fix: Qwen2.5 generation_config ships with max_length=32768 which
        # conflicts with max_new_tokens and floods logs with warnings.
        # Clearing it makes max_new_tokens the single authoritative limit.
        self._model.generation_config.max_length = None

        # Left-padding is required for correct batched causal LM generation.
        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Model ready for inference.")

    # ── Prompt template (must match train.py exactly) ────────────────
    @staticmethod
    def _build_prompt(message: str) -> str:
        return (
            "You are a banking assistant. "
            "Classify the following customer query into one of the 77 banking intent categories.\n\n"
            f"Query: {message}\n\n"
            "Intent:"
        )

    def __call__(self, message: str) -> str:  # noqa: D102
        """Classify a single customer message and return the intent label.

        Args:
            message: Raw customer query string.

        Returns:
            Predicted intent label as a string (e.g. ``"lost_or_stolen_card"``). 
        """
        return self._batch_predict([message])[0]

    def _batch_predict(self, messages: list[str]) -> list[str]:
        """Run batched generation for a list of messages in a single GPU call.

        Args:
            messages: List of raw customer query strings.

        Returns:
            List of predicted intent label strings, one per input message.
        """
        prompts = [self._build_prompt(m) for m in messages]
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                do_sample=self._do_sample,
            )

        labels: list[str] = []
        for output in outputs:
            decoded = self._tokenizer.decode(output, skip_special_tokens=True)
            label = decoded.split("Intent:")[-1].strip().split()[0]
            labels.append(label)
        return labels


# ------------------------------------------------------------------
# Batch inference on test.csv
# ------------------------------------------------------------------
def run_batch_inference(cfg: dict) -> None:
    """Run inference on the full test split and write a CSV result file."""
    test_path = os.environ.get(
        "TEST_DATA_PATH", _resolve(cfg["test_data_path"])
    )
    output_path = os.environ.get(
        "OUTPUT_PATH", _resolve(cfg["output_path"])
    )
    config_file = cfg["_config_file"]  # injected by load_config()
    batch_size: int = cfg.get("batch_size", 32)

    # Load test data
    print(f"Reading test data from: {test_path}")
    test_df = pd.read_csv(test_path)

    # Initialise the classifier (loads model once)
    clf = IntentClassification(config_file)

    # Predict in true batches: one GPU generate() call per batch
    predictions: list[str] = []
    total = len(test_df)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = test_df["text"].iloc[start:end].tolist()
        predictions.extend(clf._batch_predict(batch))
        print(f"  Processed {end}/{total} samples...")

    # Attach predictions and persist
    test_df["predicted_label"] = predictions

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    test_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Quick accuracy check (if ground-truth labels are present in test.csv)
    if "label_name" in test_df.columns:
        accuracy = (test_df["predicted_label"] == test_df["label_name"]).mean()
        print(f"Accuracy on test set: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    elif "label" in test_df.columns:
        print("Tip: add a 'label_name' column to test.csv for automatic accuracy reporting.")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    _default_config = os.path.normpath(
        os.path.join(_project_root(), "configs", "inference.yml")
    )
    parser = argparse.ArgumentParser(
        description="Run batch inference for banking intent classification."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_default_config,
        help="Path to YAML inference config (default: configs/inference.yml)",
    )
    return parser.parse_args()


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_file"] = path  # pass config path through for IntentClassification
    return cfg


if __name__ == "__main__":
    args = _parse_args()
    print(f"Loading inference config from: {args.config}")
    cfg = _load_config(args.config)
    run_batch_inference(cfg)
