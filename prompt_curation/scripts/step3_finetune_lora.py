"""
Step 3 — LoRA fine-tuning on Mistral-7B
========================================
Fine-tunes Mistral-7B-Instruct-v0.2 using LoRA on curated
prompt-instruction pairs from a chosen template.

Two runs planned:
  LoRA-A: fine-tune on template A instructions (baseline quality)
  LoRA-H: fine-tune on template H instructions (best style+specificity)

After fine-tuning, re-run step1 on the fine-tuned model and compare
attention mass before vs after — testing invariant localised grounding.

Usage:
    export CUDA_VISIBLE_DEVICES=1
    python3 step3_finetune_lora.py --template A --output ../models/lora_A/
    python3 step3_finetune_lora.py --template H --output ../models/lora_H/
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from torch.utils.data import Dataset
import numpy as np


REFUSAL_PHRASES = [
    "i'm an ai", "i cannot", "i don't have",
    "as an ai", "language model", "i am an ai", "i'm unable"
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class InstructionDataset(Dataset):
    def __init__(self, records, tokenizer, max_length=256):
        self.samples  = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for r in records:
            prompt      = r["prompt"]
            instruction = r.get("instruction", "").strip()

            # Skip refusals and empty instructions
            if not instruction:
                continue
            if any(p in instruction.lower() for p in REFUSAL_PHRASES):
                continue
            if len(instruction.split()) < 5:
                continue

            # Format: prompt + instruction + EOS
            full_text = prompt + instruction + tokenizer.eos_token
            self.samples.append(full_text)

        print(f"  Dataset: {len(self.samples)} clean samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        labels         = input_ids.clone()
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Template: {args.template}")
    print(f"Output:   {args.output}")

    # Load training data
    print("\nLoading training data...")
    with open(args.results_json) as f:
        data = json.load(f)

    if args.template not in data["per_template"]:
        raise ValueError(f"Template {args.template} not found in results JSON")

    records = data["per_template"][args.template]
    print(f"  Total records: {len(records)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    print("\nBuilding dataset...")
    dataset = InstructionDataset(records, tokenizer, max_length=args.max_length)

    if len(dataset) < 10:
        raise ValueError(f"Too few clean samples: {len(dataset)}")

    # Train/val split
    val_size   = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Load model in 4-bit
    print("\nLoading Mistral-7B in 4-bit...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        quantization_config=bnb,
        device_map={"": device},
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="none",
        dataloader_pin_memory=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    print("\nStarting LoRA fine-tuning...")
    trainer.train()

    # Save adapter
    adapter_path = out_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"\nAdapter saved: {adapter_path}")

    # Save training summary
    summary = {
        "template":      args.template,
        "base_model":    "mistralai/Mistral-7B-Instruct-v0.2",
        "train_samples": len(train_ds),
        "val_samples":   len(val_ds),
        "epochs":        args.epochs,
        "lora_r":        16,
        "lora_alpha":    32,
        "adapter_path":  str(adapter_path),
    }
    with open(out_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {out_dir / 'training_summary.json'}")
    print("\nNext: run step1 on fine-tuned model and compare attention mass")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True, choices=["A","B","C","D","E","F","G","H","I"],
                   help="Which template's instructions to use for fine-tuning")
    p.add_argument("--results_json",
                   default="../results/template_comparison_979.json")
    p.add_argument("--output", required=True,
                   help="Output directory for LoRA adapter")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_length", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
