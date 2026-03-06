#!/usr/bin/env python
"""
Baseline  ──  GSM8K Full Fine-tuning (재사용 가능)
====================================================
Hugging Face Transformers 를 사용하여
임의의 Causal LM 을 GSM8K 로 full fine-tune 한다.

Usage:
    python finetune.py --config config_llama_instruct.yaml
"""

import argparse
import json
import os
import time

import torch
import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Config Loader
# ══════════════════════════════════════════════════════════════════════════════
def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Baseline GSM8K Full Fine-tuning")
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  데이터 전처리
# ══════════════════════════════════════════════════════════════════════════════
def build_dataset(tokenizer, cfg: dict):
    """GSM8K → 토큰화된 학습 데이터셋 반환."""
    data_cfg = cfg["data"]
    max_len  = cfg["training"]["max_length"]

    raw = load_dataset(data_cfg["dataset"], "main", split=data_cfg["split_train"])
    if data_cfg.get("max_train_samples"):
        raw = raw.select(range(min(data_cfg["max_train_samples"], len(raw))))

    def tokenize(example):
        text = (
            f"Question: {example['question']}\n"
            f"Step-by-step solution:\n{example['answer']}{tokenizer.eos_token}"
        )
        enc = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
        )
        enc["labels"] = enc["input_ids"].copy()
        return enc

    ds = raw.map(tokenize, remove_columns=raw.column_names, num_proc=4)
    ds.set_format("torch")
    return ds


# ══════════════════════════════════════════════════════════════════════════════
#  메인 학습
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    model_name = cfg["model"]["name"]
    short_name = cfg["model"]["short_name"]
    train_cfg  = cfg["training"]
    out_dir    = cfg["output"]["dir"]
    device     = cfg.get("device", "cuda")

    os.makedirs(out_dir, exist_ok=True)

    print(f"╔{'═' * 55}╗")
    print(f"║  Full Fine-tuning: {short_name:<35} ║")
    print(f"╚{'═' * 55}╝")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model (전체 파라미터 학습) ────────────────────────────────────────
    # device_map="auto" conflicts with DDP; let Trainer handle placement
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if train_cfg["fp16"] else torch.float32,
        trust_remote_code=True,
        device_map=None if is_distributed else "auto",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()  # VRAM 절약

    num_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {num_params:,}")
    print(f"  Trainable params : {train_params:,} (100.00%)")

    # ── Dataset ───────────────────────────────────────────────────────────
    print("Preparing GSM8K dataset …")
    train_ds = build_dataset(tokenizer, cfg)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # ── Training Args ─────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_grad_norm=train_cfg["max_grad_norm"],
        fp16=train_cfg["fp16"],
        logging_steps=train_cfg["log_interval"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # ── 저장 ──────────────────────────────────────────────────────────────
    final_dir = os.path.join(out_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    meta = {
        "model": model_name,
        "short_name": short_name,
        "training_time_sec": elapsed,
        "training_time_min": elapsed / 60,
        "method": "full_finetune",
    }
    with open(os.path.join(out_dir, "train_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✓ Fine-tuning complete  ({elapsed / 60:.1f} min)")
    print(f"  Model saved → {final_dir}\n")


if __name__ == "__main__":
    main()
