#!/usr/bin/env python
"""
Baseline  ──  GSM8K Evaluation (재사용 가능)
=============================================
Fine-tune 된 Causal LM 의 GSM8K 정답률 + 추론 시간(Mean / Std) 측정.

Usage:
    python evaluate.py --config config_llama_instruct.yaml
"""

import argparse
import json
import math
import os
import re
import time

import torch
import yaml
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ══════════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════════
def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Baseline GSM8K Evaluation")
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  유틸리티
# ══════════════════════════════════════════════════════════════════════════════
def extract_numerical_answer(text: str) -> str:
    """GSM8K 형식 #### 뒤 숫자 추출."""
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "").strip() if nums else ""


# ══════════════════════════════════════════════════════════════════════════════
#  평가
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_gsm8k(model, tokenizer, cfg: dict, device):
    """GSM8K 정답률 + 문제당 추론 시간 측정."""
    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluate"]
    max_new  = eval_cfg.get("max_new_tokens", 256)

    test = load_dataset(data_cfg["dataset"], "main", split=data_cfg["split_test"])
    max_eval = data_cfg.get("max_eval_samples")
    if max_eval:
        test = test.select(range(min(max_eval, len(test))))

    model.eval()
    correct, total = 0, 0
    details: list[dict] = []
    latencies: list[float] = []

    short_name = cfg["model"]["short_name"]

    print(f"\n{'─' * 60}")
    print(f"  GSM8K Eval: {short_name}  │  {len(test)} samples")
    print(f"{'─' * 60}")

    for i, item in enumerate(test):
        prompt = f"Question: {item['question']}\nStep-by-step solution:\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        # ── 시간 측정 시작 ──
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,          # greedy
            pad_token_id=tokenizer.pad_token_id,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()
        # ── 시간 측정 끝 ──

        latency = t_end - t_start
        latencies.append(latency)

        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        gold = extract_numerical_answer(item["answer"])
        pred = extract_numerical_answer(gen_text)

        is_correct = pred == gold
        if is_correct:
            correct += 1
        total += 1

        details.append({
            "index": i,
            "question": item["question"],
            "gold": gold,
            "pred": pred,
            "pred_text": gen_text,
            "correct": is_correct,
            "latency_sec": latency,
        })

        if (i + 1) % 50 == 0:
            acc_so_far = correct / total * 100
            avg_lat    = np.mean(latencies)
            print(
                f"    [{i + 1:>5}/{len(test)}]  "
                f"Acc: {acc_so_far:.1f}%  "
                f"Avg latency: {avg_lat:.3f}s"
            )

    # ── 통계 계산 ─────────────────────────────────────────────────────────
    accuracy   = correct / total * 100
    lat_arr    = np.array(latencies)
    lat_mean   = float(np.mean(lat_arr))
    lat_std    = float(np.std(lat_arr, ddof=1)) if len(lat_arr) > 1 else 0.0
    lat_median = float(np.median(lat_arr))

    print(f"\n  ★  {short_name}")
    print(f"     Accuracy          : {accuracy:.2f}%  ({correct}/{total})")
    print(f"     Latency mean      : {lat_mean:.4f} s")
    print(f"     Latency std       : {lat_std:.4f} s")
    print(f"     Latency median    : {lat_median:.4f} s\n")

    summary = {
        "model": cfg["model"]["name"],
        "short_name": short_name,
        "gsm8k_accuracy": accuracy,
        "total": total,
        "correct": correct,
        "latency_mean_sec": lat_mean,
        "latency_std_sec": lat_std,
        "latency_median_sec": lat_median,
    }
    return summary, details


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    cfg  = load_cfg(args.config)

    model_name = cfg["model"]["name"]
    short_name = cfg["model"]["short_name"]
    out_dir    = cfg["output"]["dir"]
    final_dir  = os.path.join(out_dir, "final")
    device     = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tok_dir = final_dir if os.path.isdir(final_dir) else model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model 로드 ────────────────────────────────────────────────────────
    if os.path.isdir(final_dir):
        model = AutoModelForCausalLM.from_pretrained(
            final_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        print(f"✓ Loaded fine-tuned checkpoint ← {final_dir}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
        )
        print(f"⚠ No fine-tuned checkpoint — evaluating base model: {model_name}")

    # ── 평가 ──────────────────────────────────────────────────────────────
    summary, details = evaluate_gsm8k(model, tokenizer, cfg, device)

    # ── 저장 ──────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)

    result_path = os.path.join(out_dir, "eval_results.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary → {result_path}")

    detail_path = os.path.join(out_dir, "eval_details.jsonl")
    with open(detail_path, "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  Details → {detail_path}")


if __name__ == "__main__":
    main()
