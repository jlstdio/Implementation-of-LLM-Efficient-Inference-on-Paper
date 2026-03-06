import argparse
import json
import os
import re
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import Config, load_config
from TLM import (
    ElastiLM_TLM,
    add_slo_tokens,
    compress_prompt,
    level_to_ratio,
    RATIOS,
)
from model_elasticalize import (
    ElasticLlamaWrapper,
    load_elastic_model,
)

def load_eval_dataset(name: str, max_samples: Optional[int] = None):
    items = []

    if name == "arc_easy":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        for row in ds:
            choices = row["choices"]["text"]
            label_list = row["choices"]["label"]
            answer_key = row["answerKey"]
            try:
                answer_idx = label_list.index(answer_key)
            except ValueError:
                answer_idx = ord(answer_key) - ord("A") if answer_key.isalpha() else 0
            items.append({
                "question": row["question"],
                "choices": choices,
                "answer_idx": answer_idx,
            })

    elif name == "piqa":
        ds = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
        for row in ds:
            items.append({
                "question": row["goal"],
                "choices": [row["sol1"], row["sol2"]],
                "answer_idx": row["label"],
            })

    elif name == "mmlu_pro":
        try:
            ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
            for row in ds:
                items.append({
                    "question": row["question"],
                    "choices": row["options"],
                    "answer_idx": row.get("answer_index", 0),
                })
        except Exception:
            ds = load_dataset("cais/mmlu", "all", split="test")
            for row in ds:
                items.append({
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer_idx": row["answer"],
                })

    else:
        raise ValueError(f"Unknown dataset: {name}")

    if max_samples and len(items) > max_samples:
        items = items[:max_samples]

    return items


def format_n_shot_prompt(
    question: str,
    choices: list[str],
    n_shot_examples: list[dict],
) -> str:
    parts = []

    for ex in n_shot_examples:
        choice_str = "\n".join(
            [f"({chr(65+i)}) {c}" for i, c in enumerate(ex["choices"])]
        )
        answer_letter = chr(65 + ex["answer_idx"])
        parts.append(
            f"Question: {ex['question']}\n{choice_str}\nAnswer: ({answer_letter})"
        )

    choice_str = "\n".join(
        [f"({chr(65+i)}) {c}" for i, c in enumerate(choices)]
    )
    parts.append(f"Question: {question}\n{choice_str}\nAnswer:")

    return "\n\n".join(parts)


def extract_answer_letter(text: str, num_choices: int) -> str:
    valid = set(chr(65 + i) for i in range(num_choices))

    # 패턴: (A), A), A.
    m = re.search(r"\(([A-J])\)", text)
    if m and m.group(1) in valid:
        return m.group(1)

    for ch in text.strip():
        if ch.upper() in valid:
            return ch.upper()

    return ""


@torch.no_grad()
def evaluate_accuracy(
    model,
    tokenizer,
    dataset_name: str,
    device: torch.device,
    n_shot: int = 5,
    max_new_tokens: int = 256,
    max_samples: Optional[int] = None,
    model_label: str = "model",
) -> dict:
    items = load_eval_dataset(dataset_name, max_samples)
    if len(items) == 0:
        return {"accuracy": 0.0, "total": 0}

    # select n-shot
    n_shot_examples = items[:n_shot]
    eval_items = items[n_shot:]

    if len(eval_items) == 0:
        eval_items = items

    correct, total = 0, 0
    latencies = []
    details = []

    for i, item in enumerate(eval_items):
        prompt = format_n_shot_prompt(
            item["question"], item["choices"], n_shot_examples
        )
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=2048,
        ).to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = time.perf_counter() - t_start
        latencies.append(latency)

        gen_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        pred = extract_answer_letter(gen_text, len(item["choices"]))
        gold = chr(65 + item["answer_idx"])

        is_correct = pred == gold
        if is_correct:
            correct += 1
        total += 1

        details.append({
            "index": i,
            "question": item["question"][:200],
            "gold": gold,
            "pred": pred,
            "correct": is_correct,
            "latency_sec": latency,
        })

        if (i + 1) % 100 == 0:
            acc = correct / total * 100
            avg_lat = np.mean(latencies)
            print(
                f"    [{i+1:>5}/{len(eval_items)}]  "
                f"Acc: {acc:.1f}%  Avg lat: {avg_lat:.3f}s"
            )

    accuracy = correct / max(total, 1) * 100
    lat_arr = np.array(latencies)
    lat_mean = float(np.mean(lat_arr)) if len(lat_arr) > 0 else 0.0
    lat_std = float(np.std(lat_arr, ddof=1)) if len(lat_arr) > 1 else 0.0
    lat_median = float(np.median(lat_arr)) if len(lat_arr) > 0 else 0.0

    print(f"\n  ★  [{model_label}] {dataset_name}")
    print(f"     Accuracy     : {accuracy:.2f}%  ({correct}/{total})")
    print(f"     Latency mean : {lat_mean:.4f}s")
    print(f"     Latency std  : {lat_std:.4f}s")

    return {
        "model": model_label,
        "dataset": dataset_name,
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "n_shot": n_shot,
        "latency_mean_sec": lat_mean,
        "latency_std_sec": lat_std,
        "latency_median_sec": lat_median,
        "details": details,
    }

@torch.no_grad()
def evaluate_slo_compliance(
    model,
    tokenizer,
    tlm: ElastiLM_TLM,
    tlm_tokenizer,
    wrapper: ElasticLlamaWrapper,
    device: torch.device,
    slo_scenarios: list[dict],
    dataset_name: str = "arc_easy",
    max_samples: int = 200,
    model_label: str = "model",
) -> list[dict]:
    
    items = load_eval_dataset(dataset_name, max_samples)
    results = []

    for scenario in slo_scenarios:
        ttft_target_ms = scenario["ttft_ms"]
        tpot_target_ms = scenario["tpot_ms"]

        ttft_pass, tpot_pass, total = 0, 0, 0
        ttft_list, tpot_list = [], []

        slo_token = f"[TTFT_{ttft_target_ms}]"

        for item in items[:max_samples]:
            prompt = f"Question: {item['question']}\nAnswer:"

            # TLM selection
            tlm_input = tlm_tokenizer(
                slo_token + " " + prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(device)

            tlm.eval()
            strategy = tlm.predict_strategy(
                tlm_input["input_ids"], tlm_input["attention_mask"]
            )

            # LLM inference & TTFT measure
            llm_input = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=512,
            ).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model.generate(
                **llm_input,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            ttft = (time.perf_counter() - t0) * 1000
            ttft_list.append(ttft)

            # TPOT
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            full_outputs = model.generate(
                **llm_input,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time = (time.perf_counter() - t1) * 1000
            num_new = full_outputs.shape[1] - llm_input["input_ids"].shape[1]
            tpot = total_time / max(num_new, 1)
            tpot_list.append(tpot)

            if ttft <= ttft_target_ms:
                ttft_pass += 1
            if tpot <= tpot_target_ms:
                tpot_pass += 1
            total += 1

        ttft_arr = np.array(ttft_list)
        tpot_arr = np.array(tpot_list)

        result = {
            "model": model_label,
            "slo_ttft_ms": ttft_target_ms,
            "slo_tpot_ms": tpot_target_ms,
            "ttft_compliance_pct": ttft_pass / max(total, 1) * 100,
            "tpot_compliance_pct": tpot_pass / max(total, 1) * 100,
            "ttft_mean_ms": float(np.mean(ttft_arr)),
            "ttft_p50_ms": float(np.median(ttft_arr)),
            "ttft_p99_ms": float(np.percentile(ttft_arr, 99)),
            "tpot_mean_ms": float(np.mean(tpot_arr)),
            "tpot_p50_ms": float(np.median(tpot_arr)),
            "tpot_p99_ms": float(np.percentile(tpot_arr, 99)),
            "total": total,
        }
        results.append(result)

        print(
            f"  SLO(TTFT≤{ttft_target_ms}ms, TPOT≤{tpot_target_ms}ms): "
            f"TTFT={result['ttft_compliance_pct']:.1f}%  "
            f"TPOT={result['tpot_compliance_pct']:.1f}%  "
            f"(mean TTFT={result['ttft_mean_ms']:.1f}ms, TPOT={result['tpot_mean_ms']:.1f}ms)"
        )

    return results

@torch.no_grad()
def measure_overhead(
    tlm: ElastiLM_TLM,
    tlm_tokenizer,
    device: torch.device,
    num_runs: int = 100,
) -> dict:

    print(f"\n{'─' * 60}")
    print(f"  Overhead Measurement ({num_runs} runs)")
    print(f"{'─' * 60}")

    tlm.eval()
    dummy = tlm_tokenizer(
        "[TTFT_100] This is a test prompt for measuring overhead.",
        return_tensors="pt", max_length=128, truncation=True, padding="max_length",
    ).to(device)

    # Warm up
    for _ in range(10):
        _ = tlm(dummy["input_ids"], dummy["attention_mask"])

    if device.type == "cuda":
        torch.cuda.synchronize()

    tlm_times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = tlm(dummy["input_ids"], dummy["attention_mask"])
        if device.type == "cuda":
            torch.cuda.synchronize()
        tlm_times.append((time.perf_counter() - t0) * 1000)

    tlm_arr = np.array(tlm_times)

    if device.type == "cuda":
        mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    else:
        mem_allocated = 0.0
        mem_reserved = 0.0

    result = {
        "tlm_inference_mean_ms": float(np.mean(tlm_arr)),
        "tlm_inference_std_ms": float(np.std(tlm_arr)),
        "tlm_inference_p50_ms": float(np.median(tlm_arr)),
        "tlm_inference_p99_ms": float(np.percentile(tlm_arr, 99)),
        "gpu_mem_allocated_gb": mem_allocated,
        "gpu_mem_reserved_gb": mem_reserved,
        "num_runs": num_runs,
    }

    print(f"  TLM inference: {result['tlm_inference_mean_ms']:.2f} ± "
          f"{result['tlm_inference_std_ms']:.2f} ms  (p99: {result['tlm_inference_p99_ms']:.2f} ms)")
    print(f"  GPU memory allocated: {mem_allocated:.2f} GB")
    print(f"  GPU memory reserved:  {mem_reserved:.2f} GB")

    return result


# ═══════════════════════════════════════════════════════════════════════
#  비교 테이블 출력
# ═══════════════════════════════════════════════════════════════════════

def print_accuracy_table(all_results: list[dict]):
    """같은 모델 내(Base vs ElastiLM) 공평한 비교 테이블 출력."""
    if not all_results:
        return

    # 데이터셋별 그룹핑
    datasets = sorted(set(r.get("dataset", "?") for r in all_results))

    # Base(full) 정확도를 기준으로 delta 계산
    base_acc = {}
    for r in all_results:
        if "(full)" in r.get("model", ""):
            base_acc[r["dataset"]] = r["accuracy"]

    print()
    print("╔════════════════════════════════════════════════════════════════════════════════════╗")
    print("║              ElastiLM  ──  Intra-Model Fair Comparison (Base vs ElastiLM)         ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    for ds in datasets:
        ds_results = [r for r in all_results if r.get("dataset") == ds]
        print(f"║  Dataset: {ds:<72}  ║")
        print(f"║  {'Model':<28} {'Acc(%)':>8} {'Δ Acc':>8} {'N-shot':>7} "
              f"{'Lat-Mean':>10} {'Lat-Med':>10}  ║")
        print(f"║  {'─' * 79}  ║")

        b_acc = base_acc.get(ds, 0.0)
        for r in sorted(ds_results, key=lambda x: -x["accuracy"]):
            model = r.get("model", "?")[:28]
            acc = r.get("accuracy", 0.0)
            delta = acc - b_acc if b_acc > 0 else 0.0
            delta_str = f"{delta:>+7.2f}%" if "(full)" not in r.get("model", "") else f"{'baseline':>8}"
            ns = r.get("n_shot", 0)
            lat_m = r.get("latency_mean_sec", 0.0)
            lat_med = r.get("latency_median_sec", 0.0)
            print(f"║  {model:<28} {acc:>7.2f}% {delta_str} {ns:>7} "
                  f"{lat_m:>9.4f}s {lat_med:>9.4f}s  ║")
        print(f"║{'':83}║")

    print("╚════════════════════════════════════════════════════════════════════════════════════╝")


def print_slo_table(all_results: list[dict]):
    """SLO 준수율 테이블 출력."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║                    ElastiLM  SLO  Compliance  Table                    ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║ {'Model':<20} {'TTFT-SLO':>9} {'TPOT-SLO':>9} "
          f"{'TTFT%':>7} {'TPOT%':>7} {'TTFT-mean':>10} {'TPOT-mean':>10} ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")

    for r in all_results:
        model = r.get("model", "?")[:20]
        print(f"║ {model:<20} {r['slo_ttft_ms']:>7}ms {r['slo_tpot_ms']:>7}ms "
              f"{r['ttft_compliance_pct']:>6.1f}% {r['tpot_compliance_pct']:>6.1f}% "
              f"{r['ttft_mean_ms']:>8.1f}ms {r['tpot_mean_ms']:>8.1f}ms ║")

    print("╚══════════════════════════════════════════════════════════════════════════╝")


@torch.no_grad()
def evaluate_online_inference(
    tlm: ElastiLM_TLM,
    tlm_tokenizer,
    wrapper: ElasticLlamaWrapper,
    llm_tokenizer,
    device: torch.device,
    dataset_name: str = "arc_easy",
    n_shot: int = 5,
    max_new_tokens: int = 256,
    max_samples: Optional[int] = None,
    slo_token: str = "[TTFT_100]",
) -> dict:
    items = load_eval_dataset(dataset_name, max_samples)
    n_shot_examples = items[:n_shot]
    eval_items = items[n_shot:] if len(items) > n_shot else items

    correct, total = 0, 0
    latencies = []

    print(f"\n{'─' * 60}")
    print(f"  [ElastiLM Online] {dataset_name}  │  {len(eval_items)} samples")
    print(f"{'─' * 60}")

    tlm.eval()

    for i, item in enumerate(eval_items):
        prompt = format_n_shot_prompt(
            item["question"], item["choices"], n_shot_examples
        )

        # 1. TLM 추론
        tlm_input = tlm_tokenizer(
            slo_token + " " + prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(device)

        strategy = tlm.predict_strategy(
            tlm_input["input_ids"], tlm_input["attention_mask"]
        )

        prompt_ratio = level_to_ratio(strategy["prompt_level"][0].item())
        model_ratio = level_to_ratio(strategy["model_level"][0].item())

        # 2. 프롬프트 압축
        compressed_ids, compressed_mask = compress_prompt(
            tlm_input["input_ids"],
            strategy["token_scores"],
            prompt_ratio,
            tlm_input["attention_mask"],
        )

        # 3. Elastic Llama 추론 (압축된 프롬프트 → 다시 LLM 토크나이저로 인코딩)
        compressed_text = tlm_tokenizer.decode(compressed_ids[0], skip_special_tokens=True)
        llm_input = llm_tokenizer(
            compressed_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(device)

        wrapper.set_ratio(model_ratio)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = wrapper.generate(
            input_ids=llm_input["input_ids"],
            attention_mask=llm_input["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=llm_tokenizer.pad_token_id,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()
        latency = time.perf_counter() - t0
        latencies.append(latency)

        gen_text = llm_tokenizer.decode(
            outputs[0][llm_input["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        pred = extract_answer_letter(gen_text, len(item["choices"]))
        gold = chr(65 + item["answer_idx"])

        if pred == gold:
            correct += 1
        total += 1

        if (i + 1) % 100 == 0:
            acc = correct / total * 100
            print(f"    [{i+1:>5}/{len(eval_items)}]  Acc: {acc:.1f}%  "
                  f"Avg lat: {np.mean(latencies):.3f}s")

    accuracy = correct / max(total, 1) * 100
    lat_arr = np.array(latencies) if latencies else np.array([0.0])

    print(f"[ElastiLM Online] {dataset_name}")
    print(f"Accuracy: {accuracy:.2f}%  ({correct}/{total})")
    print(f"Latency mean: {float(np.mean(lat_arr)):.4f}s")

    return {
        "model": f"{wrapper.model.config._name_or_path.split('/')[-1]} (ElastiLM-Online)" if hasattr(wrapper.model, 'config') else "ElastiLM-Online",
        "dataset": dataset_name,
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "n_shot": n_shot,
        "latency_mean_sec": float(np.mean(lat_arr)),
        "latency_std_sec": float(np.std(lat_arr, ddof=1)) if len(lat_arr) > 1 else 0.0,
        "latency_median_sec": float(np.median(lat_arr)),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ElastiLM Evaluation Pipeline")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["accuracy", "latency", "overhead", "online", "full"],
        help="평가 모드 선택",
    )
    args, _ = parser.parse_known_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ecfg = cfg.evaluate

    print(f"\n{'╔' + '═' * 58 + '╗'}")
    print(f"║{'ElastiLM  ·  Evaluation Pipeline':^58}║")
    print(f"║{'Model: ' + cfg.llm.short_name:^58}║")
    print(f"║{'Mode:  ' + args.mode:^58}║")
    print(f"{'╚' + '═' * 58 + '╝'}\n")
    print(f"  Device: {device}")

    os.makedirs(cfg.output.eval_dir, exist_ok=True)

    all_accuracy_results = []
    all_slo_results = []
    overhead_result = {}

    # ── 모델 로드 ────────────────────────────────────────────────────
    # 1) 기본 (비탄력화) 모델 — 공평한 baseline
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.llm.torch_dtype, torch.float16)

    print("  Loading base model …")
    base_tokenizer = AutoTokenizer.from_pretrained(
        cfg.llm.name, trust_remote_code=cfg.llm.trust_remote_code
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.llm.name,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.llm.trust_remote_code,
        device_map="auto" if device.type == "cuda" else None,
    )
    base_model.eval()

    # 2) 탄력화 모델 (있으면)
    wrapper, elastic_tokenizer = None, None
    elastic_model_path = os.path.join(cfg.output.elastic_dir, "elasticalized_model")
    if os.path.isdir(elastic_model_path):
        print("  Loading elasticalized model …")
        wrapper, elastic_tokenizer = load_elastic_model(cfg, device)

    # 3) TLM (있으면)
    tlm, tlm_tokenizer = None, None
    tlm_ckpt = os.path.join(cfg.output.tlm_dir, "tlm_full.pt")
    if os.path.exists(tlm_ckpt):
        print("  Loading TLM …")
        tlm_tokenizer = AutoTokenizer.from_pretrained(cfg.output.tlm_dir)
        tlm = ElastiLM_TLM(
            backbone_name=cfg.tlm.backbone,
            shared_layers=cfg.tlm.shared_layers,
            num_prompt_levels=cfg.tlm.num_prompt_levels,
            num_model_levels=cfg.tlm.num_model_levels,
        )
        tlm.load_state_dict(
            torch.load(tlm_ckpt, map_location=device, weights_only=False)
        )
        tlm = tlm.to(device)
        tlm.eval()

    # ══════════════════════════════════════════════════════════════════
    #  정확도 평가
    # ══════════════════════════════════════════════════════════════════
    if args.mode in ("accuracy", "full"):
        print(f"\n{'=' * 60}")
        print(f"  Accuracy Evaluation")
        print(f"{'=' * 60}")

        for ds_name in ecfg.datasets:
            # Baseline (풀 모델)
            result = evaluate_accuracy(
                model=base_model,
                tokenizer=base_tokenizer,
                dataset_name=ds_name,
                device=device,
                n_shot=ecfg.n_shot,
                max_new_tokens=ecfg.max_new_tokens,
                max_samples=ecfg.max_eval_samples,
                model_label=f"{cfg.llm.short_name} (full)",
            )
            all_accuracy_results.append(result)

            # 탄력화 모델 (각 ratio)
            if wrapper is not None:
                for ratio in [0.5, 0.7, 0.9]:  # 주요 ratio만 평가
                    wrapper.set_ratio(ratio)
                    ratio_result = evaluate_accuracy(
                        model=wrapper.model,
                        tokenizer=elastic_tokenizer,
                        dataset_name=ds_name,
                        device=device,
                        n_shot=ecfg.n_shot,
                        max_new_tokens=ecfg.max_new_tokens,
                        max_samples=ecfg.max_eval_samples,
                        model_label=f"{cfg.llm.short_name} (elastic-{int(ratio*100)}%)",
                    )
                    all_accuracy_results.append(ratio_result)

        print_accuracy_table(all_accuracy_results)

    # ══════════════════════════════════════════════════════════════════
    #  SLO 준수율 평가
    # ══════════════════════════════════════════════════════════════════
    if args.mode in ("latency", "full") and tlm is not None and wrapper is not None:
        print(f"\n{'=' * 60}")
        print(f"  SLO Compliance Evaluation")
        print(f"{'=' * 60}")

        slo_results = evaluate_slo_compliance(
            model=wrapper.model,
            tokenizer=elastic_tokenizer,
            tlm=tlm,
            tlm_tokenizer=tlm_tokenizer,
            wrapper=wrapper,
            device=device,
            slo_scenarios=ecfg.slo_scenarios,
            dataset_name="arc_easy",
            max_samples=200,
            model_label=cfg.llm.short_name,
        )
        all_slo_results.extend(slo_results)
        print_slo_table(all_slo_results)

    # ══════════════════════════════════════════════════════════════════
    #  오버헤드 측정
    # ══════════════════════════════════════════════════════════════════
    if args.mode in ("overhead", "full") and tlm is not None:
        overhead_result = measure_overhead(
            tlm=tlm,
            tlm_tokenizer=tlm_tokenizer,
            device=device,
        )

    # ══════════════════════════════════════════════════════════════════
    #  온라인 추론 평가
    # ══════════════════════════════════════════════════════════════════
    if args.mode in ("online", "full") and tlm is not None and wrapper is not None:
        print(f"\n{'=' * 60}")
        print(f"  Online Inference Evaluation")
        print(f"{'=' * 60}")

        for ds_name in ecfg.datasets:
            online_result = evaluate_online_inference(
                tlm=tlm,
                tlm_tokenizer=tlm_tokenizer,
                wrapper=wrapper,
                llm_tokenizer=elastic_tokenizer,
                device=device,
                dataset_name=ds_name,
                n_shot=ecfg.n_shot,
                max_new_tokens=ecfg.max_new_tokens,
                max_samples=ecfg.max_eval_samples,
            )
            all_accuracy_results.append(online_result)

        if all_accuracy_results:
            print_accuracy_table(all_accuracy_results)

    # ══════════════════════════════════════════════════════════════════
    #  결과 저장
    # ══════════════════════════════════════════════════════════════════
    summary = {
        "model": cfg.llm.name,
        "short_name": cfg.llm.short_name,
        "mode": args.mode,
        "accuracy_results": [
            {k: v for k, v in r.items() if k != "details"}
            for r in all_accuracy_results
        ],
        "slo_results": all_slo_results,
        "overhead": overhead_result,
    }

    result_path = os.path.join(cfg.output.eval_dir, f"eval_results_{cfg.llm.short_name}.json")
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Results saved → {result_path}")

    # 상세 결과 저장
    if all_accuracy_results:
        detail_path = os.path.join(cfg.output.eval_dir, f"eval_details_{cfg.llm.short_name}.jsonl")
        with open(detail_path, "w") as f:
            for r in all_accuracy_results:
                for d in r.get("details", []):
                    d["model"] = r["model"]
                    d["dataset"] = r["dataset"]
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"  ✓ Details saved → {detail_path}")

    print(f"\n  ★  Evaluation complete!")
    print()


if __name__ == "__main__":
    main()
