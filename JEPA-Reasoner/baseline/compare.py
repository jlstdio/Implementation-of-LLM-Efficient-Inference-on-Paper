#!/usr/bin/env python
"""
Baseline  ──  모든 모델의 평가 결과를 취합하여
비교 테이블(JSON + 텍스트)을 생성한다.

Usage:
    python compare.py
"""

import json
import os
import glob

import numpy as np


RESULT_DIR = "./checkpoints/baseline"
JEPA_RESULT = "../checkpoints/finetune/eval_results.json"


def load_all_results():
    results = []

    # ── Baseline 모델 ──
    pattern = os.path.join(RESULT_DIR, "*/eval_results.json")
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            r = json.load(f)
        r["source_path"] = path
        results.append(r)

    # ── JEPA-Reasoner ──
    jepa_path = os.path.join(os.path.dirname(__file__), JEPA_RESULT)
    if os.path.exists(jepa_path):
        with open(jepa_path) as f:
            r = json.load(f)
        r["source_path"] = jepa_path
        if "short_name" not in r:
            r["short_name"] = "JEPA-Reasoner"
        results.append(r)

    return results


def print_comparison(results: list[dict]):
    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║                    GSM8K  Comparison  Table                             ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║ {'Model':<28} {'Acc(%)':>8} {'Lat-Mean(s)':>12} {'Lat-Std(s)':>11} {'Lat-Med(s)':>11} ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")

    for r in results:
        name    = r.get("short_name", r.get("model", "?"))[:28]
        acc     = r.get("gsm8k_accuracy", 0.0)
        lat_m   = r.get("latency_mean_sec", 0.0)
        lat_s   = r.get("latency_std_sec", 0.0)
        lat_med = r.get("latency_median_sec", 0.0)
        print(f"║ {name:<28} {acc:>7.2f}% {lat_m:>11.4f} {lat_s:>10.4f} {lat_med:>10.4f} ║")

    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()


def main():
    results = load_all_results()

    if not results:
        print("⚠ No eval_results.json found. Run evaluations first.")
        return

    print_comparison(results)

    # JSON 으로도 저장
    out_path = os.path.join(os.path.dirname(__file__), "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved comparison → {out_path}")


if __name__ == "__main__":
    main()
