import glob
import json
import os
from collections import defaultdict
import numpy as np

EVAL_DIR = "./checkpoints/eval"

def load_all_results() -> list[dict]:
    results = []
    pattern = os.path.join(EVAL_DIR, "eval_results_*.json")

    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)

        base_model = data.get("short_name", "?")

        for acc in data.get("accuracy_results", []):
            rec = {
                "model": acc.get("model", base_model),
                "base_model": base_model,
                "dataset": acc.get("dataset", "?"),
                "accuracy": acc.get("accuracy", 0.0),
                "n_shot": acc.get("n_shot", 0),
                "latency_mean_sec": acc.get("latency_mean_sec", 0.0),
                "latency_std_sec": acc.get("latency_std_sec", 0.0),
                "latency_median_sec": acc.get("latency_median_sec", 0.0),
                "total": acc.get("total", 0),
                "correct": acc.get("correct", 0),
            }
            results.append(rec)

    return results


def load_slo_results() -> list[dict]:
    results = []
    pattern = os.path.join(EVAL_DIR, "eval_results_*.json")

    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)

        for slo in data.get("slo_results", []):
            slo["base_model"] = data.get("short_name", "?")
            results.append(slo)

    return results


def load_overhead_results() -> dict:
    results = {}
    pattern = os.path.join(EVAL_DIR, "eval_results_*.json")

    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        overhead = data.get("overhead", {})
        if overhead:
            results[data.get("short_name", "?")] = overhead

    return results


def print_intra_model_comparison(results: list[dict]):
    if not results:
        return

    groups = defaultdict(list)
    for r in results:
        groups[r["base_model"]].append(r)

    for base_name, group_results in sorted(groups.items()):
        datasets = sorted(set(r["dataset"] for r in group_results))

        base_acc = {}
        base_lat = {}
        for r in group_results:
            if "(full)" in r.get("model", ""):
                base_acc[r["dataset"]] = r["accuracy"]
                base_lat[r["dataset"]] = r["latency_mean_sec"]

        # Base vs ElastiLM  (Intra-Model Fair Comparison)
        for ds in datasets:
            ds_results = [r for r in group_results if r["dataset"] == ds]
            b_acc = base_acc.get(ds, 0.0)
            b_lat = base_lat.get(ds, 0.0)

            print(f"║  Dataset: {ds:<74}║")
            print(f"║  {'Variant':<30} {'Acc(%)':>8} {'Acc':>8} "
                  f"{'Lat(s)':>10} {'Speedup':>9} {'N-shot':>7}  ║")
            print(f"║  {'─' * 82}  ║")

            for r in sorted(ds_results, key=lambda x: -x["accuracy"]):
                name = r["model"][:30]
                acc = r["accuracy"]
                lat = r["latency_mean_sec"]
                if "(full)" in r.get("model", ""):
                    delta_str = f"{'baseline':>8}"
                else:
                    delta = acc - b_acc
                    delta_str = f"{delta:>+7.2f}%"
                if "(full)" in r.get("model", "") or b_lat <= 0:
                    speed_str = f"{'1.00x':>9}"
                else:
                    speedup = b_lat / lat if lat > 0 else 0.0
                    speed_str = f"{speedup:>8.2f}x"

                ns = r.get("n_shot", 0)
                print(f"║  {name:<30} {acc:>7.2f}% {delta_str} "
                      f"{lat:>9.4f}s {speed_str} {ns:>7}  ║")



def print_slo_comparison(results: list[dict]):    
    if not results:
        return

    groups = defaultdict(list)
    for r in results:
        groups[r.get("base_model", "?")].append(r)

    for base_name, group_results in sorted(groups.items()):
        print()
        print(f"╔═══════════════════════════════════════════════════════════════════════════════════════╗")
        print(f"║  {base_name}  ──  SLO Compliance{'':>52}║")
        print(f"╠═══════════════════════════════════════════════════════════════════════════════════════╣")
        print(f"║  {'Variant':<20} {'TTFT-SLO':>9} {'TPOT-SLO':>9} "
              f"{'TTFT%':>7} {'TPOT%':>7} {'TTFT-mean':>10} {'TPOT-mean':>10}      ║")
        print(f"║  {'─' * 82}  ║")

        for r in group_results:
            model = r.get("model", "?")[:20]
            print(f"║  {model:<20} {r['slo_ttft_ms']:>7}ms {r['slo_tpot_ms']:>7}ms "
                  f"{r['ttft_compliance_pct']:>6.1f}% {r['tpot_compliance_pct']:>6.1f}% "
                  f"{r['ttft_mean_ms']:>8.1f}ms {r['tpot_mean_ms']:>8.1f}ms      ║")

        print(f"╚═══════════════════════════════════════════════════════════════════════════════════════╝")


def print_overhead_comparison(overheads: dict):
    if not overheads:
        return

    print()
    print(f"╔═══════════════════════════════════════════════════════════════════════════════════════╗")
    print(f"║  TLM Overhead Summary{'':>63}║")
    print(f"╠═══════════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  {'Model':<20} {'TLM-Mean(ms)':>13} {'TLM-P99(ms)':>13} "
          f"{'GPU-Alloc(GB)':>14} {'GPU-Rsrv(GB)':>14}  ║")
    print(f"║  {'─' * 82}  ║")

    for name, oh in sorted(overheads.items()):
        print(f"║  {name:<20} {oh.get('tlm_inference_mean_ms',0):>12.2f} "
              f"{oh.get('tlm_inference_p99_ms',0):>12.2f} "
              f"{oh.get('gpu_mem_allocated_gb',0):>13.2f} "
              f"{oh.get('gpu_mem_reserved_gb',0):>13.2f}  ║")

    print(f"╚═══════════════════════════════════════════════════════════════════════════════════════╝")


def main():
    results = load_all_results()
    slo_results = load_slo_results()
    overhead_results = load_overhead_results()

    if not results and not slo_results:
        print("No evaluation results found")
        return

    print_intra_model_comparison(results)
    print_slo_comparison(slo_results)
    print_overhead_comparison(overhead_results)

    out_path = os.path.join(EVAL_DIR, "comparison_results.json")
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "description": "Intra-model fair comparison: Base vs ElastiLM (same model family)",
            "accuracy_results": results,
            "slo_results": slo_results,
            "overhead_results": overhead_results,
        }, f, indent=2, ensure_ascii=False)
        
    print(f"Saved comparison → {out_path}")


if __name__ == "__main__":
    main()
