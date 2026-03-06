#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  ElastiLM  ──  Intra-Model Fair Comparison
#  같은 모델 내에서 Base(원본) vs ElastiLM(탄력화) 비교:
#    • Llama 3.2 3B : CUDA 1,2  →  Base 3B vs ElastiLM 3B
#    • Llama 3.1 8B : CUDA 3,4  →  Base 8B vs ElastiLM 8B
#
#  Usage:
#    bash run_compare.sh              # 두 모델 순차 평가 (기본 GPU 할당)
#    bash run_compare.sh 3b           # 3B만 평가 (CUDA 1,2)
#    bash run_compare.sh 8b           # 8B만 평가 (CUDA 3,4)
#    bash run_compare.sh all          # 3B → 8B 순차 평가 후 비교
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TARGET="${1:-all}"   # "all" | "3b" | "8b" | config yaml path

# ── GPU 할당 (고정) ──────────────────────────────────────────────────
GPUS_3B="1,2"
GPUS_8B="3,4"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║   ElastiLM  ·  Intra-Model Fair Comparison                   ║"
echo "║   Base(원본) vs ElastiLM(탄력화) ── 같은 모델끼리 비교       ║"
echo "║   3B → CUDA ${GPUS_3B}   /   8B → CUDA ${GPUS_8B}                        ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

run_eval() {
    local cfg=$1
    local name=$2
    local gpus=$3
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  ${name}  ──  Base vs ElastiLM (CUDA ${gpus})"
    echo "════════════════════════════════════════════════════════════"
    CUDA_VISIBLE_DEVICES="$gpus" python evaluate.py --config "$cfg" --mode full
}

case "$TARGET" in
    3b|3B)
        run_eval "config_llama32_3b.yaml" "Llama 3.2 3B" "$GPUS_3B"
        ;;
    8b|8B)
        run_eval "config.yaml" "Llama 3.1 8B" "$GPUS_8B"
        ;;
    all)
        run_eval "config_llama32_3b.yaml" "Llama 3.2 3B" "$GPUS_3B"
        run_eval "config.yaml" "Llama 3.1 8B" "$GPUS_8B"
        ;;
    *)
        # 직접 config 파일 지정 시
        run_eval "$TARGET" "$TARGET" "${2:-0}"
        ;;
esac

# ── 통합 비교 (같은 모델끼리 그룹핑) ─────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Intra-Model Comparison (3B는 3B끼리, 8B는 8B끼리)"
echo "════════════════════════════════════════════════════════════"
python compare.py

echo ""
echo "  ✓ Comparison complete!"
