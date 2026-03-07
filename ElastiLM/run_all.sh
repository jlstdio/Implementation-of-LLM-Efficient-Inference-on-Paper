#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  ElastiLM  ──  Full Pipeline (3B + 8B 동시 실행)
#
#  GPU 할당 (고정, 단일 GPU):
#    • Llama 3.2 3B  →  CUDA 1  (~12GB fp32)
#    • Llama 3.1 8B  →  CUDA 2  (~32GB fp32)
#
#  두 모델을 서로 다른 GPU에서 **동시에** 학습 & 평가합니다.
#  완료 후 compare.py로 같은 모델끼리 공평 비교 테이블을 출력합니다.
#
#  Usage:
#    bash run_all.sh              # 3B + 8B 동시 (학습 + 평가)
#    bash run_all.sh train        # 학습만 (Step 0~3)
#    bash run_all.sh eval         # 평가만 (Step 4)
#    bash run_all.sh 3b           # 3B만 (CUDA 1)
#    bash run_all.sh 8b           # 8B만 (CUDA 2)
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-all}"   # all | train | eval | 3b | 8b

# ── GPU 할당 (고정, 단일 GPU) ────────────────────────────────────────
GPUS_3B="1"
GPUS_8B="2"
CONFIG_3B="config_llama32_3b.yaml"
CONFIG_8B="config.yaml"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║         ElastiLM  ·  Full Pipeline (Parallel)               ║"
echo "║                                                             ║"
echo "║   Llama 3.2 3B  →  CUDA ${GPUS_3B}                                ║"
echo "║   Llama 3.1 8B  →  CUDA ${GPUS_8B}                                ║"
echo "║                                                             ║"
echo "║   Mode : ${MODE}                                                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"

# ── 단일 모델 풀 파이프라인 (학습 + 평가) ────────────────────────────
run_full_pipeline() {
    local config=$1
    local name=$2
    local gpus=$3
    local log_file="${LOG_DIR}/${name// /_}_${TIMESTAMP}.log"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ${name}  ──  Starting on CUDA ${gpus}"
    echo "  Log → ${log_file}"
    echo "═══════════════════════════════════════════════════════════════"

    bash run_train.sh "$config" "$gpus" 2>&1 | tee "$log_file"
}

# ── 학습만 (Step 0~3) ────────────────────────────────────────────────
run_train_only() {
    local config=$1
    local name=$2
    local gpus=$3
    local log_file="${LOG_DIR}/${name// /_}_train_${TIMESTAMP}.log"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ${name}  ──  Training only on CUDA ${gpus}"
    echo "  Log → ${log_file}"
    echo "═══════════════════════════════════════════════════════════════"

    export CUDA_VISIBLE_DEVICES="$gpus"

    (
        echo "[0] Elasticalize …"
        python model_elasticalize.py --config "$config"

        echo "[1] LoRA Recovery …"
        python train.py --config "$config" --phase lora

        echo "[2] TLM Score-head …"
        python train.py --config "$config" --phase score

        echo "[3] TLM Decision-head …"
        python train.py --config "$config" --phase decision

        echo "  ✓ Training complete for ${name}"
    ) 2>&1 | tee "$log_file"
}

# ── 평가만 ────────────────────────────────────────────────────────────
run_eval_only() {
    local config=$1
    local name=$2
    local gpus=$3
    local log_file="${LOG_DIR}/${name// /_}_eval_${TIMESTAMP}.log"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ${name}  ──  Evaluation only on CUDA ${gpus}"
    echo "  Log → ${log_file}"
    echo "═══════════════════════════════════════════════════════════════"

    CUDA_VISIBLE_DEVICES="$gpus" python evaluate.py --config "$config" --mode full 2>&1 | tee "$log_file"
}

# ── 메인 로직 ─────────────────────────────────────────────────────────
case "$MODE" in
    all)
        # 3B와 8B를 동시에 실행 (백그라운드 + wait)
        run_full_pipeline "$CONFIG_3B" "Llama_3.2_3B" "$GPUS_3B" &
        PID_3B=$!

        run_full_pipeline "$CONFIG_8B" "Llama_3.1_8B" "$GPUS_8B" &
        PID_8B=$!

        echo ""
        echo "  ⏳ Waiting for both models to finish …"
        echo "     3B (PID ${PID_3B}) on CUDA ${GPUS_3B}"
        echo "     8B (PID ${PID_8B}) on CUDA ${GPUS_8B}"

        FAIL=0
        wait $PID_3B || { echo "  ✗ 3B pipeline failed"; FAIL=1; }
        wait $PID_8B || { echo "  ✗ 8B pipeline failed"; FAIL=1; }

        if [[ $FAIL -ne 0 ]]; then
            echo "  ⚠ Some pipelines failed. Check logs in ${LOG_DIR}/"
        fi
        ;;

    train)
        # 학습만 동시 실행
        run_train_only "$CONFIG_3B" "Llama_3.2_3B" "$GPUS_3B" &
        PID_3B=$!

        run_train_only "$CONFIG_8B" "Llama_3.1_8B" "$GPUS_8B" &
        PID_8B=$!

        echo ""
        echo "  ⏳ Waiting for training to finish …"
        wait $PID_3B || echo "  ✗ 3B training failed"
        wait $PID_8B || echo "  ✗ 8B training failed"
        ;;

    eval)
        # 평가만 동시 실행
        run_eval_only "$CONFIG_3B" "Llama_3.2_3B" "$GPUS_3B" &
        PID_3B=$!

        run_eval_only "$CONFIG_8B" "Llama_3.1_8B" "$GPUS_8B" &
        PID_8B=$!

        echo ""
        echo "  ⏳ Waiting for evaluation to finish …"
        wait $PID_3B || echo "  ✗ 3B evaluation failed"
        wait $PID_8B || echo "  ✗ 8B evaluation failed"
        ;;

    3b|3B)
        run_full_pipeline "$CONFIG_3B" "Llama_3.2_3B" "$GPUS_3B"
        ;;

    8b|8B)
        run_full_pipeline "$CONFIG_8B" "Llama_3.1_8B" "$GPUS_8B"
        ;;

    *)
        echo "Usage: bash run_all.sh [all|train|eval|3b|8b]"
        exit 1
        ;;
esac

# ── 비교 테이블 출력 ─────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Intra-Model Comparison (3B는 3B끼리, 8B는 8B끼리)"
echo "═══════════════════════════════════════════════════════════════"
python compare.py

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ✓ All done!                                                ║"
echo "║  Logs   → ${LOG_DIR}/                                      ║"
echo "║  Results → ./checkpoints/eval/                              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
