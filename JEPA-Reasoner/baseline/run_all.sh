#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  Baseline  ──  4 모델 GSM8K Fine-tune + Evaluation + 비교
#
#  Models:
#    1. meta-llama/Llama-3.2-1B-Instruct
#    2. meta-llama/Llama-3.2-1B
#    3. google/gemma-3-1b-it
#    4. google/gemma-3-1b-pt
#
#  + JEPA-Reasoner 평가 결과와 최종 비교 테이블 출력
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   Baseline Comparison  ·  GSM8K Fine-tune & Evaluate     ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# ── 0. 의존 패키지 설치 ─────────────────────────────────────────────
echo ""
echo "[0/6] Installing dependencies …"
pip install --quiet --upgrade \
    torch \
    transformers \
    datasets \
    accelerate \
    sentencepiece \
    pyyaml \
    numpy

# ── 1. 데이터셋 사전 다운로드 ────────────────────────────────────────
echo ""
echo "[1/6] Pre-downloading GSM8K …"
python -c "
from datasets import load_dataset
_ = load_dataset('openai/gsm8k', 'main', split='train')
_ = load_dataset('openai/gsm8k', 'main', split='test')
print('  ✓ GSM8K ready')
"

# ── 모델 목록 (config 파일) ──────────────────────────────────────────
CONFIGS=(
    "config_llama_instruct.yaml"
    "config_llama_base.yaml"
    "config_gemma_it.yaml"
    "config_gemma_pt.yaml"
)

NAMES=(
    "Llama-3.2-1B-Instruct"
    "Llama-3.2-1B"
    "Gemma-3-1B-IT"
    "Gemma-3-1B-PT"
)

TOTAL=${#CONFIGS[@]}

# ── Auto-detect available GPUs ────────────────────────────────────────
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
NUM_GPUS=${NUM_GPUS:-1}
echo ""
echo "  Detected ${NUM_GPUS} GPU(s)"

# ── 2. Fine-tune 루프 ────────────────────────────────────────────────
echo ""
echo "[2/6] Fine-tuning ${TOTAL} baseline models …"
for idx in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$idx]}"
    name="${NAMES[$idx]}"
    step=$(( idx + 1 ))
    echo ""
    echo "  ── [${step}/${TOTAL}] Fine-tune: ${name} ──"
    accelerate launch --num_processes=$NUM_GPUS finetune.py --config "$cfg"
done
echo ""
echo "  ✓ All fine-tuning complete"

# ── 3. Evaluate 루프 ─────────────────────────────────────────────────
echo ""
echo "[3/6] Evaluating ${TOTAL} baseline models on GSM8K …"
for idx in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$idx]}"
    name="${NAMES[$idx]}"
    step=$(( idx + 1 ))
    echo ""
    echo "  ── [${step}/${TOTAL}] Evaluate: ${name} ──"
    python evaluate.py --config "$cfg"
done
echo ""
echo "  ✓ All evaluations complete"

# ── 4. JEPA-Reasoner 평가 ────────────────────────────────────────────
echo ""
echo "[4/6] Evaluating JEPA-Reasoner on GSM8K …"
cd "$SCRIPT_DIR/.."
if [ -f "evaluate.py" ]; then
    python evaluate.py --config config.yaml
    echo "  ✓ JEPA-Reasoner evaluation complete"
else
    echo "  ⚠ ../evaluate.py not found, skipping JEPA-Reasoner eval"
fi
cd "$SCRIPT_DIR"

# ── 5. 비교 테이블 ───────────────────────────────────────────────────
echo ""
echo "[5/6] Generating comparison table …"
python compare.py

# ── 6. 완료 ──────────────────────────────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  All done!                                               ║"
echo "║                                                          ║"
echo "║  Results per model : ./checkpoints/baseline/<model>/     ║"
echo "║  Comparison JSON   : ./comparison_results.json           ║"
echo "║  JEPA results      : ../checkpoints/finetune/            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
