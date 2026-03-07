#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  ElastiLM  ──  Full Pipeline  (Llama 3.1 8B)
#
#  Step 0 : Elasticalize   (중요도 프로파일링 + 순열 재정렬)
#  Step 1 : LoRA Recovery   (Alpaca-cleaned,  ratio별 LoRA)
#  Step 2 : TLM Score-head  (MeetingBank,  토큰 중요도)
#  Step 3 : TLM Decision    (MMLU-Pro,  Self-induced Labeling)
#  Step 4 : Evaluation      (ARC-E, PIQA, MMLU-Pro  5-shot)
#
#  모든 하이퍼파라미터는  config.yaml  에서 관리됩니다.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-config.yaml}"
GPUS="${2:-}"                     # 두 번째 인자: 사용할 단일 GPU (예: "1" 또는 "2")

# ── GPU 설정 (단일 GPU) ──────────────────────────────────────────────
if [[ -n "$GPUS" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
fi

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║         ElastiLM  ·  Full Training Pipeline              ║"
echo "║         Config : ${CONFIG}                               ║"
echo "║         GPU    : CUDA ${CUDA_VISIBLE_DEVICES:-all}       ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# ── Step 0: Elasticalize ─────────────────────────────────────────────
echo ""
echo "[0/4] Step 0 — Elasticalize model (importance profiling + reorder) …"
python model_elasticalize.py --config "$CONFIG"
echo "  ✓ Step 0 complete"

# ── Step 1: LoRA Recovery ────────────────────────────────────────────
echo ""
echo "[1/4] Step 1 — LoRA Recovery Training (Alpaca-cleaned) …"
python train.py --config "$CONFIG" --phase lora
echo "  ✓ Step 1 complete"

# ── Step 2: TLM Score-head ───────────────────────────────────────────
echo ""
echo "[2/4] Step 2 — TLM Score-head Training (MeetingBank) …"
python train.py --config "$CONFIG" --phase score
echo "  ✓ Step 2 complete"

# ── Step 3: TLM Decision-head ────────────────────────────────────────
echo ""
echo "[3/4] Step 3 — TLM Decision-head Training (Self-induced) …"
python train.py --config "$CONFIG" --phase decision
echo "  ✓ Step 3 complete"

# ── Step 4: Evaluation ───────────────────────────────────────────────
echo ""
echo "[4/4] Step 4 — Full Evaluation …"
python evaluate.py --config "$CONFIG" --mode full
echo "  ✓ Step 4 complete"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  All done!                                               ║"
echo "║  Config           : ${CONFIG}                            ║"
echo "║  Elastic ckpt     : ./checkpoints/elastic/               ║"
echo "║  LoRA ckpts       : ./checkpoints/lora/                  ║"
echo "║  TLM ckpts        : ./checkpoints/tlm/                   ║"
echo "║  Eval results     : ./checkpoints/eval/                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
