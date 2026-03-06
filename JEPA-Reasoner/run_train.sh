#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  JEPA-Reasoner  ──  Full Training Pipeline
#  Phase 1 : Pretrain     (C4 + Wikitext,  300 k steps)
#  Phase 2 : Math FT      (GSM8K,  42 k steps)
#  Phase 3 : SST          (GSM8K,  Self-Supervised Latent Alignment)
#  Phase 4 : Talker       (GSM8K,  DualTalker)
#  Phase 5 : Evaluation   (GSM8K test set)
#
#  모든 하이퍼파라미터는  config.yaml  에서 관리됩니다.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${1:-config.yaml}"        # 첫 번째 인자로 config 경로 지정 가능

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║       JEPA-Reasoner  ·  Full Training Pipeline           ║"
echo "║       Config : ${CONFIG}                                 ║"
echo "╚═══════════════════════════════════════════════════════════╝"

# ── 0. 의존 패키지 설치 ─────────────────────────────────────────────
echo ""
echo "[0/4] Installing Python dependencies …"
pip install --quiet --upgrade \
    torch \
    transformers \
    datasets \
    sentencepiece \
    pyyaml

# ── 1. 데이터셋 사전 다운로드 ────────────────────────────────────────
echo ""
echo "[1/4] Pre-downloading datasets …"
python -c "
from datasets import load_dataset
print('  → Wikitext-103 (streaming, 캐시 확인)')
_ = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=True)
print('  → GSM8K (train + test)')
_ = load_dataset('openai/gsm8k', 'main', split='train')
_ = load_dataset('openai/gsm8k', 'main', split='test')
print('  → C4 는 학습 중 스트리밍됩니다.')
print('  ✓ 데이터셋 준비 완료')
"

# ── 2. Phase 1: Pretraining ──────────────────────────────────────────
echo ""
echo "[2/4] Phase 1 — Pretraining on C4 + Wikitext …"
python pretrain.py --config "$CONFIG"
echo "  ✓ Phase 1 complete"

# ── 3. Phase 2–4: Fine-tuning ────────────────────────────────────────
echo ""
echo "[3/4] Phase 2-4 — Fine-tuning on GSM8K …"
python finetuning.py --config "$CONFIG"
echo "  ✓ Phase 2-4 complete"

# ── 4. Evaluation ────────────────────────────────────────────────────
echo ""
echo "[4/4] Evaluation on GSM8K test set …"
python evaluate.py --config "$CONFIG"
echo "  ✓ Evaluation complete"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  All done!                                               ║"
echo "║  Config        : ${CONFIG}                               ║"
echo "║  Pretrain ckpt : ./checkpoints/pretrain/                 ║"
echo "║  Finetune ckpt : ./checkpoints/finetune/                 ║"
echo "║  Eval results  : ./checkpoints/finetune/eval_results.json║"
echo "╚═══════════════════════════════════════════════════════════╝"
