#!/usr/bin/env python
"""
JEPA-Reasoner  ──  Phase 2-4: Fine-tuning on GSM8K
====================================================
Phase 2 — Math Fine-tuning       (42,000 steps, next-token prediction)
Phase 3 — Self-Supervised Training (SST, Scaled Cosine Loss + EMA)
Phase 4 — Talker Training         (DualTalker, Cross-Entropy)

평가(Evaluation)는 evaluate.py 에서 별도 실행.

Usage:
    python finetuning.py --config config.yaml
"""

import copy
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from jepa_reasoner import JEPAReasoner, DualTalker
from config import Config, get_config_from_cli


# ══════════════════════════════════════════════════════════════════════════════
#  Datasets
# ══════════════════════════════════════════════════════════════════════════════
class GSM8KCausalDataset(Dataset):
    """Phase 2 용 — 전체 시퀀스(질문+풀이)에 대한 next-token prediction 데이터셋."""

    def __init__(self, split: str, tokenizer, max_length: int = 512):
        super().__init__()
        self.pad_id = tokenizer.pad_token_id

        raw = load_dataset("openai/gsm8k", "main", split=split)

        self.inputs, self.targets = [], []
        for item in raw:
            text = (
                f"Question: {item['question']}\n"
                f"Step-by-step solution:\n{item['answer']}{tokenizer.eos_token}"
            )
            ids = tokenizer.encode(text, max_length=max_length + 1, truncation=True)
            ids = ids + [self.pad_id] * (max_length + 1 - len(ids))
            self.inputs.append(torch.tensor(ids[:-1], dtype=torch.long))
            self.targets.append(torch.tensor(ids[1:], dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class GSM8KSeq2SeqDataset(Dataset):
    """Phase 3-4 용 — (질문 → 풀이) 쌍 데이터셋."""

    def __init__(self, split: str, tokenizer, max_q_len: int = 256, max_a_len: int = 256):
        super().__init__()
        self.pad_id = tokenizer.pad_token_id

        raw = load_dataset("openai/gsm8k", "main", split=split)

        self.questions, self.answers = [], []
        self.raw_answers: list[str] = []

        for item in raw:
            q_text = f"Question: {item['question']}\nAnswer:"
            a_text = f"{item['answer']}{tokenizer.eos_token}"

            q_ids = tokenizer.encode(q_text, max_length=max_q_len, truncation=True)
            a_ids = tokenizer.encode(a_text, max_length=max_a_len, truncation=True)

            q_ids += [self.pad_id] * (max_q_len - len(q_ids))
            a_ids += [self.pad_id] * (max_a_len - len(a_ids))

            self.questions.append(torch.tensor(q_ids, dtype=torch.long))
            self.answers.append(torch.tensor(a_ids, dtype=torch.long))
            self.raw_answers.append(item["answer"])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  Loss Utilities
# ══════════════════════════════════════════════════════════════════════════════
def scaled_cosine_loss(pred: torch.Tensor, target: torch.Tensor, k: float = 4.0) -> torch.Tensor:
    """Scaled Cosine Distance Loss  (논문 Eq. 1)."""
    sim = F.cosine_similarity(pred, target, dim=-1)
    return (k * (1.0 - sim)).mean()


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, momentum: float = 0.98):
    """Target Encoder 의 EMA 업데이트."""
    for o_p, t_p in zip(online.parameters(), target.parameters()):
        t_p.data.mul_(momentum).add_(o_p.data, alpha=1.0 - momentum)


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2 — Math Fine-tuning
# ══════════════════════════════════════════════════════════════════════════════
def phase2_math_finetune(reasoner: JEPAReasoner, loader: DataLoader, cfg: Config, device):
    p2 = cfg.phase2

    vocab_size = reasoner.embedding.num_embeddings
    embed_dim  = reasoner.embedding.embedding_dim

    lm_head = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
    lm_head.weight = reasoner.embedding.weight

    optimizer = torch.optim.AdamW(reasoner.parameters(), lr=p2.learning_rate, weight_decay=0.01)
    pad_id    = loader.dataset.pad_id if hasattr(loader.dataset, "pad_id") else -100
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    reasoner.train()
    step = 0
    running_loss = 0.0
    t0 = time.time()

    print(f"\n{'─' * 55}")
    print(f"  Phase 2: Math Fine-tuning  │  Target {p2.max_steps:,} steps")
    print(f"{'─' * 55}")

    while step < p2.max_steps:
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)

            latents = reasoner(inp, reasoning_steps=1).squeeze(1)
            logits  = lm_head(latents)
            loss    = criterion(logits.view(-1, vocab_size), tgt.view(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reasoner.parameters(), 1.0)
            optimizer.step()

            step += 1
            running_loss += loss.item()

            if step % p2.log_interval == 0:
                avg = running_loss / p2.log_interval
                print(f"    Step [{step:>6,}/{p2.max_steps:,}]  Loss: {avg:.4f}")
                running_loss = 0.0

            if step >= p2.max_steps:
                break

    print(f"  ✓ Phase 2 complete  ({(time.time() - t0) / 60:.1f} min)")
    del lm_head
    return reasoner


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3 — Self-Supervised Training (SST)
# ══════════════════════════════════════════════════════════════════════════════
def phase3_sst(reasoner: JEPAReasoner, loader: DataLoader, cfg: Config, device):
    p3 = cfg.phase3

    # L2 정규화 레이어 활성화
    reasoner.hybrid_norm.rms_norm.weight.requires_grad = True

    target_enc = copy.deepcopy(reasoner.embedding).to(device)
    for p in target_enc.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(reasoner.parameters(), lr=p3.learning_rate, weight_decay=0.01)

    reasoner.train()
    t0 = time.time()

    print(f"\n{'─' * 55}")
    print(f"  Phase 3: SST (Self-Supervised Training)  │  {p3.epochs} epochs")
    print(f"{'─' * 55}")

    for epoch in range(1, p3.epochs + 1):
        epoch_loss, n = 0.0, 0
        for q_ids, a_ids in loader:
            q_ids, a_ids = q_ids.to(device), a_ids.to(device)

            pred = reasoner(q_ids, reasoning_steps=1).squeeze(1)

            with torch.no_grad():
                target = target_enc(a_ids)
                target = F.normalize(target, p=2, dim=-1)

            ml = min(pred.size(1), target.size(1))
            loss = scaled_cosine_loss(pred[:, :ml, :], target[:, :ml, :], k=p3.cosine_k)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reasoner.parameters(), 1.0)
            optimizer.step()

            update_ema(reasoner.embedding, target_enc, momentum=p3.ema_momentum)

            epoch_loss += loss.item()
            n += 1

        print(f"    Epoch {epoch}/{p3.epochs}  SST Loss: {epoch_loss / max(n, 1):.4f}")

    print(f"  ✓ Phase 3 complete  ({(time.time() - t0) / 60:.1f} min)")
    del target_enc
    return reasoner


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 4 — Talker Training
# ══════════════════════════════════════════════════════════════════════════════
def phase4_talker(reasoner: JEPAReasoner, talker: DualTalker, loader: DataLoader, cfg: Config, device):
    p4 = cfg.phase4

    vocab_size = talker.lm_head.out_features
    pad_id     = loader.dataset.pad_id if hasattr(loader.dataset, "pad_id") else -100

    reasoner.eval()
    for p in reasoner.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(talker.parameters(), lr=p4.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    talker.train()
    t0 = time.time()

    print(f"\n{'─' * 55}")
    print(f"  Phase 4: Talker Training  │  {p4.epochs} epochs")
    print(f"{'─' * 55}")

    for epoch in range(1, p4.epochs + 1):
        epoch_loss, n = 0.0, 0
        for q_ids, a_ids in loader:
            q_ids, a_ids = q_ids.to(device), a_ids.to(device)

            with torch.no_grad():
                latents = reasoner(q_ids, reasoning_steps=1).squeeze(1)

            dec_in  = a_ids[:, :-1]
            dec_tgt = a_ids[:, 1:]

            logits = talker(latents, dec_in)
            loss   = criterion(logits.reshape(-1, vocab_size), dec_tgt.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(talker.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        print(f"    Epoch {epoch}/{p4.epochs}  Talker Loss: {epoch_loss / max(n, 1):.4f}")

    print(f"  ✓ Phase 4 complete  ({(time.time() - t0) / 60:.1f} min)")
    return talker


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════
def main(cfg: Config):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # ── Reasoner 로드 ─────────────────────────────────────────────────────
    reasoner = JEPAReasoner(
        vocab_size=vocab_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        ffn_dim=cfg.model.ffn_dim,
        num_layers=cfg.model.num_layers,
    ).to(device)

    ckpt_path = cfg.finetune.pretrained_ckpt
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            reasoner.load_state_dict(ckpt["model_state_dict"])
        else:
            reasoner.load_state_dict(ckpt)
        print(f"✓ Loaded pretrained checkpoint: {ckpt_path}")
    else:
        print("⚠ No pretrained checkpoint — training from scratch.")

    # ── DualTalker ────────────────────────────────────────────────────────
    talker = DualTalker(
        vocab_size=vocab_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.talker.num_heads,
        ffn_dim=cfg.model.ffn_dim,
        num_enc_layers=cfg.talker.enc_layers,
        num_dec_layers=cfg.talker.dec_layers,
    ).to(device)

    r_params = sum(p.numel() for p in reasoner.parameters())
    t_params = sum(p.numel() for p in talker.parameters())
    print(f"Reasoner params: {r_params:,}  |  Talker params: {t_params:,}")

    # ── 데이터셋 ──────────────────────────────────────────────────────────
    print("\nDownloading / Loading GSM8K …")
    causal_ds  = GSM8KCausalDataset("train", tokenizer, cfg.data.max_length)
    seq2seq_ds = GSM8KSeq2SeqDataset("train", tokenizer, cfg.data.max_q_len, cfg.data.max_a_len)

    causal_loader = DataLoader(
        causal_ds,
        batch_size=cfg.phase2.batch_size,
        shuffle=True,
        num_workers=cfg.phase2.num_workers,
        pin_memory=True,
    )
    seq2seq_loader = DataLoader(
        seq2seq_ds,
        batch_size=cfg.phase2.batch_size,
        shuffle=True,
        num_workers=cfg.phase2.num_workers,
        pin_memory=True,
    )
    print(f"GSM8K train examples: {len(causal_ds)}")

    out_dir = cfg.finetune.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Phase 2 ───────────────────────────────────────────────────────────
    reasoner = phase2_math_finetune(reasoner, causal_loader, cfg, device)
    torch.save(reasoner.state_dict(), os.path.join(out_dir, "reasoner_after_phase2.pt"))

    # ── Phase 3 ───────────────────────────────────────────────────────────
    reasoner = phase3_sst(reasoner, seq2seq_loader, cfg, device)
    torch.save(reasoner.state_dict(), os.path.join(out_dir, "reasoner_after_phase3.pt"))

    # ── Phase 4 ───────────────────────────────────────────────────────────
    talker = phase4_talker(reasoner, talker, seq2seq_loader, cfg, device)
    torch.save(talker.state_dict(), os.path.join(out_dir, "talker_after_phase4.pt"))

    print(f"\n{'=' * 55}")
    print(f"  Fine-tuning (Phase 2-4) complete!")
    print(f"  Checkpoints → {out_dir}/")
    print(f"  Run evaluation:  python evaluate.py --config config.yaml")
    print(f"{'=' * 55}\n")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main(get_config_from_cli("JEPA-Reasoner GSM8K Fine-tuning (Phase 2-4)"))
