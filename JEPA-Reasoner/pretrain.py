#!/usr/bin/env python
"""
JEPA-Reasoner  ──  Phase 1: Pretraining on C4 + Wikitext
=========================================================
* 300,000 steps (논문 명시)
* 목적: 기본 언어 능력 및 세상 지식 습득
* L2 정규화 비활성화, Tied Embedding LM Head 사용
* C4 70 % + Wikitext-103 30 % 혼합 스트리밍 데이터

Usage:
    python pretrain.py --config config.yaml
"""

import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

from jepa_reasoner import JEPAReasoner
from config import Config, get_config_from_cli


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class StreamingPretrainDataset(IterableDataset):
    """C4 + Wikitext-103 을 결합한 스트리밍 사전학습 데이터셋.

    텍스트를 토큰화한 뒤 `max_length` 단위로 잘라서
    (input_ids, target_ids) 쌍을 생성한다 (next-token prediction).
    """

    def __init__(self, tokenizer, max_length: int = 512, c4_ratio: float = 0.7, seed: int = 42):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.c4_ratio = c4_ratio
        self.seed = seed

    def _create_stream(self):
        c4 = load_dataset(
            "allenai/c4", "en",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
        wiki = load_dataset(
            "wikitext", "wikitext-103-raw-v1",
            split="train",
            streaming=True,
        )
        combined = interleave_datasets(
            [c4, wiki],
            probabilities=[self.c4_ratio, 1.0 - self.c4_ratio],
            seed=self.seed,
            stopping_strategy="all_exhausted",
        )
        return combined

    def __iter__(self):
        stream = self._create_stream()
        buffer: list[int] = []

        for example in stream:
            text = example.get("text", "")
            if not text or len(text.strip()) < 20:
                continue

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.max_length + 1:
                chunk = buffer[: self.max_length + 1]
                buffer = buffer[self.max_length :]

                input_ids  = torch.tensor(chunk[:-1], dtype=torch.long)
                target_ids = torch.tensor(chunk[1:],  dtype=torch.long)
                yield input_ids, target_ids


# ──────────────────────────────────────────────────────────────────────────────
# LR Scheduler
# ──────────────────────────────────────────────────────────────────────────────
def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, max_steps: int):
    """Warmup → Cosine decay 스케줄러."""

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Pretraining
# ──────────────────────────────────────────────────────────────────────────────
def pretrain(cfg: Config):
    pt  = cfg.pretrain          # PretrainConfig
    mc  = cfg.model             # ModelConfig
    dc  = cfg.data              # DataConfig

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────
    model = JEPAReasoner(
        vocab_size=vocab_size,
        embed_dim=mc.embed_dim,
        num_heads=mc.num_heads,
        ffn_dim=mc.ffn_dim,
        num_layers=mc.num_layers,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # [Phase-1 설정 1] L2 정규화 비활성화
    model.hybrid_norm.rms_norm.weight.requires_grad = False

    # [Phase-1 설정 2] Tied Embeddings
    temp_lm_head = nn.Linear(mc.embed_dim, vocab_size, bias=False).to(device)
    temp_lm_head.weight = model.embedding.weight

    # ── Dataset & DataLoader ──────────────────────────────────────────────
    dataset = StreamingPretrainDataset(
        tokenizer=tokenizer,
        max_length=dc.max_length,
        c4_ratio=dc.c4_ratio,
        seed=dc.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=pt.batch_size,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pt.learning_rate,
        weight_decay=pt.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, pt.warmup_steps, pt.max_steps)
    criterion = nn.CrossEntropyLoss()

    # ── Mixed Precision ───────────────────────────────────────────────────
    use_amp = pt.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # ── 출력 디렉토리 ─────────────────────────────────────────────────────
    os.makedirs(pt.output_dir, exist_ok=True)

    meta = {"vocab_size": vocab_size, "num_params": num_params}
    with open(os.path.join(pt.output_dir, "pretrain_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── 학습 루프 ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Phase 1: Pretraining  |  Target: {pt.max_steps:,} steps")
    print(f"  Batch: {pt.batch_size}  |  SeqLen: {dc.max_length}  |  FP16: {use_amp}")
    print(f"{'=' * 60}\n")

    model.train()
    global_step = 0
    running_loss = 0.0
    log_steps = 0
    start_time = time.time()

    while global_step < pt.max_steps:
        for input_ids, target_ids in dataloader:
            input_ids  = input_ids.to(device)
            target_ids = target_ids.to(device)

            if use_amp:
                with torch.amp.autocast("cuda"):
                    latents = model(input_ids, reasoning_steps=1).squeeze(1)
                    logits  = temp_lm_head(latents)
                    loss    = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), pt.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                latents = model(input_ids, reasoning_steps=1).squeeze(1)
                logits  = temp_lm_head(latents)
                loss    = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), pt.max_grad_norm)
                optimizer.step()

            scheduler.step()
            global_step += 1
            running_loss += loss.item()
            log_steps += 1

            # ── 로깅 ──
            if global_step % pt.log_interval == 0:
                avg_loss = running_loss / log_steps
                elapsed  = time.time() - start_time
                speed    = global_step / elapsed
                lr_now   = scheduler.get_last_lr()[0]
                eta_h    = (pt.max_steps - global_step) / max(speed, 1e-9) / 3600

                print(
                    f"Step [{global_step:>7,}/{pt.max_steps:,}]  "
                    f"Loss: {avg_loss:.4f}  LR: {lr_now:.2e}  "
                    f"Speed: {speed:.1f} steps/s  ETA: {eta_h:.1f}h"
                )
                running_loss = 0.0
                log_steps = 0

            # ── 체크포인트 ──
            if global_step % pt.save_interval == 0:
                ckpt_path = os.path.join(pt.output_dir, f"checkpoint_step{global_step}.pt")
                torch.save(
                    {
                        "step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"  ✓ Saved checkpoint → {ckpt_path}")

            if global_step >= pt.max_steps:
                break

    # ── 최종 저장 ─────────────────────────────────────────────────────────
    final_path = os.path.join(pt.output_dir, "jepa_pretrained_final.pt")
    torch.save(
        {"step": global_step, "model_state_dict": model.state_dict()},
        final_path,
    )
    total_h = (time.time() - start_time) / 3600
    print(f"\n{'=' * 60}")
    print(f"  Pretraining complete!  ({total_h:.1f} h)")
    print(f"  Final model → {final_path}")
    print(f"{'=' * 60}\n")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pretrain(get_config_from_cli("JEPA-Reasoner Phase 1: Pretraining"))
