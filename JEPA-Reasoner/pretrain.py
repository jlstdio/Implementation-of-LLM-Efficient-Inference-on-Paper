import json
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from accelerate import Accelerator

from jepa_reasoner import JEPAReasoner
from config import Config, get_config_from_cli


class StreamingPretrainDataset(IterableDataset):
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

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def pretrain(cfg: Config):
    pt  = cfg.pretrain
    mc  = cfg.model
    dc  = cfg.data

    # ── Accelerator (handles DDP + mixed precision) ───────────────────
    mixed_precision = "fp16" if pt.fp16 else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"Device: {device}  |  Num GPUs: {accelerator.num_processes}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    model = JEPAReasoner(
        vocab_size=vocab_size,
        embed_dim=mc.embed_dim,
        num_heads=mc.num_heads,
        ffn_dim=mc.ffn_dim,
        num_layers=mc.num_layers,
    )

    num_params = sum(p.numel() for p in model.parameters())
    model.hybrid_norm.rms_norm.weight.requires_grad = False

    per_device_batch = pt.batch_size // accelerator.num_processes
    assert pt.batch_size % accelerator.num_processes == 0, (
        f"batch_size ({pt.batch_size}) must be divisible by "
        f"num_processes ({accelerator.num_processes})"
    )

    dataset = StreamingPretrainDataset(
        tokenizer=tokenizer,
        max_length=dc.max_length,
        c4_ratio=dc.c4_ratio,
        seed=dc.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=pt.learning_rate,
        weight_decay=pt.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, pt.warmup_steps, pt.max_steps)
    criterion = nn.CrossEntropyLoss()

    # ── Prepare with Accelerator (DDP + device placement) ─────────────
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # LM head shares embedding weights (access via unwrapped model)
    raw_model = accelerator.unwrap_model(model)
    temp_lm_head = nn.Linear(mc.embed_dim, vocab_size, bias=False).to(device)
    temp_lm_head.weight = raw_model.embedding.weight

    if accelerator.is_main_process:
        os.makedirs(pt.output_dir, exist_ok=True)
        meta = {"vocab_size": vocab_size, "num_params": num_params}
        with open(os.path.join(pt.output_dir, "pretrain_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    if accelerator.is_main_process:
        print(f"\n{'=' * 60}")
        print(f"  Phase 1: Pretraining  |  Target: {pt.max_steps:,} steps")
        print(
            f"  Batch: {per_device_batch} x {accelerator.num_processes} GPUs"
            f" = {pt.batch_size}  |  SeqLen: {dc.max_length}  |  FP16: {pt.fp16}"
        )
        print(f"{'=' * 60}\n")

    model.train()
    global_step = 0
    running_loss = 0.0
    log_steps = 0
    start_time = time.time()

    while global_step < pt.max_steps:
        for input_ids, target_ids in dataloader:
            with accelerator.autocast():
                latents = model(input_ids, reasoning_steps=1).squeeze(1)
                logits  = temp_lm_head(latents)
                loss    = criterion(logits.view(-1, vocab_size), target_ids.view(-1))

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), pt.max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += loss.item()
            log_steps += 1

            if global_step % pt.log_interval == 0 and accelerator.is_main_process:
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

            if global_step % pt.save_interval == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    ckpt_path = os.path.join(pt.output_dir, f"checkpoint_step{global_step}.pt")
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save(
                        {
                            "step": global_step,
                            "model_state_dict": unwrapped.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        ckpt_path,
                    )
                    print(f"  ✓ Saved checkpoint → {ckpt_path}")

            if global_step >= pt.max_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = os.path.join(pt.output_dir, "jepa_pretrained_final.pt")
        unwrapped = accelerator.unwrap_model(model)
        torch.save(
            {"step": global_step, "model_state_dict": unwrapped.state_dict()},
            final_path,
        )
        # total_h = (time.time() - start_time) / 3600

if __name__ == "__main__":
    pretrain(get_config_from_cli("JEPA-Reasoner Phase 1: Pretraining"))
