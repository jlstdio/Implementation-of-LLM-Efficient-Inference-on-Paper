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
from accelerate import Accelerator

from jepa_reasoner import JEPAReasoner, DualTalker
from config import Config, get_config_from_cli


class GSM8KCausalDataset(Dataset):

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


def scaled_cosine_loss(pred: torch.Tensor, target: torch.Tensor, k: float = 4.0) -> torch.Tensor:
    sim = F.cosine_similarity(pred, target, dim=-1)
    return (k * (1.0 - sim)).mean()


@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, momentum: float = 0.98):
    for o_p, t_p in zip(online.parameters(), target.parameters()):
        t_p.data.mul_(momentum).add_(o_p.data, alpha=1.0 - momentum)


def phase2_math_finetune(reasoner: JEPAReasoner, loader: DataLoader, cfg: Config, accelerator: Accelerator):
    p2 = cfg.phase2
    device = accelerator.device

    vocab_size = reasoner.embedding.num_embeddings
    embed_dim  = reasoner.embedding.embedding_dim

    optimizer = torch.optim.AdamW(reasoner.parameters(), lr=p2.learning_rate, weight_decay=0.01)
    pad_id    = loader.dataset.pad_id if hasattr(loader.dataset, "pad_id") else -100
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    reasoner, optimizer = accelerator.prepare(reasoner, optimizer)

    raw_reasoner = accelerator.unwrap_model(reasoner)
    lm_head = nn.Linear(embed_dim, vocab_size, bias=False).to(device)
    lm_head.weight = raw_reasoner.embedding.weight

    reasoner.train()
    step = 0
    running_loss = 0.0
    t0 = time.time()

    if accelerator.is_main_process:
        print(f"GSM Fine-tuning  │  Target {p2.max_steps:,} steps")

    while step < p2.max_steps:
        for inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)

            with accelerator.autocast():
                latents = reasoner(inp, reasoning_steps=1).squeeze(1)
                logits  = lm_head(latents)
                loss    = criterion(logits.view(-1, vocab_size), tgt.view(-1))

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(reasoner.parameters(), 1.0)
            optimizer.step()

            step += 1
            running_loss += loss.item()

            if step % p2.log_interval == 0 and accelerator.is_main_process:
                avg = running_loss / p2.log_interval
                print(f"Step [{step:>6,}/{p2.max_steps:,}]  Loss: {avg:.4f}")
                running_loss = 0.0

            if step >= p2.max_steps:
                break
    del lm_head
    reasoner = accelerator.unwrap_model(reasoner)
    accelerator.free_memory()
    return reasoner


def phase3_sst(reasoner: JEPAReasoner, loader: DataLoader, cfg: Config, accelerator: Accelerator):
    p3 = cfg.phase3
    device = accelerator.device

    reasoner.hybrid_norm.rms_norm.weight.requires_grad = True

    target_enc = copy.deepcopy(reasoner.embedding).to(device)
    for p in target_enc.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(reasoner.parameters(), lr=p3.learning_rate, weight_decay=0.01)

    reasoner, optimizer = accelerator.prepare(reasoner, optimizer)

    reasoner.train()
    t0 = time.time()

    if accelerator.is_main_process:
        print(f"SST (Self-Supervised Training) │ {p3.epochs} epochs")

    for epoch in range(1, p3.epochs + 1):
        if hasattr(loader, 'set_epoch'):
            loader.set_epoch(epoch)
        epoch_loss, n = 0.0, 0
        for q_ids, a_ids in loader:
            q_ids, a_ids = q_ids.to(device), a_ids.to(device)

            with accelerator.autocast():
                pred = reasoner(q_ids, reasoning_steps=1).squeeze(1)

                with torch.no_grad():
                    target = target_enc(a_ids)
                    target = F.normalize(target, p=2, dim=-1)

                ml = min(pred.size(1), target.size(1))
                loss = scaled_cosine_loss(pred[:, :ml, :], target[:, :ml, :], k=p3.cosine_k)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(reasoner.parameters(), 1.0)
            optimizer.step()

            # EMA update
            raw_reasoner = accelerator.unwrap_model(reasoner)
            update_ema(raw_reasoner.embedding, target_enc, momentum=p3.ema_momentum)

            epoch_loss += loss.item()
            n += 1

        if accelerator.is_main_process:
            print(f"Epoch {epoch}/{p3.epochs}  SST Loss: {epoch_loss / max(n, 1):.4f}")

    del target_enc
    reasoner = accelerator.unwrap_model(reasoner)
    accelerator.free_memory()
    return reasoner

def phase4_talker(reasoner: JEPAReasoner, talker: DualTalker, loader: DataLoader, cfg: Config, accelerator: Accelerator):
    p4 = cfg.phase4
    device = accelerator.device

    vocab_size = talker.lm_head.out_features
    pad_id     = loader.dataset.pad_id if hasattr(loader.dataset, "pad_id") else -100

    # Freeze reasoner
    reasoner.eval()
    for p in reasoner.parameters():
        p.requires_grad = False
    reasoner = reasoner.to(device)

    optimizer = torch.optim.AdamW(talker.parameters(), lr=p4.learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    talker, optimizer = accelerator.prepare(talker, optimizer)

    talker.train()
    t0 = time.time()

    for epoch in range(1, p4.epochs + 1):
        if hasattr(loader, 'set_epoch'):
            loader.set_epoch(epoch)
        epoch_loss, n = 0.0, 0
        for q_ids, a_ids in loader:
            q_ids, a_ids = q_ids.to(device), a_ids.to(device)

            with torch.no_grad():
                with accelerator.autocast():
                    latents = reasoner(q_ids, reasoning_steps=1).squeeze(1)

            with accelerator.autocast():
                dec_in  = a_ids[:, :-1]
                dec_tgt = a_ids[:, 1:]

                logits = talker(latents, dec_in)
                loss   = criterion(logits.reshape(-1, vocab_size), dec_tgt.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(talker.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        if accelerator.is_main_process:
            print(f"Epoch {epoch}/{p4.epochs}  Talker Loss: {epoch_loss / max(n, 1):.4f}")

    talker = accelerator.unwrap_model(talker)
    accelerator.free_memory()
    return talker

def main(cfg: Config):
    accelerator = Accelerator()
    # device = accelerator.device

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    reasoner = JEPAReasoner(
        vocab_size=vocab_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.model.num_heads,
        ffn_dim=cfg.model.ffn_dim,
        num_layers=cfg.model.num_layers,
    )

    ckpt_path = cfg.finetune.pretrained_ckpt
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            reasoner.load_state_dict(ckpt["model_state_dict"])
        else:
            reasoner.load_state_dict(ckpt)
        if accelerator.is_main_process:
            print(f"Loaded pretrained checkpoint: {ckpt_path}")

    talker = DualTalker(
        vocab_size=vocab_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.talker.num_heads,
        ffn_dim=cfg.model.ffn_dim,
        num_enc_layers=cfg.talker.enc_layers,
        num_dec_layers=cfg.talker.dec_layers,
    )

    causal_ds  = GSM8KCausalDataset("train", tokenizer, cfg.data.max_length)
    seq2seq_ds = GSM8KSeq2SeqDataset("train", tokenizer, cfg.data.max_q_len, cfg.data.max_a_len)

    per_device_batch = cfg.phase2.batch_size // accelerator.num_processes
    assert cfg.phase2.batch_size % accelerator.num_processes == 0, (
        f"batch_size ({cfg.phase2.batch_size}) must be divisible by "
        f"num_processes ({accelerator.num_processes})"
    )

    causal_loader = DataLoader(
        causal_ds,
        batch_size=per_device_batch,
        shuffle=True,
        num_workers=cfg.phase2.num_workers,
        pin_memory=True,
    )
    seq2seq_loader = DataLoader(
        seq2seq_ds,
        batch_size=per_device_batch,
        shuffle=True,
        num_workers=cfg.phase2.num_workers,
        pin_memory=True,
    )

    # Prepare dataloaders
    causal_loader, seq2seq_loader = accelerator.prepare(causal_loader, seq2seq_loader)

    out_dir = cfg.finetune.output_dir
    if accelerator.is_main_process:
        os.makedirs(out_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # math finetune    
    reasoner = phase2_math_finetune(reasoner, causal_loader, cfg, accelerator)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(reasoner.state_dict(), os.path.join(out_dir, "reasoner_after_phase2.pt"))

    # Phase 3
    reasoner = phase3_sst(reasoner, seq2seq_loader, cfg, accelerator)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(reasoner.state_dict(), os.path.join(out_dir, "reasoner_after_phase3.pt"))

    # Phase 4
    talker = phase4_talker(reasoner, talker, seq2seq_loader, cfg, accelerator)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(talker.state_dict(), os.path.join(out_dir, "talker_after_phase4.pt"))

if __name__ == "__main__":
    main(get_config_from_cli("JEPA-Reasoner GSM8K Fine-tuning"))
