import argparse
import json
import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model

from config import Config, load_config
from TLM import (
    ElastiLM_TLM,
    add_slo_tokens,
    compress_prompt,
    level_to_ratio,
    ratio_to_level,
    RATIOS,
)
from model_elasticalize import (
    ElasticLlamaWrapper,
    load_elastic_model,
    apply_lora_to_model,
)


# ═══════════════════════════════════════════════════════════════════════
#  유틸리티
# ═══════════════════════════════════════════════════════════════════════

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, max_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════════════
#  Phase 1: LoRA Recovery (Alpaca-cleaned)
# ═══════════════════════════════════════════════════════════════════════

class AlpacaDataset(Dataset):
    """Alpaca-cleaned 데이터셋 for LoRA recovery."""

    def __init__(self, tokenizer, max_length: int = 512, max_samples: int = 50000,
                 vocab_size: int | None = None):
        super().__init__()
        # pad_token_id가 None이면 eos_token_id를 사용
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.pad_id = tokenizer.pad_token_id
        self.vocab_size = vocab_size

        raw = load_dataset("yahma/alpaca-cleaned", split="train")
        if max_samples and len(raw) > max_samples:
            raw = raw.select(range(max_samples))

        self.inputs, self.labels, self.masks = [], [], []
        for item in raw:
            # Alpaca 형식 → 프롬프트 + 응답
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            output = item.get("output", "")

            if inp:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}{tokenizer.eos_token}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}{tokenizer.eos_token}"

            # HuggingFace CausalLM은 내부에서 labels를 shift하므로 수동 shift 불필요
            ids = tokenizer.encode(text, max_length=max_length, truncation=True)
            seq_len = len(ids)

            # vocab_size 범위를 벗어나는 토큰 ID가 있으면 해당 위치를 무시
            if self.vocab_size is not None:
                labels = [
                    tid if 0 <= tid < self.vocab_size else -100
                    for tid in ids
                ]
            else:
                labels = list(ids)

            # 패딩: input_ids는 pad_token_id, labels는 -100 (ignore_index)
            input_ids = ids + [self.pad_id] * (max_length - seq_len)
            labels = labels + [-100] * (max_length - seq_len)
            attn_mask = [1] * seq_len + [0] * (max_length - seq_len)

            self.inputs.append(torch.tensor(input_ids, dtype=torch.long))
            self.labels.append(torch.tensor(labels, dtype=torch.long))
            self.masks.append(torch.tensor(attn_mask, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.masks[idx]


def train_lora_recovery(cfg: Config, accelerator: Accelerator):
    """
    Phase 1: 탄력화된 Llama 의 각 ratio별 LoRA 어댑터를 학습하여
    가지치기로 인한 정확도 손실을 보강합니다.
    """
    tcfg = cfg.train_lora
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"\n{'═' * 60}")
        print(f"  Phase 1: LoRA Recovery Training")
        print(f"  Model: {cfg.llm.name}")
        print(f"  Ratios: {cfg.elastic.ratios}")
        print(f"{'═' * 60}\n")

    # ── 탄력화 모델 로드 (tokenizer + vocab 확인) ────────────────────
    # device_map="auto"는 accelerator.prepare()와 충돌 → 학습 시에는 비활성화
    wrapper, tokenizer = load_elastic_model(cfg, device, use_device_map=False)

    model_vocab_size = wrapper.model.config.vocab_size
    lm_head_out = wrapper.model.lm_head.out_features
    tok_vocab_size = len(tokenizer)

    if accelerator.is_main_process:
        print(f"    Model  config.vocab_size : {model_vocab_size}")
        print(f"    Model  lm_head.out_feat  : {lm_head_out}")
        print(f"    Tokenizer len            : {tok_vocab_size}")

    # tokenizer vocab > model vocab → 임베딩 리사이즈 (cross_entropy OOR 방지)
    if tok_vocab_size > model_vocab_size:
        if accelerator.is_main_process:
            print(f"    ⚠ Resizing embeddings {model_vocab_size} → {tok_vocab_size}")
        wrapper.model.resize_token_embeddings(tok_vocab_size)
        model_vocab_size = tok_vocab_size

    # ── 데이터셋 (한 번만 생성) ──────────────────────────────────────
    dataset = AlpacaDataset(
        tokenizer=tokenizer,
        max_length=cfg.data.max_length,
        max_samples=cfg.data.alpaca_max_samples,
        vocab_size=model_vocab_size,
    )

    # ── 데이터 사전 검증 ─────────────────────────────────────────────
    if accelerator.is_main_process:
        all_labels = torch.cat([ds_lbl for _, ds_lbl, _ in dataset])
        valid = all_labels[all_labels != -100]
        print(f"    Dataset: {len(dataset)} samples, "
              f"label range [{valid.min().item()}, {valid.max().item()}], "
              f"must be < {model_vocab_size}")
        if valid.max().item() >= model_vocab_size:
            raise ValueError(
                f"Label {valid.max().item()} >= vocab_size {model_vocab_size}! "
                f"Tokenizer/model vocab mismatch."
            )

    per_device_batch = tcfg.batch_size // accelerator.num_processes
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    loader = accelerator.prepare(loader)

    # ── 각 ratio 별 LoRA 학습 ────────────────────────────────────────
    # 매 ratio마다 깨끗한 base model을 재로드합니다.
    # merge_and_unload()는 LoRA 가중치를 base model에 영구적으로 합치므로
    # 다음 ratio 학습 시 오염된 base 위에 중첩 LoRA가 적용되어
    # 모델 구조가 망가집니다 (8B → 3.2B 파라미터 축소 등).
    for ratio in cfg.elastic.ratios:
        ratio_str = f"{int(ratio * 100)}"

        if accelerator.is_main_process:
            print(f"\n  ── LoRA training for ratio={ratio:.0%} ──")

        # 매 ratio마다 깨끗한 base model 재로드
        wrapper_fresh, _ = load_elastic_model(cfg, device, use_device_map=False)
        base_model = wrapper_fresh.model

        # tokenizer/embedding 일관성
        if tok_vocab_size > base_model.config.vocab_size:
            base_model.resize_token_embeddings(tok_vocab_size)

        peft_model = apply_lora_to_model(base_model, cfg)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, peft_model.parameters()),
            lr=tcfg.learning_rate,
            weight_decay=tcfg.weight_decay,
        )

        max_steps = tcfg.epochs * len(loader) // tcfg.gradient_accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(optimizer, tcfg.warmup_steps, max_steps)

        peft_model, optimizer, scheduler = accelerator.prepare(
            peft_model, optimizer, scheduler
        )

        peft_model.train()
        global_step = 0
        running_loss = 0.0
        t0 = time.time()

        for epoch in range(1, tcfg.epochs + 1):
            for step_in_epoch, (inp, lbl, mask) in enumerate(loader):
                with accelerator.autocast():
                    outputs = peft_model(
                        input_ids=inp,
                        attention_mask=mask,
                        labels=lbl,
                    )
                    loss = outputs.loss / tcfg.gradient_accumulation_steps

                accelerator.backward(loss)
                running_loss += loss.item()

                if (step_in_epoch + 1) % tcfg.gradient_accumulation_steps == 0:
                    accelerator.clip_grad_norm_(peft_model.parameters(), tcfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % tcfg.log_interval == 0 and accelerator.is_main_process:
                        avg_loss = running_loss / tcfg.log_interval
                        lr_now = scheduler.get_last_lr()[0]
                        elapsed = time.time() - t0
                        print(
                            f"    Ratio {ratio_str}%  Step [{global_step:>6,}]  "
                            f"Loss: {avg_loss:.4f}  LR: {lr_now:.2e}  "
                            f"Elapsed: {elapsed:.0f}s"
                        )
                        running_loss = 0.0

        # LoRA 어댑터 저장
        if accelerator.is_main_process:
            lora_path = os.path.join(cfg.output.lora_dir, f"lora_ratio_{ratio_str}")
            os.makedirs(lora_path, exist_ok=True)
            unwrapped = accelerator.unwrap_model(peft_model)
            unwrapped.save_pretrained(lora_path)
            print(f"    ✓ Saved LoRA adapter → {lora_path}")

        # 정리: 모델 전체 해제 (다음 ratio에서 깨끗하게 재로드)
        del peft_model, optimizer, scheduler, base_model, wrapper_fresh
        accelerator.free_memory()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: TLM Score-head Training (MeetingBank)
# ═══════════════════════════════════════════════════════════════════════

class ScoreHeadDataset(Dataset):
    """
    MeetingBank 기반 토큰 중요도 데이터셋.
    요약에 포함된 토큰은 Retain(1), 나머지는 Discard(0)로 라벨링합니다.
    """

    def __init__(self, tokenizer, max_length: int = 512, split: str = "train"):
        super().__init__()
        self.pad_id = tokenizer.pad_token_id

        # MeetingBank: 회의 내용 + 요약
        try:
            raw = load_dataset("huuuyeah/MeetingBank", split=split)
        except Exception:
            # fallback: CNN/DailyMail
            raw = load_dataset("cnn_dailymail", "3.0.0", split="train")

        self.input_ids_list = []
        self.labels_list = []
        self.attention_masks = []

        for item in raw:
            # 원문 / 요약 추출
            source = item.get("source", item.get("article", ""))
            summary = item.get("summary", item.get("highlights", ""))

            if not source or not summary:
                continue

            src_ids = tokenizer.encode(source, max_length=max_length, truncation=True)
            sum_ids = set(tokenizer.encode(summary, add_special_tokens=False))

            # 라벨: summary에 속하는 토큰 → 1 (Retain), 아니면 → 0 (Discard)
            labels = [1 if tid in sum_ids else 0 for tid in src_ids]

            # 패딩
            pad_len = max_length - len(src_ids)
            input_ids = src_ids + [self.pad_id] * pad_len
            labels = labels + [-100] * pad_len  # 패딩은 무시
            mask = [1] * len(src_ids) + [0] * pad_len

            self.input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
            self.labels_list.append(torch.tensor(labels, dtype=torch.long))
            self.attention_masks.append(torch.tensor(mask, dtype=torch.long))

            if len(self.input_ids_list) >= 10000:
                break

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels_list[idx],
        }


def train_score_head(cfg: Config, accelerator: Accelerator):
    """
    Phase 2: TLM Score-head 학습.
    각 토큰이 요약/답변 생성에 중요한지 이진 분류.
    """
    tcfg = cfg.train_score
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"\n{'═' * 60}")
        print(f"  Phase 2: TLM Score-head Training (MeetingBank)")
        print(f"{'═' * 60}\n")

    # ── TLM 로드 ─────────────────────────────────────────────────────
    tlm_tokenizer = AutoTokenizer.from_pretrained(cfg.tlm.backbone)
    tlm = ElastiLM_TLM(
        backbone_name=cfg.tlm.backbone,
        shared_layers=cfg.tlm.shared_layers,
        num_prompt_levels=cfg.tlm.num_prompt_levels,
        num_model_levels=cfg.tlm.num_model_levels,
    )
    tlm_tokenizer = add_slo_tokens(tlm_tokenizer, tlm, cfg.tlm.slo_tokens)

    # ── 데이터셋 ─────────────────────────────────────────────────────
    dataset = ScoreHeadDataset(
        tokenizer=tlm_tokenizer,
        max_length=cfg.data.max_length,
        split=cfg.data.meetingbank_split,
    )

    per_device_batch = tcfg.batch_size // accelerator.num_processes
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        tlm.parameters(),
        lr=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
    )
    max_steps = tcfg.epochs * len(loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, tcfg.warmup_steps, max_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    tlm, optimizer, loader, scheduler = accelerator.prepare(
        tlm, optimizer, loader, scheduler
    )

    tlm.train()
    global_step = 0
    running_loss = 0.0
    t0 = time.time()

    for epoch in range(1, tcfg.epochs + 1):
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            with accelerator.autocast():
                token_scores, _, _ = tlm(input_ids, attention_mask)
                # token_scores: [B, L, 2], labels: [B, L]
                loss = criterion(
                    token_scores.view(-1, 2),
                    labels.view(-1),
                )

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(tlm.parameters(), tcfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += loss.item()

            if global_step % tcfg.log_interval == 0 and accelerator.is_main_process:
                avg = running_loss / tcfg.log_interval
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(
                    f"    Epoch {epoch}/{tcfg.epochs}  Step [{global_step:>6,}]  "
                    f"Loss: {avg:.4f}  LR: {lr_now:.2e}  Elapsed: {elapsed:.0f}s"
                )
                running_loss = 0.0

    # 저장
    if accelerator.is_main_process:
        os.makedirs(cfg.output.tlm_dir, exist_ok=True)
        ckpt_path = os.path.join(cfg.output.tlm_dir, "tlm_score_head.pt")
        unwrapped = accelerator.unwrap_model(tlm)
        torch.save(unwrapped.state_dict(), ckpt_path)
        tlm_tokenizer.save_pretrained(cfg.output.tlm_dir)
        print(f"  ✓ Saved TLM score-head → {ckpt_path}")

    return accelerator.unwrap_model(tlm)


# ═══════════════════════════════════════════════════════════════════════
#  Phase 3: TLM Decision-head Training (Self-induced Labeling)
# ═══════════════════════════════════════════════════════════════════════

class DecisionLabelDataset(Dataset):
    """Self-induced label 데이터셋."""

    def __init__(self, label_file: str, tokenizer, max_length: int = 512):
        super().__init__()
        self.pad_id = tokenizer.pad_token_id

        with open(label_file) as f:
            self.records = [json.loads(line) for line in f]

        self.input_ids_list = []
        self.attention_masks = []
        self.prompt_labels = []
        self.model_labels = []

        for rec in self.records:
            text = rec["slo_token"] + " " + rec["prompt"]
            enc = tokenizer(
                text, max_length=max_length, truncation=True,
                padding="max_length", return_tensors="pt",
            )
            self.input_ids_list.append(enc["input_ids"].squeeze(0))
            self.attention_masks.append(enc["attention_mask"].squeeze(0))
            self.prompt_labels.append(rec["best_prompt_level"])
            self.model_labels.append(rec["best_model_level"])

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids_list[idx],
            "attention_mask": self.attention_masks[idx],
            "prompt_label": self.prompt_labels[idx],
            "model_label": self.model_labels[idx],
        }


def generate_self_induced_labels(
    cfg: Config,
    device: torch.device,
):
    """
    MMLU-Pro를 사용하여 self-induced labeling 수행.
    각 프롬프트 + SLO 조합에 대해 최적 (prompt_level, model_level) 라벨 생성.
    """
    label_dir = cfg.train_decision.label_cache_dir
    os.makedirs(label_dir, exist_ok=True)
    label_file = os.path.join(label_dir, f"decision_labels_{cfg.llm.short_name}.jsonl")

    if os.path.exists(label_file):
        print(f"  ✓ Label file already exists: {label_file}")
        return label_file

    print("  Generating self-induced labels …")

    # 대상 LLM 로드 (가벼운 평가용)
    wrapper, llm_tokenizer = load_elastic_model(cfg, device)

    # MMLU-Pro 데이터
    try:
        mmlu = load_dataset("TIGER-Lab/MMLU-Pro", split=cfg.data.mmlu_pro_split)
    except Exception:
        mmlu = load_dataset("cais/mmlu", "all", split="test")

    slo_tokens = cfg.tlm.slo_tokens
    records = []

    for idx, item in enumerate(mmlu):
        if idx >= 500:  # 라벨 생성 상한
            break

        question = item.get("question", item.get("input", ""))
        choices = item.get("options", item.get("choices", []))
        answer_idx = item.get("answer_index", item.get("answer", 0))
        if isinstance(answer_idx, str):
            answer_idx = ord(answer_idx.upper()) - ord('A')

        if not question or not choices:
            continue

        # 선택지 포맷
        choice_str = "\n".join(
            [f"({chr(65+i)}) {c}" for i, c in enumerate(choices)]
        )
        prompt = f"Question: {question}\n{choice_str}\nAnswer:"

        for slo_token in slo_tokens:
            best_p, best_m = None, None
            min_cost = float("inf")

            for p_idx, p_ratio in enumerate(RATIOS):
                for m_idx, m_ratio in enumerate(RATIOS):
                    # 비용 = 작을수록 가벼움
                    cost = p_ratio * m_ratio

                    if cost >= min_cost:
                        continue

                    # 간이 정답 확인 (프로토타입: 풀 모델로 생성)
                    try:
                        enc = llm_tokenizer(
                            prompt, return_tensors="pt",
                            max_length=cfg.data.max_length, truncation=True,
                        ).to(device)
                        with torch.no_grad():
                            out = wrapper.generate(
                                input_ids=enc["input_ids"],
                                attention_mask=enc["attention_mask"],
                                max_new_tokens=32,
                            )
                        gen_text = llm_tokenizer.decode(out[0], skip_special_tokens=True)

                        # 정답 확인
                        pred_letter = ""
                        for ch in gen_text.upper():
                            if ch in "ABCDEFGHIJ":
                                pred_letter = ch
                                break

                        correct_letter = chr(65 + answer_idx)
                        if pred_letter == correct_letter:
                            min_cost = cost
                            best_p = p_idx
                            best_m = m_idx
                    except Exception:
                        continue

            if best_p is None:
                best_p = len(RATIOS) - 1  # fallback: 100%
                best_m = len(RATIOS) - 1

            records.append({
                "prompt": prompt[:512],
                "slo_token": slo_token,
                "best_prompt_level": best_p,
                "best_model_level": best_m,
            })

        if (idx + 1) % 50 == 0:
            print(f"    Labeled {idx + 1} questions ({len(records)} records)")

    # 저장
    with open(label_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"  ✓ Generated {len(records)} labels → {label_file}")
    return label_file


def train_decision_head(cfg: Config, accelerator: Accelerator):
    """
    Phase 3: TLM Decision-head 학습.
    Self-induced labeling 결과를 사용하여 전략 분류기를 학습합니다.
    """
    tcfg = cfg.train_decision
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"\n{'═' * 60}")
        print(f"  Phase 3: TLM Decision-head Training (Self-induced)")
        print(f"{'═' * 60}\n")

    # ── Self-induced label 생성 (메인 프로세스만) ─────────────────────
    if accelerator.is_main_process:
        label_file = generate_self_induced_labels(cfg, device)
    accelerator.wait_for_everyone()
    label_file = os.path.join(
        tcfg.label_cache_dir, f"decision_labels_{cfg.llm.short_name}.jsonl"
    )

    # ── TLM 로드 (Score-head 가중치 포함) ────────────────────────────
    tlm_tokenizer = AutoTokenizer.from_pretrained(cfg.output.tlm_dir)
    tlm = ElastiLM_TLM(
        backbone_name=cfg.tlm.backbone,
        shared_layers=cfg.tlm.shared_layers,
        num_prompt_levels=cfg.tlm.num_prompt_levels,
        num_model_levels=cfg.tlm.num_model_levels,
    )
    score_ckpt = os.path.join(cfg.output.tlm_dir, "tlm_score_head.pt")
    if os.path.exists(score_ckpt):
        tlm.load_state_dict(torch.load(score_ckpt, map_location="cpu", weights_only=False))
        print(f"  ✓ Loaded score-head weights: {score_ckpt}")

    # ── 데이터셋 ─────────────────────────────────────────────────────
    dataset = DecisionLabelDataset(
        label_file=label_file,
        tokenizer=tlm_tokenizer,
        max_length=cfg.data.max_length,
    )

    per_device_batch = tcfg.batch_size // accelerator.num_processes
    loader = DataLoader(
        dataset,
        batch_size=per_device_batch,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Decision-head 파라미터만 학습 (Score-head freeze)
    for name, param in tlm.named_parameters():
        if "score" in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, tlm.parameters()),
        lr=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
    )
    max_steps = tcfg.epochs * len(loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, tcfg.warmup_steps, max_steps)
    criterion = nn.CrossEntropyLoss()

    tlm, optimizer, loader, scheduler = accelerator.prepare(
        tlm, optimizer, loader, scheduler
    )

    tlm.train()
    global_step = 0
    running_loss = 0.0
    t0 = time.time()

    for epoch in range(1, tcfg.epochs + 1):
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            p_labels = batch["prompt_label"]
            m_labels = batch["model_label"]

            with accelerator.autocast():
                _, prompt_logits, model_logits = tlm(input_ids, attention_mask)
                loss_p = criterion(prompt_logits, p_labels)
                loss_m = criterion(model_logits, m_labels)
                loss = loss_p + loss_m

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(tlm.parameters(), tcfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += loss.item()

            if global_step % tcfg.log_interval == 0 and accelerator.is_main_process:
                avg = running_loss / tcfg.log_interval
                lr_now = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                print(
                    f"    Epoch {epoch}/{tcfg.epochs}  Step [{global_step:>6,}]  "
                    f"Loss: {avg:.4f}  (P:{loss_p.item():.3f} M:{loss_m.item():.3f})  "
                    f"LR: {lr_now:.2e}  Elapsed: {elapsed:.0f}s"
                )
                running_loss = 0.0

    # 저장
    if accelerator.is_main_process:
        ckpt_path = os.path.join(cfg.output.tlm_dir, "tlm_full.pt")
        unwrapped = accelerator.unwrap_model(tlm)
        torch.save(unwrapped.state_dict(), ckpt_path)
        print(f"  ✓ Saved TLM (full) → {ckpt_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ElastiLM Training Pipeline")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["all", "lora", "score", "decision"],
        help="학습 단계 선택 (default: all)",
    )
    args, _ = parser.parse_known_args()
    cfg = load_config(args.config)

    accelerator = Accelerator(
        mixed_precision="fp16" if cfg.train_lora.fp16 else "no"
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"\n{'╔' + '═' * 58 + '╗'}")
        print(f"║{'ElastiLM  ·  Training Pipeline':^58}║")
        print(f"║{'Model: ' + cfg.llm.short_name:^58}║")
        print(f"║{'Phase: ' + args.phase:^58}║")
        print(f"{'╚' + '═' * 58 + '╝'}\n")

    # Phase 실행
    if args.phase in ("all", "lora"):
        train_lora_recovery(cfg, accelerator)
        accelerator.wait_for_everyone()

    if args.phase in ("all", "score"):
        train_score_head(cfg, accelerator)
        accelerator.wait_for_everyone()

    if args.phase in ("all", "decision"):
        train_decision_head(cfg, accelerator)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f"\n  ★  Training complete!")
        print(f"     TLM checkpoints   : {cfg.output.tlm_dir}")
        print(f"     LoRA checkpoints  : {cfg.output.lora_dir}")
        print()


if __name__ == "__main__":
    main()
