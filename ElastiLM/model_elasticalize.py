"""
1. 중요도 프로파일링 (Gradient × Weight)
2. Attention 헤드 / MLP 뉴런 단위 재정렬
3. 앵커 레이어 보호
4. 런타임 슬라이싱 → Elastic Sub-model
5. LoRA Recovery (peft)
"""
import os
import math
import json
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoConfig)
from peft import LoraConfig, get_peft_model, PeftModel
from config import Config


class SimpleTextDataset(Dataset):

    def __init__(self, tokenizer, dataset_name: str, max_length: int, max_samples: int):
        super().__init__()
        from datasets import load_dataset

        if dataset_name == "wikitext":
            raw = load_dataset(
                "wikitext", "wikitext-103-raw-v1",
                split="train", streaming=True,
            )
        else:
            raw = load_dataset(dataset_name, split="train", streaming=True)

        self.samples = []
        for i, item in enumerate(raw):
            if i >= max_samples:
                break
            text = item.get("text", "")
            if len(text.strip()) < 20:
                continue
            enc = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            self.samples.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        attention_mask[i, :L] = b["attention_mask"]
        labels[i, :L-1] = b["input_ids"][1:]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def compute_importance_scores(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:

    model.train()
    importance = {
        name: torch.zeros_like(param, device="cpu")
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    num_batches = 0

    # device_map 모델인 경우 입력은 첫 번째 레이어의 device로 보냄
    if hasattr(model, "hf_device_map"):
        input_device = next(model.parameters()).device
    else:
        input_device = device

    for batch in dataloader:
        batch = {k: v.to(input_device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and name in importance:
                    importance[name] += torch.abs(param.grad * param.data).cpu()
        model.zero_grad()
        num_batches += 1

    if num_batches > 0:
        for name in importance:
            importance[name] /= num_batches

    model.eval()
    return importance

def _get_head_dim_info(config):
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = config.hidden_size // num_heads
    return num_heads, num_kv_heads, head_dim


def reorder_mlp_units(
    layer: nn.Module,
    importance: dict[str, torch.Tensor],
    layer_idx: int,
    prefix: str = "model.layers",
):
    gate_key = f"{prefix}.{layer_idx}.mlp.gate_proj.weight"
    up_key = f"{prefix}.{layer_idx}.mlp.up_proj.weight"
    down_key = f"{prefix}.{layer_idx}.mlp.down_proj.weight"

    if gate_key not in importance:
        return

    # 뉴런별 중요도 합산
    mlp_imp = (
        importance[gate_key].sum(dim=1)   # gate_proj: [inter, hidden] → sum over hidden
        + importance[up_key].sum(dim=1)   # up_proj:   [inter, hidden] → sum over hidden
        + importance[down_key].sum(dim=0) # down_proj: [hidden, inter] → sum over inter
    )

    sorted_indices = torch.argsort(mlp_imp, descending=True)

    with torch.no_grad():
        layer.mlp.gate_proj.weight.copy_(
            layer.mlp.gate_proj.weight.data[sorted_indices, :]
        )
        layer.mlp.up_proj.weight.copy_(
            layer.mlp.up_proj.weight.data[sorted_indices, :]
        )
        layer.mlp.down_proj.weight.copy_(
            layer.mlp.down_proj.weight.data[:, sorted_indices]
        )


def reorder_attention_units(
    layer: nn.Module,
    importance: dict[str, torch.Tensor],
    layer_idx: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    prefix: str = "model.layers",
):
    """
    Attention 순열 불변 단위 재정렬 (GQA 구조 처리).
    Q 헤드를 GQA 그룹 단위로 묶어 중요도를 계산하고 재정렬합니다.
    """
    q_key = f"{prefix}.{layer_idx}.self_attn.q_proj.weight"
    k_key = f"{prefix}.{layer_idx}.self_attn.k_proj.weight"
    v_key = f"{prefix}.{layer_idx}.self_attn.v_proj.weight"
    o_key = f"{prefix}.{layer_idx}.self_attn.o_proj.weight"

    if q_key not in importance:
        return

    group_size = num_heads // num_kv_heads  # Q 헤드 수 / KV 헤드 수

    # KV 그룹별 중요도 계산
    group_imp = torch.zeros(num_kv_heads)
    q_imp = importance[q_key]  # [num_heads * head_dim, hidden]
    k_imp = importance[k_key]  # [num_kv_heads * head_dim, hidden]
    v_imp = importance[v_key]  # [num_kv_heads * head_dim, hidden]
    o_imp = importance[o_key]  # [hidden, num_heads * head_dim]

    for g in range(num_kv_heads):
        # Q 헤드 범위: [g*group_size*head_dim, (g+1)*group_size*head_dim)
        q_start = g * group_size * head_dim
        q_end = (g + 1) * group_size * head_dim
        # KV 헤드 범위: [g*head_dim, (g+1)*head_dim)
        kv_start = g * head_dim
        kv_end = (g + 1) * head_dim

        group_imp[g] = (
            q_imp[q_start:q_end].sum()
            + k_imp[kv_start:kv_end].sum()
            + v_imp[kv_start:kv_end].sum()
            + o_imp[:, q_start:q_end].sum()
        )

    sorted_groups = torch.argsort(group_imp, descending=True)

    # Q 인덱스 재구성
    q_indices = []
    kv_indices = []
    for g in sorted_groups:
        g = g.item()
        for h in range(group_size):
            head_idx = g * group_size + h
            q_indices.extend(range(head_idx * head_dim, (head_idx + 1) * head_dim))
        kv_indices.extend(range(g * head_dim, (g + 1) * head_dim))

    q_indices = torch.tensor(q_indices, dtype=torch.long)
    kv_indices = torch.tensor(kv_indices, dtype=torch.long)

    with torch.no_grad():
        layer.self_attn.q_proj.weight.copy_(
            layer.self_attn.q_proj.weight.data[q_indices, :]
        )
        layer.self_attn.k_proj.weight.copy_(
            layer.self_attn.k_proj.weight.data[kv_indices, :]
        )
        layer.self_attn.v_proj.weight.copy_(
            layer.self_attn.v_proj.weight.data[kv_indices, :]
        )
        layer.self_attn.o_proj.weight.copy_(
            layer.self_attn.o_proj.weight.data[:, q_indices]
        )


# ═══════════════════════════════════════════════════════════════════════
#  3. 앵커 레이어 식별
# ═══════════════════════════════════════════════════════════════════════

def identify_anchor_layers(
    importance: dict[str, torch.Tensor],
    num_layers: int,
    anchor_top_pct: float = 0.2,
    prefix: str = "model.layers",
) -> list[int]:
    """
    레이어별 총 중요도를 계산하고 상위 anchor_top_pct%를 앵커 레이어로 지정.
    앵커 레이어는 모든 하위 모델에서 100% 유지됩니다.
    """
    layer_importance = torch.zeros(num_layers)
    for name, imp in importance.items():
        for idx in range(num_layers):
            if f"{prefix}.{idx}." in name:
                layer_importance[idx] += imp.sum().item()

    num_anchor = max(1, int(num_layers * anchor_top_pct))
    _, top_indices = layer_importance.topk(num_anchor)
    anchor_layers = sorted(top_indices.tolist())

    return anchor_layers


# ═══════════════════════════════════════════════════════════════════════
#  4. Elastic Sub-model 슬라이싱
# ═══════════════════════════════════════════════════════════════════════

class ElasticLlamaWrapper(nn.Module):
    """
    탄력화된 Llama 모델 래퍼.
    재정렬된 가중치에서 ratio 기반 슬라이싱으로 하위 모델을 구성합니다.
    LoRA 어댑터도 ratio 별로 관리합니다.
    """

    def __init__(
        self,
        model: nn.Module,
        anchor_layers: list[int],
        ratios: list[float],
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.model = model
        self.anchor_layers = set(anchor_layers)
        self.ratios = ratios
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.current_ratio = 1.0

        # LoRA 어댑터 저장소 (ratio → PeftModel state)
        self.lora_adapters: dict[float, Optional[str]] = {}

    def set_ratio(self, ratio: float):
        """현재 활성 비율 설정 (실제 슬라이싱은 forward에서)."""
        self.current_ratio = ratio

    def get_sliced_intermediate_size(self, ratio: float) -> int:
        """MLP 중간 차원의 슬라이싱 크기."""
        return int(self.intermediate_size * ratio)

    def get_sliced_num_heads(self, ratio: float) -> tuple[int, int]:
        """헤드 수의 슬라이싱 (GQA 그룹 단위)."""
        num_kv_active = max(1, int(self.num_kv_heads * ratio))
        group_size = self.num_heads // self.num_kv_heads
        num_q_active = num_kv_active * group_size
        return num_q_active, num_kv_active

    def register_lora(self, ratio: float, lora_path: str):
        """특정 ratio 에 대한 LoRA 체크포인트 경로 등록."""
        self.lora_adapters[ratio] = lora_path

    @torch.no_grad()
    def forward_with_slicing(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ratio: Optional[float] = None,
    ):
        """
        슬라이싱된 하위 모델로 forward.
        NOTE: 실제 프로덕션에서는 가중치 포인터 이동만으로 구현하지만,
              여기서는 프로토타입으로 인덱싱 기반 forward를 수행합니다.
        """
        if ratio is None:
            ratio = self.current_ratio

        # ratio == 1.0 이면 풀 모델
        if ratio >= 1.0:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # 부분 모델 forward 는 full model generate를 사용 (프로토타입)
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        **kwargs,
    ):
        """생성 인터페이스."""
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════
#  5. 전체 탄력화 파이프라인
# ═══════════════════════════════════════════════════════════════════════

def elasticalize_model(
    cfg: Config,
    device: torch.device,
    save: bool = True,
) -> tuple[ElasticLlamaWrapper, AutoTokenizer]:
    """
    전체 탄력화 파이프라인 실행:
    1) 모델 로드  2) 중요도 프로파일링  3) 재정렬  4) 앵커 레이어 식별
    5) ElasticLlamaWrapper 생성

    Returns:
        wrapper   : ElasticLlamaWrapper
        tokenizer : AutoTokenizer
    """
    print(f"\n{'=' * 60}")
    print(f"  ElastiLM  ·  Elasticalization Pipeline")
    print(f"  Model: {cfg.llm.name}")
    print(f"{'=' * 60}\n")

    # ── 1. 모델 & 토크나이저 로드 ────────────────────────────────────
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.llm.torch_dtype, torch.float16)

    print("  [1/5] Loading model & tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.llm.name, trust_remote_code=cfg.llm.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 중요도 프로파일링을 위해 float32로 로드, device_map="auto"로 2 GPU 분산
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm.name,
        torch_dtype=torch.float32,
        trust_remote_code=cfg.llm.trust_remote_code,
        device_map="auto" if device.type == "cuda" else None,
    )

    model_config = model.config
    num_layers = model_config.num_hidden_layers
    num_heads, num_kv_heads, head_dim = _get_head_dim_info(model_config)
    intermediate_size = model_config.intermediate_size

    print(f"        Layers={num_layers}  Heads={num_heads}  KV-Heads={num_kv_heads}  "
          f"HeadDim={head_dim}  IntermDim={intermediate_size}")

    # ── 2. 중요도 프로파일링 ─────────────────────────────────────────
    print("  [2/5] Computing importance scores …")

    # Gradient checkpointing 활성화 (multi-GPU backward 안전 + 메모리 절약)
    model.gradient_checkpointing_enable()
    model.train()

    ds = SimpleTextDataset(
        tokenizer=tokenizer,
        dataset_name=cfg.elastic.importance_dataset,
        max_length=cfg.data.max_length,
        max_samples=cfg.elastic.importance_samples,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.elastic.importance_batch,
        collate_fn=_collate_fn,
        shuffle=False,
    )
    importance = compute_importance_scores(model, loader, device)
    print(f"        Profiled {len(ds)} samples, {len(importance)} parameters")

    # Gradient checkpointing 비활성화 (프로파일링 완료)
    model.gradient_checkpointing_disable()

    # ── 3. 순열 불변 단위 재정렬 ─────────────────────────────────────
    print("  [3/5] Reordering permutation-invariant units …")
    model.eval()

    # device_map 분산 모델을 CPU로 합침 (재정렬 시 텐서 일관성)
    model = model.cpu()

    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        reorder_mlp_units(layer, importance, layer_idx)
        reorder_attention_units(
            layer, importance, layer_idx,
            num_heads, num_kv_heads, head_dim,
        )

    # 추론용 dtype으로 변환 (float16 / bfloat16)
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.llm.torch_dtype, torch.float16)
    model = model.to(torch_dtype)

    # ── 4. 앵커 레이어 식별 ──────────────────────────────────────────
    print("  [4/5] Identifying anchor layers …")
    anchor_layers = identify_anchor_layers(
        importance, num_layers, cfg.elastic.anchor_top_pct
    )
    print(f"        Anchor layers ({len(anchor_layers)}): {anchor_layers}")

    # ── 5. Wrapper 생성 ──────────────────────────────────────────────
    print("  [5/5] Creating ElasticLlamaWrapper …")
    wrapper = ElasticLlamaWrapper(
        model=model,
        anchor_layers=anchor_layers,
        ratios=cfg.elastic.ratios,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
    )

    # ── 저장 ─────────────────────────────────────────────────────────
    if save:
        os.makedirs(cfg.output.elastic_dir, exist_ok=True)
        meta = {
            "model_name": cfg.llm.name,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate_size,
            "anchor_layers": anchor_layers,
            "ratios": cfg.elastic.ratios,
        }
        meta_path = os.path.join(cfg.output.elastic_dir, "elastic_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # 재정렬된 모델 저장
        ckpt_path = os.path.join(cfg.output.elastic_dir, "elasticalized_model")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        print(f"  ✓ Saved elasticalized model → {ckpt_path}")
        print(f"  ✓ Saved metadata            → {meta_path}")

    print(f"\n{'=' * 60}")
    print(f"  Elasticalization complete!")
    print(f"{'=' * 60}\n")

    return wrapper, tokenizer


# ═══════════════════════════════════════════════════════════════════════
#  6. LoRA Recovery 적용
# ═══════════════════════════════════════════════════════════════════════

def apply_lora_to_model(
    model: nn.Module,
    cfg: Config,
) -> PeftModel:
    """LoRA 어댑터를 모델에 적용 (peft 라이브러리)."""
    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()
    return peft_model


def load_elastic_model(
    cfg: Config,
    device: torch.device,
) -> tuple[ElasticLlamaWrapper, AutoTokenizer]:
    """저장된 탄력화 모델을 로드."""
    ckpt_path = os.path.join(cfg.output.elastic_dir, "elasticalized_model")
    meta_path = os.path.join(cfg.output.elastic_dir, "elastic_meta.json")

    with open(meta_path) as f:
        meta = json.load(f)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(cfg.llm.torch_dtype, torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.llm.trust_remote_code,
        device_map="auto" if device.type == "cuda" else None,
    )

    wrapper = ElasticLlamaWrapper(
        model=model,
        anchor_layers=meta["anchor_layers"],
        ratios=meta["ratios"],
        num_heads=meta["num_heads"],
        num_kv_heads=meta["num_kv_heads"],
        head_dim=meta["head_dim"],
        intermediate_size=meta["intermediate_size"],
    )

    # LoRA 어댑터 등록
    for ratio in meta["ratios"]:
        ratio_str = f"{int(ratio * 100)}"
        lora_path = os.path.join(cfg.output.lora_dir, f"lora_ratio_{ratio_str}")
        if os.path.isdir(lora_path):
            wrapper.register_lora(ratio, lora_path)

    return wrapper, tokenizer


# ═══════════════════════════════════════════════════════════════════════
#  CLI entrypoint
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from config import get_config_from_cli
    cfg = get_config_from_cli("ElastiLM – Elasticalize Model")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    elasticalize_model(cfg, device, save=True)
