from __future__ import annotations
import argparse
import os
from dataclasses import dataclass, field
from typing import Optional, List
import yaml

@dataclass
class LLMConfig:
    name:            str  = "meta-llama/Llama-3.1-8B"
    short_name:      str  = "Llama-3.1-8B"
    torch_dtype:     str  = "float16"
    trust_remote_code: bool = True


@dataclass
class TLMConfig:
    backbone:            str  = "google/mobilebert-uncased"
    shared_layers:       int  = 12        # 하위 공유 레이어 수
    num_prompt_levels:   int  = 9         # 프롬프트 압축 단계 (20%~100%, 10% 간격)
    num_model_levels:    int  = 9         # 모델 크기 단계
    slo_tokens:          list = field(default_factory=lambda: [
        "[TTFT_50]", "[TTFT_100]", "[TTFT_200]", "[TTFT_500]",
        "[TPOT_20]", "[TPOT_40]", "[TPOT_80]", "[TPOT_160]",
    ])


@dataclass
class ElasticConfig:
    ratios:              list  = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    anchor_top_pct:      float = 0.20     # 상위 20% 레이어는 앵커(100% 유지)
    importance_dataset:  str   = "wikitext"
    importance_samples:  int   = 512      # 중요도 프로파일링에 사용할 샘플 수
    importance_batch:    int   = 4


@dataclass
class LoRAConfig:
    r:                   int   = 8
    alpha:               int   = 16
    dropout:             float = 0.05
    target_modules:      list  = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


@dataclass
class DataConfig:
    max_length:          int   = 512
    seed:                int   = 42
    # Score-head 학습: MeetingBank
    meetingbank_split:   str   = "train"
    # Decision-head 학습: MMLU-Pro
    mmlu_pro_split:      str   = "test"
    # LoRA Recovery: Alpaca-cleaned
    alpaca_max_samples:  int   = 50000


@dataclass
class TrainScoreConfig:
    """TLM Score-head 학습 설정."""
    epochs:              int   = 5
    batch_size:          int   = 16
    learning_rate:       float = 2e-5
    weight_decay:        float = 0.01
    warmup_steps:        int   = 500
    max_grad_norm:       float = 1.0
    log_interval:        int   = 50


@dataclass
class TrainDecisionConfig:
    """TLM Decision-head 학습 (Self-induced Labeling) 설정."""
    epochs:              int   = 5
    batch_size:          int   = 8
    learning_rate:       float = 2e-5
    weight_decay:        float = 0.01
    warmup_steps:        int   = 200
    max_grad_norm:       float = 1.0
    log_interval:        int   = 50
    label_cache_dir:     str   = "./checkpoints/decision_labels"


@dataclass
class TrainLoRAConfig:
    """LoRA Recovery 학습 설정."""
    epochs:              int   = 3
    batch_size:          int   = 4
    learning_rate:       float = 1e-4
    weight_decay:        float = 0.01
    warmup_steps:        int   = 200
    max_grad_norm:       float = 1.0
    fp16:                bool  = True
    log_interval:        int   = 100
    gradient_accumulation_steps: int = 4


@dataclass
class EvaluateConfig:
    """평가 설정."""
    datasets:            list  = field(default_factory=lambda: [
        "arc_easy", "piqa", "mmlu_pro",
    ])
    n_shot:              int   = 5
    max_new_tokens:      int   = 256
    max_eval_samples:    Optional[int] = None
    # SLO 시나리오
    slo_scenarios:       list  = field(default_factory=lambda: [
        {"ttft_ms": 50,  "tpot_ms": 20},
        {"ttft_ms": 100, "tpot_ms": 40},
        {"ttft_ms": 200, "tpot_ms": 80},
        {"ttft_ms": 500, "tpot_ms": 160},
    ])
    measure_overhead:    bool  = True


@dataclass
class OutputConfig:
    """출력 경로 설정."""
    base_dir:            str = "./checkpoints"
    elastic_dir:         str = "./checkpoints/elastic"
    lora_dir:            str = "./checkpoints/lora"
    tlm_dir:             str = "./checkpoints/tlm"
    eval_dir:            str = "./checkpoints/eval"


@dataclass
class Config:
    llm:             LLMConfig            = field(default_factory=LLMConfig)
    tlm:             TLMConfig            = field(default_factory=TLMConfig)
    elastic:         ElasticConfig        = field(default_factory=ElasticConfig)
    lora:            LoRAConfig           = field(default_factory=LoRAConfig)
    data:            DataConfig           = field(default_factory=DataConfig)
    train_score:     TrainScoreConfig     = field(default_factory=TrainScoreConfig)
    train_decision:  TrainDecisionConfig  = field(default_factory=TrainDecisionConfig)
    train_lora:      TrainLoRAConfig      = field(default_factory=TrainLoRAConfig)
    evaluate:        EvaluateConfig       = field(default_factory=EvaluateConfig)
    output:          OutputConfig         = field(default_factory=OutputConfig)
    device:          str                  = "cuda"


# ── YAML ↔ dataclass 변환 ────────────────────────────────────────────
def _dict_to_dataclass(dc_cls, d: dict):
    if d is None:
        return dc_cls()
    filtered = {k: v for k, v in d.items() if k in dc_cls.__dataclass_fields__}
    return dc_cls(**filtered)


def load_config(path: str) -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        llm             = _dict_to_dataclass(LLMConfig,            raw.get("llm")),
        tlm             = _dict_to_dataclass(TLMConfig,            raw.get("tlm")),
        elastic         = _dict_to_dataclass(ElasticConfig,        raw.get("elastic")),
        lora            = _dict_to_dataclass(LoRAConfig,           raw.get("lora")),
        data            = _dict_to_dataclass(DataConfig,           raw.get("data")),
        train_score     = _dict_to_dataclass(TrainScoreConfig,     raw.get("train_score")),
        train_decision  = _dict_to_dataclass(TrainDecisionConfig,  raw.get("train_decision")),
        train_lora      = _dict_to_dataclass(TrainLoRAConfig,      raw.get("train_lora")),
        evaluate        = _dict_to_dataclass(EvaluateConfig,       raw.get("evaluate")),
        output          = _dict_to_dataclass(OutputConfig,         raw.get("output")),
        device          = raw.get("device", "cuda"),
    )
    return cfg


def get_config_from_cli(description: str = "") -> Config:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to YAML config file  (default: ./config.yaml)",
    )
    args, _ = parser.parse_known_args()
    cfg = load_config(args.config)
    return cfg
