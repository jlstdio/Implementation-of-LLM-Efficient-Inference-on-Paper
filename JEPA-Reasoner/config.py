"""
JEPA-Reasoner  ──  Configuration Loader
========================================
YAML 기반 설정 파일을 파이썬 dataclass 로 변환.
모든 스크립트(pretrain / finetuning / evaluate)가 공유한다.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

import yaml


# ══════════════════════════════════════════════════════════════════════════════
#  Dataclass 정의
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelConfig:
    embed_dim:  int = 384
    num_heads:  int = 16
    ffn_dim:    int = 1536
    num_layers: int = 18


@dataclass
class TalkerConfig:
    num_heads:  int = 8
    enc_layers: int = 4
    dec_layers: int = 4


@dataclass
class TokenizerConfig:
    name: str = "gpt2"


@dataclass
class DataConfig:
    max_length: int   = 512
    max_q_len:  int   = 256
    max_a_len:  int   = 256
    c4_ratio:   float = 0.7
    seed:       int   = 42


@dataclass
class PretrainConfig:
    max_steps:      int   = 300_000
    batch_size:     int   = 32
    learning_rate:  float = 1e-4
    weight_decay:   float = 0.01
    warmup_steps:   int   = 2000
    max_grad_norm:  float = 1.0
    fp16:           bool  = True
    log_interval:   int   = 100
    save_interval:  int   = 10_000
    output_dir:     str   = "./checkpoints/pretrain"


@dataclass
class Phase2Config:
    max_steps:      int   = 42_000
    learning_rate:  float = 1e-4
    batch_size:     int   = 16
    num_workers:    int   = 4
    log_interval:   int   = 100


@dataclass
class Phase3Config:
    epochs:         int   = 5
    learning_rate:  float = 1e-4
    ema_momentum:   float = 0.98
    cosine_k:       float = 4.0


@dataclass
class Phase4Config:
    epochs:         int   = 5
    learning_rate:  float = 1e-4


@dataclass
class FinetuneConfig:
    pretrained_ckpt: str = "./checkpoints/pretrain/jepa_pretrained_final.pt"
    output_dir:      str = "./checkpoints/finetune"


@dataclass
class EvaluateConfig:
    max_samples:     Optional[int] = None
    max_new_tokens:  int           = 256
    output_dir:      str           = "./checkpoints/finetune"
    reasoner_ckpt:   str           = "./checkpoints/finetune/reasoner_after_phase3.pt"
    talker_ckpt:     str           = "./checkpoints/finetune/talker_after_phase4.pt"


@dataclass
class Config:
    """전체 설정을 하나로 묶는 최상위 dataclass."""
    model:      ModelConfig      = field(default_factory=ModelConfig)
    talker:     TalkerConfig     = field(default_factory=TalkerConfig)
    tokenizer:  TokenizerConfig  = field(default_factory=TokenizerConfig)
    data:       DataConfig       = field(default_factory=DataConfig)
    pretrain:   PretrainConfig   = field(default_factory=PretrainConfig)
    phase2:     Phase2Config     = field(default_factory=Phase2Config)
    phase3:     Phase3Config     = field(default_factory=Phase3Config)
    phase4:     Phase4Config     = field(default_factory=Phase4Config)
    finetune:   FinetuneConfig   = field(default_factory=FinetuneConfig)
    evaluate:   EvaluateConfig   = field(default_factory=EvaluateConfig)
    device:     str              = "cuda"


# ══════════════════════════════════════════════════════════════════════════════
#  로더
# ══════════════════════════════════════════════════════════════════════════════
def _dict_to_dataclass(dc_cls, d: dict):
    """딕셔너리를 dataclass 인스턴스로 안전하게 변환."""
    if d is None:
        return dc_cls()
    filtered = {k: v for k, v in d.items() if k in dc_cls.__dataclass_fields__}
    return dc_cls(**filtered)


def load_config(path: str) -> Config:
    """YAML 파일로부터 Config 객체를 생성한다."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        model     = _dict_to_dataclass(ModelConfig,     raw.get("model")),
        talker    = _dict_to_dataclass(TalkerConfig,    raw.get("talker")),
        tokenizer = _dict_to_dataclass(TokenizerConfig, raw.get("tokenizer")),
        data      = _dict_to_dataclass(DataConfig,      raw.get("data")),
        pretrain  = _dict_to_dataclass(PretrainConfig,  raw.get("pretrain")),
        phase2    = _dict_to_dataclass(Phase2Config,    raw.get("phase2")),
        phase3    = _dict_to_dataclass(Phase3Config,    raw.get("phase3")),
        phase4    = _dict_to_dataclass(Phase4Config,    raw.get("phase4")),
        finetune  = _dict_to_dataclass(FinetuneConfig,  raw.get("finetune")),
        evaluate  = _dict_to_dataclass(EvaluateConfig,  raw.get("evaluate")),
        device    = raw.get("device", "cuda"),
    )
    return cfg


def get_config_from_cli(description: str = "") -> Config:
    """CLI 인자 ``--config`` 로 YAML 경로를 받아 Config 를 반환한다."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to YAML config file  (default: ./config.yaml)",
    )
    args, _ = parser.parse_known_args()
    cfg = load_config(args.config)
    return cfg
