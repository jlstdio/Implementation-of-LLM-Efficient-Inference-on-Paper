import copy
from typing import Optional

import torch
import torch.nn as nn
from transformers import MobileBertModel, AutoTokenizer


class ElastiLM_TLM(nn.Module):
    def __init__(
        self,
        backbone_name: str = "google/mobilebert-uncased",
        shared_layers: int = 12,
        num_prompt_levels: int = 9,
        num_model_levels: int = 9,
    ):
        super().__init__()

        self.backbone = MobileBertModel.from_pretrained(backbone_name)
        hidden_size = self.backbone.config.hidden_size  # 512

        self.embeddings = self.backbone.embeddings
        all_layers = list(self.backbone.encoder.layer)
        self.shared_layers_list = nn.ModuleList(all_layers[:shared_layers])

        top_layers = all_layers[shared_layers:]
        self.score_head_layers = nn.ModuleList(
            [copy.deepcopy(l) for l in top_layers]
        )
        self.decision_head_layers = nn.ModuleList(
            [copy.deepcopy(l) for l in top_layers]
        )

        self.score_proj = nn.Linear(hidden_size, 2)

        self.prompt_decision_proj = nn.Linear(hidden_size, num_prompt_levels)
        self.model_decision_proj = nn.Linear(hidden_size, num_model_levels)

        self.hidden_size = hidden_size
        self.num_prompt_levels = num_prompt_levels
        self.num_model_levels = num_model_levels

    def _encode_shared(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.embeddings(input_ids)

        if attention_mask is not None:
            extended = attention_mask[:, None, None, :].to(dtype=x.dtype)
            extended = (1.0 - extended) * -10000.0
        else:
            extended = None

        for layer in self.shared_layers_list:
            x = layer(x, attention_mask=extended)[0]

        return x, extended

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        shared_out, extended_mask = self._encode_shared(input_ids, attention_mask)

        score_x = shared_out
        for layer in self.score_head_layers:
            score_x = layer(score_x, attention_mask=extended_mask)[0]
        token_scores = self.score_proj(score_x)  # B, L, 2

        dec_x = shared_out
        for layer in self.decision_head_layers:
            dec_x = layer(dec_x, attention_mask=extended_mask)[0]
        cls_out = dec_x[:, 0, :]

        prompt_level = self.prompt_decision_proj(cls_out) 
        model_level = self.model_decision_proj(cls_out)

        return token_scores, prompt_level, model_level

    def predict_strategy(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        self.eval()
        with torch.no_grad():
            token_scores, prompt_logits, model_logits = self.forward(
                input_ids, attention_mask
            )
        prompt_level = torch.argmax(prompt_logits, dim=-1)
        model_level = torch.argmax(model_logits, dim=-1)

        return {
            "token_scores": token_scores,
            "prompt_level": prompt_level,
            "model_level": model_level,
        }


def add_slo_tokens(
    tokenizer: AutoTokenizer,
    model: ElastiLM_TLM,
    slo_tokens: list[str],
) -> AutoTokenizer:

    num_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": slo_tokens}
    )
    if num_added > 0:
        old_emb = model.embeddings.word_embeddings
        old_weight = old_emb.weight.data
        new_size = len(tokenizer)

        new_emb = nn.Embedding(new_size, old_emb.embedding_dim)
        new_emb.weight.data[: old_weight.size(0)] = old_weight
        
        mean_vec = old_weight.mean(dim=0)
        for i in range(old_weight.size(0), new_size):
            new_emb.weight.data[i] = mean_vec

        model.embeddings.word_embeddings = new_emb

    return tokenizer


def compress_prompt(
    input_ids: torch.Tensor,
    token_scores: torch.Tensor,
    target_ratio: float,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    retain_probs = torch.softmax(token_scores, dim=-1)[:, :, 1]

    B, L = input_ids.shape
    k = max(1, int(L * target_ratio))

    _, topk_indices = retain_probs.topk(k, dim=-1, sorted=True)
    topk_indices, _ = topk_indices.sort(dim=-1)

    compressed_ids = torch.gather(input_ids, 1, topk_indices)
    if attention_mask is not None:
        compressed_mask = torch.gather(attention_mask, 1, topk_indices)
    else:
        compressed_mask = torch.ones_like(compressed_ids)

    return compressed_ids, compressed_mask

RATIOS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def level_to_ratio(level: int) -> float:
    return RATIOS[min(level, len(RATIOS) - 1)]


def ratio_to_level(ratio: float) -> int:
    diffs = [abs(r - ratio) for r in RATIOS]
    return diffs.index(min(diffs))
