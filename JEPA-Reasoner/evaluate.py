import json
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer

from jepa_reasoner import JEPAReasoner, DualTalker
from config import Config, get_config_from_cli


def extract_numerical_answer(text: str) -> str:
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # fallback: 마지막 숫자
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    return nums[-1].replace(",", "").strip() if nums else ""


@torch.no_grad()
def generate_answer(
    reasoner: JEPAReasoner,
    talker: DualTalker,
    tokenizer,
    question: str,
    device: torch.device,
    max_new_tokens: int = 256,
) -> tuple[str, float]:

    reasoner.eval()
    talker.eval()

    prompt = f"Question: {question}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # starting timer
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    latents = reasoner(input_ids, reasoning_steps=1).squeeze(1)   # (1, Q, D)

    memory = talker.encoder(latents)

    eos_id = tokenizer.eos_token_id
    generated = torch.tensor([[eos_id]], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        tgt_emb  = talker.embedding(generated)
        seq_len  = generated.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        out    = talker.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        logits = talker.lm_head(out[:, -1, :])
        nxt    = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, nxt], dim=1)
        if nxt.item() == eos_id:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    latency = time.perf_counter() - t_start
    # end of timer

    text = tokenizer.decode(generated[0, 1:], skip_special_tokens=True)
    return text, latency

def evaluate_gsm8k(
    reasoner: JEPAReasoner,
    talker: DualTalker,
    tokenizer,
    device: torch.device,
    max_samples: int | None = None,
    max_new_tokens: int = 256,
) -> tuple[float, list[dict]]:
    
    # accuracy (float): 0-100 %
    # details  (list[dict]): result per question (w/ latency)
    
    test = load_dataset("openai/gsm8k", "main", split="test")
    if max_samples:
        test = test.select(range(min(max_samples, len(test))))

    correct, total = 0, 0
    details: list[dict] = []
    latencies: list[float] = []

    t0 = time.time()

    for i, item in enumerate(test):
        gold = extract_numerical_answer(item["answer"])
        pred_text, latency = generate_answer(
            reasoner, talker, tokenizer,
            item["question"], device,
            max_new_tokens=max_new_tokens,
        )
        pred = extract_numerical_answer(pred_text)
        latencies.append(latency)

        is_correct = pred == gold
        if is_correct:
            correct += 1
        total += 1

        details.append({
            "index": i,
            "question": item["question"],
            "gold": gold,
            "pred": pred,
            "pred_text": pred_text,
            "correct": is_correct,
            "latency_sec": latency,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (len(test) - i - 1) / max(speed, 1e-9)
            avg_lat = np.mean(latencies)
            print(
                f"    [{i + 1:>5}/{len(test)}]  "
                f"Acc: {correct / total * 100:.1f}%  "
                f"Avg lat: {avg_lat:.3f}s  "
                f"Speed: {speed:.1f} q/s  ETA: {eta:.0f}s"
            )


    acc        = correct / total * 100
    lat_arr    = np.array(latencies)
    lat_mean   = float(np.mean(lat_arr))
    lat_std    = float(np.std(lat_arr, ddof=1)) if len(lat_arr) > 1 else 0.0
    lat_median = float(np.median(lat_arr))
    total_sec  = time.time() - t0

    print(f"\n  ★  JEPA-Reasoner  GSM8K Results")
    print(f"     Accuracy          : {acc:.2f}%  ({correct}/{total})")
    print(f"     Latency mean      : {lat_mean:.4f} s")
    print(f"     Latency std       : {lat_std:.4f} s")
    print(f"     Latency median    : {lat_median:.4f} s")
    print(f"     Total time        : {total_sec:.0f} s\n")

    return acc, details


def main(cfg: Config):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    ).to(device)

    ckpt_r = cfg.evaluate.reasoner_ckpt
    if os.path.exists(ckpt_r):
        state = torch.load(ckpt_r, map_location=device, weights_only=False)
        if "model_state_dict" in state:
            reasoner.load_state_dict(state["model_state_dict"])
        else:
            reasoner.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Reasoner checkpoint not found: {ckpt_r}")

    talker = DualTalker(
        vocab_size=vocab_size,
        embed_dim=cfg.model.embed_dim,
        num_heads=cfg.talker.num_heads,
        ffn_dim=cfg.model.ffn_dim,
        num_enc_layers=cfg.talker.enc_layers,
        num_dec_layers=cfg.talker.dec_layers,
    ).to(device)

    ckpt_t = cfg.evaluate.talker_ckpt
    if os.path.exists(ckpt_t):
        talker.load_state_dict(
            torch.load(ckpt_t, map_location=device, weights_only=False)
        )
    else:
        raise FileNotFoundError(f"Talker checkpoint not found: {ckpt_t}")

    accuracy, details = evaluate_gsm8k(
        reasoner, talker, tokenizer, device,
        max_samples=cfg.evaluate.max_samples,
        max_new_tokens=cfg.evaluate.max_new_tokens,
    )

    lat_arr    = np.array([d["latency_sec"] for d in details])
    lat_mean   = float(np.mean(lat_arr))
    lat_std    = float(np.std(lat_arr, ddof=1)) if len(lat_arr) > 1 else 0.0
    lat_median = float(np.median(lat_arr))

    os.makedirs(cfg.evaluate.output_dir, exist_ok=True)

    result_path = os.path.join(cfg.evaluate.output_dir, "eval_results.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                "model": "JEPA-Reasoner",
                "short_name": "JEPA-Reasoner",
                "gsm8k_accuracy": accuracy,
                "total": len(details),
                "correct": sum(1 for d in details if d["correct"]),
                "latency_mean_sec": lat_mean,
                "latency_std_sec": lat_std,
                "latency_median_sec": lat_median,
            },
            f, indent=2,
        )

    detail_path = os.path.join(cfg.evaluate.output_dir, "eval_details.jsonl")
    with open(detail_path, "w") as f:
        for d in details:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    cfg = get_config_from_cli("JEPA-Reasoner GSM8K Evaluation")
    main(cfg)
