"""
Precompute & cache the SDXL text-conditioning embeddings for condition training.

Only the adapters train, so the text branch is frozen. Each panel's caption is fixed, and the
only randomness is `t_drop_rate` (caption -> "" with small prob). So we encode every caption
once here, plus one empty-caption embedding, and cache:
  - text_embeds  = concat(te1.hidden_states[-2], te2.hidden_states[-2])  -> [77, 2048]
  - pooled       = te2 pooled output                                     -> [1280]
Training then drops both text encoders from VRAM and, per step, picks the cached caption embed
or the cached empty embed according to t_drop. Frees ~1.6GB of resident encoder weights.

Identical to train.py's text path (tokenizer max_length=77, truncation, penultimate hidden
states, te2 pooled), using the SAME base model (WAI) so embeddings match exactly.

Usage:
    python -m diffsensei.cli.train.precompute_text_embeds \
        --config configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml
"""
import os
import sys
import json
import argparse

import torch
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def text_cache_dir_for(config):
    base = config.train_data.get("text_cache_dir", None)
    return base or os.path.join("data", "text_cache", "wai")


def encode(caption, tok, tok2, te, te2, device):
    ids = tok(caption, max_length=tok.model_max_length, padding="max_length",
              truncation=True, return_tensors="pt").input_ids.to(device)
    ids2 = tok2(caption, max_length=tok2.model_max_length, padding="max_length",
                truncation=True, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = te(ids, output_hidden_states=True)
        out2 = te2(ids2, output_hidden_states=True)
    text_embeds = torch.cat([out.hidden_states[-2], out2.hidden_states[-2]], dim=-1)  # [1,77,2048]
    pooled = out2[0]                                                                  # [1,1280]
    return text_embeds.squeeze(0).half().cpu(), pooled.squeeze(0).half().cpu()


def main(args):
    config = OmegaConf.load(args.config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    base = config.model.pretrained_model_path

    tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")
    te = CLIPTextModel.from_pretrained(base, subfolder="text_encoder").to(device).eval()
    te2 = CLIPTextModelWithProjection.from_pretrained(base, subfolder="text_encoder_2").to(device).eval()
    te.requires_grad_(False); te2.requires_grad_(False)

    out_dir = text_cache_dir_for(config)
    os.makedirs(out_dir, exist_ok=True)

    # empty caption (for t_drop)
    em, ep = encode("", tok, tok2, te, te2, device)
    torch.save({"text_embeds": em, "pooled": ep}, os.path.join(out_dir, "empty.pt"))

    with open(config.train_data.ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    n_done = n_skip = 0
    for ann_idx, ann in enumerate(tqdm(annotations, desc="precompute text")):
        for frame_idx, frame in enumerate(ann["frames"]):
            out_path = os.path.join(out_dir, f"{ann_idx}_{frame_idx}.pt")
            if os.path.exists(out_path) and not args.force:
                n_skip += 1
                continue
            t, p = encode(frame.get("caption", ""), tok, tok2, te, te2, device)
            torch.save({"text_embeds": t, "pooled": p}, out_path)
            n_done += 1

    print(f"\ndone. cached={n_done} skipped(existing)={n_skip}\ntext cache dir: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    main(args)
