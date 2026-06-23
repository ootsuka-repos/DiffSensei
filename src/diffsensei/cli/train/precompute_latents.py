"""
Precompute & cache the target-panel VAE latents for condition training.

Because only the adapters (IP / dialog / Resampler) are trained, the diffusion *target* —
the VAE latent of each panel — is frozen and DETERMINISTIC per (page, frame): the panel crop
+ center-resize to its bucket is fixed, and `mask_dialog` is the only optional transform.
So we encode every panel once here and cache (mean, std, crop_coords, sizes). Training then
skips the VAE entirely: it samples `latents = (mean + std*eps) * scaling_factor` from the cache.

This removes the fp32 VAE encode — the single biggest activation hog and the usual 1024 OOM
trigger — from the training loop, and frees the VAE weights from VRAM.

The target-image pipeline here is byte-for-byte the same as the dataset's __getitem__
(crop -> resize_and_center_crop -> image_transform), and uses the SAME size_buckets filtered
by `max_bucket_size`, so every frame maps to the identical bucket/resolution as training.

Usage:
    python -m diffsensei.cli.train.precompute_latents \
        --config configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml
"""
import os
import sys
import json
import argparse

import torch
from PIL import Image
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from diffusers import AutoencoderKL

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from diffsensei.datasets.utils import size_buckets, get_bucket_size, resize_and_center_crop, mask_dialogs_from_image
from diffsensei.datasets.dataset_size_bucket import image_transform


def cache_dir_for(config):
    base = config.train_data.get("latent_cache_dir", None)
    if base:
        return base
    mb = config.train_data.get("max_bucket_size", 512)
    return os.path.join("data", "latent_cache", f"wai_maxb{mb}")


def filtered_buckets(config):
    # Keep tiers whose SQUARE side <= max_bucket_size (must match train.py's filter exactly so
    # every frame maps to the identical bucket/resolution).
    mb = config.train_data.get("max_bucket_size", None)
    if mb is None:
        return size_buckets
    keep = [t for t in size_buckets if t["size"] <= mb]
    return keep or size_buckets


def main(args):
    config = OmegaConf.load(args.config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    vae = AutoencoderKL.from_pretrained(config.model.pretrained_model_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.to(device, dtype=torch.float32)   # one-time; fp32 for best latent fidelity
    vae.enable_slicing()
    vae.enable_tiling()

    ann_path = config.train_data.ann_path
    image_root = config.train_data.image_root
    mask_dialog = config.train_data.get("mask_dialog", False)
    buckets = filtered_buckets(config)
    out_dir = cache_dir_for(config)
    os.makedirs(out_dir, exist_ok=True)

    with open(ann_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    n_done = n_skip = n_fail = 0
    for ann_idx, ann in enumerate(tqdm(annotations, desc="precompute latents")):
        page = None
        for frame_idx, frame in enumerate(ann["frames"]):
            out_path = os.path.join(out_dir, f"{ann_idx}_{frame_idx}.pt")
            if os.path.exists(out_path) and not args.force:
                n_skip += 1
                continue
            x1, y1, x2, y2 = frame["bbox"]
            w, h = x2 - x1, y2 - y1
            if w < 8 or h < 8:
                n_fail += 1
                continue
            try:
                if page is None:
                    page = Image.open(os.path.join(image_root, ann["image_path"])).convert("RGB")
                    if mask_dialog:
                        page = mask_dialogs_from_image(page, ann)
                bh, bw, _ = get_bucket_size(h, w, buckets)
                img = page.crop([x1, y1, x2, y2])
                img, crop_coords = resize_and_center_crop(img, (bh, bw))
                px = image_transform(img).unsqueeze(0).to(device, dtype=torch.float32)
                with torch.no_grad():
                    dist = vae.encode(px).latent_dist
                torch.save({
                    "mean": dist.mean.squeeze(0).half().cpu(),
                    "std": dist.std.squeeze(0).half().cpu(),
                    "crop_coords": [int(crop_coords[0]), int(crop_coords[1])],
                    "orig": [int(h), int(w)],
                    "target": [int(bh), int(bw)],
                }, out_path)
                n_done += 1
            except Exception as e:
                print(f"  skip ({ann['image_path']} f{frame_idx}): {e}")
                n_fail += 1

    print(f"\ndone. cached={n_done} skipped(existing)={n_skip} failed={n_fail}\ncache dir: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true", help="recompute even if a cache file exists")
    args = ap.parse_args()
    main(args)
