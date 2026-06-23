"""
Inference with a checkpoint produced by diffsensei.cli.train.train.

Usage:
    python -m diffsensei.cli.inference.inference_trained \
        --config configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml \
        --ckpt logs/diffsensei/self_finetune_wai_condition_5060ti/<ts>/epoch-1/ckpt.pth \
        --input_json configs/inference/eval_input.json \
        --output_dir outputs --tag epoch1
"""
import os
import sys
import json
import argparse

import torch
from omegaconf import OmegaConf

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from diffsensei.inference import build_pipeline, load_ip_images, resolve_weight_dtype


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = OmegaConf.load(args.config)
    weight_dtype = resolve_weight_dtype(cfg, device)
    with open(args.input_json, "r", encoding="utf-8") as f:
        sample = json.load(f)

    pipeline = build_pipeline(cfg, args.ckpt, weight_dtype, device)
    print("trained pipeline ready")

    generator = torch.Generator(device).manual_seed(args.seed)
    images = pipeline(
        prompt=sample["prompt"], prompt_2=sample["prompt"],
        height=sample.get("height", 512), width=sample.get("width", 512),
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt, negative_prompt_2=args.negative_prompt,
        num_samples=args.num_samples, generator=generator,
        ip_images=load_ip_images(sample.get("ip_images", [])), ip_image_embeds=None,
        ip_bbox=[list(b) for b in sample.get("ip_bbox", [])], ip_scale=args.ip_scale,
        dialog_bbox=[list(b) for b in sample.get("dialog_bbox", [])],
    ).images

    os.makedirs(args.output_dir, exist_ok=True)
    for i, image in enumerate(images):
        out = os.path.join(args.output_dir, f"{args.tag}_{i}.png")
        image.save(out)
        print(f"saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="trained")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--ip_scale", type=float, default=0.6)
    parser.add_argument("--negative_prompt", type=str,
                        default="think lines, pure black background, colored, lowres, bad anatomy, worst quality, low quality")
    args = parser.parse_args()
    main(args)