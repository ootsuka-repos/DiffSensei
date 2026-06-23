"""
Convert a single-file SDXL checkpoint (e.g. WAI Illustrious, an anime/bishoujo
specialized SDXL) into a diffusers folder so it can be used as the training base
(`pretrained_model_path`).

Usage:
    python -m diffsensei.cli.train.prepare_wai \
        --single_file checkpoints/waiIllustriousSDXL_v170.safetensors \
        --out checkpoints/wai-illustrious-diffusers
"""
import os
import argparse

import torch
from diffusers import StableDiffusionXLPipeline


def main(args):
    if os.path.exists(os.path.join(args.out, "unet", "diffusion_pytorch_model.safetensors")):
        print(f"[skip] {args.out} already prepared")
        return
    print(f"Loading single-file checkpoint: {args.single_file}", flush=True)
    pipe = StableDiffusionXLPipeline.from_single_file(args.single_file, torch_dtype=torch.float16)
    os.makedirs(args.out, exist_ok=True)
    print(f"Saving diffusers layout to: {args.out}", flush=True)
    pipe.save_pretrained(args.out)
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_file", type=str, default="checkpoints/waiIllustriousSDXL_v170.safetensors")
    parser.add_argument("--out", type=str, default="checkpoints/wai-illustrious-diffusers")
    args = parser.parse_args()
    main(args)
