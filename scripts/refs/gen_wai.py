"""
Generate character reference standing illustrations with WAI Illustrious SDXL (text2img).

Usage:
    python -m scripts.refs.gen_wai --character "shigure ui (vtuber)" --n 5 --out outputs/ref_shigure
"""
import os
import sys
import argparse

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.append(os.getcwd())

QUALITY = "masterpiece, best quality, amazing quality, very aesthetic, absurdres"
NEG = ("bad quality, worst quality, worst detail, sketch, censored, lowres, "
       "bad anatomy, bad hands, missing fingers, extra digits, fewer digits, "
       "jpeg artifacts, signature, watermark, username, blurry, text")


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    model = args.model
    if os.path.isdir(model):
        pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=dtype)
    else:
        pipe = StableDiffusionXLPipeline.from_single_file(model, torch_dtype=dtype)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)

    if args.prompt:
        prompt = f"{QUALITY}, {args.character}, 1girl, solo, full body, standing, looking at viewer, simple background, white background, {args.prompt}"
    else:
        prompt = (f"{QUALITY}, {args.character}, 1girl, solo, full body, standing, "
                  f"looking at viewer, simple background, white background, cowboy shot to full body")
    print("prompt:", prompt)

    os.makedirs(args.out, exist_ok=True)
    for i in range(args.n):
        g = torch.Generator(device).manual_seed(args.seed + i)
        img = pipe(
            prompt=prompt, negative_prompt=NEG,
            width=args.width, height=args.height,
            num_inference_steps=args.steps, guidance_scale=args.cfg,
            generator=g,
        ).images[0]
        p = os.path.join(args.out, f"ref_{i}.png")
        img.save(p)
        print("saved:", p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", required=True, help="danbooru character tag, e.g. 'shigure ui (vtuber)'")
    ap.add_argument("--prompt", default=None, help="extra outfit/scene tags appended after the character")
    ap.add_argument("--model", default="checkpoints/wai-illustrious-diffusers")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="outputs/ref_shigure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=1216)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=5.0)
    args = ap.parse_args()
    main(args)