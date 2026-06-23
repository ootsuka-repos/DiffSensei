"""
Reproduce training-data panels with the exact training-time conditioning.

Usage:
    python -m diffsensei.cli.eval.reproduce --config <cfg> --ckpt <epoch-N/ckpt.pth> \
        --ann data/annotations/train.json --image_root data --page 80 --out outputs/repro.png
"""
import os
import sys
import json
import argparse

import torch
from PIL import Image
from omegaconf import OmegaConf

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from diffsensei.datasets.utils import get_relative_bbox
from diffsensei.inference import build_pipeline, frame_gen_size, infer_eval_dtype


def main(args):
    cfg = OmegaConf.load(args.config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    wd = infer_eval_dtype(cfg, device)

    ann = json.load(open(args.ann, encoding="utf-8"))
    if args.page is not None:
        page = ann[args.page]
    else:
        page = next(p for p in ann if sum(1 for f in p["frames"] if f["characters"]) >= 3)
    page_img = Image.open(os.path.join(args.image_root, page["image_path"])).convert("RGB")
    print(f"page: {page['image_path']}  frames={len(page['frames'])}")

    pipe = build_pipeline(cfg, args.ckpt, wd, device)
    gen = torch.Generator(device).manual_seed(args.seed)

    rows = []
    for fi, fr in enumerate(page["frames"]):
        fb = fr["bbox"]
        fw, fh = fb[2] - fb[0], fb[3] - fb[1]
        if fw < 16 or fh < 16:
            continue
        gt = page_img.crop(tuple(fb))
        gw, gh = frame_gen_size(fw, fh, cfg, args.min_bucket_size)

        ip_images, ip_bbox = [], []
        for ch in fr["characters"][: cfg.model.max_num_ips]:
            ip_images.append(page_img.crop(tuple(ch["bbox"])).convert("L").convert("RGB"))
            ip_bbox.append(get_relative_bbox(fb, ch["bbox"]))
        dialog_bbox = [get_relative_bbox(fb, d["bbox"]) for d in fr["dialogs"][: cfg.model.max_num_dialogs]]

        img = pipe(
            prompt=fr.get("caption", ""), prompt_2=fr.get("caption", ""),
            height=gh, width=gw, num_inference_steps=args.steps, guidance_scale=7.5,
            negative_prompt=args.neg, negative_prompt_2=args.neg,
            num_samples=1, generator=gen,
            ip_images=ip_images, ip_image_embeds=None,
            ip_bbox=ip_bbox, ip_scale=args.ip_scale,
            dialog_bbox=dialog_bbox,
        ).images[0]

        H = 320
        g1 = gt.resize((max(1, int(gt.width * H / gt.height)), H))
        g2 = img.resize((max(1, int(img.width * H / img.height)), H))
        row = Image.new("RGB", (g1.width + g2.width + 12, H), "white")
        row.paste(g1, (0, 0)); row.paste(g2, (g1.width + 12, 0))
        rows.append(row)
        print(f"  panel {fi}: gen={gw}x{gh} GT={gt.size} chars={len(ip_images)} dialogs={len(dialog_bbox)}")

    if not rows:
        raise SystemExit("no usable panels on this page")
    W = max(r.width for r in rows)
    canvas = Image.new("RGB", (W, sum(r.height for r in rows) + 8 * len(rows)), "gray")
    y = 0
    for r in rows:
        canvas.paste(r, (0, y)); y += r.height + 8
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    canvas.save(args.out)
    print("saved (left=GT, right=generated):", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ann", default="data/annotations/train.json")
    ap.add_argument("--image_root", default="data")
    ap.add_argument("--page", type=int, default=None)
    ap.add_argument("--out", default="outputs/reproduce.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--ip_scale", type=float, default=0.7)
    ap.add_argument("--min_bucket_size", type=int, default=None,
                    help="eval bucket tier square side (default: eval.min_bucket_size or 1280)")
    ap.add_argument("--neg", type=str, default="colored, lowres, bad anatomy, worst quality, low quality")
    args = ap.parse_args()
    main(args)