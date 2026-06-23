"""
Compose a full manga PAGE from a trained DiffSensei checkpoint.

DiffSensei generates ONE panel at a time. This script loads the trained pipeline once,
generates each panel from a page spec (prompt / reference characters / character bbox /
dialog bbox), composites the panels onto a white page canvas with black borders, and
overlays the (real, readable) Japanese dialogue into the speech-bubble regions with PIL
— the diffusion model only reserves the bubble area, it cannot render text itself.

Usage:
    python -m scripts.demo.make_page \
        --config configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml \
        --ckpt logs/.../epoch-1/ckpt.pth \
        --spec scripts/demo/eval_page.json --out outputs/page.png
"""
import os
import sys
import json
import argparse

import torch
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.getcwd())

from scripts.demo.inference_trained import build_pipeline, load_ip_images


def load_font(size):
    for path in [r"C:\Windows\Fonts\msgothic.ttc", r"C:\Windows\Fonts\meiryo.ttc",
                 r"C:\Windows\Fonts\YuGothM.ttc", r"C:\Windows\Fonts\msmincho.ttc"]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def wrap_text(draw, text, font, max_w):
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur); cur = ""; continue
        test = cur + ch
        if draw.textlength(test, font=font) <= max_w:
            cur = test
        else:
            lines.append(cur); cur = ch
    if cur:
        lines.append(cur)
    return lines


def draw_bubble(page_draw, abox, text, font):
    x1, y1, x2, y2 = abox
    page_draw.ellipse([x1, y1, x2, y2], fill="white", outline="black", width=3)
    pad = (x2 - x1) * 0.12
    lines = wrap_text(page_draw, text, font, (x2 - x1) - 2 * pad)
    lh = (font.size + 4)
    th = lh * len(lines)
    ty = (y1 + y2) / 2 - th / 2
    for ln in lines:
        tw = page_draw.textlength(ln, font=font)
        page_draw.text(((x1 + x2) / 2 - tw / 2, ty), ln, fill="black", font=font)
        ty += lh


def main(args):
    cfg = OmegaConf.load(args.config)
    spec = json.load(open(args.spec, encoding="utf-8"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    wd = torch.float16 if device.startswith("cuda") else torch.float32

    pipe = build_pipeline(cfg, args.ckpt, wd, device)
    print("pipeline ready; generating page panels ...")

    W, H = spec["page"]["width"], spec["page"]["height"]
    page = Image.new("RGB", (W, H), "white")
    pdraw = ImageDraw.Draw(page)
    gen = torch.Generator(device).manual_seed(args.seed)

    for i, p in enumerate(spec["panels"]):
        px1, py1, px2, py2 = p["page_bbox"]
        pw, ph = px2 - px1, py2 - py1
        gw = max(256, min(args.max_side, (pw // 8) * 8))
        gh = max(256, min(args.max_side, (ph // 8) * 8))
        img = pipe(
            prompt=p["prompt"], prompt_2=p["prompt"], height=gh, width=gw,
            num_inference_steps=args.steps, guidance_scale=7.5,
            negative_prompt=args.neg, negative_prompt_2=args.neg,
            num_samples=1, generator=gen,
            ip_images=load_ip_images(p.get("ip_images", [])), ip_image_embeds=None,
            ip_bbox=[list(b) for b in p.get("ip_bbox", [])], ip_scale=p.get("ip_scale", 0.7),
            dialog_bbox=[list(b) for b in p.get("dialog_bbox", [])],
        ).images[0]
        page.paste(img.resize((pw, ph)), (px1, py1))
        pdraw.rectangle([px1, py1, px2 - 1, py2 - 1], outline="black", width=5)
        print(f"  panel {i+1}/{len(spec['panels'])} done")

        for dlg in p.get("dialogs", []):
            bx = dlg["bbox"]
            abox = [px1 + bx[0] * pw, py1 + bx[1] * ph, px1 + bx[2] * pw, py1 + bx[3] * ph]
            fsize = max(12, int((abox[3] - abox[1]) * 0.16))
            draw_bubble(pdraw, abox, dlg["text"], load_font(fsize))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    page.save(args.out)
    print("saved page:", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", default="outputs/page.png")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max_side", type=int, default=640)
    ap.add_argument("--neg", type=str,
                    default="text, speech bubble, colored, lowres, bad anatomy, worst quality, low quality")
    args = ap.parse_args()
    main(args)
