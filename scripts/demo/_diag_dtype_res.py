"""一時診断: eval の「汚さ」が fp16/低解像度のサンプル生成設定由来かを実証する。
同じコマ・同じ条件付け・同じ seed で (A) 現状の fp16 / <=512 と (B) bf16 / <=1024 を生成し、
GT と横並びで比較保存する。学習済み重みや作画基盤(WAI)は両者で完全に同一。
実行: $env:CUDA_VISIBLE_DEVICES="1"; python -m scripts.demo._diag_dtype_res
"""
import os, sys, json
os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.getcwd())

import gc
import torch
from PIL import Image, ImageDraw
from omegaconf import OmegaConf
from scripts.demo.inference_trained import build_pipeline
from src.datasets.utils import get_relative_bbox

CFG = "configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml"
CKPT = "logs/diffsensei/self_finetune_wai_condition_5060ti/2026-06-23-09-23-34/epoch-29/ckpt.pth"
ANN = "data/annotations/train.json"
ROOT = "data"
PAGE = 113
NEG = "colored, lowres, bad anatomy, worst quality, low quality"


def mul8(v, lo, hi):
    return max(lo, min(hi, (int(v) // 8) * 8))


cfg = OmegaConf.load(CFG)
ann = json.load(open(ANN, encoding="utf-8"))
page = ann[PAGE]
page_img = Image.open(os.path.join(ROOT, page["image_path"])).convert("RGB")
print("page:", page["image_path"], "frames:", len(page["frames"]))

# 最初の「キャラあり・十分な大きさ」のコマを選ぶ
fr = None
for f in page["frames"]:
    fb = f["bbox"]
    if f["characters"] and (fb[2] - fb[0]) >= 180 and (fb[3] - fb[1]) >= 180:
        fr = f
        break
if fr is None:
    fr = max(page["frames"], key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))

fb = fr["bbox"]
fw, fh = fb[2] - fb[0], fb[3] - fb[1]
gt = page_img.crop(tuple(fb))
print("frame bbox:", fb, "size:", (fw, fh), "chars:", len(fr["characters"]), "dialogs:", len(fr["dialogs"]))

ip_images, ip_bbox = [], []
for ch in fr["characters"][: cfg.model.max_num_ips]:
    ip_images.append(page_img.crop(tuple(ch["bbox"])).convert("L").convert("RGB"))
    ip_bbox.append(get_relative_bbox(fb, ch["bbox"]))
dialog_bbox = [get_relative_bbox(fb, d["bbox"]) for d in fr["dialogs"][: cfg.model.max_num_dialogs]]
prompt = fr.get("caption", "")


def gen(dtype, lo, hi):
    dev = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 の下では物理GPU1
    pipe = build_pipeline(cfg, CKPT, dtype, dev)
    g = torch.Generator(dev).manual_seed(0)
    w, h = mul8(fw, lo, hi), mul8(fh, lo, hi)
    print(f"  gen dtype={dtype} size={(w, h)}")
    img = pipe(
        prompt=prompt, prompt_2=prompt, height=h, width=w,
        num_inference_steps=28, guidance_scale=7.5,
        negative_prompt=NEG, negative_prompt_2=NEG,
        num_samples=1, generator=g,
        ip_images=ip_images, ip_image_embeds=None,
        ip_bbox=ip_bbox, ip_scale=0.7, dialog_bbox=dialog_bbox,
    ).images[0]
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return img


print("== A: fp16 / <=512 (現状の eval 設定) ==")
a = gen(torch.float16, 256, 512)
print("== B: bf16 / <=1024 (改善案) ==")
b = gen(torch.bfloat16, 512, 1024)

H = 420
rs = lambda im: im.resize((max(1, int(im.width * H / im.height)), H))
panels = [("GT", gt), ("fp16 / <=512 (now)", a), ("bf16 / <=1024 (fix)", b)]
ims = [rs(im) for _, im in panels]
pad, lab = 10, 26
W = sum(i.width for i in ims) + pad * (len(ims) + 1)
canvas = Image.new("RGB", (W, H + lab + pad), "white")
d = ImageDraw.Draw(canvas)
x = pad
for (name, _), im in zip(panels, ims):
    canvas.paste(im, (x, lab))
    d.text((x, 6), name, fill="black")
    x += im.width + pad
os.makedirs("outputs", exist_ok=True)
out = "outputs/_diag_dtype_res.png"
canvas.save(out)
print("saved:", out)
