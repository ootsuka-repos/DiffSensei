"""
Fully-automatic annotation of raw manga pages into the DiffSensei training format.

For every image under `--image_root`, we run the Magi manga model
(`ragavsachdeva/magi`) to detect:
  - panels        -> become DiffSensei "frames"
  - characters    -> per-frame characters, with a stable `id` from Magi's
                     character clustering (same person across panels = same id)
  - texts         -> per-frame "dialogs" (speech / text boxes)
and optionally tag each panel with WD14 (via `imgutils`) to produce a caption.

Output: a single JSON list at `--ann_path`, each element being one page:
    {
      "image_path": "<relative to image_root>",
      "frames": [
        {
          "bbox": [x1, y1, x2, y2],                       # panel, page pixel coords
          "caption": "manga, monochrome, 1girl, ...",
          "characters": [{"id": int, "bbox": [...], "type": 0}],
          "dialogs": [{"bbox": [...]}]
        }, ...
      ]
    }

All bboxes are absolute page pixel coordinates (the DiffSensei dataset converts
them to panel-relative coordinates at load time).

Usage:
    python -m scripts.dataset.auto_annotate \
        --image_root data \
        --ann_path data/annotations/train.json \
        --caption wd14
"""
import os
import sys
import glob
import json
import argparse

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch

sys.path.append(os.getcwd())

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg", ".bmp", ".gif")
MAGI_REPO = "ragavsachdeva/magi"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def list_images(image_root):
    files = []
    for ext in IMAGE_EXTS:
        files += glob.glob(os.path.join(image_root, "**", "*" + ext), recursive=True)
        files += glob.glob(os.path.join(image_root, "*" + ext))
    # unique + stable order
    files = sorted(set(os.path.normpath(f) for f in files))
    return files


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = int(round(max(0, min(x1, w))))
    y1 = int(round(max(0, min(y1, h))))
    x2 = int(round(max(0, min(x2, w))))
    y2 = int(round(max(0, min(y2, h))))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def center_inside(inner, outer):
    cx, cy = box_center(inner)
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def assign_to_panel(box, panels):
    """Return index of the smallest panel whose region contains the box center."""
    best, best_area = -1, None
    for i, p in enumerate(panels):
        if center_inside(box, p):
            a = box_area(p)
            if best_area is None or a < best_area:
                best, best_area = i, a
    return best


# --------------------------------------------------------------------------- #
# captioning (WD14 tags via imgutils, optional)
# --------------------------------------------------------------------------- #
class Captioner:
    def __init__(self, mode="wd14", style_prefix="manga, monochrome, greyscale", max_tags=20):
        self.mode = mode
        self.style_prefix = style_prefix
        self.max_tags = max_tags
        self._wd14 = None
        if mode == "wd14":
            try:
                from imgutils.tagging import get_wd14_tags  # noqa
                self._wd14 = get_wd14_tags
            except Exception as e:
                print(f"[caption] WD14 unavailable ({e}); falling back to style prefix only.")
                self._wd14 = None

    def __call__(self, panel_pil):
        prefix = self.style_prefix
        if self.mode == "none" or self._wd14 is None:
            return prefix
        try:
            rating, general, characters = self._wd14(panel_pil)
            tags = sorted(general.items(), key=lambda x: -x[1])[: self.max_tags]
            tag_str = ", ".join(t.replace("_", " ") for t, _ in tags)
            char_str = ", ".join(c.replace("_", " ") for c in characters)
            parts = [p for p in [prefix, char_str, tag_str] if p]
            return ", ".join(parts)
        except Exception:
            return prefix


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading Magi from {MAGI_REPO} ...", flush=True)
    from transformers import AutoModel
    magi = AutoModel.from_pretrained(MAGI_REPO, trust_remote_code=True).to(device).eval()

    captioner = Captioner(mode=args.caption, style_prefix=args.style_prefix, max_tags=args.max_tags)

    images = list_images(args.image_root)
    if args.limit > 0:
        images = images[: args.limit]
    print(f"Found {len(images)} images under {args.image_root}")

    annotations = []
    n_frames = n_chars = n_dialogs = 0

    for img_path in tqdm(images, desc="annotate"):
        rel_path = os.path.relpath(img_path, args.image_root).replace("\\", "/")
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"skip {rel_path}: cannot open ({e})")
            continue
        W, H = pil.size
        arr = np.array(pil)

        try:
            with torch.no_grad():
                result = magi.predict_detections_and_associations(
                    [arr],
                    character_detection_threshold=args.char_thr,
                    panel_detection_threshold=args.panel_thr,
                    text_detection_threshold=args.text_thr,
                )[0]
        except Exception as e:
            print(f"skip {rel_path}: magi failed ({e})")
            continue

        panels = [clamp_box(b, W, H) for b in result.get("panels", [])]
        characters = [clamp_box(b, W, H) for b in result.get("characters", [])]
        char_ids = result.get("character_cluster_labels", list(range(len(characters))))
        texts = [clamp_box(b, W, H) for b in result.get("texts", [])]

        # fallback: if no panel detected, treat the whole page as a single panel
        if len(panels) == 0:
            panels = [[0, 0, W, H]]

        # drop degenerate panels
        panels = [p for p in panels if box_area(p) >= args.min_panel_area]
        if len(panels) == 0:
            panels = [[0, 0, W, H]]

        # build per-panel frames
        frames = [{"bbox": p, "characters": [], "dialogs": [], "caption": ""} for p in panels]

        for cbox, cid in zip(characters, char_ids):
            if box_area(cbox) < args.min_char_area:
                continue
            pi = assign_to_panel(cbox, panels)
            if pi < 0:
                continue
            frames[pi]["characters"].append({"id": int(cid), "bbox": cbox, "type": 0})

        for tbox in texts:
            pi = assign_to_panel(tbox, panels)
            if pi < 0:
                continue
            frames[pi]["dialogs"].append({"bbox": tbox})

        # caption each panel (crop from the page)
        for fr in frames:
            x1, y1, x2, y2 = fr["bbox"]
            if x2 - x1 < 8 or y2 - y1 < 8:
                fr["caption"] = args.style_prefix
                continue
            panel_crop = pil.crop((x1, y1, x2, y2))
            fr["caption"] = captioner(panel_crop)

        # keep only frames with a sensible size
        frames = [f for f in frames if (f["bbox"][2] - f["bbox"][0]) >= 8 and (f["bbox"][3] - f["bbox"][1]) >= 8]
        if len(frames) == 0:
            continue

        annotations.append({"image_path": rel_path, "frames": frames})
        n_frames += len(frames)
        n_chars += sum(len(f["characters"]) for f in frames)
        n_dialogs += sum(len(f["dialogs"]) for f in frames)

    os.makedirs(os.path.dirname(os.path.abspath(args.ann_path)), exist_ok=True)
    with open(args.ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=1)

    print(
        f"\nDone. pages={len(annotations)} frames={n_frames} "
        f"characters={n_chars} dialogs={n_dialogs}\nWrote {args.ann_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, default="data")
    parser.add_argument("--ann_path", type=str, default="data/annotations/train.json")
    parser.add_argument("--caption", type=str, default="wd14", choices=["wd14", "none"])
    parser.add_argument("--style_prefix", type=str, default="manga, monochrome, greyscale")
    parser.add_argument("--max_tags", type=int, default=20)
    parser.add_argument("--char_thr", type=float, default=0.30)
    parser.add_argument("--panel_thr", type=float, default=0.20)
    parser.add_argument("--text_thr", type=float, default=0.25)
    parser.add_argument("--min_panel_area", type=int, default=1024)
    parser.add_argument("--min_char_area", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0, help="limit number of images (0 = all)")
    args = parser.parse_args()
    main(args)
