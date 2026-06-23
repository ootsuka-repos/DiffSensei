"""
Build a CLEAN character reference image for DiffSensei IP conditioning.

Instead of using Magi's loose character bbox (which includes panel background), this
picks a prominent single-character instance from the annotations, then runs imgutils'
anime segmentation (isnetis) to cut the character out and place it on a white
background. The result is a clean "who should appear" reference for the page.

Usage:
    # auto-pick the most prominent character from the dataset:
    python -m scripts.dataset.extract_ref --ann data/annotations/train.json --image_root data --out outputs/page_ref.png
    # or extract from a specific image you choose:
    python -m scripts.dataset.extract_ref --source path/to/char.png --out outputs/page_ref.png
"""
import os
import sys
import json
import argparse

from PIL import Image

sys.path.append(os.getcwd())


def candidate_crops(ann_path, image_root, work=None, skip_first=1):
    """Single-character crops, largest first (prominent characters make better refs).

    If `work` is given, only consider pages under that data/<work>/ folder, and skip the
    first `skip_first` pages of the work (the cover / color title pages = bad training data).
    """
    ann = json.load(open(ann_path, encoding="utf-8"))
    if work is not None:
        pages = sorted([p for p in ann if p["image_path"].replace("\\", "/").split("/")[0] == str(work)],
                       key=lambda p: p["image_path"])
        pages = pages[skip_first:]
    else:
        pages = ann
    cands = []
    for page in pages:
        for fr in page["frames"]:
            if len(fr["characters"]) != 1:
                continue
            x1, y1, x2, y2 = fr["characters"][0]["bbox"]
            w, h = x2 - x1, y2 - y1
            if w < 120 or h < 120:
                continue
            if not (0.4 < w / h < 1.1):
                continue
            cands.append((w * h, page["image_path"], [x1, y1, x2, y2]))
    cands.sort(reverse=True)
    return cands


IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg", ".bmp")


def best_portrait_from_work(image_root, work, skip_first=2):
    """Annotation-free: scan data/<work>/ pages directly, detect characters with imgutils,
    and return the cleanest single-character upper-body portrait found in the work.

    Picks the largest halfbody region that contains exactly one face (a clear solo
    character), skipping the first `skip_first` pages (covers / color title pages).
    """
    import glob
    from imgutils.detect import detect_faces, detect_halfbody

    folder = os.path.join(image_root, str(work))
    files = []
    for ext in IMAGE_EXTS:
        files += glob.glob(os.path.join(folder, "*" + ext))
    files = sorted(files)[skip_first:]

    best = None  # (area, path, box)
    for fp in files:
        img = Image.open(fp).convert("RGB")
        half = detect_halfbody(img)
        if not half:
            continue
        for (hx1, hy1, hx2, hy2), _, _ in half:
            hb = (int(hx1), int(hy1), int(hx2), int(hy2))
            sub = img.crop(hb)
            faces = detect_faces(sub)
            if len(faces) != 1:          # want exactly one clear face (solo, clean)
                continue
            area = (hb[2] - hb[0]) * (hb[3] - hb[1])
            if best is None or area > best[0]:
                best = (area, fp, hb)
    if best is None:
        raise SystemExit(f"work {work}: no clean single-character portrait found")
    _, fp, hb = best
    print(f"work {work}: picked {os.path.relpath(fp, image_root)} box={hb}")
    return Image.open(fp).convert("RGB").crop(hb)


def portrait_from_crop(crop):
    """Return a clean head+upper-body portrait crop if a face is present, else None."""
    from imgutils.detect import detect_faces, detect_halfbody
    faces = detect_faces(crop)
    if not faces:
        return None
    half = detect_halfbody(crop)
    W, H = crop.size
    if half:
        (hx1, hy1, hx2, hy2), _, _ = max(half, key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]))
    else:
        # fall back to expanding the face box into an upper-body region
        (fx1, fy1, fx2, fy2), _, _ = max(faces, key=lambda b: (b[0][2] - b[0][0]) * (b[0][3] - b[0][1]))
        fw, fh = fx2 - fx1, fy2 - fy1
        hx1, hy1, hx2, hy2 = fx1 - fw, fy1 - 0.6 * fh, fx2 + fw, fy2 + 3.2 * fh
    box = (max(0, int(hx1)), max(0, int(hy1)), min(W, int(hx2)), min(H, int(hy2)))
    if box[2] - box[0] < 80 or box[3] - box[1] < 80:
        return None
    return crop.crop(box)


def main(args):
    from imgutils.segment import segment_rgba_with_isnetis

    if args.source:
        crop = Image.open(args.source).convert("RGB")
        print(f"source image: {args.source} {crop.size}")
        portrait = portrait_from_crop(crop) or crop
    elif args.work is not None:
        # annotation-free: scan the work's pages directly with imgutils detection
        portrait = best_portrait_from_work(args.image_root, args.work, skip_first=args.skip_first)
    else:
        portrait = None
        for area, img, bbox in candidate_crops(args.ann, args.image_root):
            crop = Image.open(os.path.join(args.image_root, img)).convert("RGB").crop(tuple(bbox))
            p = portrait_from_crop(crop)
            if p is not None:
                print(f"picked source: {img} bbox={bbox} -> portrait {p.size}")
                portrait = p
                break
        if portrait is None:
            raise SystemExit("no single-character crop with a detectable face found")

    mask, rgba = segment_rgba_with_isnetis(portrait)   # rgba: character on transparent bg

    # composite onto a white background (manga pages are white)
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    out = Image.alpha_composite(white, rgba).convert("RGB")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.save(args.out)
    print(f"saved clean reference: {args.out} {out.size}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="data/annotations/train.json")
    ap.add_argument("--image_root", default="data")
    ap.add_argument("--source", default=None, help="use this image instead of auto-picking")
    ap.add_argument("--work", default=None, help="only pick from data/<work>/ (e.g. 1..5)")
    ap.add_argument("--skip_first", type=int, default=1, help="skip first N pages of the work (covers)")
    ap.add_argument("--out", default="outputs/page_ref.png")
    args = ap.parse_args()
    main(args)
