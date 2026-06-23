"""
Fetch the parts of the released DiffSensei model needed for fine-tuning / released
inference, into `checkpoints/diffsensei/image_generator`:
  - unet/pytorch_model.bin            (~11.6 GB, the trained manga UNet)
  - image_proj_model/pytorch_model.bin (~336 MB, the trained Resampler)

The rest of the pipeline (vae / text encoders / tokenizers / scheduler) is reused
from the SDXL base, and the CLIP image encoder + Magi crop encoder from the HF
cache, so the giant MLLM (SEED-X, ~55GB) and duplicate encoders are NOT downloaded.

Resilient (curl with resume + stall-retry); idempotent (skips present files).

Usage:
    python -m scripts.train.prepare_diffsensei
"""
import os
import subprocess

from huggingface_hub import hf_hub_url, hf_hub_download

REPO = "jianzongwu/DiffSensei"
OUT = os.path.join("checkpoints", "diffsensei", "image_generator")
FILES = [
    "unet/pytorch_model.bin",
    "image_proj_model/pytorch_model.bin",
]
# expected minimum sizes (bytes) to consider a file complete
MIN_SIZE = {
    "unet/pytorch_model.bin": 11_000_000_000,
    "image_proj_model/pytorch_model.bin": 300_000_000,
}


def _curl_download(url, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    cmd = [
        "curl", "-L", "--fail",
        "--retry", "60", "--retry-all-errors", "--retry-delay", "3",
        "-C", "-",
        "--speed-limit", "30000", "--speed-time", "30",
        "-o", target, url,
    ]
    subprocess.run(cmd, check=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in FILES:
        target = os.path.join(OUT, f)
        if os.path.exists(target) and os.path.getsize(target) >= MIN_SIZE.get(f, 1):
            print(f"[skip] {f} already present ({os.path.getsize(target)} bytes)")
            continue
        print(f"[download] {f}", flush=True)
        url = hf_hub_url(REPO, "image_generator/" + f)
        try:
            _curl_download(url, target)
        except Exception as e:
            print(f"  curl failed ({e}); falling back to hf_hub_download ...")
            hf_hub_download(REPO, "image_generator/" + f, local_dir="checkpoints/diffsensei")

    print(f"\nReleased DiffSensei (unet + image_proj) ready at: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
