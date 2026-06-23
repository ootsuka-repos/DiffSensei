"""Resolution and dtype helpers for training-time eval sample generation."""

import torch

from src.datasets.utils import get_bucket_size, size_buckets


def infer_eval_dtype(cfg, device):
    if not device.startswith("cuda"):
        return torch.float32
    dtype_override = cfg.get("eval", {}).get("dtype", None)
    if dtype_override == "fp16":
        return torch.float16
    if dtype_override == "fp32":
        return torch.float32
    mp = cfg.get("mixed_precision", "bf16")
    if mp == "bf16":
        return torch.bfloat16
    if mp in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def eval_size_buckets(cfg, min_bucket_size=None):
    min_tier = min_bucket_size or cfg.get("eval", {}).get("min_bucket_size", 1280)
    tier = next((t for t in size_buckets if t["size"] == min_tier), None)
    if tier is None:
        tier = max(size_buckets, key=lambda t: t["size"])
    return [tier]


def frame_gen_size(fw, fh, cfg, min_bucket_size=None):
    gh, gw, _ = get_bucket_size(fh, fw, eval_size_buckets(cfg, min_bucket_size))
    return gw, gh