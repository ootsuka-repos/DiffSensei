"""GPU-free import and helper tests for src/inference layout."""

import importlib
import inspect
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_src_inference_public_exports():
    from src.inference import (
        build_pipeline,
        eval_size_buckets,
        frame_gen_size,
        infer_eval_dtype,
        load_ip_images,
        resolve_weight_dtype,
    )

    assert callable(build_pipeline)
    assert callable(load_ip_images)
    assert callable(resolve_weight_dtype)
    assert callable(frame_gen_size)
    assert callable(infer_eval_dtype)
    assert callable(eval_size_buckets)


def test_resolve_weight_dtype_matches_mixed_precision():
    from src.inference import resolve_weight_dtype

    cfg_bf16 = OmegaConf.create({"mixed_precision": "bf16"})
    cfg_fp16 = OmegaConf.create({"mixed_precision": "fp16"})
    assert resolve_weight_dtype(cfg_bf16, "cuda:0") is torch.bfloat16
    assert resolve_weight_dtype(cfg_fp16, "cuda:0") is torch.float16
    assert resolve_weight_dtype(cfg_bf16, "cpu") is torch.float32


def test_frame_gen_size_uses_1280_tier():
    from src.inference import frame_gen_size
    from src.datasets.utils import get_bucket_size, size_buckets

    cfg = OmegaConf.create({"eval": {"min_bucket_size": 1280}})
    fw, fh = 731, 558
    gw, gh = frame_gen_size(fw, fh, cfg)
    tier1280 = [t for t in size_buckets if t["size"] == 1280]
    expected_gh, expected_gw, _ = get_bucket_size(fh, fw, tier1280)
    assert (gw, gh) == (expected_gw, expected_gh)
    assert max(gw, gh) >= 1024


def test_build_pipeline_is_in_src_inference_pipeline_module():
    from src.inference.pipeline import build_pipeline as direct_build
    from src.inference import build_pipeline as exported_build

    assert direct_build is exported_build
    assert "ckpt_path" in inspect.signature(direct_build).parameters


def test_eval_subprocess_module_path():
    train_py = REPO_ROOT / "scripts" / "train" / "train.py"
    text = train_py.read_text(encoding="utf-8")
    assert '"scripts.eval.reproduce"' in text or "'scripts.eval.reproduce'" in text


def test_cli_modules_are_importable():
    for mod in (
        "scripts.eval.reproduce",
        "scripts.inference.inference_trained",
        "scripts.inference.make_page",
        "scripts.refs.gen_wai",
    ):
        importlib.import_module(mod)