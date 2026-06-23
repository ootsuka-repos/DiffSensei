"""Build a DiffSenseiPipeline from a stage-2 training checkpoint."""

import os

import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection, AutoModel

from diffsensei.models.resampler import Resampler
from diffsensei.models.unet import UNetMangaModel
from diffsensei.pipelines.pipeline_diffsensei import DiffSenseiPipeline


def resolve_weight_dtype(cfg, device):
    if not device.startswith("cuda"):
        return torch.float32
    mp = cfg.get("mixed_precision", "bf16")
    if mp == "bf16":
        return torch.bfloat16
    if mp in ("fp16", "float16"):
        return torch.float16
    return torch.float32


def build_pipeline(cfg, ckpt_path, weight_dtype, device):
    m = cfg.model
    unet = UNetMangaModel.from_pretrained(m.pretrained_model_path, subfolder="unet", torch_dtype=weight_dtype)
    unet.set_manga_modules(
        max_num_ips=m.max_num_ips,
        num_vision_tokens=m.num_vision_tokens,
        max_num_dialogs=m.max_num_dialogs,
        dialog_bbox_encode_type=m.get("dialog_bbox_encode_type", "mask"),
        use_context=m.get("context_adapter", False),
    )

    ds_path = m.get("diffsensei_pretrained_path", None)
    if ds_path is not None:
        ds_unet = torch.load(os.path.join(ds_path, "unet", "pytorch_model.bin"), map_location="cpu")
        if m.get("diffsensei_ip_only", False):
            ds_unet = {k: v for k, v in ds_unet.items() if ("_ip" in k or "dialog" in k)}
            unet.load_state_dict(ds_unet, strict=False)
        else:
            unet.load_state_dict(ds_unet)
        del ds_unet

    if m.unet_trained_parameters == "lora":
        from peft import LoraConfig
        unet.add_adapter(LoraConfig(
            r=m.lora_rank, lora_alpha=m.lora_rank, init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        ))

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(m.image_encoder_path, torch_dtype=weight_dtype)
    magi_image_encoder = AutoModel.from_pretrained(m.magi_image_encoder_path, trust_remote_code=True).crop_embedding_model
    magi_image_encoder = magi_image_encoder.to(device=device, dtype=weight_dtype)

    image_proj_model = Resampler(
        dim=1280, depth=4, dim_head=64, heads=20,
        num_queries=m.num_vision_tokens, num_dummy_tokens=m.num_dummy_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim, ff_mult=4,
        magi_embedding_dim=magi_image_encoder.config.hidden_size, use_magi=True,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    unet.load_state_dict(ckpt["unet_trained"], strict=False)
    image_proj_model.load_state_dict(ckpt["image_proj"])

    image_proj_model = image_proj_model.to(device=device, dtype=weight_dtype)

    pipeline = DiffSenseiPipeline.from_pretrained(
        m.pretrained_model_path, unet=unet, image_encoder=image_encoder, torch_dtype=weight_dtype,
    )
    pipeline.progress_bar_config = {"disable": False}
    pipeline.register_manga_modules(image_proj_model=image_proj_model, magi_image_encoder=magi_image_encoder)
    pipeline.to(device=device, dtype=weight_dtype)
    return pipeline


def load_ip_images(paths):
    return [Image.open(p).convert("L").convert("RGB") for p in paths]