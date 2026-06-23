"""
Inference with a checkpoint produced by scripts/train/train.py, rebuilding the
EXACT training-time model from that run's config (base UNet, optional DiffSensei
IP graft, optional LoRA) and overlaying the trained weights (`unet_trained` + `image_proj`).

Usage:
    python -m scripts.demo.inference_trained \
        --config configs/train/diffsensei/self_finetune_wai_condition_5060ti.yaml \
        --ckpt logs/diffsensei/self_finetune_wai_condition_5060ti/<ts>/epoch-1/ckpt.pth \
        --input_json scripts/demo/eval_input.json \
        --output_dir outputs --tag epoch1
"""
import os
import sys
import json
import argparse

import torch
from PIL import Image
from omegaconf import OmegaConf
from transformers import CLIPVisionModelWithProjection, AutoModel

os.environ.setdefault("USE_LIBUV", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, os.getcwd())

from src.models.unet import UNetMangaModel
from src.models.resampler import Resampler
from src.pipelines.pipeline_diffsensei import DiffSenseiPipeline


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

    # Re-create the frozen DiffSensei IP graft so the trained ckpt sits on the same base
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

    # Overlay the TRAINED weights from this run
    ckpt = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = unet.load_state_dict(ckpt["unet_trained"], strict=False)
    image_proj_model.load_state_dict(ckpt["image_proj"])
    print(f"loaded trained ckpt: {len(ckpt['unet_trained'])} unet tensors, image_proj ok")

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


def main(args):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    weight_dtype = torch.float16 if device.startswith("cuda") else torch.float32

    cfg = OmegaConf.load(args.config)
    with open(args.input_json, "r", encoding="utf-8") as f:
        sample = json.load(f)

    pipeline = build_pipeline(cfg, args.ckpt, weight_dtype, device)
    print("trained pipeline ready")

    generator = torch.Generator(device).manual_seed(args.seed)
    images = pipeline(
        prompt=sample["prompt"], prompt_2=sample["prompt"],
        height=sample.get("height", 512), width=sample.get("width", 512),
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
        negative_prompt=args.negative_prompt, negative_prompt_2=args.negative_prompt,
        num_samples=args.num_samples, generator=generator,
        ip_images=load_ip_images(sample.get("ip_images", [])), ip_image_embeds=None,
        ip_bbox=[list(b) for b in sample.get("ip_bbox", [])], ip_scale=args.ip_scale,
        dialog_bbox=[list(b) for b in sample.get("dialog_bbox", [])],
    ).images

    os.makedirs(args.output_dir, exist_ok=True)
    for i, image in enumerate(images):
        out = os.path.join(args.output_dir, f"{args.tag}_{i}.png")
        image.save(out)
        print(f"saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--tag", type=str, default="trained")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--ip_scale", type=float, default=0.6)
    parser.add_argument("--negative_prompt", type=str,
                        default="think lines, pure black background, colored, lowres, bad anatomy, worst quality, low quality")
    args = parser.parse_args()
    main(args)
