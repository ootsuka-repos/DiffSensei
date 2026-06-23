import os
# Windows: torch's distributed TCPStore defaults to libuv, which Windows PyTorch is built
# without. If unset, Accelerator() tries to init it and the process exits silently at startup
# (just `python -m scripts.train.train ...` returns to the prompt). Must be set before torch is
# imported (transformers below pulls in torch). setdefault so an explicit override still wins.
os.environ.setdefault("USE_LIBUV", "0")
# Reduce CUDA memory fragmentation: let the allocator grow/shrink segments instead of
# reserving fixed blocks. Lets larger resolutions fit in the same 16GB by reclaiming
# "reserved but unallocated" gaps. Harmless if the user already set it. Must be set
# before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import logging
from omegaconf import OmegaConf
from datetime import datetime
import time
import gc
import subprocess
import itertools
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, AutoModel
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

import diffusers
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.training_utils import cast_training_params
from peft import LoraConfig

from src.models.unet import UNetMangaModel
from src.models.resampler import Resampler
from src.models.utils import load_ip_adapter, load_unet, load_ckpt, compute_ip_contrastive_loss, compute_ip_contrastive_loss_slow
from src.datasets.utils import size_buckets
from src.datasets.dataset_size_bucket import MangaTrainSizeBucketDataset, BucketBatchSampler, collate_fn
from scripts.utils import print_gpu_memory_usage


logger = get_logger(__name__, log_level="INFO")
logging.getLogger('PIL').setLevel(logging.WARNING)


def launch_eval(config, config_path, log_dir, ckpt_path, tag):
    """Fire-and-forget test inference of the just-saved checkpoint, on a SEPARATE GPU so it
    never competes with training for VRAM. REPRODUCES an actual TRAINING-DATA page with the
    exact training-time conditioning (caption / ip_bbox / dialog_bbox / self-condition refs)
    and saves a GT-vs-generated comparison into <log_dir>/samples/<tag>.png. The eval page is
    chosen from the same annotation file the model is training on. Controlled by `eval`."""
    ev = config.get("eval", None)
    if ev is None or not ev.get("enable", False):
        return None
    out_dir = os.path.join(log_dir, "samples")
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(ev.get("gpu", 1))   # default: GPU1 (training uses GPU0)
    env["HF_HUB_OFFLINE"] = "1"; env["TRANSFORMERS_OFFLINE"] = "1"
    out_path = os.path.join(out_dir, f"{tag}.png")
    cmd = [
        sys.executable, "-m", "scripts.demo.reproduce",
        "--config", config_path, "--ckpt", ckpt_path,
        "--ann", config.train_data.ann_path, "--image_root", config.train_data.image_root,
        "--out", out_path,
        "--steps", str(ev.get("steps", 28)),
        "--ip_scale", str(ev.get("ip_scale", 0.7)),
    ]
    if ev.get("page", None) is not None:
        cmd += ["--page", str(ev.get("page"))]
    try:
        logger.info(f"[eval] reproducing a training page (GT vs gen) -> {out_path} (GPU {env['CUDA_VISIBLE_DEVICES']})")
        return subprocess.Popen(cmd, env=env)
    except Exception as e:
        logger.warning(f"[eval] failed to launch: {e}")
        return None


def mean_multiple_ip_embeds(image_embeds, ip_exists, config, bsz):
    """
    image_embeds: [bsz * max_num_ip_sources, num_dummy_tokens + max_num_ips * num_vision_tokens, cross_attn_dim]
    """
    ip_image_embeds = image_embeds[:, config.model.num_dummy_tokens:, :]
    ip_image_embeds = ip_image_embeds.view(bsz, config.train_data.max_num_ip_sources, config.model.max_num_ips, config.model.num_vision_tokens, -1).transpose(1, 2).contiguous() # (bsz, max_num_ips, max_num_ip_sources, num_vision_tokens, dim)
    
    ip_mask = ip_exists.unsqueeze(-1).to(ip_image_embeds.device, dtype=ip_image_embeds.dtype) # (bsz, max_num_ips, max_num_ip_sources, 1)
    if len(ip_image_embeds.shape) == 5:
        ip_mask = ip_mask.unsqueeze(-1)
    masked_ip_image_embeds = ip_image_embeds * ip_mask
    # Sum along the num_sources axis and divide by the number of valid sources (avoid dividing by zero)
    valid_sources_count = ip_mask.sum(dim=2).clamp(min=1) # shape (bsz, max_num_ips). Clamp to avoid division by zero
    ip_image_embeds = masked_ip_image_embeds.sum(dim=2) / valid_sources_count # (bsz, max_num_ips, num_vision_tokens, dim)

    ip_image_embeds = ip_image_embeds.view(bsz, config.model.max_num_ips * config.model.num_vision_tokens, -1)
    image_embeds = image_embeds.view(bsz, config.train_data.max_num_ip_sources, *image_embeds.shape[1:])[:, 0, :, :]
    image_embeds[:, config.model.num_dummy_tokens:, :] = ip_image_embeds

    return image_embeds


def main(args):
    # Load and merge config
    config = OmegaConf.load(args.config_path)
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    args_conf = OmegaConf.create(args_dict)
    config = OmegaConf.merge(config, args_conf)
    if args.resume_log_dir is not None:
        # Resume
        log_dir = args.resume_log_dir
        OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))
    else:
        # Load config and create log folder
        config_name = args.config_path.split("/")[-1][:-5]
        config_folder = args.config_path.split("/")[-2]
        if config.exp_name:
            log_folder = f"{config_name}_{config.exp_name}"
        else:
            log_folder = f"{config_name}"
        
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
    )

    # Resolve the half-precision dtype early so we can cast frozen weights and save VRAM.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process and args.resume_log_dir is None:
        log_dir = os.path.join("logs", config_folder, log_folder, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
        OmegaConf.save(config, os.path.join(log_dir, "config.yaml"))
    accelerator.wait_for_everyone()
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info("\n" + "\n".join([f"{k}\t{v}" for k, v in OmegaConf.to_container(config, resolve=True).items()]))
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set the training seed
    set_seed(config.seed)

    # Load pretrained models
    tokenizer = CLIPTokenizer.from_pretrained(config.model.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config.model.pretrained_model_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(config.model.pretrained_model_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(config.model.pretrained_model_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(config.model.pretrained_model_path, subfolder="vae")
    noise_scheduler = DDPMScheduler.from_pretrained(config.model.pretrained_model_path, subfolder="scheduler")
    unet = UNetMangaModel.from_pretrained(config.model.pretrained_model_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(config.model.image_encoder_path)
    if config.model.magi_image_encoder_path is not None:
        magi_image_encoder = AutoModel.from_pretrained(config.model.magi_image_encoder_path, trust_remote_code=True).crop_embedding_model
        magi_image_encoder.requires_grad_(False)
    else:
        magi_image_encoder = None
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)

    if config.model.manga_t2i_model_path is not None:
        load_unet(unet, config.model.manga_t2i_model_path)

    # Init adapter modules
    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=config.model.num_vision_tokens,
        num_dummy_tokens=config.model.num_dummy_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
        magi_embedding_dim=magi_image_encoder.config.hidden_size if magi_image_encoder is not None else None,
        use_magi=config.model.magi_image_encoder_path is not None
    )

    # Register manga condition modules in unet
    unet.set_manga_modules(
        max_num_ips=config.model.max_num_ips,
        num_vision_tokens=config.model.num_vision_tokens,
        dialog_bbox_encode_type=config.model.dialog_bbox_encode_type,
        max_num_dialogs=config.model.max_num_dialogs,
        use_context=config.model.context_adapter
    )

    # Optionally graft the released DiffSensei character-injection (`*_ip`) + dialog
    # modules onto the (WAI) base as initialization, so we fine-tune from a working
    # IP setup instead of random init. With `diffsensei_ip_only`, the base drawing
    # weights stay as the (WAI) base, and only IP/dialog modules are replaced.
    diffsensei_path = config.model.get("diffsensei_pretrained_path", None)
    if diffsensei_path is not None:
        ds_unet = torch.load(os.path.join(diffsensei_path, "unet", "pytorch_model.bin"), map_location="cpu")
        if config.model.get("diffsensei_ip_only", False):
            ds_unet = {k: v for k, v in ds_unet.items() if ("_ip" in k or "dialog" in k)}
            missing, unexpected = unet.load_state_dict(ds_unet, strict=False)
            logger.info(f"grafted DiffSensei IP/dialog: {len(ds_unet)} tensors, unexpected={len(unexpected)}")
        else:
            unet.load_state_dict(ds_unet)
        del ds_unet
        proj_path = os.path.join(diffsensei_path, "image_proj_model", "pytorch_model.bin")
        if os.path.exists(proj_path):
            ds_proj = torch.load(proj_path, map_location="cpu")
            missing, unexpected = image_proj_model.load_state_dict(ds_proj, strict=False)
            logger.info(f"grafted DiffSensei Resampler: missing={len(missing)} unexpected={len(unexpected)}")
            del ds_proj

    if config.get("gradient_checkpointing", False):
        unet.enable_gradient_checkpointing()

    # Initialize Lora if use
    if config.model.unet_trained_parameters == 'lora':
        for name, param in unet.named_parameters():
            if '_ip' not in name:
                param.requires_grad_(False)
        if accelerator.mixed_precision == "fp16":
            unet.to(accelerator.device, dtype=weight_dtype)
        unet_lora_config = LoraConfig(
            r=config.model.lora_rank,
            lora_alpha=config.model.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        unet.add_adapter(unet_lora_config)
        if accelerator.mixed_precision == "fp16":
            cast_training_params(unet, dtype=torch.float32)

    if config.model.pretrained_ip_adapter_path is not None:
        load_ip_adapter(image_proj_model, unet, config.model.pretrained_ip_adapter_path)

    if config.model.manga_pretrained_model_path is not None:
        load_ckpt(image_proj_model, unet, config.model.manga_pretrained_model_path)
    
    # Load resume checkpoints if resume
    if args.resume_ckpt is not None:
        # Resume the trained weights (image_proj + unet_trained) from a specific ckpt.pth,
        # e.g. logs/.../epoch-34/ckpt.pth. Continues fine-tuning the same modules.
        load_ckpt(image_proj_model, unet, args.resume_ckpt)
        logger.info(f"resumed trained weights from {args.resume_ckpt}")
    elif args.resume_log_dir is not None:
        all_ckpt_steps = [d for d in os.listdir(log_dir) if d.startswith("step-")]
        last_ckpt_step = int(sorted(all_ckpt_steps, key=lambda x: int(x.split("-")[1]))[-1].split('-')[-1])
        load_ckpt(image_proj_model, unet, os.path.join(log_dir, f"step-{last_ckpt_step}", "ckpt.pth"))

    # Define which parameters to train
    unet_trained_parameters = []
    unet_trained_state_dict = {}
    unet_trained_parameter_names = []
    total_trained_parameter_size = 0
    for name, param in unet.named_parameters():
        is_train = False
        if config.model.unet_trained_parameters == 'full':
            is_train = True
        elif config.model.unet_trained_parameters == 'lora':
            # print(f"{name}: {param.shape}, requires_grad: {param.requires_grad}")
            if param.requires_grad == True:
                is_train = True
        elif config.model.unet_trained_parameters == 'new':
            if '_ip' in name or 'dialog' in name:
                is_train = True
        elif config.model.unet_trained_parameters == 'ip':
            if '_ip' in name:
                is_train = True
        else:
            raise NotImplementedError(f"The trained parameters type {config.model.unet_trained_parameters} is not implemented yet!")

        # For partial training ('new'/'ip'), freeze non-trained params and cast them to
        # half precision to fit a 16GB GPU. Trainable params stay fp32 for stable optim.
        if config.model.unet_trained_parameters in ('new', 'ip'):
            param.requires_grad_(is_train)
            if not is_train:
                param.data = param.data.to(weight_dtype)

        if is_train:
            unet_trained_parameters.append(param)
            unet_trained_state_dict[name] = param
            unet_trained_parameter_names.append(name)
            total_trained_parameter_size += param.numel() * param.element_size()

    logger.info(f"Total trained parameters in unet: {(total_trained_parameter_size / (1024 * 1024)):.2f} MB")
    
    trained_parameters = itertools.chain(image_proj_model.parameters(), unet_trained_parameters)
    trained_state_dict = {"image_proj": image_proj_model.state_dict(), "unet_trained": unet_trained_state_dict}

    # Optimizer (8-bit AdamW from bitsandbytes optionally, to save optimizer-state VRAM).
    # `paged: true` uses PagedAdamW8bit: optimizer state lives in unified memory and is paged
    # to CPU on demand, so a transient VRAM spike near the limit doesn't OOM the whole run.
    if config.optimizer.get("use_8bit_adam", False):
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.PagedAdamW8bit if config.optimizer.get("paged", False) else bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(
        trained_parameters,
        lr=config.optimizer.learning_rate,
        betas=(config.optimizer.adam_beta1, config.optimizer.adam_beta2),
        weight_decay=config.optimizer.adam_weight_decay,
        eps=config.optimizer.adam_epsilon,
    )

    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        config.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
        num_cycles=config.lr_scheduler.lr_num_cycles,
        power=config.lr_scheduler.lr_power,
    )

    # DataLoader
    # Cap training resolution to fit a 16GB GPU. `size_buckets` has tiers keyed by their square
    # side ("size": 256/512/1024/1280): a tier's square bucket is size x size, with aspect-ratio
    # variants of the same area. We keep tiers whose SQUARE side <= max_bucket_size, so e.g.
    # max_bucket_size=1024 trains squares up to 1024x1024 (SDXL native), 1280 up to ~1280x1280.
    # (Earlier this filtered on max *side*, which made "1024" secretly cap squares at 512.)
    max_bucket = config.train_data.get("max_bucket_size", 512)
    if max_bucket is not None:
        train_size_buckets = [t for t in size_buckets if t["size"] <= max_bucket]
        if not train_size_buckets:
            train_size_buckets = size_buckets
    else:
        train_size_buckets = size_buckets
    tiers_str = ",".join(str(t["size"]) for t in train_size_buckets)
    logger.info(f"using {len(train_size_buckets)}/{len(size_buckets)} bucket tiers (square <= {max_bucket}; tiers: {tiers_str})")

    # Optional precomputed VAE latent cache: drops the VAE from the training loop entirely
    # (no fp32 encode, no VAE weights on GPU) — the biggest single VRAM saving at high res.
    use_latent_cache = config.train_data.get("use_latent_cache", False)
    latent_cache_dir = None
    if use_latent_cache:
        latent_cache_dir = config.train_data.get("latent_cache_dir", None) or os.path.join(
            "data", "latent_cache", f"wai_maxb{config.train_data.get('max_bucket_size', 512)}")
        assert os.path.isdir(latent_cache_dir), (
            f"latent cache dir not found: {latent_cache_dir}. "
            f"Run: python -m scripts.train.precompute_latents --config {args.config_path}")
        logger.info(f"using precomputed latent cache: {latent_cache_dir} (VAE removed from training loop)")

    # Optional precomputed text-embedding cache: drops both text encoders from VRAM (~1.6GB).
    use_text_cache = config.train_data.get("use_text_cache", False)
    text_cache_dir = None
    if use_text_cache:
        text_cache_dir = config.train_data.get("text_cache_dir", None) or os.path.join("data", "text_cache", "wai")
        assert os.path.isdir(text_cache_dir), (
            f"text cache dir not found: {text_cache_dir}. "
            f"Run: python -m scripts.train.precompute_text_embeds --config {args.config_path}")
        logger.info(f"using precomputed text cache: {text_cache_dir} (text encoders removed from training loop)")

    train_dataset = MangaTrainSizeBucketDataset(
        ann_path=config.train_data.ann_path,
        image_root=config.train_data.image_root,
        size_buckets=train_size_buckets,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        t_drop_rate=config.train_data.t_drop_rate,
        i_drop_rate=config.train_data.i_drop_rate,
        c_drop_rate=config.train_data.c_drop_rate,
        max_num_ips=config.model.max_num_ips,
        max_num_ip_sources=config.train_data.max_num_ip_sources,
        max_num_dialogs=config.model.max_num_dialogs,
        mask_dialog=config.train_data.mask_dialog,
        load_context_image=config.model.context_adapter,
        ip_self_condition_rate=config.train_data.ip_self_condition_rate,
        min_ip_height=config.train_data.min_ip_height,
        min_ip_width=config.train_data.min_ip_width,
        latent_cache_dir=latent_cache_dir,
        text_cache_dir=text_cache_dir,
    )
    batch_sampler = BucketBatchSampler(
        dataset=train_dataset,
        batch_size=config.train_batch_size
    )
    _nw = config.train_data.get("num_workers", 8)
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=_nw * accelerator.num_processes,
        persistent_workers=_nw > 0,
        collate_fn=collate_fn,
    )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoders,
    # image_encoder) to half-precision; they are only used for inference. (weight_dtype was
    # resolved right after the accelerator was created.)
    #
    # VAE: the fp32 VAE encode is the single biggest activation hog at high resolution and the
    # usual 16GB OOM trigger. Run it in bf16 by default (bf16's wide exponent range avoids the
    # fp16-VAE NaN problem), and tile+slice the encode so its peak memory stays flat as the
    # resolution grows. This is what makes the 1024 tier fit.
    vae_dtype = weight_dtype
    _vd = config.train_data.get("vae_dtype", "bf16")
    vae_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(_vd, weight_dtype)
    if use_latent_cache:
        # latents come from the cache; keep the VAE off the GPU (only its scaling_factor is used).
        vae_scaling_factor = vae.config.scaling_factor
        vae.to("cpu")
    else:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae_scaling_factor = vae.config.scaling_factor
        if config.train_data.get("vae_tiling", True):
            vae.enable_slicing()
            vae.enable_tiling()
    if use_text_cache:
        # text embeds come from the cache; keep both text encoders off the GPU.
        text_encoder.to("cpu")
        text_encoder_2.to("cpu")
    else:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    if magi_image_encoder is not None:
        magi_image_encoder.to(accelerator.device, dtype=weight_dtype)

    # Prepare everything with accelerator
    image_proj_model, unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        image_proj_model, unet, optimizer, lr_scheduler, train_dataloader
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("manga")
        tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Base batch size per device = {config.train_batch_size}")
    logger.info(f"  Batch number per epoch = {len(train_dataloader)}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {config.max_train_steps}")
    
    if args.resume_log_dir is not None:
        global_step = last_ckpt_step
    else:
        global_step = 0

    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=global_step,
        desc="Step",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # # Save pre-trained model weights if necessary
    if 0 in config.checkpointing_steps and accelerator.is_main_process and config.checkpoints_total_limit != 0 and global_step == 0:
        step_dir = os.path.join(log_dir, f"step-{global_step}")
        save_path = os.path.join(step_dir, "ckpt.pth")
        os.makedirs(step_dir, exist_ok=True)
        trained_state_dict = {"image_proj": accelerator.unwrap_model(image_proj_model).state_dict(), "unet_trained": unet_trained_state_dict} # must add, to update the state_dict of image_proj_model
        torch.save(trained_state_dict, save_path)
        logger.info(f"Saved state to {save_path}")

    if hasattr(train_dataset, "buckets"):
        for bucket_key, value in train_dataset.buckets.items():
            logger.info(f"{bucket_key}: {len(value)}")

    # Epoch-based training. One epoch = one full pass over the dataloader.
    steps_per_epoch = max(1, len(train_dataloader) // config.gradient_accumulation_steps)
    if config.get("num_train_epochs", None):
        num_train_epochs = int(config.num_train_epochs)
    else:
        num_train_epochs = max(1, (config.max_train_steps + steps_per_epoch - 1) // steps_per_epoch)
    save_every = int(config.get("checkpointing_epochs", 1))
    logger.info(f"  Epochs = {num_train_epochs} (steps/epoch ~ {steps_per_epoch}); checkpoint+eval every {save_every} epoch(s)")

    unet.train()
    eval_procs = []
    for epoch in range(num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            with accelerator.accumulate(image_proj_model, unet):
                # Convert images to latent space (or sample from the precomputed latent cache).
                with torch.no_grad():
                    if batch.get("latent_mean", None) is not None:
                        mean = batch["latent_mean"].to(accelerator.device, dtype=torch.float32)
                        std = batch["latent_std"].to(accelerator.device, dtype=torch.float32)
                        latents = mean + std * torch.randn_like(mean)   # reparameterized VAE sample
                    else:
                        latents = vae.encode(batch["images"].to(accelerator.device, dtype=vae_dtype)).latent_dist.sample()
                    latents = latents * vae_scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample the noise
                noise = torch.randn_like(latents)

                # Sample a random timestep for each image
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Encode IP images
                with torch.no_grad():
                    if config.model.ip_adapter_plus:
                        image_embeds = image_encoder(batch["ip_images"].to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2] # [bsz * max_num_ips * max_num_ip_sources, sequence_length, dim]
                    else:
                        image_embeds = image_encoder(batch["ip_images"].to(accelerator.device, dtype=weight_dtype)).image_embeds
                    image_embeds = image_embeds.view(bsz, config.model.max_num_ips, config.train_data.max_num_ip_sources, *image_embeds.shape[1:]).transpose(1, 2).contiguous().view(bsz * config.train_data.max_num_ip_sources, config.model.max_num_ips, *image_embeds.shape[1:]).to(dtype=torch.float32)

                    if magi_image_encoder is not None:
                        magi_image_embeds = magi_image_encoder(batch["magi_ip_images"].to(accelerator.device, dtype=weight_dtype)).last_hidden_state[:, 0]
                        magi_image_embeds = magi_image_embeds.view(bsz, config.model.max_num_ips, config.train_data.max_num_ip_sources, *magi_image_embeds.shape[1:]).transpose(1, 2).contiguous().view(bsz * config.train_data.max_num_ip_sources, config.model.max_num_ips, *magi_image_embeds.shape[1:]).to(dtype=torch.float32)
                    else:
                        magi_image_embeds = None
                
                image_embeds = image_proj_model(image_embeds, magi_image_embeds) # [bsz * max_num_ip_sources, num_dummy_tokens + max_num_ips * num_vision_tokens, cross_attn_dim]

                # Compute IP image embeds contrastive loss
                if config.model.ip_contrastive_loss == "fast":
                    loss_ip_contrastive = compute_ip_contrastive_loss(image_embeds[:, config.model.num_dummy_tokens:, :], batch["ip_exists"], config, bsz)
                elif config.model.ip_contrastive_loss == "slow":
                    loss_ip_contrastive = compute_ip_contrastive_loss_slow(image_embeds[:, config.model.num_dummy_tokens:, :], batch["ip_exists"], config, bsz)
                else:
                    loss_ip_contrastive = torch.Tensor([0.0]).to(device=accelerator.device, dtype=image_embeds.dtype)

                # Mean the max_num_ip_sources dimension
                image_embeds = mean_multiple_ip_embeds(image_embeds, batch["ip_exists"], config, bsz) # [bsz, num_dummy_tokens + max_num_ips * num_vision_tokens, cross_attn_dim]

                # Text condition: from cache, or encode live.
                if batch.get("text_embeds", None) is not None:
                    text_embeds = batch["text_embeds"].to(accelerator.device, dtype=weight_dtype)
                    pooled_text_embeds = batch["pooled_text_embeds"].to(accelerator.device, dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                        encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)

                # Concat other embeddings into text_embeds
                encoder_hidden_states = torch.cat([text_embeds, image_embeds], dim=1)

                # Prepare SDXL extra conditions
                # Transfer dialog bbox into positional embeddings and concat to add_time_ids
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                if config.model.dialog_bbox_encode_type == "aug":
                    unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids, "dialog_bbox": batch["dialog_bbox"].to(accelerator.device)}
                else:
                    unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                
                # Predict the noise
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    added_cond_kwargs=unet_added_cond_kwargs,
                    cross_attention_kwargs={"bbox": batch["ip_bbox"], "aspect_ratio": latents.shape[-2] / latents.shape[-1]},
                    dialog_bbox=batch["dialog_bbox"].to(accelerator.device),
                ).sample
                
                # Compute the MSE loss
                loss_diffusion = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                loss = loss_diffusion + config.model.ip_contrastive_loss_weight * loss_ip_contrastive

                # Backward
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            # (checkpointing + test inference happen at the end of each epoch, below)

            avg_loss_diffusion = accelerator.gather(loss_diffusion.unsqueeze(0)).mean().detach().item()
            avg_ip_contrastive_loss = accelerator.gather(loss_ip_contrastive.unsqueeze(0)).mean().detach().item()
            
            logs = {
                "Diffusion Loss": f"{avg_loss_diffusion:.4f}",
                "IP Contastive Loss": f"{avg_ip_contrastive_loss:.4f}",
                "Step Time": f"{time.perf_counter() - begin:.2f}s",
                "Data Time": f"{load_data_time:.2f}s",
            } 
            progress_bar.set_postfix(**logs)
            
            if accelerator.is_main_process:
                tb_writer.add_scalar("Diffusion Loss", avg_loss_diffusion, global_step)
                tb_writer.add_scalar("IP Contastive Loss", avg_ip_contrastive_loss, global_step)

            begin = time.perf_counter()
            # print_gpu_memory_usage(accelerator.local_process_index)

        # --- End of epoch: save checkpoint + fire a test inference on a separate GPU ---
        if accelerator.is_main_process and config.checkpoints_total_limit != 0 and (epoch + 1) % save_every == 0:
            ep_dir = os.path.join(log_dir, f"epoch-{epoch + 1}")
            os.makedirs(ep_dir, exist_ok=True)
            save_path = os.path.join(ep_dir, "ckpt.pth")
            trained_state_dict = {"image_proj": accelerator.unwrap_model(image_proj_model).state_dict(), "unet_trained": unet_trained_state_dict}
            torch.save(trained_state_dict, save_path)
            logger.info(f"[epoch {epoch + 1}/{num_train_epochs}] saved {save_path} (global_step={global_step})")
            eval_procs = [p for p in eval_procs if p.poll() is None]   # reap finished evals
            proc = launch_eval(config, args.config_path, log_dir, save_path, f"epoch-{epoch + 1}")
            if proc is not None:
                eval_procs.append(proc)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    accelerator.end_training()
    logger.info(f"The End")


if __name__ == "__main__":
    """
    nohup accelerate launch \
        --multi_gpu \
        -m scripts.train.train \
        --config_path configs/train/diffsensei/self_0.5.yaml \
        > nohup/train.out 2>&1 &
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--inference_config_path", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default="")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=-1, help="-1 means no limit")
    parser.add_argument("--resume_log_dir", type=str, default=None)
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="resume trained weights from a specific ckpt.pth (e.g. logs/.../epoch-34/ckpt.pth)")
    args = parser.parse_args()
    
    main(args)