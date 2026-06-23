import os
import numpy as np
from PIL import Image
from safetensors import safe_open

import torch
import torch.nn.functional as F


def _gather_ip_embeds(ip_image_embeds, ip_exists, config, bsz):
    """
    Reshape per-source IP token embeddings into per-(sample, character) mean embeddings,
    keeping only characters that actually exist.

    Args:
        ip_image_embeds: [bsz * max_num_ip_sources, max_num_ips * num_vision_tokens, dim]
        ip_exists: [bsz, max_num_ips, max_num_ip_sources]
    Returns:
        embeds: [N, dim] L2-normalized mean embedding per valid character
        labels: [N] integer label = global character index (sample_idx * max_num_ips + ip_idx)
    """
    max_num_ips = config.model.max_num_ips
    num_vision_tokens = config.model.num_vision_tokens
    max_num_ip_sources = config.train_data.max_num_ip_sources
    dim = ip_image_embeds.shape[-1]

    # -> [bsz, max_num_ip_sources, max_num_ips, num_vision_tokens, dim]
    embeds = ip_image_embeds.view(bsz, max_num_ip_sources, max_num_ips, num_vision_tokens, dim)
    # mean over the vision-token axis -> one vector per (sample, source, character)
    embeds = embeds.mean(dim=3)  # [bsz, max_num_ip_sources, max_num_ips, dim]
    embeds = embeds.transpose(1, 2).contiguous()  # [bsz, max_num_ips, max_num_ip_sources, dim]

    mask = ip_exists.to(embeds.device, dtype=embeds.dtype)  # [bsz, max_num_ips, max_num_ip_sources]
    valid = mask.sum(dim=2)  # [bsz, max_num_ips]
    # mean over the available sources of each character
    summed = (embeds * mask.unsqueeze(-1)).sum(dim=2)  # [bsz, max_num_ips, dim]
    char_embeds = summed / valid.clamp(min=1).unsqueeze(-1)  # [bsz, max_num_ips, dim]

    keep = valid > 0  # [bsz, max_num_ips]
    if keep.sum() == 0:
        return None, None

    selected = char_embeds[keep]  # [N, dim]
    labels = torch.arange(bsz * max_num_ips, device=embeds.device).view(bsz, max_num_ips)[keep]
    selected = F.normalize(selected, dim=-1)
    return selected, labels


def compute_ip_contrastive_loss(ip_image_embeds, ip_exists, config, bsz, temperature=0.07):
    """
    InfoNCE-style contrastive loss that pulls together the embeddings of the same character
    (different source crops) and pushes apart different characters within the batch.
    Vectorized ("fast") implementation.
    """
    embeds, labels = _gather_ip_embeds(ip_image_embeds, ip_exists, config, bsz)
    if embeds is None or embeds.shape[0] < 2:
        return torch.zeros([], device=ip_image_embeds.device, dtype=ip_image_embeds.dtype)

    logits = embeds @ embeds.t() / temperature  # [N, N]
    n = logits.shape[0]
    eye = torch.eye(n, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(eye, float("-inf"))

    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    same = same & (~eye)

    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    has_pos = same.any(dim=1)
    if has_pos.sum() == 0:
        return torch.zeros([], device=ip_image_embeds.device, dtype=ip_image_embeds.dtype)

    pos_log_prob = (log_prob * same).sum(dim=1) / same.sum(dim=1).clamp(min=1)
    loss = -pos_log_prob[has_pos].mean()
    return loss


def compute_ip_contrastive_loss_slow(ip_image_embeds, ip_exists, config, bsz, temperature=0.07):
    """Readable per-anchor implementation of `compute_ip_contrastive_loss` (same objective)."""
    embeds, labels = _gather_ip_embeds(ip_image_embeds, ip_exists, config, bsz)
    if embeds is None or embeds.shape[0] < 2:
        return torch.zeros([], device=ip_image_embeds.device, dtype=ip_image_embeds.dtype)

    n = embeds.shape[0]
    losses = []
    for i in range(n):
        sim = embeds[i] @ embeds.t() / temperature  # [N]
        mask_self = torch.ones(n, device=embeds.device, dtype=torch.bool)
        mask_self[i] = False
        pos_mask = (labels == labels[i]) & mask_self
        if pos_mask.sum() == 0:
            continue
        denom = torch.logsumexp(sim[mask_self], dim=0)
        pos_log_prob = sim[pos_mask] - denom
        losses.append(-pos_log_prob.mean())

    if len(losses) == 0:
        return torch.zeros([], device=ip_image_embeds.device, dtype=ip_image_embeds.dtype)
    return torch.stack(losses).mean()


def get_generator(seed, device):
    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator


def load_unet(unet, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location="cpu")
    unet.load_state_dict(state_dict["unet_trained"], strict=False)


def load_ip_adapter(image_proj_model, unet, ckpt_path):
    if os.path.splitext(ckpt_path)[-1] == ".safetensors":
        state_dict = {"image_proj": {}, "ip_adapter": {}}
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("image_proj."):
                    state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                elif key.startswith("ip_adapter."):
                    state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    ori_param_weights_sum = torch.sum(torch.stack([torch.sum(p) for p in image_proj_model.parameters()]))
    image_proj_model.load_state_dict(state_dict["image_proj"], strict=False)
    new_param_weights_sum = torch.sum(torch.stack([torch.sum(p) for p in image_proj_model.parameters()]))

    if ori_param_weights_sum == new_param_weights_sum:
        print(f"Weights of image_proj_model did not change!")
    
    if unet is not None:
        ip_layers = torch.nn.ModuleList(unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"])

    del state_dict


def load_ckpt(image_proj_model, unet, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    image_proj_ckpt = {}

    for key in checkpoint['image_proj'].keys():
        if key.startswith("module."):
            image_proj_ckpt[key.replace("module.", "")] = checkpoint["image_proj"][key]
        else:
            image_proj_ckpt[key] = checkpoint["image_proj"][key]
    del checkpoint['image_proj']

    image_proj_model.load_state_dict(image_proj_ckpt, strict=True)
    unet.load_state_dict(checkpoint['unet_trained'], strict=False)


def load_ckpt_mllm(unet, agent_model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    unet.load_state_dict(checkpoint["unet_trained"], strict=False)
    agent_model.load_state_dict(checkpoint["agent_model"], strict=False)
