"""
Resolve the local Hugging Face cache paths for every checkpoint DiffSensei needs,
so training can run fully offline from already-downloaded models (no `checkpoints/`
folder required).

Models used (all expected to be present in the HF cache):
  - stabilityai/stable-diffusion-xl-base-1.0   (SDXL base: vae/unet/text_encoders/tokenizers/scheduler)
  - h94/IP-Adapter                             (image_encoder dir + ip-adapter-plus_sdxl_vit-h.safetensors)
  - ragavsachdeva/magi                         (Magi: crop-embedding encoder, used via trust_remote_code)

Usage (standalone): prints resolved paths as JSON.
    python -m diffsensei.cli.train.hf_paths
"""
import os
import json

# Use the local cache only; never hit the network during training setup.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from huggingface_hub import snapshot_download, hf_hub_download

SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_PLUS_FILE = "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
MAGI_REPO = "ragavsachdeva/magi"


def sdxl_path():
    """
    Local SDXL base 1.0 (diffusers layout).

    Prefer the assembled `checkpoints/sdxl-base-1.0` (built by prepare_sdxl.py with
    real weights); fall back to the HF cache snapshot if that is complete.
    """
    local = os.path.join("checkpoints", "sdxl-base-1.0")
    if os.path.exists(os.path.join(local, "unet", "diffusion_pytorch_model.safetensors")):
        return os.path.abspath(local)
    return snapshot_download(SDXL_REPO, local_files_only=True)


def ip_adapter_image_encoder_path():
    """Local dir of the IP-Adapter CLIP-ViT-H image encoder."""
    snap = snapshot_download(
        IP_ADAPTER_REPO, local_files_only=True, allow_patterns=["models/image_encoder/*"]
    )
    return os.path.join(snap, "models", "image_encoder")


def ip_adapter_plus_file():
    """Local path of the ip-adapter-plus_sdxl_vit-h.safetensors weights."""
    return hf_hub_download(IP_ADAPTER_REPO, IP_ADAPTER_PLUS_FILE, local_files_only=True)


def magi_path():
    """Repo id for Magi (loaded with trust_remote_code; served from cache offline)."""
    # Verify it is cached, then return the repo id (AutoModel resolves it from cache).
    snapshot_download(MAGI_REPO, local_files_only=True)
    return MAGI_REPO


def resolve_all():
    return {
        "sdxl": sdxl_path(),
        "ip_adapter_image_encoder": ip_adapter_image_encoder_path(),
        "ip_adapter_plus_file": ip_adapter_plus_file(),
        "magi": magi_path(),
    }


if __name__ == "__main__":
    print(json.dumps(resolve_all(), indent=2, ensure_ascii=False))
