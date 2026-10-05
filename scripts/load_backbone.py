"""Load the released Eye-MESD ViT-L/14 backbone.

The checkpoint is the EMA teacher backbone at iteration 479999, stored as a
flat state dict (no optimizer, no FSDP shard prefixes). Build the network
with block_chunks=0 so parameter names match blocks.{i}.* .
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dinov2.models.vision_transformer import vit_large


def build_backbone(img_size: int = 224):
    return vit_large(
        img_size=img_size,
        patch_size=14,
        init_values=1.0e-5,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
    )


def load_state_dict(path: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "teacher" in state and isinstance(state["teacher"], dict):
        state = state["teacher"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    cleaned = {}
    for key, value in state.items():
        name = key
        for prefix in ("teacher.backbone.", "student.backbone.", "backbone.", "module."):
            if name.startswith(prefix):
                name = name[len(prefix):]
        cleaned[name] = value
    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/Eye_mesd_pretrained.pth",
    )
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    model = build_backbone(args.img_size)
    state = load_state_dict(args.checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_param = sum(v.numel() for v in model.state_dict().values())
    print(f"loaded keys: {len(state)}")
    print(f"parameters: {n_param}")
    print(f"embed_dim: {model.embed_dim}  depth: {model.n_blocks}  patch: {model.patch_size}")
    print(f"pos_embed: {tuple(model.pos_embed.shape)}")
    print(f"missing: {len(missing)}  unexpected: {len(unexpected)}")
    if missing:
        print("missing sample:", missing[:8])
    if unexpected:
        print("unexpected sample:", unexpected[:8])
    if missing or unexpected:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
