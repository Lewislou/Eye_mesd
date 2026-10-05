# Eye-MESD

Eye-MESD is an ophthalmic vision foundation model. A ViT-L/14 student is pretrained with DINO, iBOT, and KoLeo, while a GRPO router distills three frozen teachers: EyeCLIP, RETFound, and DINOv2-Giant.

The training code is based on [DINOv2](https://github.com/facebookresearch/dinov2).

## Pretrained model

| Model | Architecture | Download |
| --- | --- | --- |
| Eye-MESD | ViT-L/14 | [Eye_mesd_pretrained.pth](https://github.com/Lewislou/Eye_mesd/releases/download/v1.0/Eye_mesd_pretrained.pth) |

SHA256: `28b1630da44f7870a65f36dcbb8255a5ff3f87da78075a40048d8b3e3fb7f7fe`

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Tested with PyTorch 2.1.2 and xFormers 0.0.23.post1.

## Usage

```python
import torch
from dinov2.models.vision_transformer import vit_large

model = vit_large(
    img_size=224,
    patch_size=14,
    init_values=1e-5,
    block_chunks=0,
)
state = torch.load("Eye_mesd_pretrained.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

`pos_embed` matches a 224×224 input. Larger inputs use the model's position-embedding interpolation.

## Training

Put the image list at `data/images.json` (a list of paths, or a dict whose keys are paths) and set the teacher checkpoints in [`configs/vitl14.yaml`](configs/vitl14.yaml):

- `student.pretrained_weights`: DINOv2 ViT-L/14
- `distill.eyeclip_weights`: EyeCLIP
- `distill.retfound_weights`: RETFound
- `distill.giant_weights`: DINOv2 ViT-g/14

```bash
torchrun --nproc_per_node=2 dinov2/train/train.py \
  --config-file configs/vitl14.yaml \
  --output-dir output
```

The released recipe uses 2 GPUs, batch size 80 per GPU, and 100 epochs. See the config for the optimizer, crops, and loss weights.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

DINOv2 portions remain under the copyright of Meta Platforms, Inc. EyeCLIP, RETFound, and DINOv2-Giant weights are not included and stay under their own licenses.

## Citation

```bibtex
@misc{eyemesd2026,
  title={Eye-MESD},
  author={Lou, Lewis},
  year={2026},
  url={https://github.com/Lewislou/Eye_mesd}
}
```
