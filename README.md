# Eye-MESD

Eye-MESD pretrains a ViT-L/14 on ophthalmic images. The objective combines DINO, iBOT, and KoLeo with a GRPO router that distills three frozen teachers: EyeCLIP, RETFound, and a giant ViT.

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
from eye_mesd import vit_large

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

The released checkpoint matches a 224×224 input. Larger inputs use position-embedding interpolation.

## Training

Image paths go in `data/images.json`, as a list or as a dict keyed by path. Teacher checkpoints and the student initialization are set in [`configs/vitl14.yaml`](configs/vitl14.yaml).

```bash
bash scripts/train.sh
```

The released recipe uses 2 GPUs, batch size 80 per GPU, and 100 epochs.

## License

Apache License 2.0. See [LICENSE](LICENSE). Teacher checkpoints are not included.

## Citation

```bibtex
@misc{eyemesd2026,
  title={Eye-MESD},
  author={Lou, Lewis},
  year={2026},
  url={https://github.com/Lewislou/Eye_mesd}
}
```
