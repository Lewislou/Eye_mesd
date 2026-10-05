# Eye-MESD

ViT-L/14 pretrained on ophthalmic images.

Download the weights: [Eye_mesd_pretrained.pth](https://github.com/Lewislou/Eye_mesd/releases/download/v1.0/Eye_mesd_pretrained.pth)

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

```python
import torch
from eye_mesd import vit_large

model = vit_large(img_size=224, patch_size=14, init_values=1e-5, block_chunks=0)
state = torch.load("Eye_mesd_pretrained.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
