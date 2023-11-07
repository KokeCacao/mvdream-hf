---
language: 
  - en
thumbnail: "https://huggingface.co/KokeCacao/mvdream-hf/resolve/main/doc/image_1.png"
tags:
- diffusers
---

# MVDream-HF

<p align="center">
  <img src="https://huggingface.co/KokeCacao/mvdream-hf/resolve/main/doc/image_0.png" height="256">
  <img src="https://huggingface.co/KokeCacao/mvdream-hf/resolve/main/doc/image_1.png" height="256">
  <img src="https://huggingface.co/KokeCacao/mvdream-hf/resolve/main/doc/image_2.png" height="256">
  <img src="https://huggingface.co/KokeCacao/mvdream-hf/resolve/main/doc/image_3.png" height="256">
</p>

A huggingface implementation of MVDream with 4 views, used for quick one-line download. See [huggingface repo](https://huggingface.co/KokeCacao/mvdream-hf/tree/main) that hosts sd-v1.5 version and [huggingface repo](https://huggingface.co/KokeCacao/mvdream-base-hf) for sd-v2.1 version. See [github repo](https://github.com/KokeCacao/mvdream-hf) for convertion code.

Note that the original paper presents the sd-v2.1 version. Images above are generated with sd-v2.1 version.

## Convert Original Weights to Diffusers

Download original MVDream checkpoint through one of the following sources:

```bash
# for sd-v1.5 (recommended for production)
wget https://huggingface.co/MVDream/MVDream/resolve/main/sd-v1.5-4view.pt
wget https://raw.githubusercontent.com/bytedance/MVDream/main/mvdream/configs/sd-v1.yaml

# for sd-v2.1 (recommended for publication)
wget https://huggingface.co/MVDream/MVDream/resolve/main/sd-v2.1-base-4view.pt
wget https://raw.githubusercontent.com/bytedance/MVDream/main/mvdream/configs/sd-v2-base.yaml
```

Hugging Face diffusers weights are converted by script:
```bash
python ./scripts/convert_mvdream_to_diffusers.py --checkpoint_path ./sd-v1.5-4view.pt --dump_path . --original_config_file ./sd-v1.yaml --test
python ./scripts/convert_mvdream_to_diffusers.py --checkpoint_path ./sd-v2.1-base-4view.pt --dump_path . --original_config_file ./sd-v2-base.yaml --test
```
