# Convert original weights to diffusers

Download original MVDream checkpoint under `ckpts` through one of the following sources:

```bash
# for sd-v1.5 (recommended for production)
wget https://huggingface.co/MVDream/MVDream/resolve/main/sd-v1.5-4view.pt .
wget https://raw.githubusercontent.com/bytedance/MVDream/main/mvdream/configs/sd-v1.yaml .

# for sd-v2.1 (recommended for publication)
wget https://huggingface.co/MVDream/MVDream/resolve/main/sd-v2.1-base-4view.pt .
wget https://raw.githubusercontent.com/bytedance/MVDream/main/mvdream/configs/sd-v2-base.yaml .
```

Hugging Face diffusers weights are converted by script:
```bash
mkdir converted
python ./scripts/convert_mvdream_to_diffusers.py --checkpoint_path ./sd-v1.5-4view.pt --dump_path ./converted --original_config_file ./sd-v1.yaml
```
