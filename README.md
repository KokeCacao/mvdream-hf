# MVDream-HF

<p align="center">
  <img src="https://huggingface.co/KokeCacao/mvdream-hf/resolve/main/doc/image_0.png" height="256">
  <img src="https://huggingface.co/KokeCacao/mvdream-hf/resolve/main/doc/image_1.png" height="256">
  <img src="https://huggingface.co/KokeCacao/mvdream-hf/resolve/main/doc/image_2.png" height="256">
  <img src="https://huggingface.co/KokeCacao/mvdream-hf/resolve/main/doc/image_3.png" height="256">
</p>

A huggingface implementation of MVDream with 4 views, used for quick one-line download. See [huggingface repo](https://huggingface.co/KokeCacao/mvdream-hf/tree/main) that hosts sd-v1.5 version and [huggingface repo](https://huggingface.co/KokeCacao/mvdream-base-hf) for sd-v2.1 version. See [github repo](https://github.com/KokeCacao/mvdream-hf) for convertion code.

Note that the original paper presents the sd-v2.1 version. Images above are generated with sd-v2.1 version.

## Quick Start

First install `kokikit` by `pip install "kokikit @ git+https://github.com/KokeCacao/kokikit.git"`.
Then copy paste the following inference code and run it. There should be 4 images generated in the current directory.

```python
import torch

from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from kokikit.models import MultiViewUNetWrapperModel, get_camera
from kokikit.diffusion import predict_noise_mvdream
from typing import List, Union, Optional


class MVDream():

    def __init__(self, mvdream_path: str = "KokeCacao/mvdream-hf", dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cuda")):
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            mvdream_path,
            subfolder='text_encoder',
            device_map={'': 0},
        ) # type: ignore
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(mvdream_path, subfolder='tokenizer', use_fast=False)

        self.unet_mvdream: MultiViewUNetWrapperModel = MultiViewUNetWrapperModel.from_pretrained(mvdream_path, subfolder="unet") # type: ignore
        self.unet_mvdream = self.unet_mvdream.to(device=device)

        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(mvdream_path, subfolder="vae") # type: ignore
        self.vae = self.vae.to(device=device, dtype=dtype) # type: ignore

        self.scheduler: DDIMScheduler = DDIMScheduler.from_pretrained(mvdream_path, subfolder="scheduler", torch_dtype=dtype) # type: ignore
        self.scheduler.betas = self.scheduler.betas.to(dtype=dtype)
        self.scheduler.alphas = self.scheduler.alphas.to(dtype=dtype)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(dtype=dtype)

    def get_camera(self, batch_size: int = 4, device: torch.device = torch.device("cuda")):
        return get_camera(batch_size).to(device=device)

    def get_text_embedding(
            self,
            prompts: Union[List[str], str] = "ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions",
            device: torch.device = torch.device("cuda"),
    ):
        if isinstance(prompts, str):
            return_type = "tensor"
            batch_size = 1
            prompts = [prompts]
        elif isinstance(prompts, list):
            return_type = "list"
            batch_size = len(prompts)
        else:
            raise ValueError(f"Prompt type {type(prompts)} is not supported")

        tokenizer = self.tokenizer
        text_encoder = self.text_encoder
        text_encoder = text_encoder.to(device=device) # type: ignore

        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(prompts, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1:-1])
            print(f"The following part of your input was truncated because CLIP can only handle sequences up to"
                  f" {tokenizer.model_max_length} tokens: {removed_text}")

        prompt_embeds = text_encoder(text_input_ids.to(device),)
        prompt_embeds = prompt_embeds[0]

        if return_type == "tensor":
            return prompt_embeds[0]
        elif return_type == "list":
            return [prompt_embeds[i] for i in range(batch_size)]
        else:
            raise ValueError(f"Return type {return_type} is not supported")

    def get_noise(
            self,
            random_seed: Optional[int] = None,
            image_height: int = 256,
            image_width: int = 256,
            batch_size: int = 1,
            channel: int = 4,
            vae_scale: int = 8,
            scheduler: Optional[SchedulerMixin] = None,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cuda"),
    ):
        init_noise_sigma = 1.0
        if scheduler is not None and hasattr(scheduler, "init_noise_sigma"):
            init_noise_sigma = scheduler.init_noise_sigma # type: ignore

        if random_seed is None:
            return torch.randn(batch_size, channel, image_height // vae_scale, image_width // vae_scale, dtype=dtype, device=device) * init_noise_sigma
        torch.manual_seed(random_seed)
        return torch.randn(batch_size, channel, image_height // vae_scale, image_width // vae_scale, dtype=dtype, device=device) * init_noise_sigma

    def execute(
        self,
        prompt_embeds_pos: Union[List[torch.Tensor], torch.Tensor, List[str]],
        prompt_embeds_neg: Union[List[torch.Tensor], torch.Tensor, List[str]],
        latents_original: torch.Tensor = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        unet: MultiViewUNetWrapperModel = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        scheduler: DDIMScheduler = None, # type: ignore (wait for https://peps.python.org/pep-0671/)
        cfg: float = 7.5,
        vae: Optional[AutoencoderKL] = None,
        diffusion_steps: int = 50,
        n_views: int = 4,
        reconstruction_loss: bool = True,
        cfg_rescale: float = 0.5,
    ) -> torch.Tensor:

        vae = self.vae
        unet = self.unet_mvdream
        scheduler = self.scheduler

        if isinstance(prompt_embeds_pos, list) and isinstance(prompt_embeds_pos[0], str):
            prompt_embeds_pos = self.get_text_embedding(prompts=prompt_embeds_pos) # type: ignore
        if isinstance(prompt_embeds_neg, list) and isinstance(prompt_embeds_neg[0], str):
            prompt_embeds_neg = self.get_text_embedding(prompts=prompt_embeds_neg) # type: ignore

        if latents_original is None:
            latents_original = self.get_noise(batch_size=4, image_height=256, image_width=256)
        camera_embeddings = self.get_camera()

        # correctly format input tensors to [77, 768] x4
        if isinstance(prompt_embeds_pos, torch.Tensor):
            if prompt_embeds_pos.shape[0] == n_views:
                prompt_embeds_pos = prompt_embeds_pos.unbind(0) # type: ignore
            else:
                prompt_embeds_pos = [prompt_embeds_pos] * n_views
        if isinstance(prompt_embeds_neg, torch.Tensor):
            if prompt_embeds_neg.shape[0] == n_views:
                prompt_embeds_neg = prompt_embeds_neg.unbind(0) # type: ignore
            else:
                prompt_embeds_neg = [prompt_embeds_neg] * n_views
        if len(prompt_embeds_pos) == 1 and len(prompt_embeds_neg) == 1:
            prompt_embeds_pos = prompt_embeds_pos * n_views
            prompt_embeds_neg = prompt_embeds_neg * n_views
        if len(prompt_embeds_pos) != n_views or len(prompt_embeds_neg) != n_views:
            raise ValueError(f"prompt_embeds_pos and prompt_embeds_neg must be of length {n_views}")
        if prompt_embeds_pos[0].shape[0] != prompt_embeds_neg[0].shape[0]:
            raise ValueError(f"prompt_embeds_pos and prompt_embeds_neg must have the same batch size")
        if prompt_embeds_pos[0].shape[0] == 1:
            prompt_embeds_pos = [p.squeeze(0) for p in prompt_embeds_pos]
            prompt_embeds_neg = [p.squeeze(0) for p in prompt_embeds_neg]

        device = latents_original.device
        scheduler.set_timesteps(diffusion_steps, device=device)

        with torch.no_grad():
            pbar = tqdm(scheduler.timesteps - 1)
            latents_noised = latents_original
            for step, time in enumerate(pbar):

                latents_noised = latents_original
                noise_pred, noise_pred_x0 = predict_noise_mvdream(
                    unet_mvdream=unet,
                    latents_noised=latents_noised, # [4, 4, 32, 32]
                    text_embeddings_conditional=prompt_embeds_pos, # [77, 768] x4 # type: ignore
                    text_embeddings_unconditional=prompt_embeds_neg, # [77, 768] x4 # type: ignore
                    camera_embeddings=camera_embeddings, # [4, 16]
                    cfg=cfg,
                    t=time,
                    scheduler=scheduler,
                    n_views=n_views,
                    reconstruction_loss=reconstruction_loss,
                    cfg_rescale=cfg_rescale,
                )
                latents_noised = scheduler.step(noise_pred, time, latents_noised).prev_sample # type: ignore
                latents_original = latents_noised

                if vae is not None and (step % 10 == 0 or step == len(pbar) - 1):
                    _ = latents_noised if noise_pred_x0 is None else noise_pred_x0 # use x0 instead of actual image if available
                    image_batch = vae.decode(1 / vae.config['scaling_factor'] * _.clone().detach()).sample # [B, C, H, W] # type: ignore
                    image_batch = (image_batch / 2 + 0.5).clamp(0, 1) * 255
                    image_batch = image_batch.permute(0, 2, 3, 1).detach().cpu() # [B, C, H, W] -> [B, H, W, C]
        result = self.decode(latents_noised, to_image=True)
        return result

    def decode(self, latents: torch.Tensor, to_image: bool):
        image_batch = self.vae.decode(1 / self.vae.config['scaling_factor'] * latents.clone().detach()).sample # [B, C, H, W] # type: ignore
        image_batch = image_batch.nan_to_num()
        image_batch = image_batch.clamp(-1, 1)

        if to_image:
            image_batch = (image_batch / 2 + 0.5).clamp(0, 1) * 255
            image_batch = image_batch.permute(0, 2, 3, 1).detach().cpu() # [B, C, H, W] -> [B, H, W, C]
        return image_batch


if __name__ == '__main__':
    import argparse
    import numpy as np
    import PIL.Image
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_pos", type=str, default="a minecraft house, front")
    parser.add_argument("--prompt_neg", type=str, default="ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions")
    args = parser.parse_args()

    mvdream = MVDream()
    images = mvdream.execute(prompt_embeds_pos=[args.prompt_pos], prompt_embeds_neg=[args.prompt_neg])
    for i, image in enumerate(images):
        image = PIL.Image.fromarray(image.numpy().astype(np.uint8))
        image.save(f"mvdream_{i}.png")
```

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
