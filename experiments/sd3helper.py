""" Module to help with Stable Diffusion 3 representation extraction from the denoising transformer blocks for existing images.

Usage:
```python
    from sd3helper import SD3
    sd = SD3()
    img = sd('A photo of a cat holding a sign that says "Hello World!"')
    representation = sd.get_repr(img)
    print(representation.shape)
```

Author: Jonas Loos
"""

import torch
from diffusers import StableDiffusion3Pipeline
from contextlib import ExitStack


class SD3:
    ''' Helper class for Stable Diffusion 3 representation extraction from the denoising transformer blocks. '''
    def __init__(self, device = 'cuda'):
        self.device = device
        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16).to(self.device)

    def __call__(self, prompt: str, **kwargs):
        ''' Generate an image from a prompt.

        Args:
            prompt: The prompt to use for generating the image.
            **kwargs: Additional arguments to pass to the pipeline.
        '''
        return self.pipe(prompt=prompt, **kwargs).images[0]
    
    @torch.no_grad()
    def encode_latents(self, img):
        ''' Encode an image to latents.

        Args:
            img: The (PIL) image to encode.
        '''
        img_tensor = self.pipe.image_processor.preprocess(img).to(self.device)
        return (self.pipe.vae.encode(img_tensor.to(dtype=next(self.pipe.vae.modules()).dtype)).latent_dist.sample().to(dtype=self.pipe.dtype) - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents):
        ''' Decode latents to an image.

        Args:
            latents: The latents to decode.
        '''
        tmp = self.pipe.vae.decode(((latents) / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor, return_dict=False)[0]
        return self.pipe.image_processor.postprocess(tmp)[0]

    @torch.no_grad()
    def get_repr(self, img, step: int = 950, prompt: str = ''):
        ''' Get the SD3 intermediate representations from the denoising transformer blocks for a given image.

        Args:
            img: The (PIL) image to get representations for.
            step: The timestep to get representations for. Has to be between 0 (more noise) and 999 (less noise).
            prompt: The prompt to use for encoding.
        '''
        # prepare timestep and prompt
        self.pipe.scheduler.set_timesteps(1000, device=self.device)
        timestep = self.pipe.scheduler.timesteps[step]
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(prompt=prompt, prompt_2=None, prompt_3=None)

        # encode image and add noise
        latents = self.encode_latents(img)
        latents = self.pipe.scheduler.scale_noise(latents, timestep=timestep, noise=torch.randn_like(latents))

        # extract representations
        reprs = []
        with ExitStack() as stack, torch.no_grad():
            for x in self.pipe.transformer.transformer_blocks:
                stack.enter_context(x.register_forward_hook(lambda _, input, output: reprs.append(output[1])))
            self.pipe.transformer(hidden_states=latents, timestep=torch.tensor(timestep.expand(latents.shape[0]), device=self.device), encoder_hidden_states=prompt_embeds, pooled_projections=pooled_prompt_embeds)

        # fix representation shape
        num_tokens = reprs[0].shape[1]
        potential_repr_shapes = [(i, num_tokens//i) for i in range(1, num_tokens) if num_tokens % i == 0]
        repr_shape = min(potential_repr_shapes, key=lambda x: abs(x[1]/x[0] - img.size[0] / img.size[1]))
        reprs = torch.stack(reprs).reshape(len(reprs), *repr_shape, 1536)

        return reprs
