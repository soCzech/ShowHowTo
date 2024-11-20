import os
import torch
import argparse
import numpy as np

from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange, repeat

from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.utils import instantiate_from_config


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


@torch.no_grad()
def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.,
                           unconditional_guidance_scale=1.0, **kwargs):
    ddim_sampler = DDIMSampler(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([1.] * batch_size, dtype=torch.long, device=model.device)

    img = videos[:, :, 0]  # bchw
    img_emb = model.embedder(img)  # blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    (_B, _, _T, _, _), tB = videos.shape, cond_emb.shape[0]
    if tB != _B:  # in case we have multiple prompts for a single video
        assert _B * _T == tB, f"{_B} * {_T} != {tB}"
        img_emb = img_emb.repeat_interleave(repeats=_T, dim=0)
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}

    z = get_latent_z(model, videos)  # b c t h w
    img_cat_cond = z[:, :, :1, :, :]
    img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
    cond["c_concat"] = [img_cat_cond]  # b c 1 h w
    
    if unconditional_guidance_scale != 1.0:
        uc_emb = model.get_learned_conditioning(batch_size * [""])

        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {
            "c_crossattn": [torch.cat([uc_emb,uc_img_emb], dim=1)],
            "c_concat": [img_cat_cond]
        }
    else:
        uc = None

    z0 = None
    cond_mask = None
    x_T = None
    timesteps = None

    batch_variants = []
    for _ in range(n_samples):
        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=batch_size,
                                             shape=noise_shape[1:],
                                             verbose=True,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             mask=cond_mask,
                                             x0=cond_z0,
                                             fs=fs,
                                             x_T=x_T,
                                             timesteps=timesteps,
                                             **kwargs)
        # reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)

    # variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def main(args):
    config = OmegaConf.load("./configs/inference_256_v1.1.yaml")["model"]
    model = instantiate_from_config(config)
    model = model.cuda()

    state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with open(args.prompt_file, "r") as f:
        dataset = []
        for line in f.readlines():
            if line.strip() == "" or line.startswith("#"):
                continue
            image_path, prompts = line.split(args.delimiter, 1)
            prompts = [p.strip().strip('"') for p in prompts.split(args.delimiter)]

            dataset.append((image_path.strip(), prompts))

    for idx, (image_path, prompts) in enumerate(dataset):
        print(f"Processing {idx + 1}/{len(dataset)}: {image_path}", flush=True)
        assert os.path.exists(image_path), f"Image not found: {image_path}"

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if w > h:
            img = img.crop(((w - h) // 2, 0, h + (w - h) // 2, h))
        elif h > w:
            img = img.crop((0, (h - w) // 2, w, w + (h - w) // 2))
        img = img.resize((256, 256))

        n_frames = len(prompts)
        noise_shape = [1, 4, n_frames, 32, 32]  # B, C, T, H, W

        torch_img = torch.from_numpy(np.array(img).copy()).permute(2, 0, 1).float().div_(255 / 2).sub_(1)
        torch_img = torch_img.unsqueeze(1)  # add temp dimension: 3, 1, 256, 256
        torch_img = torch_img.unsqueeze(0)  # add batch size: 1, 3, 1, 256, 256
        torch_img = repeat(torch_img, 'b c t h w -> b c (repeat t) h w', repeat=n_frames)

        torch_img = torch_img.cuda()
        samples = image_guided_synthesis(
            model, prompts, torch_img, noise_shape, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
            unconditional_guidance_scale=args.unconditional_guidance_scale)

        # n_samples=1, B=1, C, T, h, w
        samples = samples.squeeze(0).squeeze(0)
        samples = samples.clamp_(-1, 1).add_(1.).mul_(255 / 2)
        samples = samples.to(torch.uint8).permute(1, 2, 3, 0)
        samples = samples.cpu().numpy()

        output_image = Image.fromarray(np.concatenate(samples, axis=1))
        os.makedirs(args.output_dir, exist_ok=True)
        output_image.save(os.path.join(args.output_dir, f"{idx:05d}.jpg"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True, help="text file with image paths and prompts")
    parser.add_argument("--delimiter", type=str, required=True, help="delimiter for image paths and prompts")
    parser.add_argument("--output_dir", type=str, default="output", help="output directory for generated images")

    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM")
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5, help="prompt classifier-free guidance")
    args = parser.parse_args()

    main(args)
