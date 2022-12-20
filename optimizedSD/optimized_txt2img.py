import argparse, os, re, io
import torch
import numpy as np
from random import randint
from dataclasses import dataclass
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from optimizedSD.optimUtils import split_weighted_subprompts, logger
from transformers import logging
# from samplers import CompVisDenoiser
logging.set_verbosity_error()

@dataclass
class Opt:
    prompt: str
    ddim_steps: int
    fixed_code: bool
    ddim_eta: float
    n_iter: int
    H: int
    W: int
    C: int
    f: int
    n_samples: int
    n_rows: int
    scale: float
    device: str
    from_file: str
    seed: int
    unet_bs: int
    turbo: bool
    precision: str
    sampler: str
    ckpt: str

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

async def generateImages(data: dict, sio, sid):
    config = "../stable-diffusion/optimizedSD/v1-inference.yaml"
    ckpt = "../stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt"

    opt = Opt(
        prompt=data.get("prompt", "Bear riding a tricycle"),
        ddim_steps=data.get("steps", 5),
        fixed_code=None,
        ddim_eta=0.0,
        n_iter=data.get("iterations", 1),
        H=data.get("height", 512),
        W=data.get("width", 512),
        C=4,
        f=8,
        n_samples=data.get("samples", 9),
        n_rows=0,
        device="cuda",
        scale=data.get("scale", 7.5),
        from_file=None,
        seed=data.get("seed", None),
        unet_bs=1,
        turbo=True,
        precision="autocast",
        sampler="plms",
        ckpt=data.get("ckpt", ckpt),
    )

    if opt.seed == None:
        opt.seed = randint(0, 1000000)
    seed_everything(opt.seed)


    sd = load_model_from_config(f"{opt.ckpt}")
    li, lo = [], []
    for key, value in sd.items():
        sp = key.split(".")
        if (sp[0]) == "model":
            if "input_blocks" in sp:
                li.append(key)
            elif "middle_block" in sp:
                li.append(key)
            elif "time_embed" in sp:
                li.append(key)
            else:
                lo.append(key)
    for key in li:
        sd["model1." + key[6:]] = sd.pop(key)
    for key in lo:
        sd["model2." + key[6:]] = sd.pop(key)

    config = OmegaConf.load(f"{config}")

    model = instantiate_from_config(config.modelUNet)
    _, _ = model.load_state_dict(sd, strict=False)
    model.eval()
    model.unet_bs = opt.unet_bs
    model.cdevice = opt.device
    model.turbo = opt.turbo
    
    modelCS = instantiate_from_config(config.modelCondStage)
    _, _ = modelCS.load_state_dict(sd, strict=False)
    modelCS.eval()
    modelCS.cond_stage_model.device = opt.device
    
    modelFS = instantiate_from_config(config.modelFirstStage)
    _, _ = modelFS.load_state_dict(sd, strict=False)
    modelFS.eval()
    del sd

    if opt.device != "cpu" and opt.precision == "autocast":
        model.half()
        modelCS.half()

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=opt.device)


    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        assert opt.prompt is not None
        prompt = opt.prompt
        print(f"Using prompt: {prompt}")
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            text = f.read()
            print(f"Using prompt: {text.strip()}")
            data = text.splitlines()
            data = batch_size * list(data)
            data = list(chunk(sorted(data), batch_size))


    if opt.precision == "autocast" and opt.device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext
    
    seeds = ""
    with torch.no_grad():
    
        all_samples = list()
        for n in trange(opt.n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    modelCS.to(opt.device)
                    uc = None
                    if opt.scale != 1.0:
                        uc = modelCS.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
    
                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = modelCS.get_learned_conditioning(prompts)

                    shape = [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f]

                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelCS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    async def callback (i):
                        await sio.emit("status", data={"step": i + 1, "total": opt.ddim_steps}, room=sid)
                        await sio.sleep(0)
                        

                    samples_ddim = await model.sample(
                        S=opt.ddim_steps,
                        conditioning=c,
                        seed=opt.seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=opt.scale,
                        unconditional_conditioning=uc,
                        eta=opt.ddim_eta,
                        x_T=start_code,
                        sampler = opt.sampler,
                        callback=callback
                    )

                    modelFS.to(opt.device)

                    print(samples_ddim.shape)
                    print("saving images")
                    images = []
                    for i in range(batch_size):
    
                        x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_sample = 255. * rearrange(x_sample[0].cpu().numpy(), 'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        image_bytes = io.BytesIO()
                        image.save(image_bytes, format='PNG')
                        images.append({
                            "seed": opt.seed,
                            "steps": opt.ddim_steps,
                            "data": image_bytes.getvalue()
                        })
                        opt.seed+=1


                    if opt.device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        modelFS.to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)
                    del samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    return images