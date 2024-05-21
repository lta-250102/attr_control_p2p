import sys
import os
import json
import hydra
import torch
import random
import traceback
import numpy as np
from PIL import Image
from datetime import datetime
from omegaconf import DictConfig
import matplotlib.pyplot as plt
from attribute_control import EmbeddingDelta
from attribute_control.model import SDXL, SD15, ModelBase
from attribute_control.prompt_utils import get_mask, get_mask_regex



attrs_40 = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
            'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
            'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
            'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

def get_delta(path, dims):
    delta = EmbeddingDelta(dims)
    state_dict = torch.load(path)
    delta.load_state_dict(state_dict['delta'])
    delta = delta.to('cuda:0')
    return delta

glob_attrs = ['Bald', 'Young', 'Pale_Skin', 'Heavy_Makeup', 'Smiling', 'Wavy_Hair', 'Chubby']
ops = ['Young', 'Big_Lips', 'Narrow_Eyes']

def get_pattern_target(prompt, attr_name):
    if attr_name == 'Narrow_Eyes':
        pattern_target = r'\b(eyes)\b'
        obj = 'eyes'
    elif attr_name in ['Big_Nose', 'Pointy_Nose']:
        pattern_target = r'\b(nose)\b'
        obj = 'nose'
    elif attr_name == 'Big_Lips':
        pattern_target = r'\b(lips)\b'
        obj = 'lips'
    elif attr_name in ['Bushy_Eyebrows', 'Arched_Eyebrows']:
        pattern_target = r'\b(eyebrows)\b'
        obj = 'eyebrows'

    if attr_name in glob_attrs:
        if 'woman' in prompt:
            pattern_target = r'\b(woman)\b'
            obj = 'woman'
        elif ' man' in prompt:
            pattern_target = r'\b(man)\b'
            obj = 'man'
        elif 'lady' in prompt:
            pattern_target = r'\b(lady)\b'
            obj = 'lady'
        elif 'female' in prompt:
            pattern_target = r'\b(female)\b'
            obj = 'female'
        elif 'guy' in prompt:
            pattern_target = r'\b(guy)\b'
            obj = 'guy'
        elif ' she' in prompt:
            pattern_target = r'\b(she)\b'
            obj = 'she'
        elif ' he ' in prompt:
            pattern_target = r'\b(he)\b'
            obj = 'he'
        elif 'person' in prompt:
            pattern_target = r'\b(person)\b'
            obj = 'person'
        elif 'adult' in prompt:
            pattern_target = r'\b(adult)\b'
            obj = 'adult'
    
    return pattern_target, obj

def gen(caps, model, guidance_scale, num_inference_steps, seeds=None):
    if seeds == None:
        seeds = []
        for i in range(len(caps)):
            seeds.append(random.randint(1, 1000000))
    prompts = ["a portrait photo with high facial detailed of a person with all eyes, nose, eyebrows and lips." + cap.lower() for cap in caps]
    embs = [model.embed_prompt(prompt) for prompt in prompts]
    ori_images = model.sample_gen(embs=embs, embs_neg=[None] * len(embs), guidance_scale=guidance_scale, 
                                  generator=[torch.manual_seed(seed) for seed in seeds], 
                                  num_inference_steps=num_inference_steps)
    return ori_images

def apply_deltas(emb, alphas: dict, prompt, deltas: dict):
    embs = [emb]
    for attr_name, alpha in alphas.items():
        pattern_target, _ = get_pattern_target(prompt, attr_name)
        characterwise_mask = get_mask_regex(prompt, pattern_target)
        embs.append(deltas[attr_name].apply(embs[-1], characterwise_mask, alpha))
    return embs[1:]

def edit(cap, ori_image, model, alphas: dict, deltas: dict, delay_relative, guidance_scale, num_inference_steps, seed=None):
    if seed == None:
        seed = random.randint(1, 1000000)
    prompt = "a portrait photo with high facial detailed of a person with all eyes, nose, eyebrows and lips." + cap.lower()
    emb = model.embed_prompt(prompt)
    embs = apply_deltas(emb, alphas, prompt, deltas)
    imgs = []
    for emb in embs:
        img = model.sample_edit(
            images=[ori_image],
            embs=[emb],
            embs_neg=[None],
            delay_relative=delay_relative,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed),
            num_inference_steps=num_inference_steps,
        )[0]
        imgs.append(img)
    return imgs

def both(cap, model, alphas: dict, deltas: dict, delay_relative, guidance_scale, num_inference_steps, seed=None):
    seed = random.randint(1, 1000000)
    prompt = "a portrait photo with high facial detailed of a person with all eyes, nose, eyebrows and lips." + cap.lower()
    emb = model.embed_prompt(prompt)
    ori_image = model.sample(embs=[emb], embs_neg=[None], guidance_scale=guidance_scale, generator=torch.manual_seed(seed), num_inference_steps=num_inference_steps)[0]
    imgs = [ori_image]
    embs = apply_deltas(emb, alphas, prompt, deltas)
    for emb in embs:
        img = model.sample_edit(
            images=[ori_image],
            embs=[emb],
            embs_neg=[None],
            delay_relative=delay_relative,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed),
            num_inference_steps=num_inference_steps,
        )[0]
        imgs.append(img)
    return imgs


@hydra.main(config_path="../configs", config_name="sample")
@torch.no_grad()
def main(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg)
    mode = cfg.mode
    model: ModelBase = cfg.model
    if mode != 'gen':
        delta_path = cfg.delta_path
        deltas = {
            'Bald': get_delta(f'{delta_path}person_bald.pt', model.dims),
            'Young': get_delta(f'{delta_path}person_age.pt', model.dims),
            'Pale_Skin': get_delta(f'{delta_path}person_pale.pt', model.dims),
            'Heavy_Makeup': get_delta(f'{delta_path}person_makeup.pt', model.dims),
            'Smiling': get_delta(f'{delta_path}person_smile.pt', model.dims),
            'Wavy_Hair': get_delta(f'{delta_path}person_curly_hair.pt', model.dims),
            'Chubby': get_delta(f'{delta_path}person_width.pt', model.dims),
            'Narrow_Eyes': get_delta(f'{delta_path}big_eyes.pt', model.dims),
            'Big_Nose': get_delta(f'{delta_path}big_nose.pt', model.dims),
            'Big_Lips': get_delta(f'{delta_path}thin_lips.pt', model.dims),
            'Bushy_Eyebrows': get_delta(f'{delta_path}bushy_eyebrows.pt', model.dims),
            'Pointy_Nose': get_delta(f'{delta_path}pointy_nose.pt', model.dims),
            'Arched_Eyebrows': get_delta(f'{delta_path}arched_eyebrows.pt', model.dims),
        }

    if mode == 'gen':
        img = gen([cfg.prompt], model, cfg.guidance_scale, cfg.num_inference_steps, [cfg.get('seed', None)])[0]
        os.makedirs(f'{cfg.out_dir}{mode}/', exist_ok=True)
        img_name = cfg.get('img_name', datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
        img.save(f'{cfg.out_dir}{mode}/{img_name}.jpg')
    elif mode == 'edit':
        ori_img = Image.open(cfg.ori_img_path)
        alphas: dict = dict(zip(cfg.deltas, cfg.alphas))
        imgs = edit(cfg.prompt, ori_img, model, alphas, deltas, cfg.delay_relative, cfg.guidance_scale, cfg.num_inference_steps, cfg.get('seed', None))
        img_name = cfg.get('img_name', datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
        os.makedirs(f'{cfg.out_dir}{mode}/{img_name}/', exist_ok=True)
        for i in range(len(imgs)):
            attr_name = list(alphas.keys())[i]
            imgs[i].save(f'{cfg.out_dir}{mode}/{img_name}/{i+1}_{attr_name}_{alphas[attr_name]}.jpg')
    elif mode == 'both':
        alphas: dict = dict(zip(cfg.deltas, cfg.alphas))
        imgs = both(cfg.prompt, model, alphas, deltas, cfg.delay_relative, cfg.guidance_scale, cfg.num_inference_steps, cfg.get('seed', None))
        img_name = cfg.get('img_name', datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
        os.makedirs(f'{cfg.out_dir}{mode}/{img_name}/', exist_ok=True)
        for i in range(len(imgs)):
            attr_name = list(alphas.keys())[i] if i > 0 else 'ori'
            alphas['ori'] = 0
            imgs[i].save(f'{cfg.out_dir}{mode}/{img_name}/{i}_{attr_name}_{alphas[attr_name]}.jpg')
    else: 
        print("Invalid mode!!")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
