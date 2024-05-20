import sys
import os
import json
import hydra
import torch
import random
import traceback
import numpy as np
import matplotlib.pyplot as plt
from attribute_control import EmbeddingDelta
from attribute_control.model import SDXL, SD15
from attribute_control.model import ModelBase
from attribute_control.prompt_utils import get_mask, get_mask_regex

torch.set_float32_matmul_precision('high')

attrs_40 = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
            'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
            'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
            'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
delta_attrs = ['Bald', 'Young', 'Pale_Skin', 'Heavy_Makeup', 'Smiling', 'Wavy_Hair', 'Chubby', 
               'Narrow_Eyes', 'Big_Nose', 'Big_Lips', 'Bushy_Eyebrows', 'Arched_Eyebrows', 'Pointy_Nose']

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

def apply_deltas(attr, emb, delta_names, prompt, deltas):
    embs = [emb]
    alphas = {}
    for attr_name in delta_names:
        alpha = attr[attrs_40.index(attr_name)] * 2
        if attr_name in ops:
            alpha = -alpha
        pattern_target, _ = get_pattern_target(prompt, attr_name)
        characterwise_mask = get_mask_regex(prompt, pattern_target)
        embs.append(deltas[attr_name].apply(embs[-1], characterwise_mask, alpha))
        alphas[attr_name] = alpha
    return embs, alphas

def demo_delay(attr, cap, file: str, out_path: str, delta_attr_name, model, deltas, delay_relative):
    try:
        seed = random.randint(1, 1000000)
        prompt = "a portrait photo with high facial detailed of a person with all eyes, nose, eyebrows and lips." + cap.lower()
        embs, alphas = apply_deltas(attr, model.embed_prompt(prompt), delta_attr_name, prompt, deltas)
        imgs = []
        for emb in embs:
            img = model.sample_delayed(
                embs=[emb],
                embs_unmodified=[embs[0]],
                embs_neg=[None],
                delay_relative=delay_relative,
                guidance_scale=7.5,
                generator=torch.manual_seed(seed),
                num_inference_steps=30,
            )[0]
            imgs.append(img)
        os.makedirs(f'{out_path}{file.replace(".jpg", "")}/', exist_ok=True)
        for i in range(len(imgs)):
            imgs[i].save(f'{out_path}{file.replace(".jpg", "")}/{i}_{delta_attrs[i-1] if i > 0 else "ori"}_{alphas[delta_attrs[i-1]] if i > 0 else 0}.jpg')
    except Exception as e:
        traceback.print_exc()

def demo(attr, cap, file: str, out_path: str, delta_attr_name, model, deltas, delay_relative):
    try:
        seed = random.randint(1, 1000000)
        prompt = "a portrait photo with high facial detailed of a person with all eyes, nose, eyebrows and lips." + cap.lower()
        embs, alphas = apply_deltas(attr, model.embed_prompt(prompt), delta_attr_name, prompt, deltas)
        ori_image = model.sample(embs=[embs[0]], embs_neg=[None], guidance_scale=7.5, generator=torch.manual_seed(seed), num_inference_steps=30,)[0]
        imgs = [ori_image]
        for emb in embs[1:]:
            img = model.sample_edit(
                images=[ori_image],
                embs=[emb],
                embs_neg=[None],
                delay_relative=delay_relative,
                guidance_scale=7.5,
                generator=torch.manual_seed(seed),
                num_inference_steps=30,
            )[0]
            imgs.append(img)
        os.makedirs(f'{out_path}{file.replace(".jpg", "")}/', exist_ok=True)
        for i in range(len(imgs)):
            imgs[i].save(f'{out_path}{file.replace(".jpg", "")}/{i}_{delta_attrs[i-1] if i > 0 else "ori"}_{alphas[delta_attrs[i-1]] if i > 0 else 0}.jpg')
    except Exception as e:
        traceback.print_exc()

@hydra.main(config_path="configs", config_name="gen_demo")
@torch.no_grad()
def main():
    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model
    delay_relative = cfg.delay_relative
    deltas = {
        'Bald': get_delta('./pretrained_deltas/person_bald.pt', model.dims),
        'Young': get_delta('./pretrained_deltas/person_age.pt', model.dims),
        'Pale_Skin': get_delta('./pretrained_deltas/person_pale.pt', model.dims),
        'Heavy_Makeup': get_delta('./pretrained_deltas/person_makeup.pt', model.dims),
        'Smiling': get_delta('./pretrained_deltas/person_smile.pt', model.dims),
        'Wavy_Hair': get_delta('./pretrained_deltas/person_curly_hair.pt', model.dims),
        'Chubby': get_delta('./pretrained_deltas/person_width.pt', model.dims),
        'Narrow_Eyes': get_delta('./pretrained_deltas/big_eyes.pt', model.dims),
        'Big_Nose': get_delta('./pretrained_deltas/big_nose.pt', model.dims),
        'Big_Lips': get_delta('./pretrained_deltas/thin_lips.pt', model.dims),
        'Bushy_Eyebrows': get_delta('./pretrained_deltas/bushy_eyebrows.pt', model.dims),
        'Pointy_Nose': get_delta('./pretrained_deltas/pointy_nose.pt', model.dims),
        'Arched_Eyebrows': get_delta('./pretrained_deltas/arched_eyebrows.pt', model.dims),
    }

    dataset = cfg.dataset
    attrs = json.load(open(cfg.attrs_path))
    captions = json.load(open(cfg.caps_path))
    out_path = f'{cfg.out_dir}{dataset}/demo/'
    os.makedirs(out_path, exist_ok=True)
    n = cfg.n
    i = 0
    for file, attr in attrs.items():
        i += 1
        if i >= n: break
        demo(attr=attr, cap=captions[file], file=file, out_path=out_path, delta_attr_name=delta_attrs, 
             model=model, deltas=deltas, delay_relative=delay_relative)


if __name__ == "__main__":
    main()