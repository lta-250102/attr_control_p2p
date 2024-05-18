import sys
import os
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from attribute_control import EmbeddingDelta
from attribute_control.model import SDXL, SD15
from attribute_control.prompt_utils import get_mask, get_mask_regex

torch.set_float32_matmul_precision('high')

DEVICE = 'cuda:0'
DTYPE = torch.float16

model = SDXL(
    pipeline_type='diffusers.StableDiffusionXLPipeline',
    model_name='stabilityai/stable-diffusion-xl-base-1.0',
    pipe_kwargs={ 'torch_dtype': DTYPE, 'variant': 'fp16', 'use_safetensors': True },
    device=DEVICE
)

delay_relative = 0.20

attrs_40 = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
            'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
            'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
            'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
delta_attrs = ['Bald', 'Young', 'Pale_Skin', 'Heavy_Makeup', 'Smiling', 'Wavy_Hair', 'Chubby', 
               'Narrow_Eyes', 'Big_Nose', 'Big_Lips', 'Bushy_Eyebrows', 'Arched_Eyebrows', 'Pointy_Nose']

def get_delta(path):
    delta = EmbeddingDelta(model.dims)
    state_dict = torch.load(path)
    delta.load_state_dict(state_dict['delta'])
    delta = delta.to(DEVICE)
    return delta

deltas = {
    'Bald': get_delta('./pretrained_deltas/person_bald.pt'),
    'Young': get_delta('./pretrained_deltas/person_age.pt'),
    'Pale_Skin': get_delta('./pretrained_deltas/person_pale.pt'),
    'Heavy_Makeup': get_delta('./pretrained_deltas/person_makeup.pt'),
    'Smiling': get_delta('./pretrained_deltas/person_smile.pt'),
    'Wavy_Hair': get_delta('./pretrained_deltas/person_curly_hair.pt'),
    'Chubby': get_delta('./pretrained_deltas/person_width.pt'),
    'Narrow_Eyes': get_delta('./pretrained_deltas/big_eyes.pt'),
    'Big_Nose': get_delta('./pretrained_deltas/big_nose.pt'),
    'Big_Lips': get_delta('./pretrained_deltas/thin_lips.pt'),
    'Bushy_Eyebrows': get_delta('./pretrained_deltas/bushy_eyebrows.pt'),
    'Pointy_Nose': get_delta('./pretrained_deltas/pointy_nose.pt'),
    'Arched_Eyebrows': get_delta('./pretrained_deltas/arched_eyebrows.pt'),
}

glob_attrs = ['Bald', 'Young', 'Pale_Skin', 'Heavy_Makeup', 'Smiling', 'Wavy_Hair', 'Chubby']
ops = ['Young', 'Big_Lips', 'Narrow_Eyes']

dir_prompt = {
    'Bald': {
        'pos': 'bald',
        'neg': 'bearded'
    },
    'Young': {
        'pos': 'young',
        'neg': 'old'
    },
    'Pale_Skin': {
        'pos': 'pale',
        'neg': 'ruddy'
    },
    'Heavy_Makeup': {
        'pos': 'heavy makeup',
        'neg': 'slightly makeup'
    },
    'Smiling': {
        'pos': 'smiling',
        'neg': 'angry'
    },
    'Wavy_Hair': {
        'pos': 'wavy hair',
        'neg': 'straight hair'
    },
    'Chubby': {
        'pos': 'chubby',
        'neg': 'lean'
    },
}


def get_pattern_target(prompt, attr_name):
    if attr_name == 'Narrow_Eyes':
        return r'\b(eyes)\b'
    elif attr_name in ['Big_Nose', 'Pointy_Nose']:
        return r'\b(nose)\b'
    elif attr_name == 'Big_Lips':
        return r'\b(lips)\b'
    elif attr_name in ['Bushy_Eyebrows', 'Arched_Eyebrows']:
        return r'\b(eyebrows)\b'
    
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

def apply_deltas(attr, emb, delta_names, prompt):
    embs = [emb]
    alphas = {}
    for attr_name in delta_names:
        alpha = attr[attrs_40.index(attr_name)] * 2
        if attr_name in ops:
            alpha = -alpha
        pattern_target = get_pattern_target(prompt, attr_name)
        characterwise_mask = get_mask_regex(prompt, pattern_target)
        embs.append(deltas[attr_name].apply(embs[-1], characterwise_mask, alpha))
        alphas[attr_name] = alpha
    return embs, alphas

# def gen_sample(prompt, delta_name, pattern_target, alphas):
#     seed = random.randint(1, 1000000)
#     characterwise_mask = get_mask_regex(prompt, pattern_target)
#     embs = [model.embed_prompt(prompt)]
#     for alpha in alphas:
#         embs.append(deltas[delta_name].apply(embs[0], characterwise_mask, alpha))
#     imgs = []
#     for emb in embs[1:]:
#         img = model.sample_delayed(
#             embs=[emb],
#             embs_unmodified=[embs[0]],
#             embs_neg=[None],
#             delay_relative=delay_relative,
#             generator=torch.manual_seed(seed),
#             guidance_scale=7.5,
#             num_inference_steps=30,
#         )[0]
#         imgs.append(img)
#     return imgs

def demo(attr, cap, file: str, out_path: str, delta_attr_name):
    try:
        seed = random.randint(1, 1000000)
        prompt = "a portrait photo with high facial detailed of a person with all eyes, nose, eyebrows and lips." + cap.lower()
        embs, alphas = apply_deltas(attr, model.embed_prompt(prompt), delta_attr_name, prompt)
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
        print(e.__traceback__)

# def eva_delay(attr, cap, file: str, out_path: str):
#     try:
#         seed = random.randint(1, 1000000)
#         prompt = "a portrait photo with high facial detailed. " + cap.lower()
#         delta_attr_name = random.choice(delta_attrs)
#         embs, _ = apply_deltas(attr, model.embed_prompt(prompt), [delta_attr_name], prompt)
#         ori_img = model.sample_delayed(
#                   embs=[embs[0]],
#                   embs_unmodified=[embs[0]],
#                   embs_neg=[None],
#                   delay_relative=0,
#                   guidance_scale=7.5,
#                   generator=torch.manual_seed(seed),
#                   num_inference_steps=30,
#         )[0].resize((256, 256))
#         img_delay = model.sample_delayed(
#                 embs=[embs[1]],
#                 embs_unmodified=[embs[0]],
#                 embs_neg=[None],
#                 delay_relative=delay_relative,
#                 guidance_scale=7.5,
#                 generator=torch.manual_seed(seed),
#                 num_inference_steps=30,
#         )[0].resize((256, 256))
#         img = model.sample_delayed(
#                 embs=[embs[1]],
#                 embs_unmodified=[embs[0]],
#                 embs_neg=[None],
#                 delay_relative=0,
#                 guidance_scale=7.5,
#                 generator=torch.manual_seed(seed),
#                 num_inference_steps=30,
#         )[0].resize((256, 256))
#         for sub_dir in ['origin', 'normal', 'delay']:
#             os.makedirs(f'{out_path}{sub_dir}/', exist_ok=True)
#         ori_img.save(f'{out_path}origin/{file}')
#         img_delay.save(f'{out_path}delay/{file}')
#         img.save(f'{out_path}normal/{file}')
#     except Exception as e:
#         print(e)


def main():
    dataset = 'celebA'
    attrs = json.load(open(f'./data/{dataset}/attrs.json'))
    captions = json.load(open(f'./data/{dataset}/captions.json'))
    out_path = f'./output/{dataset}/demo/'
    os.makedirs(out_path, exist_ok=True)
    n = 1000
    i = 0
    for file, attr in attrs.items():
        i += 1
        if i >= n: break
        demo(attr=attr, cap=captions[file], file=file, out_path=out_path, delta_attr_name=delta_attrs)


if __name__ == "__main__":
    main()