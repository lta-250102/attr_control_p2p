from typing import List, Dict
from itertools import product
from pathlib import Path
from tqdm.auto import tqdm
import hydra
import torch
import random
from omegaconf import DictConfig
from PIL import Image
from attribute_control import EmbeddingDelta
from attribute_control.model import ModelBase
from attribute_control.prompt_utils import get_mask_regex


@hydra.main(config_path="configs", config_name="learn_delta_naive_clip")
@torch.no_grad()
def main(cfg: DictConfig):
    print(cfg)
    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model
    prompts: List[Dict[str, str]] = cfg.prompts
    prefixes: List[str] = cfg.prefixes
    targets = []
    ex_prompts = []

    # Compute the deltas for each prompt pair
    deltas = []
    for prefix, d_prompt in tqdm(product(prefixes, prompts), total=(len(prefixes) * len(prompts))):
        targets.append(d_prompt['prompt_target'])
        ex_prompts.append(f'{prefix} {d_prompt["prompt_target"]}')
        target_token_embs = { }
        for direction in ['prompt_positive', 'prompt_negative']:
            emb = model.embed_prompt(f'{prefix} {d_prompt[direction]}')
            tokenwise_masks = emb.get_tokenwise_mask(get_mask_regex(emb.prompt, d_prompt[direction]))
            # Retrieve last token that is part of the target word
            target_token_embs[direction] = { encoder: embedding[len(tokenwise_masks[encoder]) - 1 - tokenwise_masks[encoder][::-1].index(True)] for encoder, embedding in emb.tokenwise_embeddings.items() }
        # Eq. 2
        deltas.append({ encoder: target_token_embs['prompt_positive'][encoder] - target_token_embs['prompt_negative'][encoder] for encoder in emb.tokenwise_embeddings })

    # Compute the average delta
    delta = EmbeddingDelta(model.dims)
    for encoder in delta.tokenwise_delta:
        delta.tokenwise_delta[encoder].copy_(torch.stack([d[encoder] for d in deltas]).mean(dim=0))

    output_dir = Path('./checkpoints')
    checkpoint_path = output_dir / f'delta.pt'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'delta': delta.cpu().state_dict(),
    }, checkpoint_path)

    i = random.randint(0, len(targets)-1)
    delta = delta.to('cuda')
    prompt = "4k, crime, portrait, detail, a portrait photo with high facial detailed of a person. " + ex_prompts[i].lower() 
    pattern_target = fr'\b({targets[i]})\b'
    characterwise_mask = get_mask_regex(prompt, pattern_target)
    emb = model.embed_prompt(prompt) 
    imgs = []
    for alpha in [0, 2]:
        img: Image.Image = model.sample_delayed(
            embs=[delta.apply(emb, characterwise_mask, alpha)],
            embs_unmodified=[emb],
            embs_neg=[None],
            delay_relative=0.2,
            generator=torch.manual_seed(42),
            guidance_scale=7.5,
            num_inference_steps=30,
        )[0]
        imgs.append(img)
        img.save(Path('./demo' / f'{alpha}_{cfg.tag}'))

if __name__ == "__main__":
    main()
