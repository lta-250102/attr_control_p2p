# Face image generation attributes control

This repository contains an implementation of the paper "Continuous, Subject-Specific Attribute Control in T2I Models by Identifying Semantic Directions".

We present a simple, straight-forward method for enabling fine-grained control over attribute expression in T2I (diffusion) models in a subject-specific manner.
We identify meaningful directions in the tokenwise prompt embedding space that enable modulating single attributes for specific subjects without adapting the T2I model.

![teaser](./docs/static/images/teaser.png)

## 🚀 Usage
### Setup
Just clone the repo and install the requirements via `pip install -r requirements.txt`, then you're ready to go. For usage, see the examples below, everything else that's needed (model checkpoints) will be downloaded automatically.

### Inference
For inference, just start with one of the notebook at [`notebooks`](https://github.com/CompVis/attribute-control/tree/main/notebooks) for a minimal example.

The second option, run:
```shell
python src/gen_demo.py
```

We provide a range of learned deltas for SDXL at [`pretrained_deltas`](https://github.com/lta-250102/attr_control_p2p/tree/master/pretrained_deltas). These can also be used for models such as SD 1.5 or LDM3D by just loading them as usual.

### Creating new Attribute Deltas
When creating deltas for new attributes, start by creating a config for them akin to `configs/prompts/people/age.yaml`. There are multiple entries of base prompts that correspond to the attribute in a neutral, "negative", and "positive" direction. Please make sure to use the same noun for all the prompts per entry and specify it as the `pattern_target`.
You can also specify a list of prefixes that contain various other words that will be added before the main prompt to help obtain more robust deltas. The syntax used finds all sets of words enclosed in braces (e.g., `{young,old}`) and then generates all combinations of words in the braces.

#### Learning-based Method
The best method to obtain deltas is the learning-based method, although it takes substantially longer than the naive method (see below)

To obtain a delta with the naive method, use:
```shell
python src/learn_delta.py
```
This will save the delta at `outputs/learn_delta/people/age/runs/<date>/<time>/checkpoints/delta.pt`, which you can then directly use as shown in the example notebooks.

#### Naive CLIP Difference Method
The simplest method to obtain deltas is the naive CLIP difference-based method. With it, you can obtain a delta in a few seconds on a decent GPU. It is substantially worse than the proper learned method though.

To obtain a delta with the naive method, use (same arguments as for the learning-based method):
```shell
python src/learn_delta_naive_clip.py
```
This will save the delta at `outputs/learn_delta_naive_clip/people/age/runs/<date>/<time>/checkpoints/delta.pt`, which you can then directly use as shown in the example notebooks.
