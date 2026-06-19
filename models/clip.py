import os
import warnings
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from clip_lib.AnomalyCLIP import VisionTransformer
from clip_lib.CLIP import Transformer, LayerNorm
from clip_lib.model_load import _MODELS, _download, _transform
from models.prompt_ensemble import tokenize


# ──────────────────────────────────────────────
# CLIP model definition
# ──────────────────────────────────────────────

class CleanCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 image_resolution: int,
                 vision_layers: Union[tuple, int],
                 vision_width: int,
                 vision_patch_size: int,
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int):
        super().__init__()
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers, output_dim=embed_dim, heads=vision_heads,
                input_resolution=image_resolution, width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution, patch_size=vision_patch_size,
                width=vision_width, layers=vision_layers, heads=vision_heads, output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width, layers=transformer_layers, heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, feature_list=[], ori_patch=False, proj_use=True, DPAM_layer=None, ffn=False):
        return self.visual(image.type(self.dtype), feature_list,
                           ori_patch=ori_patch, proj_use=proj_use, DPAM_layer=DPAM_layer, ffn=ffn)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image, logits_per_image.t()


def build_model_new(name: str, state_dict: dict):
    vit = "visual.proj" in state_dict
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
                  for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CleanCLIP(
        embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]
    model.load_state_dict(state_dict)
    return model.eval()


def load_clean_clip(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
                    jit: bool = False, download_root: str = None):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("./clip_model"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found")

    with open(model_path, 'rb') as opened_file:
        try:
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model_new(name, state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    return model, _transform(model.input_resolution.item())


# ──────────────────────────────────────────────
# Text feature bank
# ──────────────────────────────────────────────

COMMON_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a cropped photo of a {}.",
    "a cropped photo of the {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "the {} in the image.",
    "the {} in the scene.",
    "an image of a {}.",
    "an image of the {}.",
    "a rendering of a {}.",
    "a photo of one {}.",
    "showing the {}.",
    "showing a {}.",
    "itap of a {}.",
]


def encode_and_mean(model, device, core_descriptions, templates):
    all_prompts = [tmpl.format(core) for core in core_descriptions for tmpl in templates]
    tokens = tokenize(all_prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.mean(dim=0)


def compute_text_features(model, device):
    '''Text residual: R_t = t_anom − t_normal.'''
    NOUNS = ["object"]
    NORMAL_STATES = ["", "perfect", "flawless", "clean", "undamaged"]
    ANOMALY_STATES = ["damaged"]

    with torch.no_grad():
        neutral_prompts = [f"{noun}" for noun in NOUNS]
        t_neutral = encode_and_mean(model, device, neutral_prompts, COMMON_TEMPLATES)
        t_neutral = t_neutral / t_neutral.norm(dim=-1, keepdim=True)

        normal_prompts = [f"{state} {noun}" for state in NORMAL_STATES for noun in NOUNS]
        t_normal = encode_and_mean(model, device, normal_prompts, COMMON_TEMPLATES)
        t_normal = t_normal / t_normal.norm(dim=-1, keepdim=True)

        anom_prompts = [f"{state} {noun}" for state in ANOMALY_STATES for noun in NOUNS] + \
                       [f"{noun} with {state}" for state in ANOMALY_STATES for noun in NOUNS]
        t_anom = encode_and_mean(model, device, anom_prompts, COMMON_TEMPLATES)
        t_anom = t_anom / t_anom.norm(dim=-1, keepdim=True)

        r_normal = t_normal - t_neutral
        r_normal = r_normal / r_normal.norm(dim=-1, keepdim=True)

        r_anom = t_anom - t_normal
        r_anom = r_anom / r_anom.norm(dim=-1, keepdim=True)

    return t_neutral, t_normal, t_anom, r_normal, r_anom


class TextFeatureBank:
    '''Pre-computes class-agnostic text features once.
    get_features() returns the same [D] tensors every time.
    '''
    def __init__(self, model, device):
        self.device = device
        print("Pre-computing text features...")
        t_neutral, t_normal, t_anom, r_normal, r_anom = compute_text_features(model, device)
        self.t_neutral = t_neutral.to(device)
        self.t_normal  = t_normal.to(device)
        self.t_anom    = t_anom.to(device)
        self.r_normal  = r_normal.to(device)
        self.r_anom    = r_anom.to(device)
        print(f"Text feature dim: {self.t_neutral.shape}")

    def get_features(self):
        '''Returns (t_neutral, t_normal, t_anom, r_normal, r_anom), each [D].'''
        return (self.t_neutral, self.t_normal, self.t_anom, self.r_normal, self.r_anom)
