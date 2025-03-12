
#Conversion functions for 3rd part state-dicts and non-torch native checkpoint formats.

from typing import Union

import torch
import numpy as np

from .model import CLIP
from .transformer import TextTransformer, Transformer

def convert_state_dict(model: CLIP, state_dict):
    """
    if 'image_encoder.model.patch_embed.0.rbr_conv.0.conv.weight' in state_dict:
        # Apple MobileCLIP s1 & s2 state_dicts (s0 and b not currently supported)
        state_dict = convert_mobile_clip_state_dict(model, state_dict)
    if 'image_encoder.model.patch_emb.0.block.conv.weight' in state_dict:
        # convert b model
        state_dict = convert_mobile_clip_state_dict(model, state_dict, fastvit=False)"""
    return state_dict