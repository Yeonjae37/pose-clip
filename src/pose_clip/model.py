""" CLIP Model

Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial

from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer,\
    text_global_pool
from .utils import to_2tuple


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12 # 모델의 층 개수
    width: int = 768 # 모델의 채널 수 
    head_width: int = 64 # 멀티 헤드 어텐션에서 사용되는 각 헤드의 너비
    mlp_ratio: float = 4.0 # MLP 레이어에서 확장되는 비율
    patch_size: int = 16 # ViT에서 사용하는 패치의 크기
    image_size: Union[Tuple[int, int], int] = 224 # 입력 이미지 크기

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = False
    pad_id: int = 0
    no_causal_mask: bool = False  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = False
    proj_type: str = 'linear'  # control final text projection, 'none' forces no projection
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models


def get_cast_dtype(precision: str): # 연산 데이터 타입
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str): # 입력 데이터 타입
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower( # 이미지를 벡터로 변환
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)): # vision_cfg.layers가 리스트나 튜플이면 ResNet 모델 사용
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet( # ResNet 구조를 변형한 ModifiedResNet 모델 사용
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else: # ViT 모델 사용
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm  # 부동소수점(FP16, BF16)을 사용할 경우 LayerNormFp32 사용
        if vision_cfg.norm_kwargs: # 추가적인 정규화 설정이 있으면 적용
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None: # 활성화 함수도 추가적인 설정이 있으면 적용
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = HFTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj_type=text_cfg.hf_proj_type,
            pooler_type=text_cfg.hf_pooler_type,
            pretrained=text_cfg.hf_model_pretrained,
            output_tokens=text_cfg.output_tokens,
        )
    else: # OpenCLIP의 TextTransformer 모델 사용
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if text_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
        if text_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **text_cfg.act_kwargs)

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            mlp_ratio=text_cfg.mlp_ratio,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            embed_cls=text_cfg.embed_cls,
            no_causal_mask=text_cfg.no_causal_mask,
            pad_id=text_cfg.pad_id,
            pool_type=text_cfg.pool_type,
            proj_type=text_cfg.proj_type,
            proj_bias=text_cfg.proj_bias,
            output_tokens=text_cfg.output_tokens,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
    return text


class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.transformer = text.transformer # 텍스트 인코더 저장
        self.context_length = text.context_length # 문맥 길이
        self.vocab_size = text.vocab_size # 단어 사전 크기
        self.token_embedding = text.token_embedding # 토큰 임베딩 레이어
        self.positional_embedding = text.positional_embedding # 위치 임베딩 레이어
        self.ln_final = text.ln_final # 마지막 LayerNorm 
        self.text_projection = text.text_projection # 텍스트 임베딩을 embed_dim에 맞게 변환하는 프로젝션 레이어
        self.text_pool_type = text.pool_type # 텍스트 풀링 방식 (argmax, cls, mean)
        self.register_buffer('attn_mask', text.attn_mask, persistent=False) # 어텐션 마스크 버퍼 등록

        lshape = [1] if nonscalar_logit_scale else [] 
        '''
        lshape = [1] # 1D 텐서 형태
        lshape = [] # 스칼라 값
        '''

        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale) 
        if init_logit_bias is not None: 
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True): # Gradient Checkpointing 활성화/비활성화
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self): # weight decay가 적용되지 않아야 할 파라미터들 반환
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding'} # 위치 임베딩은 weight decay가 적용되지 않음
        if hasattr(self.visual, 'no_weight_decay'): # vision 모델도 no_weight_decay를 정의하고 있으면 해당 파라미터들도 추가
            for n in self.visual.no_weight_decay():
                no_wd.add('visual.' + n)
        return no_wd

    def encode_image(self, image, normalize: bool = False): # 이미지 특징 벡터를 추출
        features = self.visual(image) # self.visual은 _build_vision_tower()에서 생성. Vision 모델에서 특징 추출
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False): # 텍스트 특징 벡터를 추출
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        x, _ = text_global_pool(x, text, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return F.normalize(x, dim=-1) if normalize else x

    def get_logits(self, image, text): # 이미지와 텍스트 간의 유사도 계산
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T # 텍스트 로짓은 이미지 로짓의 전치
        # text_logits[i, j] = image_logits[j, i]
        return image_logits, text_logits

    def forward( # 모델이 이미지나 텍스트를 입력받았을 때 실행되는 기본 동작 정의
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict: 
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16): # 모델의 가중치를 저정밀도 데이터 타입으로 변환
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l): # 각 레이어의 가중치를 변환
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)): # 1D 컨볼루션, 2D 컨볼루션, 선형 레이어인 경우
            l.weight.data = l.weight.data.to(dtype) # 가중치 데이터를 저정밀도 데이터 타입으로 변환
            if l.bias is not None: # 편향이 있는 경우
                l.bias.data = l.bias.data.to(dtype) # 편향 데이터도 저정밀도 데이터 타입으로 변환

        if isinstance(l, (nn.MultiheadAttention, Attention)): # 트랜스포머의 멀티헤드 어텐션 레이어인 경우
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                """
                in_proj_weight: 입력 가중치
                q_proj_weight: 쿼리 가중치
                k_proj_weight: 키 가중치
                v_proj_weight: 값 가중치
                in_proj_bias: 입력 편향
                bias_k: 키 편향
                bias_v: 값 편향
                """
                
                tensor = getattr(l, attr) # l 객체에서 attr 속성을 가져옴
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, (CLIP, TextTransformer)): # CLIP 모델이나 TextTransformer 모델인 경우
            # convert text nn.Parameter projections
            attr = getattr(l, "text_projection", None) # l 객체에서 text_projection 속성을 가져옴
            if attr is not None:
                attr.data = attr.data.to(dtype) # attr 데이터를 저정밀도 데이터 타입으로 변환

        if isinstance(l, VisionTransformer): # VisionTransformer 모델인 경우
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None) # l 객체에서 proj 속성을 가져옴
            if attr is not None:
                attr.data = attr.data.to(dtype) # attr 데이터를 저정밀도 데이터 타입으로 변환

    model.apply(_convert_weights) # 모델의 모든 서브모듈을 돌면서 _convert_weights 함수 적용

convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat
# covert_weights_to_fp16() 함수도 convert_weights_to_lp() 함수와 동일하게 동작하도록 별칭 설정

# used to maintain checkpoint compatibility
def convert_to_custom_text_state_dict(state_dict: dict): # CLIP의 체크포인트 포맷을 새로운 custom 포맷으로 변환
    if 'text_projection' in state_dict:
        # old format state_dict, move text tower -> .text
        # text_projection : CLIP의 텍스트 임베딩을 projection 하는 레이어, 이 키가 존재하면 old 포맷임을 의미함.
        new_state_dict = {}
        for k, v in state_dict.items():
            if any(k.startswith(p) for p in (
                'text_projection',
                'positional_embedding',
                'token_embedding',
                'transformer',
                'ln_final',
            )):
                k = 'text.' + k # 키 이름을 'text.' 접두사를 붙여 새로운 딕셔너리에 추가 (text_projection -> text.text_projection)
            new_state_dict[k] = v
        return new_state_dict
    return state_dict


def build_model_from_openai_state_dict( # OpenAI 형식의 체크포인트 포맷을 사용하여 모델 생성
        state_dict: dict,
        quick_gelu=True,
        cast_dtype=torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit: # ViT 모델인 경우, 필요한 구조 정보를 state_dict에서 추출
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
    else: # ResNet 기반 모델인 경우, 주요 파라미터 추출
        counts: list = [
            len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_size = output_width * 32

    # 텍스트 모델 파라미터 추출
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        cast_dtype=cast_dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]: # 불필요한 메타데이터 삭제
        state_dict.pop(key, None)
    convert_weights_to_fp16(model)  # OpenAI state dicts are partially converted to float16
    model.load_state_dict(state_dict)
    return model.eval()


def trace_model(model, batch_size=256, device=torch.device('cpu')): #JIT 컴파일로 모델을 최적화하고 추론 속도 높임
    model.eval() # 모델을 평가모드로 설정해 Dropout 같은 학습 연산 비활성화
    image_size = model.visual.image_size
    example_images = torch.ones((batch_size, 3, image_size, image_size), device=device) # 모의 이미지 데이터 (모든 픽셀 값이 1)
    example_text = torch.zeros((batch_size, model.context_length), dtype=torch.int, device=device) # 모의 텍스트 데이터 (모든 픽셀 값이 0)
    model = torch.jit.trace_module( # TorchScript 변환 수행
        model,
        inputs=dict(
            forward=(example_images, example_text),
            encode_text=(example_text,),
            encode_image=(example_images,)
        ))
    # trace_module()은 주어진 입력값을 통해 모델의 연산 그래프를 고정하여 Python 인터프리터 없이도 실행 가능
    model.visual.image_size = image_size
    return model


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True): # CLIP 모델에서 vision 부분의 위치 임베딩 크기를 조정함
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size) # 현재 모델의 grid 크기
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens # 새로운 임베딩 크기 계산
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    # interpolation을 사용해 새로운 위치 임베딩 생성
    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None: # 클래스 토큰이 있다면 이를 포함하여
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0) # 새로운 위치 임베딩 저장
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def resize_text_pos_embed(state_dict, model, interpolation: str = 'linear', antialias: bool = False):
    old_pos_embed = state_dict.get('positional_embedding', None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, 'positional_embedding', None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, 'positional_embedding', None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, 'text pos_embed width changed!'
    if old_num_pos == num_pos:
        return

    logging.info('Resizing text position embedding num_pos from %s to %s', old_num_pos, num_pos)
    old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
    old_pos_embed = F.interpolate(
        old_pos_embed,
        size=num_pos,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
    new_pos_embed = old_pos_embed

    state_dict['positional_embedding'] = new_pos_embed


def get_model_preprocess_cfg(model): # 모델의 이미지 전처리 설정을 가져오는 함수
    module = getattr(model, 'visual', model) 
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, 'image_size')
        if size is not None:
            preprocess_cfg['size'] = size
        mean = getattr(module, 'image_mean', None)
        if mean is not None:
            preprocess_cfg['mean'] = mean
        std = getattr(module, 'image_std', None)
        if std is not None:
            preprocess_cfg['std'] = std
    return preprocess_cfg
    '''
    {
        "size": 224,
        "mean": [0.48145466, 0.4578275, 0.40821073],  # 이미지 정규화 평균값
        "std": [0.26862954, 0.26130258, 0.27577711]   # 이미지 정규화 표준편차
    }
    '''


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]): # 모델의 이미지 전처리 설정을 업데이트
    module = getattr(model, 'visual', model)
    module.image_mean = preprocess_cfg['mean']  # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg['std']  # legacy attribute, keeping for bwd compat
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)  # new attr, package all pp cfg as dict


def get_model_tokenize_cfg(model): # 모델의 토크나이징 설정을 가져오는 함수
    module = getattr(model, 'text', model)
    cfg = {}
    context_length = getattr(module, 'context_length', None)
    if context_length is not None:
        cfg['context_length'] = context_length
    vocab_size = getattr(module, 'vocab_size', None)
    if vocab_size is not None:
        cfg['vocab_size'] = vocab_size
    return cfg