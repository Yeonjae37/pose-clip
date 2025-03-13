import json
import logging
import os
import re
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch


from .model import CLIP, convert_to_custom_text_state_dict,\
    resize_pos_embed, get_cast_dtype, resize_text_pos_embed, set_model_preprocess_cfg
#from .coca_model import CoCa
from .loss import ClipLoss, DistillClipLoss, CoCaLoss, SigLipLoss
from .pretrained import is_pretrained_cfg, get_pretrained_cfg, download_pretrained, \
    download_pretrained_from_hf, list_pretrained_tags_by_model
from .transform import image_transform_v2, AugmentationCfg, PreprocessCfg, merge_preprocess_dict, merge_preprocess_kwargs
from .tokenizer import HFTokenizer, SimpleTokenizer, DEFAULT_CONTEXT_LENGTH

HF_HUB_PREFIX = 'hf-hub:'
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_): 
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path): # 모델 구성 파일 추가(경로로 추가)
    """ add model config path or file and update registry """
    if not isinstance(path, Path): # path가 Path 객체인지 확인. Path는 pathlib 모듈의 클래스
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name): # 모델 설정 반환
    """ Fetch model config from builtin (local library) configs.
    """
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None
    
def _get_hf_config(
        model_id: str,
        cache_dir: Optional[str] = None,
):
    """ Fetch model config from HuggingFace Hub.
    """
    config_path = download_pretrained_from_hf(
        model_id,
        filename='open_clip_config.json',
        cache_dir=cache_dir,
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config

def get_tokenizer(
        model_name: str = '', # 모델 이름
        context_length: Optional[int] = None, # 토크나이저가 처리할 문장의 최대 길이
        cache_dir: Optional[str] = None, # hugging face 모델 다운로드할 캐시 디렉토리
        **kwargs, # 추가적인 인자들
):
    if model_name.startswith(HF_HUB_PREFIX): # 허깅 페이스에서 모델 가져올 경우
        model_name = model_name[len(HF_HUB_PREFIX):]
        try:
            config = _get_hf_config(model_name, cache_dir=cache_dir)['model_cfg']
        except Exception:
            tokenizer = HFTokenizer(
                model_name,
                context_length=context_length or DEFAULT_CONTEXT_LENGTH,
                cache_dir=cache_dir,
                **kwargs,
            )
            return tokenizer
    else: # 로컬 모델 설정을 가져올 경우
        config = get_model_config(model_name)
        assert config is not None, f"No valid model config found for {model_name}."

    text_config = config.get('text_cfg', {}) # 텍스트 
    if 'tokenizer_kwargs' in text_config:
        tokenizer_kwargs = dict(text_config['tokenizer_kwargs'], **kwargs)
    else:
        tokenizer_kwargs = kwargs

    if context_length is None:
        context_length = text_config.get('context_length', DEFAULT_CONTEXT_LENGTH)

    if 'hf_tokenizer_name' in text_config:
        tokenizer = HFTokenizer(
            text_config['hf_tokenizer_name'],
            context_length=context_length,
            cache_dir=cache_dir,
            **tokenizer_kwargs,
        )
    else:
        tokenizer = SimpleTokenizer(
            context_length=context_length,
            **tokenizer_kwargs,
        )

    return tokenizer

def load_state_dict( # 모델의 체크포인트 파일을 로드해서 state_dict 반환
        checkpoint_path: str,
        device='cpu',
        weights_only=True,
):
    # Check if safetensors or not and load weights accordingly
    if str(checkpoint_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path, device=device)
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
        except TypeError: # weights_only가 False인 경우
            checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]: 
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'): # "module.layer1.weight" -> "layer1.weight" 로 변경
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint( # 가중치를 모델에 적용
        model: CLIP,
        checkpoint_path: str,
        strict: bool = True, # 가중치를 모델 구조와 엄격하게 맞출지에 대한 여부
        weights_only: bool = True,
        device='cpu',
):
    if Path(checkpoint_path).suffix in ('.npz', '.npy'): #.npz, .npy 파일인 경우 SiGLIP 같은 big_vision 모델 가중치 불러옴
        # Separate path loading numpy big_vision (SigLIP) weights
        from open_clip.convert import load_big_vision_weights
        load_big_vision_weights(model, checkpoint_path)
        return {}

    state_dict = load_state_dict(checkpoint_path, device=device, weights_only=weights_only)

    # Detect & convert 3rd party state_dicts -> open_clip
    # state_dict = convert_state_dict(model, state_dict)

    # Detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)

    # correct if logit_scale differs in being scaler vs 1d param
    if 'logit_scale' in state_dict and model.logit_scale.ndim != state_dict['logit_scale'].ndim:
        state_dict['logit_scale'] = state_dict['logit_scale'].reshape(model.logit_scale.shape)

    # correct if logit_bias differs in being scaler vs 1d param
    if 'logit_bias' in state_dict and model.logit_bias.ndim != state_dict['logit_bias'].ndim:
        state_dict['logit_bias'] = state_dict['logit_bias'].reshape(model.logit_bias.shape)

    # If loading a non-SigLIP model for SigLIP training. See https://github.com/mlfoundations/open_clip/issues/712
    if 'logit_bias' not in state_dict and model.logit_bias is not None:
        state_dict["logit_bias"] = torch.zeros_like(state_dict["logit_scale"])

    # Certain text transformers no longer expect position_ids after transformers==4.31
    position_id_key = 'text.transformer.embeddings.position_ids'
    if position_id_key in state_dict and not hasattr(model, position_id_key):
        del state_dict[position_id_key]

    resize_pos_embed(state_dict, model)
    resize_text_pos_embed(state_dict, model)

    # Finally, load the massaged state_dict into model
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model( # CLIP 모델 생성
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        load_weights_only: bool = True,
        **model_kwargs,
):

    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = get_model_config(model_name)
    if model_cfg is not None:
        logging.info(f'Loaded {model_name} model config.')
    else:
        logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
        raise RuntimeError(f'Model config for {model_name} not found.')

    model_cfg = dict(model_cfg, **model_kwargs)  # merge cfg dict w/ kwargs (kwargs overrides cfg)

    model = CLIP(**model_cfg)

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16
    }.get(precision, torch.float32)

    model.to(device=device, dtype=dtype)

    if pretrained:
        if os.path.exists(pretrained):
            logging.info(f"Loading pretrained model from {pretrained}")
            load_checkpoint(model, pretrained, weights_only=load_weights_only)
        else:
            raise RuntimeError(f"Pretrained model not found at: {pretrained}")
    
    elif require_pretrained:
        raise RuntimeError(f"Pretrained model is required but no valid path was provided.")

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    force_preprocess_cfg = force_preprocess_cfg or {}
    preprocess_cfg = {
        'size': 224,
        'mode': 'RGB',
        'mean': (0.48145466, 0.4578275, 0.40821073),
        'std': (0.26862954, 0.26130258, 0.27577711),
        'interpolation': 'bicubic',
        'resize_mode': 'shortest',
        'fill_color': 0
    }

    set_model_preprocess_cfg(model, merge_preprocess_dict(preprocess_cfg, force_preprocess_cfg))
    
    return model


def create_loss(args):
    if args.distill: #DistillClipLoss(지식 증류용 손실 함수)를 반환.
        # 지식 증류 : 큰 모델(teacher)의 지식을 작은 모델(student)에게 전이하는 방식.
        return DistillClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif "coca" in args.model.lower():
        return CoCaLoss(
            caption_loss_weight=args.coca_caption_loss_weight,
            clip_loss_weight=args.coca_contrastive_loss_weight,
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
    elif args.siglip:
        assert not args.horovod, "Horovod not currently supported for SigLip"
        return SigLipLoss(
            rank=args.rank,
            world_size=args.world_size,
            dist_impl=args.loss_dist_impl,  # siglip has multiple distributed implementations to choose from
        )

    return ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )


def create_model_and_transforms(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        load_weights_only: bool = True,
        **model_kwargs,
):

    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )

    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_patch_dropout=force_patch_dropout,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        pretrained_image=pretrained_image,
        pretrained_hf=pretrained_hf,
        cache_dir=cache_dir,
        output_dict=output_dict,
        load_weights_only=load_weights_only,
        **model_kwargs,
    )

    pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)

    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )

    return model, preprocess_train, preprocess_val


def create_model_from_pretrained(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        image_mean: Optional[Tuple[float, ...]] = None,
        image_std: Optional[Tuple[float, ...]] = None,
        image_interpolation: Optional[str] = None,
        image_resize_mode: Optional[str] = None,  # only effective for inference
        return_transform: bool = True,
        cache_dir: Optional[str] = None,
        load_weights_only: bool = True,
        **model_kwargs,
):
    force_preprocess_cfg = merge_preprocess_kwargs(
        {},
        mean=image_mean,
        std=image_std,
        interpolation=image_interpolation,
        resize_mode=image_resize_mode,
    )
    
    model = create_model(
        model_name,
        pretrained,
        precision=precision,
        device=device,
        jit=jit,
        force_quick_gelu=force_quick_gelu,
        force_custom_text=force_custom_text,
        force_image_size=force_image_size,
        force_preprocess_cfg=force_preprocess_cfg,
        cache_dir=cache_dir,
        require_pretrained=True,
        load_weights_only=load_weights_only,
        **model_kwargs,
    )

    if not return_transform:
        return model

    preprocess = image_transform_v2(
        PreprocessCfg(**model.visual.preprocess_cfg),
        is_train=False,
    )

    return model, preprocess
