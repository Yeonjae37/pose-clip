# CLIP 모델을 학습할 때 사용된 데이터셋에서 추출한 평균과 표준 편차
OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

# ImageNet 데이터셋에서 사용된 평균과 표준 편차
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Inception 모델(Google의 이미지 인식 모델)에서 사용된 평균과 표준 편차
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)

# Default name for a weights file hosted on the Huggingface Hub.
HF_WEIGHTS_NAME = "open_clip_pytorch_model.bin"  # default pytorch pkl
HF_SAFE_WEIGHTS_NAME = "open_clip_model.safetensors"  # safetensors version
HF_CONFIG_NAME = 'open_clip_config.json'
