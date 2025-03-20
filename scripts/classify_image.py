import torch
import os
from PIL import Image

from pose_clip import create_model_and_transforms, get_tokenizer
from pose_clip.constants import pretrained_model_path

'''
이미지-텍스트 매칭 테스트 스크립트
'''

model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained=pretrained_model_path)

path = "data/dog.jpg"

model.eval()
tokenizer = get_tokenizer('ViT-B-32')

def get_frames(path):
    for files in os.walk(path):
        list_files = []
        if len(files) > 0:
            for file in files:
                list_files.append(file)
    return list_files

image = preprocess(Image.open(path)).unsqueeze(0)
text = tokenizer(["a snail", "a dog", "a cat"])

with torch.no_grad(), torch.amp.autocast('cuda'):
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)