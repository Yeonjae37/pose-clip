import torch
import os
import click
from PIL import Image

from pose_clip import create_model_and_transforms, get_tokenizer
from pose_clip.model import resize_text_pos_embed, resize_pos_embed


@click.command()
@click.option('--image_path', default="data/dog.jpg", help="Path to the image to classify")
@click.option('--labels', default='a snail, a dog, a cat', help="Comma-separated list of labels")
def classify_image(image_path, labels):
    """
    이미지-텍스트 매칭 스크립트
    """
    #pretrained_model_path = "models/ViT-B-32-laion2B-s34B-b79K.safetensors"
    #pretrained_model_path = "models/ViT-L-14-laion2B-s32B-b82K.safetensors"
    pretrained_model_path = "models/ViT-g-14-laion2B-s12B-b42K.safetensors"
    
    model, _, preprocess = create_model_and_transforms('ViT-g-14', pretrained=pretrained_model_path)
    #model, _, preprocess = create_model_and_transforms('ViT-g-14', pretrained=pretrained_model_path, force_image_size=224, vision_cfg=dict(patch_size=14))

    resize_text_pos_embed(model.state_dict(), model)
    resize_pos_embed(model.state_dict(), model)

    model.eval()
    tokenizer = get_tokenizer('ViT-g-14')

    image = preprocess(Image.open(image_path)).unsqueeze(0)

    text_list = labels.split(',')
    text = tokenizer(text_list)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        #image_features /= image_features.norm(dim=-1, keepdim=True)
        #text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    print(f"Label probs for {image_path}: {text_probs}")

if __name__ == "__main__":
    classify_image()