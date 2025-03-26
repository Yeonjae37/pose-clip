from faiss import write_index
from PIL import Image
from tqdm import tqdm

import argparse
import faiss
import json
import numpy as np
import os
import torch

from pose_clip import create_model_and_transforms, get_tokenizer
from pose_clip.model import resize_text_pos_embed, resize_pos_embed

def index(image_dir_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrained_model_path = "../../models/ViT-B-32-laion2B-s34B-b79K.safetensors"
    model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained=pretrained_model_path)
    model.eval()

    images = []
    image_paths = []
    img_dir_path = image_dir_path
    for animal_name in sorted(os.listdir(img_dir_path)):
        print(animal_name)
        if not os.path.isdir(os.path.join(img_dir_path, animal_name)):
            continue
        for img_file in tqdm(os.listdir(os.path.join(img_dir_path, animal_name))):
            if not img_file.endswith(".jpg"):
                continue
            image = Image.open(os.path.join(img_dir_path, animal_name, img_file)).convert("RGB")
            images.append(preprocess(image))
            image_paths.append(os.path.join(img_dir_path, animal_name, img_file))
    image_input = torch.tensor(np.stack(images)).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().numpy()

    index = faiss.IndexFlatIP(image_features.shape[1])
    index.add(image_features)
    write_index(index, "static/index.faiss")

    with open("static/image_paths.json", "w") as f:
        json.dump(image_paths, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir_path", type=str, default="static/data/images")
    args = parser.parse_args()
    index(args.image_dir_path)