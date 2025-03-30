from faiss import write_index
from PIL import Image
from tqdm import tqdm

import argparse
import faiss
import json
import numpy as np
import os
import torch
import cv2

from pose_clip import create_model_and_transforms


"""
Script for indexing video or image datasets using a CLIP-based visual encoder and FAISS.
This script extracts frame/image features using a pretrained ViT model, computes embeddings,
and builds a FAISS index for efficient similarity search.

- For video indexing, it extracts frame-level embeddings, averages them, and stores them per video.
- For image indexing, it directly computes and stores embeddings per image.
- All metadata and index files are saved in the specified output directory.
"""

def extract_image_features(image, model, device):
    image_input = torch.tensor(np.stack(image)).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy()

def save_faiss_index(features, paths_or_metadata, output_dir, index_name="index.faiss", json_name="image_paths.json"):
    os.makedirs(output_dir, exist_ok=True)
    index = faiss.IndexFlatIP(features.shape[1])
    index.add(features)
    faiss.write_index(index, os.path.join(output_dir, index_name))

    with open(os.path.join(output_dir, json_name), "w") as f:
        json.dump(paths_or_metadata, f)

def index_images(image_dir_path, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    pretrained_model_path = os.path.join(project_root, "models", "ViT-B-32-laion2B-s34B-b79K.safetensors")

    model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained=pretrained_model_path)
    model.eval()

    images = []
    image_paths = []

    for label_dir in sorted(os.listdir(image_dir_path)):
        full_path = os.path.join(image_dir_path, label_dir)
        if not os.path.isdir(full_path):
            continue
        for img_file in tqdm(os.listdir(full_path)):
            if not img_file.endswith(".jpg"):
                continue
            image_path = os.path.join(full_path, img_file)
            image = Image.open(image_path).convert("RGB")
            images.append(preprocess(image))
            image_paths.append(image_path)

    features = extract_image_features(images, model, device)
    save_faiss_index(features, image_paths, output_dir)

def index_videos(video_root_dir, output_dir=None, select_dirs=None, frame_interval=30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    pretrained_model_path = os.path.join(project_root, "models", "ViT-B-32-laion2B-s34B-b79K.safetensors")

    output_dir = os.path.join(project_root, "src", "app", "static")

    model, _, preprocess = create_model_and_transforms("ViT-B-32", pretrained=pretrained_model_path)
    model.eval()

    features = []
    video_metadata = []

    selected_dirs = select_dirs if select_dirs else os.listdir(video_root_dir)

    for action_dir in sorted(selected_dirs):
        full_dir_path = os.path.join(video_root_dir, action_dir)
        if not os.path.isdir(full_dir_path):
            continue

        for video_file in sorted(os.listdir(full_dir_path)):
            if not video_file.endswith(".mp4"):
                continue

            video_path = os.path.join(full_dir_path, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            frame_embeddings = []

            print(f"\nProcessing: {action_dir}/{video_file}")
            while True:
                success, frame = cap.read()
                if not success:
                    break
                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                processed = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    feat = model.encode_image(processed).float()
                feat /= feat.norm(dim=-1, keepdim=True)
                frame_embeddings.append(feat.cpu().numpy()[0])

                frame_idx += 1

            cap.release()

            if frame_embeddings:
                avg_embedding = np.mean(np.stack(frame_embeddings), axis=0)
                features.append(avg_embedding)
                video_metadata.append({
                    "video": video_file,
                    "action": action_dir
                })

    features = np.stack(features).astype(np.float32)
    save_faiss_index(features, video_metadata, output_dir, index_name="index.faiss", json_name="video_metadata.json")
    print(f"\nIndexed {len(video_metadata)} videos. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir_path", type=str, default=os.path.join("..", "data", "NTU_RGB+D"))
    parser.add_argument("--select_dirs", nargs="*", default=None)
    parser.add_argument("--frame_interval", type=int, default=30)
    args = parser.parse_args()

    index_videos(args.video_dir_path, select_dirs=args.select_dirs, frame_interval=args.frame_interval)