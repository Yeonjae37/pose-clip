import torch
import os
import click
import cv2
import csv
import numpy as np
from PIL import Image
from pose_clip import create_model_and_transforms, get_tokenizer


@click.command()
@click.option('--video_dir', default="data/NTU_RGB+D", help="Path to the directory containing videos.")
@click.option('--output_csv', default="data/NTU_RGB+D/NTU_RGB+D_predictions.csv", help="Path to save predictions.")
@click.option('--frame_interval', default=5, type=int, help="Extract one frame every N frames.")
@click.option('--labels', default="kicking, walking_toward_each_other, pushing", help="Comma-separated list of action labels.")
@click.option('--target_actions', default="kicking, walking toward each other, pushing", help="Comma-separated list of specific actions to process. Leave empty to process all.")
def classify_video(video_dir, output_csv, frame_interval, labels, target_actions):
    """
    이미지 인코더 기반의 영상 분류 스크립트
    """
    pretrained_model_path = "models/ViT-B-32-laion2B-s34B-b79K.safetensors"
    #pretrained_model_path = "models/ViT-g-14-laion2B-s12B-b42K.safetensors"
    
    model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained=pretrained_model_path)
    #model, _, preprocess = create_model_and_transforms('ViT-g-14', pretrained=pretrained_model_path)
    model.eval()
    tokenizer = get_tokenizer('ViT-B-32')

    text_list = labels.split(',')
    text = tokenizer(text_list)

    target_actions = [action.strip() for action in target_actions.split(",")] if target_actions else []

    results = []

    for action_label in os.listdir(video_dir):
        action_path = os.path.join(video_dir, action_label)

        action_label_with_space = action_label.replace("_", " ")
        if target_actions and action_label_with_space not in target_actions:
            print(f"Skipping action: {action_label}")
            continue

        for video_file in os.listdir(action_path):
            video_path = os.path.join(action_path, video_file)
            print(f"Processing video: {video_path}")

            frames = extract_frames(video_path, frame_interval, preprocess)

            with torch.no_grad():
                image_features = model.encode_image(frames)
                text_features = model.encode_text(text)
                #image_features /= image_features.norm(dim=-1, keepdim=True)
                #text_features /= text_features.norm(dim=-1, keepdim=True)

                text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                video_prediction = text_probs.mean(dim=0)

                predicted_label = text_list[torch.argmax(video_prediction).item()]
                predicted_label = predicted_label.replace("_", " ").strip()
                results.append([video_file, predicted_label])

                print(f"Video {video_file} predicted as: {predicted_label}")

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["video_id", "predicted_label"])
        writer.writerows(results)

    print(f"All predictions saved to {output_csv}")


def extract_frames(video_path, frame_interval, preprocess):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frames = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(preprocess(image).unsqueeze(0))

        frame_count += 1

    cap.release()

    if len(frames) > 0:
        return torch.cat(frames, dim=0)
    return torch.tensor([])


if __name__ == "__main__":
    classify_video()