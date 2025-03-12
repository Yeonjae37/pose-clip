import torch
import os
from PIL import Image
import cv2
import numpy as np

from pose_clip import create_model_and_transforms, get_tokenizer

CAMERA_URL = "http://192.168.5.61:4747/video"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.to(device).eval()

tokenizer = get_tokenizer('ViT-B-32')

text_labels = [    
    "index finger pointing forward", 
    "index finger pointing upward", 
    "index finger pointing downward",
    "spread all fingers", 
    "thumbs up", 
    "closed fist", 
    "desk"
]

text_tokens = tokenizer(text_labels).to(device)

cap = cv2.VideoCapture(CAMERA_URL)

count = 0
threshold = 0.1502

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)

    image_tensor = preprocess(image_pil).unsqueeze(0).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).softmax(dim=-1)

        best_action_idx = similarity.argmax().item()
        best_label = text_labels[best_action_idx]
        best_confidence = similarity[0, best_action_idx].item()

        if best_confidence >= threshold:
            count += 1
            print(f"detect!!!!!!! ({count}) : {best_label} / {best_confidence:.5f}")

        #count += 1
        #print(f"detect!!! ({count}) : {best_label} / {best_confidence:.5f}")

    cv2.imshow("iPad Camera Stream", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()