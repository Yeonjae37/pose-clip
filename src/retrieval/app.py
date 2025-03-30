# app_video.py
import json
import os
import torch
import numpy as np
from pose_clip import create_model_and_transforms, get_tokenizer

from faiss import read_index


class App:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        pretrained_model_path = os.path.join(project_root, "models", "ViT-B-32-laion2B-s34B-b79K.safetensors")

        self.model, _, _ = create_model_and_transforms("ViT-B-32", pretrained=pretrained_model_path)
        self.model.eval()
        self.tokenizer = get_tokenizer("ViT-B-32")

        static_dir = os.path.join(current_dir, "..", "app", "static")

        index_path = os.path.join(static_dir, "index.faiss")
        self.index = read_index(index_path)
        
        metadata_path = os.path.join(static_dir, "video_metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            print(f"Loaded {len(self.metadata)} videos from metadata")

    def search(self, search_text, topk=5):
        text_tokens = self.tokenizer([search_text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        _, indices = self.index.search(text_features.cpu().numpy(), topk)
        top_results = [self.metadata[i] for i in indices[0]]

        print(f"\nQuery: {search_text}")
        for item in top_results:
            print(f"{item['video']} ({item['action']})")

        # Evaluation
        query_label = search_text.lower().replace(" ", "_")
        correct = [item for item in top_results if query_label in item['action'].lower()]
        #precision = len(correct) / topk
        #total_gt = len([item for item in self.metadata if query_label in item['action'].lower()])
        #recall = len(correct) / total_gt if total_gt else 0.0

        #print(f"\nPrecision@{topk}: {precision:.2f}")
        #print(f" Recall@{topk}: {recall:.2f}")

        return [f"/videos/{item['action']}/{item['video']}" for item in top_results]


        
if __name__ == "__main__":
    app = App()
    while True:
        query = input("Search: ")
        if query == "exit":
            break
        results = app.search(query)
        print("Top videos:")
        for res in results:
            print(res)