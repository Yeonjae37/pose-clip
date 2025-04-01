from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "retrieval"))
from app import App

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

parser = argparse.ArgumentParser()
args = parser.parse_args()

flask_app = Flask(__name__)
app_instance = App()

@flask_app.route("/")
def index():
    return render_template("index.html")


@flask_app.route("/search")
def search():
    search_query = request.args.get("search_query", "")
    results = app_instance.search(search_query, topk=20)

    print(f"\n[Query: {search_query}] Returned Videos:")
    return jsonify(results)


def serve_video(filename):
    video_dir = os.path.join(ROOT_DIR, "data")
    return send_from_directory(video_dir, filename)

if __name__ == "__main__":
    flask_app.run(debug=True, port=5000)