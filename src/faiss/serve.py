from flask import Flask, request, render_template, jsonify, send_from_directory

from app import App
import os

flask_app = Flask(__name__)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

@flask_app.route("/")
def index():
    return render_template("index.html")

@flask_app.route("/search")
def search():
    search_query = request.args.get("search_query", "")
    
    app = App()
    results = app.search(search_query, results=5)
    results = [f"/images/{os.path.relpath(path, os.path.join(ROOT_DIR, 'data/images')).replace(os.sep, '/')}" for path in results]
    
    print("Search query:", search_query)
    print("Returned image paths:", results)

    return jsonify(results)


@flask_app.route("/images/<path:filename>")
def serve_images(filename):
    return send_from_directory(os.path.join(ROOT_DIR, "data", "images"), filename)


if __name__ == "__main__":
    flask_app.run(debug=True, port=5000)