import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.faiss.indexer import index_videos
from src.faiss.serve import serve_video, flask_app
import argparse

def main():
    parser = argparse.ArgumentParser(description='video indexing and web server')
    parser.add_argument('--video_dir_path', type=str, default="../static/data/ours/video",
                      help='indexing video directory')
    parser.add_argument('--select_dirs', type=str, default="static/faiss/ours",
                      help='select directory')
    parser.add_argument('--port', type=int, default=5000,
                      help='web server port')
    args = parser.parse_args()

    print("start video indexing...")
    index_videos(
        video_dir_path=args.video_dir_path,
        select_dirs=args.select_dirs
    )
    print("indexing done")

    print(f"start web server (port:  {args.port})")
    flask_app.run(debug=True, port=args.port)

if __name__ == "__main__":
    main()