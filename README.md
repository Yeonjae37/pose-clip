## Pose CLIP

Vision-Language-based action retrieval system with filtering for efficient ground truth generation in signal-based action recognition.

### **Project Structure**

The project is organized as follows:

- `data/video/`: Contains video files for demonstration and testing
- `nbs/` : Contains Jupyter Notebook files for prototyping
- `scripts/`: Contains experiment and test scripts
    - `unit_test/` : Scripts for unit testing
- `results/` : Stores action recognition results and outputs
- `src/`: Source code
    - `faiss/`: Code for text-based retrieval of image and video data using FAISS and PoseCLIP
    - `pose_clip/`: Core implementation of the PoseCLIP model
    - `pose_clip_train/`: Code for model fine-tuning
- `Pipfile` and `Pipfile.lock`: Python dependency management

### **Pretrained Model Weights**

To use the pretrained model, download the weight file from the following link:  [**Download Weights**](https://drive.google.com/file/d/1GFOh18QQwsAcwavmOOeg8v1ZpQmSELco/view?usp=drive_link)

Once downloaded, place the weight file in the following directory:   
`models/ViT-B-32-laion2B-s34B-b79K.safetensors`

### **FAISS-based Image Retrieval**
This project integrates [**FAISS**](https://github.com/facebookresearch/faiss) to enable fast and efficient data retrieval based on text queries using the PoseCLIP model.

To test the retrieval system:
1. Place sample images in `data/images/`
2. Run the indexing script:

    ```
    python src/faiss/index.py
    ```

3. Launch the web server:

    ```
    python src/faiss/serve.py
    ```
4. Open your browser and go to http://localhost:5000 to test text-based data search (e.g., try searching for “dog”).

### **Installation**

**Prerequisites**

- Python 3.10
- pipenv (for dependency management)

**Setup**

1. Clone the repository:
    
    ```
    git clone https://github.com/Yeonjae37/pose-clip.git
    cd pose-clip
    ```
    
2. Install dependencies using pipenv:
    
    ```
    pipenv install
    ```
    
3. Activate the virtual environment:
    
    ```
    pipenv shell
    ```
