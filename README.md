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
    - `pose_clip/`: Core implementation of the PoseCLIP model
    - `pose_clip_train/`: Code for model fine-tuning
- `Pipfile` and `Pipfile.lock`: Python dependency management

### **Pretrained Model Weights**

To use the pretrained model, download the weight file from the following link:    
**[Download Weights]** (https://drive.google.com/file/d/1GFOh18QQwsAcwavmOOeg8v1ZpQmSELco/view?usp=drive_link)

Once downloaded, place the weight file in the following directory:   
`models/ViT-B-32-laion2B-s34B-b79K/open_clip_model.safetensors`

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
