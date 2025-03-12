from setuptools import setup, find_packages

setup(
    name="pose-clip",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "Pillow",
        "open_clip_torch",
    ],
    author="Your Name",
    description="A CLIP-based pose estimation project",
    python_requires=">=3.8",
)