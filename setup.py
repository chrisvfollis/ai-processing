from setuptools import setup, find_packages

setup(
    name="ai_processing",
    version="0.1.0",
    description="Ivakt Timemanager AI Processing",
    author="Chris V. Follis",
    author_email="chrisvfollis@gmail.com",
    packages=find_packages(),
    install_requires=[
        # Deep Learning Frameworks:
        "torch",
        "torchvision",
        "torchreid",
        "onnx2torch",
        "tensorflow==2.15.0",
        "keras==2.15.0",

        # Image/Video Processing and Computer Vision:
        "opencv-contrib-python",
        "pillow",
        "deepface @ git+https://github.com/serengil/deepface.git@master",
        "facenet-pytorch",

        # Data Manipulation:
        "numpy",
        "pandas",
        "scipy",
        "openpyxl",
        "xlsxwriter",
        "h5py",

        # Networks, APIs, etc:
        "requests",
        "boto3",
        "python-dotenv",

        # Other Libraries:
        "shapely",
        "tqdm==4.43.0",
        "psutil"

    ],
    python_requires=">=3.10,<3.11",
)
