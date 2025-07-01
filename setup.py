# standard dependencies
pass
# 3rd-party dependencies
from setuptools import setup, find_packages

# internal dependencies
pass


setup(
    name='ai_processing',
    version='0.1.0',
    description='Ivakt Timemanager AI Processing',
    author='Chris V. Follis',
    author_email='chrisvfollis@gmail.com',
    packages=find_packages(),
    install_requires=[
        # Deep Learning Frameworks:
        'torch',
        'torchvision',
        'torchreid',
        'onnx2torch',

        # Image/Video Processing:
        'opencv-contrib-python',
        'pillow',

        # Data Manipulation:
        'numpy',
        'pandas',
        'pyarrow',
        'scipy',
        'openpyxl',
        'xlsxwriter',
        'h5py',
        'statsmodels',
        'scikit-learn',
        'lap',

        # Networking, APIs, Connections, etc:
        'requests',
        'boto3',
        'python-dotenv',
        'psycopg2-binary',

        # Other Libraries:
        'shapely',
        'tqdm==4.43.0',
        'psutil',
        'matplotlib',
        'pympler',
        'filterpy',
    ],
    python_requires='>=3.10,<3.11',
)
