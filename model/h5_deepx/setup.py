from setuptools import setup, find_packages

setup(
    name='h5_deepx',
    version='0.1.0',
    description='A tool to extract model structure and weights from H5 files',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'h5py',  # H5 文件依赖
        'numpy',  # NumPy 依赖
        'pyyaml',  # YAML 依赖
    ],
    entry_points={
        'console_scripts': [
            'todeepx=h5_deepx.todeepx:extract_h5_model',
        ],
    },
)