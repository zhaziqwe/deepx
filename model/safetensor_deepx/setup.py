from setuptools import setup, find_packages

setup(
    name='safetensor-deepx',
    version='0.1.0',
    description='SafeTensor support for DeepX',
    author='igor.li',
    packages=find_packages(),
    install_requires=[
        'safetensors>=0.3.0',
        'numpy>=1.19.0',
        'deepxpy>=0.1.0',
        'graphviz>=0.20.1',
    ],
    python_requires='>=3.7',
)