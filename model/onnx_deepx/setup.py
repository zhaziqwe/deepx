from setuptools import setup, find_packages

setup(
    name='onnx_deepx',
    version='0.1.0',
    description='A simple ONNX model extractor',
    author='Lipeng',
    author_email='lipeng@mirrorsoft.cn',
    packages=find_packages(),
    install_requires=[
        'onnx',  # 添加 ONNX 依赖
    ],
    entry_points={
        'console_scripts': [
            'todeepx=onnx_deepx.todeepx:extract_onnx_info',
            'toonnx=onnx_deepx.toonnx:extract_onnx_info',
        ],
    },
)