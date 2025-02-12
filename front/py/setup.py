from setuptools import setup, find_packages

setup(
    name='deepxpy',
    version='0.1.0',
    description='deepx python interface',
    author='igor.li',
    author_email='lipeng@mirrorsoft.cn',
    packages=find_packages(),
    install_requires=[
        'pyyaml',  # YAML 依赖
    ],
    description="DeepX - 高性能深度学习框架",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/array2d/deepx",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)