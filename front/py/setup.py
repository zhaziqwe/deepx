from setuptools import setup, find_packages

setup(
    name="deepx-framework",
    version="0.1.0",
    description="DeepX - 高性能深度学习框架",
    author="igor.li",
    author_email="lipeng@mirrorsoft.cn",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0", 
        "graphviz>=0.20.1",
        "networkx>=3.0.0",
        "sympy>=1.10.0",
        "pyyaml>=6.0.0",
        "jinja2>=3.0.0",
        "typing-extensions>=4.0.0",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
) 