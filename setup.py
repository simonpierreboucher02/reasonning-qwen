"""Setup script for reasoning model package."""

from setuptools import setup, find_packages

setup(
    name="reasoning_model",
    version="1.0.0",
    description="Clean implementation of reasoning models based on Qwen3",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.7.1",
        "tokenizers>=0.21.2",
        "sympy>=1.14.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "dev": [
            "jupyterlab>=4.4.7",
            "matplotlib>=3.10.7",
        ],
    },
)
