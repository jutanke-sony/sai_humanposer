from setuptools import setup, find_packages

setup(
    name="humanposer",
    version="1.0.1",
    packages=find_packages(),
    install_requires=["matplotlib", "numpy", "tqdm", "einops", "gdown"],
    python_requires=">=3.8",
)
