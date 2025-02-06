from setuptools import setup, find_packages

setup(
    name="humanposer",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["matplotlib", "numpy", "tqdm", "einops", "smplx", "chumpy"],
    python_requires=">=3.8",
)
