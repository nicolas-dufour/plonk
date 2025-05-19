from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="diff_plonk",
    version="0.3",
    description="Diffusion Geolocalization package for PLONK models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nicolas Dufour",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "transformers",
        "accelerate",
        "geoopt",
        "geos",
        "scipy==1.13.1",
        "einops",
        "torchdiffeq",
    ],
    include_package_data=True,
    extras_require={
        "train": [
            "wandb",
            "hydra-core",
            "pytorch-lightning",
            "scikit-learn",
            "reverse_geocoder",
            "matplotlib",
            "webdataset==0.2.57",
        ],
        "demo": ["streamlit", "streamlit-extras", "plotly"],
    },
)
