from setuptools import setup

setup(
    name="diff_plonk",
    version="0.1",
    description="Diffusion Geolocalization package for PLONK models",
    author="Nicolas Dufour",
    python_requires=">=3.10",
    packages=["plonk"],
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
