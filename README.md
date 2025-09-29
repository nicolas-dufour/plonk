<div align="center">

# Around the World in 80 Timesteps: <br>A Generative Approach to Global Visual Geolocation

![PLONK](/.media/teaser.png)

<a href="https://nicolas-dufour.github.io/" >Nicolas Dufour</a>, <a href="https://davidpicard.github.io/" >David Picard</a>, <a href="https://vicky.kalogeiton.info/" >Vicky Kalogeiton</a>, <a href="https://loiclandrieu.com/" >Loic Landrieu</a>
</div>

Introducing the first generative geolocation method based on diffusion and flow matching! We learn the relationship between visual content and location by denoising random locations conditionally to images.

➜ New SOTA for visual geolocation on OpenStreetView-5M, YFCC-100M, and iNat-21

➜ Generate global probability density maps and quantify localizability

➜ Introduce the problem of probabilistic visual geolocation

This repository contains the code for the paper "Around the World in 80 Timesteps: A Generative Approach to Global Visual Geolocation"

<br>
<br>

Project website: [https://nicolas-dufour.github.io/plonk](https://nicolas-dufour.github.io/plonk)

Arxiv: [https://arxiv.org/abs/2412.06781](https://arxiv.org/abs/2412.06781)

HuggingFace Collection: [https://huggingface.co/collections/nicolas-dufour/around-the-world-in-80-timesteps-6758595d634129e6fc63dad9](https://huggingface.co/collections/nicolas-dufour/around-the-world-in-80-timesteps-6758595d634129e6fc63dad9)

Demo: [https://huggingface.co/spaces/nicolas-dufour/PLONK](https://huggingface.co/spaces/nicolas-dufour/PLONK)

##  Installation
If you want to use our models, you can directly install the `diff-plonk` package:
```bash
conda create -n plonk python=3.10
conda activate plonk
pip install diff-plonk
```

For a local installation, if you want to train the model or use the demo, you can clone the repository and install the dependencies:


From the github repository:
```bash
git clone https://github.com/nicolas-dufour/plonk.git
cd plonk
conda create -n plonk python=3.10
conda activate plonk
pip install -e .
```

From pip:
```bash
conda create -n plonk python=3.10
conda activate plonk
pip install diff-plonk
```

## Models

We provide pre-trained models for the OSV-5M, YFCC-100M, and iNat-21 datasets. You can download them from the [huggingface hub](https://huggingface.co/collections/nicolas-dufour/around-the-world-in-80-timesteps-6758595d634129e6fc63dad9).

## Demo

We provide a demo of the model hosted on the following [HuggingFace Space](https://huggingface.co/spaces/nicolas-dufour/PLONK).

### Running the demo locally

```bash
pip install -e ".[demo]"
```

And then run the following command:

```bash
streamlit run plonk/demo/demo.py 
```

## Usage

To use the models, you can use our pipeline:

```python
from plonk import PlonkPipeline

pipeline = PlonkPipeline("nicolas-dufour/PLONK_YFCC")

gps_coords = pipeline(images, batch_size=1024)
```
With images being a list of PIL images or a PIL image.

3 different models are provided for each dataset:

- `nicolas-dufour/PLONK_OSV_5M`: OSV-5M
- `nicolas-dufour/PLONK_YFCC`: YFCC-100M
- `nicolas-dufour/PLONK_iNat`: iNat-21

### Baselines models
We also provide the baseline models for the OSV-5M, YFCC-100M, and iNat-21 datasets for the carthesian flow matching and diffusion models. You can download them from the [huggingface hub](https://huggingface.co/collections/nicolas-dufour/around-the-world-in-80-timesteps-6758595d634129e6fc63dad9).

Flow matching models:

- `nicolas-dufour/PLONK_OSV_5M_flow`: OSV-5M
- `nicolas-dufour/PLONK_YFCC_flow`: YFCC-100M
- `nicolas-dufour/PLONK_iNat_flow`: iNat-21

Diffusion models:

- `nicolas-dufour/PLONK_OSV_5M_diffusion`: OSV-5M
- `nicolas-dufour/PLONK_YFCC_diffusion`: YFCC-100M
- `nicolas-dufour/PLONK_iNat_diffusion`: iNat-21

## Training
### Install training dependencies
You will need to install the training dependencies:
```bash
pip install -e ".[train]"
```

### Downloading the dataset
You will need to download the OSV5M webdataset from the [https://huggingface.co/datasets/osv5m/osv5m-wds](https://huggingface.co/datasets/osv5m/osv5m-wds) in the `plonk/datasets/osv5m` folder.

### Training the model
To train the model, you can use the following command:

```bash
python plonk/train.py exp=osv_5m_geoadalnmlp_r3_small_sigmoid_flow_riemann mode=traineval experiment_name=My_OSV_5M_Experiment
```
### Evaluating the model
To evaluate the model, you can use the following command:

```bash
python plonk/train.py exp=osv_5m_geoadalnmlp_r3_small_sigmoid_flow_riemann mode=eval experiment_name=My_OSV_5M_Experiment
```

### YFCC and iNAturalist
For YFCC and iNat, script to preprocess the dataset are provided in the `plonk/data/extract_embeddings` and `plonk/data/to_webdataset` folders.

## Citation

If you find this work useful for your research or use our code, please cite our paper:

```bibtex
@article{dufour2024around,
  title={Around the World in 80 Timesteps: A Generative Approach to Global Visual Geolocation},
  author={Dufour, Nicolas and Picard, David and Kalogeiton, Vicky and Landrieu, Loic},
  journal={arXiv preprint arXiv:2412.06781},
  year={2024}
}
```
