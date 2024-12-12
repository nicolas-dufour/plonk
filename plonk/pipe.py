import torch
import random
import string
from transformers import AutoTokenizer, T5EncoderModel
from plonk.models.pretrained_models import Plonk
from plonk.models.samplers.riemannian_flow_sampler import riemannian_flow_sampler

from plonk.models.postprocessing import CartesiantoGPS

from plonk.models.schedulers import (
    SigmoidScheduler,
    LinearScheduler,
    CosineScheduler,
)
from plonk.models.preconditioning import DDPMPrecond
from torchvision import transforms
from transformers import CLIPProcessor, CLIPVisionModel
from plonk.utils.image_processing import CenterCrop
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODELS = {
    "nicolas-dufour/PLONK_YFCC": {"emb_name": "dinov2"},
    "nicolas-dufour/PLONK_OSV_5M": {
        "emb_name": "street_clip",
    },
    "nicolas-dufour/PLONK_iNaturalist": {
        "emb_name": "dinov2",
    },
}


def scheduler_fn(
    scheduler_type: str, start: float, end: float, tau: float, clip_min: float = 1e-9
):
    if scheduler_type == "sigmoid":
        return SigmoidScheduler(start, end, tau, clip_min)
    elif scheduler_type == "cosine":
        return CosineScheduler(start, end, tau, clip_min)
    elif scheduler_type == "linear":
        return LinearScheduler(clip_min=clip_min)
    else:
        raise ValueError(f"Scheduler type {scheduler_type} not supported")


class DinoV2FeatureExtractor:
    def __init__(self, device=device):
        super().__init__()
        self.device = device
        self.emb_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
        self.emb_model.eval()
        self.emb_model.to(self.device)
        self.augmentation = transforms.Compose(
            [
                CenterCrop(ratio="1:1"),
                transforms.Resize(
                    336, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __call__(self, batch):
        embs = []
        with torch.no_grad():
            for img in batch["img"]:
                emb = self.emb_model(
                    self.augmentation(img).unsqueeze(0).to(self.device)
                ).squeeze(0)
                embs.append(emb)
        batch["emb"] = torch.stack(embs)
        return batch


class StreetClipFeatureExtractor:
    def __init__(self, device=device):
        self.device = device
        self.emb_model = CLIPVisionModel.from_pretrained("geolocal/StreetCLIP").to(
            device
        )
        self.processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")

    def __call__(self, batch):
        inputs = self.processor(images=batch["img"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.emb_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]
        batch["emb"] = embeddings
        return batch


def load_prepocessing(model_name, dtype=torch.float32):
    if MODELS[model_name]["emb_name"] == "dinov2":
        return DinoV2FeatureExtractor()
    elif MODELS[model_name]["emb_name"] == "street_clip":
        return StreetClipFeatureExtractor()
    else:
        raise ValueError(f"Embedding model {MODELS[model_name]['emb_name']} not found")


class PlonkPipeline:
    """
    The CADT2IPipeline class is designed to facilitate the generation of images from text prompts using a pre-trained CAD model.
    It integrates various components such as samplers, schedulers, and post-processing techniques to produce high-quality images.

    Initialization:
        CADT2IPipeline(
            model_path,
            sampler="ddim",
            scheduler="sigmoid",
            postprocessing="sd_1_5_vae",
            scheduler_start=-3,
            scheduler_end=3,
            scheduler_tau=1.1,
            device="cuda",
        )

    Parameters:
        model_path (str): Path to the pre-trained CAD model.
        sampler (str): The sampling method to use. Options are "ddim", "ddpm", "dpm", "dpm_2S", "dpm_2M". Default is "ddim".
        scheduler (str): The scheduler type to use. Options are "sigmoid", "cosine", "linear". Default is "sigmoid".
        postprocessing (str): The post-processing method to use. Options are "consistency-decoder", "sd_1_5_vae". Default is "sd_1_5_vae".
        scheduler_start (float): Start value for the scheduler. Default is -3.
        scheduler_end (float): End value for the scheduler. Default is 3.
        scheduler_tau (float): Tau value for the scheduler. Default is 1.1.
        device (str): Device to run the model on. Default is "cuda".

    Methods:
        model(*args, **kwargs):
            Runs the preconditioning on the network with the provided arguments.

        __call__(...):
            Generates images based on the provided conditions and parameters.

            Parameters:
                cond (str or list of str): The conditioning text or list of texts.
                num_samples (int, optional): Number of samples to generate. If not provided, it is inferred from cond.
                x_N (torch.Tensor, optional): Initial noise tensor. If not provided, it is generated.
                latents (torch.Tensor, optional): Previous latents.
                num_steps (int, optional): Number of steps for the sampler. If not provided, the default is used.
                sampler (callable, optional): Custom sampler function. If not provided, the default sampler is used.
                scheduler (callable, optional): Custom scheduler function. If not provided, the default scheduler is used.
                cfg (float): Classifier-free guidance scale. Default is 15.
                guidance_type (str): Type of guidance. Default is "constant".
                guidance_start_step (int): Step to start guidance. Default is 0.
                generator (torch.Generator, optional): Random number generator.
                coherence_value (float): Doherence value for sampling. Default is 1.0.
                uncoherence_value (float): Uncoherence value for sampling. Default is 0.0.
                unconfident_prompt (str, optional): Unconfident prompt text.
                thresholding_type (str): Type of thresholding. Default is "clamp".
                clamp_value (float): Clamp value for thresholding. Default is 1.0.
                thresholding_percentile (float): Percentile for thresholding. Default is 0.995.

            Returns:
                torch.Tensor: The generated image tensor after post-processing.

        to(device):
            Moves the model and its components to the specified device.

            Parameters:
                device (str): The device to move the model to (e.g., "cuda", "cpu").

            Returns:
                CADT2IPipeline: The pipeline instance with updated device.

    Example Usage:
            pipe = CADT2IPipeline(
                "nicolas-dufour/",
            )
            pipe.to("cuda")
            image = pipe(
                "a beautiful landscape with a river and mountains",
                num_samples=4,
            )
    """

    def __init__(
        self,
        model_path,
        scheduler="sigmoid",
        scheduler_start=-7,
        scheduler_end=3,
        scheduler_tau=1.0,
        device=device,
    ):
        self.network = Plonk.from_pretrained(model_path).to(device)
        self.network.requires_grad_(False).eval()
        assert scheduler in [
            "sigmoid",
            "cosine",
            "linear",
        ], f"Scheduler {scheduler} not supported"
        self.scheduler = scheduler_fn(
            scheduler, scheduler_start, scheduler_end, scheduler_tau
        )
        self.cond_preprocessing = load_prepocessing(model_name=model_path)
        self.postprocessing = CartesiantoGPS()
        self.sampler = riemannian_flow_sampler
        self.model_path = model_path
        self.preconditioning = DDPMPrecond()
        self.device = device

    def model(self, *args, **kwargs):
        return self.preconditioning(self.network, *args, **kwargs)

    def __call__(
        self,
        images,
        batch_size=None,
        x_N=None,
        num_steps=None,
        scheduler=None,
        cfg=0,
        generator=None,
    ):
        """Sample from the model given conditioning.

        Args:
            cond: Conditioning input (image or list of images)
            batch_size: Number of samples to generate (inferred from cond if not provided)
            x_N: Initial noise tensor (generated if not provided)
            num_steps: Number of sampling steps (uses default if not provided)
            sampler: Custom sampler function (uses default if not provided)
            scheduler: Custom scheduler function (uses default if not provided)
            cfg: Classifier-free guidance scale (default 15)
            generator: Random number generator

        Returns:
            Sampled GPS coordinates after postprocessing
        """
        # Set up batch size and initial noise
        shape = [3]
        if not isinstance(images, list):
            images = [images]
        if x_N is None:
            if batch_size is None:
                if isinstance(images, list):
                    batch_size = len(images)
                else:
                    batch_size = 1
            x_N = torch.randn(
                batch_size, *shape, device=self.device, generator=generator
            )
        else:
            x_N = x_N.to(self.device)
            if x_N.ndim == 3:
                x_N = x_N.unsqueeze(0)
            batch_size = x_N.shape[0]

        # Set up batch with conditioning
        batch = {"y": x_N}
        batch["img"] = images
        batch = self.cond_preprocessing(batch)
        if len(images) > 1:
            assert len(images) == batch_size
        else:
            batch["emb"] = batch["emb"].repeat(batch_size, 1)

        # Use default sampler/scheduler if not provided
        sampler = self.sampler
        if scheduler is None:
            scheduler = self.scheduler
        # Sample from model
        if num_steps is None:
            output = sampler(
                self.model,
                batch,
                conditioning_keys="emb",
                scheduler=scheduler,
                cfg_rate=cfg,
                generator=generator,
            )
        else:
            output = sampler(
                self.model,
                batch,
                conditioning_keys="emb",
                scheduler=scheduler,
                num_steps=num_steps,
                cfg_rate=cfg,
                generator=generator,
            )

        # Apply postprocessing and return
        output = self.postprocessing(output)
        # To degrees
        output = np.degrees(output.detach().cpu().numpy())
        return output

    def to(self, device):
        self.network.to(device)
        self.postprocessing.to(device)
        self.device = torch.device(device)
        return self
