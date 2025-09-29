import torch
from plonk.models.pretrained_models import Plonk
from plonk.models.samplers.riemannian_flow_sampler import riemannian_flow_sampler
from plonk.models.samplers.flow_sampler import flow_sampler
from plonk.models.samplers.ddim import ddim_sampler

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
from plonk.utils.manifolds import Sphere
from torch.func import jacrev, vmap, vjp
from torchdiffeq import odeint
from tqdm import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MODELS = {
    "nicolas-dufour/PLONK_YFCC": {
        "emb_name": "dinov2",
        "sampler": riemannian_flow_sampler,
    },
    "nicolas-dufour/PLONK_OSV_5M": {
        "emb_name": "street_clip",
        "sampler": riemannian_flow_sampler,
    },
    "nicolas-dufour/PLONK_iNaturalist": {
        "emb_name": "dinov2",
        "sampler": riemannian_flow_sampler,
    },
    "nicolas-dufour/PLONK_YFCC_flow": {"emb_name": "dinov2", "sampler": flow_sampler},
    "nicolas-dufour/PLONK_OSV_5M_flow": {
        "emb_name": "street_clip",
        "sampler": flow_sampler,
    },
    "nicolas-dufour/PLONK_iNaturalist_flow": {
        "emb_name": "dinov2",
        "sampler": flow_sampler,
    },
    "nicolas-dufour/PLONK_YFCC_diffusion": {
        "emb_name": "dinov2",
        "sampler": ddim_sampler,
    },
    "nicolas-dufour/PLONK_OSV_5M_diffusion": {
        "emb_name": "street_clip",
        "sampler": ddim_sampler,
    },
    "nicolas-dufour/PLONK_iNaturalist_diffusion": {
        "emb_name": "dinov2",
        "sampler": ddim_sampler,
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
                # Ensure image is RGB before augmentation
                if img.mode != "RGB":
                    img = img.convert("RGB")
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


# Helper functions adapted from plonk/models/module.py
# for likelihood computation
def div_fn(u):
    """Accepts a function u:R^D -> R^D."""
    J = jacrev(u, argnums=0)
    return lambda x, y: torch.trace(J(x, y).squeeze(0))


def output_and_div(vecfield, x, y, v=None):
    if v is None:
        dx = vecfield(x, y)
        div = vmap(div_fn(vecfield))(x, y)
    else:
        vecfield_x = lambda x: vecfield(x, y)
        dx, vjpfunc = vjp(vecfield_x, x)
        vJ = vjpfunc(v)[0]
        div = torch.sum(vJ * v, dim=-1)
    return dx, div


def _gps_degrees_to_cartesian(gps_coords_deg, device):
    """Converts GPS coordinates (latitude, longitude) in degrees to Cartesian coordinates."""
    if not isinstance(gps_coords_deg, np.ndarray):
        gps_coords_deg = np.array(gps_coords_deg)
    if gps_coords_deg.ndim == 1:
        gps_coords_deg = gps_coords_deg[np.newaxis, :]

    lat_rad = np.radians(gps_coords_deg[:, 0])
    lon_rad = np.radians(gps_coords_deg[:, 1])
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    cartesian_coords = np.stack([x, y, z], axis=-1)
    return torch.tensor(cartesian_coords, dtype=torch.float32, device=device)


class PlonkPipeline:
    """
    The PlonkPipeline class is designed to perform geolocation prediction from images using a pre-trained PLONK model.
    It integrates various components such as feature extractors, samplers, and coordinate transformations to predict locations.

    Initialization:
        PlonkPipeline(
            model_path,
            scheduler="sigmoid",
            scheduler_start=-7,
            scheduler_end=3,
            scheduler_tau=1.0,
            device="cuda",
        )

    Parameters:
        model_path (str): Path to the pre-trained PLONK model.
        scheduler (str): The scheduler type to use. Options are "sigmoid", "cosine", "linear". Default is "sigmoid".
        scheduler_start (float): Start value for the scheduler. Default is -7.
        scheduler_end (float): End value for the scheduler. Default is 3.
        scheduler_tau (float): Tau value for the scheduler. Default is 1.0.
        device (str): Device to run the model on. Default is "cuda".

    Methods:
        model(*args, **kwargs):
            Runs the preconditioning on the network with the provided arguments.

        __call__(...):
            Predicts geolocation coordinates from input images.

            Parameters:
                images: Input images to predict locations for.
                batch_size (int, optional): Batch size for processing.
                x_N (torch.Tensor, optional): Initial noise tensor. If not provided, it is generated.
                num_steps (int, optional): Number of steps for the sampler.
                scheduler (callable, optional): Custom scheduler function. If not provided, the default scheduler is used.
                cfg (float): Classifier-free guidance scale. Default is 0.
                generator (torch.Generator, optional): Random number generator.

            Returns:
                torch.Tensor: Predicted latitude and longitude coordinates.

        compute_likelihood(...):
            Computes the exact log-likelihood of observing the given coordinates for the given images.

            Parameters:
                images: Input images (PIL Image or list of PIL Images). Optional if emb is provided.
                coordinates: Target GPS coordinates (latitude, longitude) in degrees.
                emb: Pre-computed embeddings. If provided, images will be ignored.
                cfg (float): Classifier-free guidance scale. Default is 0 (no guidance).
                rademacher (bool): Whether to use Rademacher estimator for divergence. Default is False.
                atol (float): Absolute tolerance for ODE solver. Default is 1e-5.
                rtol (float): Relative tolerance for ODE solver. Default is 1e-5.
                normalize_logp (bool): Whether to normalize the log-likelihood by log(2) * dim. Default is True.

        compute_likelihood_grid(...):
            Computes the likelihood of an image over a global grid of coordinates.

            Parameters:
                image: Input PIL Image.
                grid_resolution_deg (float): The resolution of the grid in degrees. Default is 10 degrees.
                batch_size (int): How many grid points to process in each batch. Adjust based on available memory. Default is 1024.
                cfg (float): Classifier-free guidance scale passed to compute_likelihood. Default is 0.

            Returns:
                tuple: (latitude_grid, longitude_grid, likelihood_grid)
                    - latitude_grid (np.ndarray): 1D array of latitudes.
                    - longitude_grid (np.ndarray): 1D array of longitudes.
                    - likelihood_grid (np.ndarray): 2D array of log-likelihoods corresponding to the lat/lon grid.

        compute_localizability(...):
            Computes the localizability of an image. We use importance sampling by sampling by the model and not the grid to have a more accurate estimate.

            Parameters:
                image: Input PIL Image.
                atol (float): Absolute tolerance for ODE solver. Default is 1e-5.
                rtol (float): Relative tolerance for ODE solver. Default is 1e-5.
                number_monte_carlo_samples (int): How many samples to use for importance sampling. Default is 1024.

            Returns:
                torch.Tensor: Localizability of the image.

    Example Usage:
        pipe = PlonkPipeline(
            "path/to/plonk/model",
        )
        pipe.to("cuda")
        coordinates = pipe(
            images,
            batch_size=32
        )
        likelihood = pipe.compute_likelihood(
            images,
            coordinates,
            cfg=0,
            rademacher=False,
        )
        localizability = pipe.compute_localizability(
            image,
            number_monte_carlo_samples=1024,
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
        self.sampler = MODELS[model_path]["sampler"]
        self.model_path = model_path
        self.preconditioning = DDPMPrecond()
        self.device = device
        # Add manifold
        self.manifold = Sphere()
        self.input_dim = 3  # Assuming 3D Cartesian coordinates for sphere

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
            images: Conditioning input (image or list of images)
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

    def compute_likelihood(
        self,
        images=None,
        coordinates=None,
        emb=None,
        cfg=0,
        rademacher=False,
        atol=1e-6,
        rtol=1e-6,
        normalize_logp=True,
    ):
        """
        Computes the exact log-likelihood of observing the given coordinates for the given images.

        Args:
            images: Input images (PIL Image or list of PIL Images). Optional if emb is provided.
            coordinates: Target GPS coordinates (latitude, longitude) in degrees.
                         Can be a list of pairs, numpy array (N, 2), or tensor (N, 2).
            emb: Pre-computed embeddings. If provided, images will be ignored.
            cfg (float): Classifier-free guidance scale. Default is 0 (no guidance).
            rademacher (bool): Whether to use Rademacher estimator for divergence. Default is False.
            atol (float): Absolute tolerance for ODE solver. Default is 1e-5.
            rtol (float): Relative tolerance for ODE solver. Default is 1e-5.
            normalize_logp (bool): Whether to normalize the log-likelihood by log(2) * dim. Default is True.
        Returns:
            torch.Tensor: Log-likelihood values for each input pair (image, coordinate).
        """
        nfe = [0]  # Counter for number of function evaluations

        # 1. Get embeddings either from images or directly from emb parameter
        if emb is not None:
            # Use provided embeddings directly
            if isinstance(emb, torch.Tensor):
                batch = {"emb": emb.to(self.device)}
            else:
                raise TypeError("emb must be a torch.Tensor")
        else:
            # Process images to get embeddings
            if not isinstance(images, list):
                images = [images]
            batch = {"img": images}
            batch = self.cond_preprocessing(batch)  # Adds 'emb' key

        # 2. Preprocess coordinates (GPS degrees -> Cartesian)
        x_1 = _gps_degrees_to_cartesian(coordinates, self.device)
        if x_1.shape[0] != batch["emb"].shape[0]:
            if x_1.shape[0] == 1:
                # Repeat coordinates if only one is provided for multiple images
                x_1 = x_1.repeat(batch["emb"].shape[0], 1)
            elif batch["emb"].shape[0] == 1:
                # Repeat embedding if only one image is provided for multiple coords
                batch["emb"] = batch["emb"].repeat(x_1.shape[0], 1)
            else:
                raise ValueError(
                    f"Batch size mismatch between images ({batch['emb'].shape[0]}) and coordinates ({x_1.shape[0]})"
                )

        # Ensure correct shapes for ODE solver
        if x_1.ndim == 1:
            x_1 = x_1.unsqueeze(0)
        if batch["emb"].ndim == 1:
            batch["emb"] = batch["emb"].unsqueeze(0)

        with torch.inference_mode(mode=False):  # Enable grads for jacobian calculation
            # Define the ODE function
            def odefunc(t, tensor):
                nfe[0] += 1
                t = t.to(tensor)
                gamma = self.scheduler(t)
                x = tensor[..., : self.input_dim]
                y = batch["emb"]  # Conditioning

                def vecfield(x_vf, y_vf):
                    batch_vecfield = {
                        "y": x_vf,
                        "emb": y_vf,
                        "gamma": gamma.reshape(-1),
                    }
                    if cfg > 0:
                        # Apply classifier-free guidance
                        batch_vecfield_uncond = {
                            "y": x_vf,
                            "emb": torch.zeros_like(y_vf),  # Null condition
                            "gamma": gamma.reshape(-1),
                        }
                        model_output_cond = self.model(batch_vecfield)
                        model_output_uncond = self.model(batch_vecfield_uncond)
                        model_output = model_output_cond + cfg * (
                            model_output_cond - model_output_uncond
                        )
                    else:
                        # Unconditional or naturally conditioned score
                        model_output = self.model(batch_vecfield)

                    # Assuming 'flow_matching' interpolant based on sampler used
                    d_gamma = self.scheduler.derivative(t).reshape(-1, 1)
                    return d_gamma * model_output

                if rademacher:
                    v = torch.randint_like(x, 2) * 2 - 1
                else:
                    v = None
                dx, div = output_and_div(vecfield, x, y, v=v)
                div = div.reshape(-1, 1)
                del t, x
                return torch.cat([dx, div], dim=-1)

            # 3. Solve the ODE
            state1 = torch.cat([x_1, torch.zeros_like(x_1[..., :1])], dim=-1)

            # Note: Using standard ODEINT here. For strict Riemannian integration,
            # a manifold-aware solver might be needed, but this follows the
            # structure from DiffGeolocalizer.compute_exact_loglikelihood more closely.
            with torch.no_grad():
                state0 = odeint(
                    odefunc,
                    state1,
                    t=torch.linspace(0, 1.0, 2).to(x_1.device),
                    atol=atol,
                    rtol=rtol,
                    method="dopri5",
                    options={"min_step": 1e-5},
                )[
                    -1
                ]  # Get the state at t=0

        x_0, logdetjac = state0[..., : self.input_dim], state0[..., -1]

        # Project final point onto the manifold (optional but good practice)
        x_0 = self.manifold.projx(x_0)

        # 4. Compute log probability
        # Log prob of base distribution (Gaussian projected onto sphere approx)
        logp0 = self.manifold.base_logprob(x_0)

        # Change of variables formula: log p(x_1) = log p(x_0) + log |det J|
        logp1 = logp0 + logdetjac

        # Optional: Normalize by log(2) * dim for bits per dimension
        if normalize_logp:
            logp1 = logp1 / (self.input_dim * np.log(2))

        print(f"Likelihood NFE: {nfe[0]}")  # Print number of function evaluations
        return logp1

    def compute_likelihood_grid(
        self,
        image,
        grid_resolution_deg=10,
        batch_size=1024,
        cfg=0,
    ):
        """
        Computes the likelihood of an image over a global grid of coordinates.

        Args:
            image: Input PIL Image.
            grid_resolution_deg (float): The resolution of the grid in degrees.
                                        Default is 10 degrees.
            batch_size (int): How many grid points to process in each batch.
                             Adjust based on available memory. Default is 1024.
            cfg (float): Classifier-free guidance scale passed to compute_likelihood.
                        Default is 0.

        Returns:
            tuple: (latitude_grid, longitude_grid, likelihood_grid)
                - latitude_grid (np.ndarray): 1D array of latitudes.
                - longitude_grid (np.ndarray): 1D array of longitudes.
                - likelihood_grid (np.ndarray): 2D array of log-likelihoods
                                                corresponding to the lat/lon grid.
        """
        # 1. Generate the grid
        latitudes = np.arange(-90, 90 + grid_resolution_deg, grid_resolution_deg)
        longitudes = np.arange(-180, 180 + grid_resolution_deg, grid_resolution_deg)
        lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

        # Flatten the grid for processing
        all_coordinates = np.vstack([lat_grid.ravel(), lon_grid.ravel()]).T
        num_points = all_coordinates.shape[0]
        print(
            f"Computing likelihood over a {latitudes.size}x{longitudes.size} grid ({num_points} points)..."
        )

        emb = self.cond_preprocessing({"img": [image]})["emb"]

        # 2. Process in batches
        all_likelihoods = []
        for i in tqdm(
            range(0, num_points, batch_size), desc="Computing Likelihood Grid"
        ):
            coord_batch = all_coordinates[i : i + batch_size]

            # Compute likelihood for the batch
            likelihood_batch = self.compute_likelihood(
                emb=emb,
                coordinates=coord_batch,
                cfg=cfg,
                rademacher=False,  # Using exact divergence is better for grid
            )
            all_likelihoods.append(likelihood_batch.detach().cpu().numpy())

        # 3. Combine and reshape results
        likelihood_flat = np.concatenate(all_likelihoods, axis=0)
        likelihood_grid = likelihood_flat.reshape(lat_grid.shape)

        # Return grid definition and likelihood values
        return latitudes, longitudes, likelihood_grid

    def compute_localizability(
        self,
        image,
        atol=1e-6,
        rtol=1e-6,
        number_monte_carlo_samples=1024,
    ):
        """
        Computes the localizability of an image. We use importance sampling by sampling by the model and not the grid to have a more accurate estimate.

        Args:
            image: Input PIL Image.
            atol (float): Absolute tolerance for ODE solver. Default is 1e-5.
            rtol (float): Relative tolerance for ODE solver. Default is 1e-5.
        """
        samples = self(image, batch_size=number_monte_carlo_samples)
        emb = self.cond_preprocessing({"img": [image]})["emb"]
        localizability = self.compute_likelihood(
            emb=emb,
            coordinates=samples,
            atol=atol,
            rtol=rtol,
            normalize_logp=False,
        )  # importance sampling of likelihood
        return localizability.mean() / (4 * torch.pi * np.log(2))

    def to(self, device):
        self.network.to(device)
        self.postprocessing.to(device)
        self.device = torch.device(device)
        return self
