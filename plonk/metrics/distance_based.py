import torch

from metrics.utils import haversine, reverse
from sklearn.metrics import pairwise_distances
from torchmetrics import Metric
import numpy as np
from plonk.utils.kde import BatchedKDE
from tqdm import tqdm


class HaversineMetrics(Metric):
    """
    Computes the average haversine distance between the predicted and ground truth points.
    Compute the accuracy given some radiuses.
    Compute the Geoguessr score given some radiuses.

    Args:
        acc_radiuses (list): list of radiuses to compute the accuracy from
        acc_area (list): list of areas to compute the accuracy from.
    """

    def __init__(
        self,
        acc_radiuses=[],
        acc_area=["country", "region", "sub-region", "city"],
        use_kde=False,
        manifold_k=3,
    ):
        super().__init__()
        self.use_kde = use_kde
        self.add_state("haversine_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("geoguessr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        for acc in acc_radiuses:
            self.add_state(
                f"close_enough_points_{acc}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
        for acc in acc_area:
            self.add_state(
                f"close_enough_points_{acc}",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
            self.add_state(
                f"count_{acc}", default=torch.tensor(0), dist_reduce_fx="sum"
            )
        self.acc_radius = acc_radiuses
        self.acc_area = acc_area
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "real_points",
            [],
            dist_reduce_fx=None,
        )
        self.add_state(
            "fake_points",
            [],
            dist_reduce_fx=None,
        )
        self.manifold_k = manifold_k

    def update(self, pred, gt):
        if self.use_kde:
            (x_mode, y_mode), kde = estimate_kde_mode(pred["gps"])
            # self.nll_sum += -torch.log(
            #     kde.score(gt["gps"].unsqueeze(1).to(pred["gps"].device))
            # ).sum()
            pred["gps"] = torch.stack([x_mode, y_mode], dim=1)
        # Handle NaN values without modifying the original inputs
        if pred["gps"].isnan().any():
            valid_mask = ~pred["gps"].isnan().any(dim=1)
            pred_gps = pred["gps"][valid_mask]
            gt_gps = gt["gps"][valid_mask]
            if len(pred_gps) == 0:  # Skip if no valid predictions remain
                return
        else:
            pred_gps = pred["gps"]
            gt_gps = gt["gps"]
        haversine_distance = haversine(pred_gps, gt_gps)
        for acc in self.acc_radius:
            self.__dict__[f"close_enough_points_{acc}"] += (
                haversine_distance < acc
            ).sum()
        if len(self.acc_area) > 0:
            area_pred, area_gt = reverse(pred_gps, gt, self.acc_area)
        for acc in self.acc_area:
            self.__dict__[f"close_enough_points_{acc}"] += (
                area_pred[acc] == area_gt["_".join(["unique", acc])]
            ).sum()
            self.__dict__[f"count_{acc}"] += len(area_gt["_".join(["unique", acc])])
        self.haversine_sum += haversine_distance.sum()
        self.geoguessr_sum += 5000 * torch.exp(-haversine_distance / 1492.7).sum()
        self.real_points.append(gt_gps)
        self.fake_points.append(pred_gps)
        self.count += pred_gps.shape[0]

    def compute(self):
        output = {
            "Haversine": self.haversine_sum / self.count,
            "Geoguessr": self.geoguessr_sum / self.count,
        }
        for acc in self.acc_radius:
            output[f"Accuracy_{acc}_km_radius"] = (
                self.__dict__[f"close_enough_points_{acc}"] / self.count
            )
        for acc in self.acc_area:
            output[f"Accuracy_{acc}"] = (
                self.__dict__[f"close_enough_points_{acc}"]
                / self.__dict__[f"count_{acc}"]
            )
        real_points = torch.cat(self.real_points, dim=0)
        fake_points = torch.cat(self.fake_points, dim=0)
        (
            output["precision"],
            output["recall"],
            output["density"],
            output["coverage"],
        ) = self.manifold_metrics(real_points, fake_points, self.manifold_k)
        return output

    def compute_pairwise_distance(self, data_x, data_y=None):
        """
        Args:
            data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
            data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Returns:
            numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
        """
        if data_y is None:
            data_y = data_x

        dists = pairwise_distances(data_x, data_y, metric="haversine", n_jobs=8)
        return dists

    def get_kth_value(self, unsorted, k, axis=-1):
        """
        Args:
            unsorted: numpy.ndarray of any dimensionality.
            k: int
        Returns:
            kth values along the designated axis.
        """
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values

    def compute_nearest_neighbour_distances(self, input_features, nearest_k):
        """
        Args:
            input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int
        Returns:
            Distances to kth nearest neighbours.
        """
        distances = self.compute_pairwise_distance(input_features)
        radii = self.get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii

    def compute_prdc(self, real_features, fake_features, nearest_k):
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int.
        Returns:
            dict of precision, recall, density, and coverage.
        """

        real_nearest_neighbour_distances = self.compute_nearest_neighbour_distances(
            real_features, nearest_k
        )
        fake_nearest_neighbour_distances = self.compute_nearest_neighbour_distances(
            fake_features, nearest_k
        )
        distance_real_fake = self.compute_pairwise_distance(
            real_features, fake_features
        )

        precision = (
            (
                distance_real_fake
                < np.expand_dims(real_nearest_neighbour_distances, axis=1)
            )
            .any(axis=0)
            .mean()
        )

        recall = (
            (
                distance_real_fake
                < np.expand_dims(fake_nearest_neighbour_distances, axis=0)
            )
            .any(axis=1)
            .mean()
        )

        density = (1.0 / float(nearest_k)) * (
            distance_real_fake
            < np.expand_dims(real_nearest_neighbour_distances, axis=1)
        ).sum(axis=0).mean()

        coverage = (
            distance_real_fake.min(axis=1) < real_nearest_neighbour_distances
        ).mean()

        return precision, recall, density, coverage

    def manifold_metrics(self, real_features, fake_features, nearest_k, num_splits=20):
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
            fake_features: torch.Tensor([N, feature_dim], dtype=torch.float32)
            nearest_k: int.
            num_splits: int. Number of splits to use for computing metrics.
        Returns:
            dict of precision, recall, density, and coverage.
        """
        real_features = real_features.chunk(num_splits, dim=0)
        fake_features = fake_features.chunk(num_splits, dim=0)
        precision, recall, density, coverage = [], [], [], []
        for real, fake in tqdm(
            zip(real_features, fake_features), desc="Computing manifold"
        ):
            p, r, d, c = self.compute_prdc(
                real.cpu().numpy(), fake.cpu().numpy(), nearest_k=nearest_k
            )
            precision.append(torch.tensor(p, device=real.device))
            recall.append(torch.tensor(r, device=real.device))
            density.append(torch.tensor(d, device=real.device))
            coverage.append(torch.tensor(c, device=real.device))
        return (
            torch.stack(precision).mean().item(),
            torch.stack(recall).mean().item(),
            torch.stack(density).mean().item(),
            torch.stack(coverage).mean().item(),
        )


def estimate_kde_mode(points):
    kde = BatchedKDE()
    kde.fit(points)
    batch_size = points.shape[0]
    X, Y, positions = batched_make_grid(points.cpu())
    X = X.to(points.device)
    Y = Y.to(points.device)
    positions = positions.to(points.device)
    Z = kde.score(positions).reshape(X.shape)

    x_mode = X.reshape(batch_size, -1)[
        torch.arange(batch_size), Z.reshape(batch_size, -1).argmax(dim=1)
    ]
    y_mode = Y.reshape(batch_size, -1)[
        torch.arange(batch_size), Z.reshape(batch_size, -1).argmax(dim=1)
    ]
    return (x_mode, y_mode), kde


def make_grid(points):
    (lat_min, long_min), _ = points.min(dim=-2)
    (lat_max, long_max), _ = points.max(dim=-2)
    x = torch.linspace(lat_min, lat_max, 100)
    y = torch.linspace(long_min, long_max, 100)
    X, Y = torch.meshgrid(x, y)
    positions = torch.vstack([X.flatten(), Y.flatten()]).transpose(-1, -2)
    return X, Y, positions


batched_make_grid = torch.vmap(make_grid)
