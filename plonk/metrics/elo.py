import os
import torch
from metrics.utils import haversine

from torchmetrics import Metric


class HaversineELOMetric(Metric):
    """
    Computes the ELO score of the current network given previous players

    Args:
        previous_players_scores (str): path to the csv containing the scores of the previous players
        previous_players_predictions (str): path to the folder containing the predictions of the previous players
        tag (str): tag of the current experiment

    """

    def __init__(self, cache_folder, tag):
        ### TODO
        pass
