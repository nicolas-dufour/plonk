import torch
import reverse_geocoder
import numpy as np


def haversine(pred, gt):
    # expects inputs to be np arrays in (lat, lon) format as radians
    # N x 2

    # calculate the difference in latitude and longitude between the predicted and ground truth points
    lat_diff = pred[:, 0] - gt[:, 0]
    lon_diff = pred[:, 1] - gt[:, 1]

    # calculate the haversine formula components
    lhs = torch.sin(lat_diff / 2) ** 2
    rhs = torch.cos(pred[:, 0]) * torch.cos(gt[:, 0]) * torch.sin(lon_diff / 2) ** 2
    a = lhs + rhs

    # calculate the final distance using the haversine formula
    c = 2 * torch.arctan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = 6371 * c

    return distance

def haversine_np(pred, gt):
    # expects inputs to be np arrays in (lat, lon) format as radians
    # N x 2

    # calculate the difference in latitude and longitude between the predicted and ground truth points
    lat_diff = pred[0] - gt[0]
    lon_diff = pred[1] - gt[1]

    # calculate the haversine formula components
    lhs = np.sin(lat_diff / 2) ** 2
    rhs = np.cos(pred[0]) * np.cos(gt[0]) * np.sin(lon_diff / 2) ** 2
    a = lhs + rhs

    # calculate the final distance using the haversine formula
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = 6371 * c

    return distance


def reverse(pred, gt, area):
    df = {}
    gt_area = {}
    nan_mask = {}
    areas = ["_".join(["unique", ar]) for ar in area]
    if "unique_continent" in areas:
        areas.remove("unique_continent")
    for ar in areas:
        inter = np.array(gt[ar])
        nan_mask[ar] = inter != "nan"
        gt_area[ar] = inter[nan_mask[ar]]
    location = reverse_geocoder.search(
        [
            (lat, lon)
            for lat, lon in zip(
                np.degrees(pred[:, 0].cpu()), np.degrees(pred[:, 1].cpu())
            )
        ]
    )
    if "continent" in area:
        continent = torch.load("continent.pt")
        inter = np.array([l.get("cc", "") for l in location])[
            nan_mask["unique_country"]
        ]
        df["continent"] = np.array([continent[i] for i in inter])
        gt_area["unique_continent"] = np.array(
            [continent[i] for i in gt_area["unique_country"]]
        )

    if "country" in area:
        df["country"] = np.array([l.get("cc", "") for l in location])[
            nan_mask["unique_country"]
        ]
    if "region" in area:
        df["region"] = np.array(
            ["_".join([l.get("admin1", ""), l.get("cc", "")]) for l in location]
        )[nan_mask["unique_region"]]
    if "sub-region" in area:
        df["sub-region"] = np.array(
            [
                "_".join([l.get("admin2", ""), l.get("admin1", ""), l.get("cc", "")])
                for l in location
            ]
        )[nan_mask["unique_sub-region"]]
    if "city" in area:
        df["city"] = np.array(
            [
                "_".join(
                    [
                        l.get("name", ""),
                        l.get("admin2", ""),
                        l.get("admin1", ""),
                        l.get("cc", ""),
                    ]
                )
                for l in location
            ]
        )[nan_mask["unique_city"]]

    return df, gt_area
