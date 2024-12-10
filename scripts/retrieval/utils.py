import os
import numpy as np
import reverse_geocoder


def get_loc(x):
    location = reverse_geocoder.search(x[0].tolist())[0]
    country = location.get("cc", "")
    region = location.get("admin1", "")
    sub_region = location.get("admin2", "")
    city = location.get("name", "")

    a = country if country != "" else None
    b, c, d = None, None, None
    if a is not None:
        b = country + "," + region if region != "" else None
        if b is not None:
            c = country + "," + region + "," + sub_region if sub_region != "" else None
            d = (
                country + "," + region + "," + sub_region + "," + city
                if city != ""
                else None
            )

    return a, b, c, d


def get_match_values(pred, gt, N, pos):
    xa, xb, xc, xd = get_loc(gt)
    ya, yb, yc, yd = get_loc(pred)

    if xa is not None:
        N["country"] += 1
        if xa == ya:
            pos["country"] += 1
        if xb is not None:
            N["region"] += 1
            if xb == yb:
                pos["region"] += 1
            if xc is not None:
                N["sub-region"] += 1
                if xc == yc:
                    pos["sub-region"] += 1
            if xd is not None:
                N["city"] += 1
                if xd == yd:
                    pos["city"] += 1


def compute_print_accuracy(N, pos):
    for k in N.keys():
        pos[k] /= N[k]

    # pretty-print accuracy in percentage with 2 floating points
    print(
        f'Accuracy: {pos["country"]*100.0:.2f} (country), {pos["region"]*100.0:.2f} (region), {pos["sub-region"]*100.0:.2f} (sub-region), {pos["city"]*100.0:.2f} (city)'
    )
    print(
        f'Haversine: {pos["haversine"]:.2f} (haversine), {pos["geoguessr"]:.2f} (geoguessr)'
    )


def get_filenames(idx):
    from autofaiss import build_index

    path = join(args.features_parent, f"features-{idx}/")
    files = [f for f in os.listdir(path)]
    full_files = [join(path, f) for f in os.listdir(path)]
    index = build_index(
        embeddings=np.concatenate([np.load(f) for f in tqdm(full_files)], axis=0),
        nb_cores=12,
        save_on_disk=False,
    )[0]
    return index, files


def normalize(x):
    lat, lon = x[:, 0], x[:, 1]
    """Used to put all lat lon inside ±90 and ±180."""
    lat = (lat + 90) % 360 - 90
    if lat > 90:
        lat = 180 - lat
        lon += 180
    lon = (lon + 180) % 360 - 180
    return np.stack([lat, lon], axis=1)


def haversine(pred, gt, N, p):
    # expects inputs to be np arrays in (lat, lon) format as radians
    # N x 2
    pred = np.radians(normalize(pred))
    gt = np.radians(normalize(gt))

    # calculate the difference in latitude and longitude between the predicted and ground truth points
    lat_diff = pred[:, 0] - gt[:, 0]
    lon_diff = pred[:, 1] - gt[:, 1]

    # calculate the haversine formula components
    lhs = np.sin(lat_diff / 2) ** 2
    rhs = np.cos(pred[:, 0]) * np.cos(gt[:, 0]) * np.sin(lon_diff / 2) ** 2
    a = lhs + rhs

    # calculate the final distance using the haversine formula
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    haversine_distance = 6371 * c[0]
    geoguessr_sum = 5000 * np.exp(-haversine_distance / 1492.7)

    N["geoguessr"] += 1
    p["geoguessr"] += geoguessr_sum

    N["haversine"] += 1
    p["haversine"] += haversine_distance
