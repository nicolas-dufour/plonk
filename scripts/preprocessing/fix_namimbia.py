from os.path import join, dirname
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Define the list of cities
    cities = [
        "Walvis Bay",
        "Keetmanshoop",
        "Warmbad",
        "Rundu",
        "Outapi",
        "Karibib",
        "Otjimbingwe",
        "Ondangwa",
        "Oranjemund",
        "Maltahohe",
        "Otavi",
        "Outjo",
        "Swakopmund",
        "Gobabis",
        "Karasburg",
        "Opuwo",
        "Hentiesbaai",
        "Katima Mulilo",
        "Oshikango",
        "Bethanie",
        "Ongandjera",
        "Mariental",
        "Bagani",
        "Nkurenkuru",
        "Usakos",
        "Rehoboth",
        "Aranos",
        "Omaruru",
        "Arandis",
        "Windhoek",
        "Khorixas",
        "Okahandja",
        "Grootfontein",
        "Tsumeb",
    ]

    csv_dtype = {"category": str, "country": str, "city": str}
    for split in ["train", "test"]:
        fp = join(
            dirname(dirname(__file__)), "datasets", "osv5m", f"{split}.csv"
        )

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(fp, dtype=csv_dtype)

        # Check if the "country" column contains any of the cities in the list
        mask = df["city"].isin(cities)

        # If a city is found, set the corresponding rows in the "country" column to 'NMB'
        df.loc[mask, "country"] = "NMB"
        assert all(map(lambda x: isinstance(x, str), df["country"].unique().tolist()))

        # Drop the columns that are all NaN
        df.dropna(subset=["id", "latitude", "longitude"], inplace=True)

        # Save the modified DataFrame back to the CSV file
        df.to_csv(fp, index=False)
