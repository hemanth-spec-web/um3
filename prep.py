"""Data preparation for the Nassau Candy factory reallocation project.

Key step: the raw Ship Date column carries a large, order-date dependent offset
(raw order->ship gaps of 904-1642 days). The offset is removed by aligning each
of the three gap clusters to its base so that the residual 'true' lead time is
0-8 days and matches the ship-mode service windows.
"""
import numpy as np
import pandas as pd
import zipcodes

FACTORIES = {
    "Lot's O' Nuts": (32.881893, -111.768036),
    "Wicked Choccy's": (32.076176, -81.088371),
    "Sugar Shack": (48.11914, -96.18115),
    "Secret Factory": (41.446333, -90.565487),
    "The Other Factory": (35.1175, -89.971107),
}

PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows": "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious": "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate": "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy": "Sugar Shack",
    "SweeTARTS": "Sugar Shack",
    "Nerds": "Sugar Shack",
    "Fun Dip": "Sugar Shack",
    "Fizzy Lifting Drinks": "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Hair Toffee": "The Other Factory",
    "Lickable Wallpaper": "Secret Factory",
    "Wonka Gum": "Secret Factory",
    "Kazookles": "The Other Factory",
}

CANADA_CITIES = {
    "Calgary": (51.0447, -114.0719), "Charlottetown": (46.2382, -63.1311),
    "Edmonton": (53.5461, -113.4938), "Halifax": (44.6488, -63.5752),
    "Moncton": (46.0878, -64.7782), "Montreal": (45.5017, -73.5673),
    "Quebec City": (46.8139, -71.2080), "Regina": (50.4452, -104.6189),
    "St. John's": (47.5615, -52.7126), "Toronto": (43.6532, -79.3832),
    "Vancouver": (49.2827, -123.1207), "Winnipeg": (49.8951, -97.1384),
}

# raw order->ship gap bases (days) for the three offset clusters
GAP_BASES = [(0, 1000, 904), (1000, 1450, 1269), (1450, 3000, 1635)]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _geocode(row):
    if row["Country/Region"] == "Canada":
        return CANADA_CITIES[row["City"]]
    m = zipcodes.matching(str(row["Postal Code"]).zfill(5))
    return (float(m[0]["lat"]), float(m[0]["long"])) if m else (np.nan, np.nan)


def load_clean(path="data.csv"):
    df = pd.read_csv(path, dtype={"Postal Code": str})
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y")
    df["raw_gap"] = (df["Ship Date"] - df["Order Date"]).dt.days

    base = np.zeros(len(df))
    for lo, hi, b in GAP_BASES:
        base[(df.raw_gap >= lo) & (df.raw_gap < hi)] = b
    df["lead_time"] = df.raw_gap - base
    df["lt_flag_clipped"] = df.lead_time < 0
    df["lead_time"] = df.lead_time.clip(lower=0)

    # geography
    geo = {}
    keys = df[["Country/Region", "City", "Postal Code"]].drop_duplicates()
    for _, r in keys.iterrows():
        geo[(r["Country/Region"], r["City"], r["Postal Code"])] = _geocode(r)
    ll = [geo[(a, b, c)] for a, b, c in zip(df["Country/Region"], df["City"], df["Postal Code"])]
    df["dest_lat"] = [x[0] for x in ll]
    df["dest_lon"] = [x[1] for x in ll]

    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)
    for f, (la, lo) in FACTORIES.items():
        df[f"dist_{f}"] = haversine_km(la, lo, df.dest_lat, df.dest_lon)
    df["distance_km"] = [df.loc[i, f"dist_{f}"] for i, f in zip(df.index, df.Factory)]
    df["margin_pct"] = df["Gross Profit"] / df["Sales"]
    df["unit_cost"] = df["Cost"] / df["Units"]
    df["Route"] = df["Factory"] + " -> " + df["Region"]
    return df


def distance_to(df, factory):
    return df[f"dist_{factory}"].values


if __name__ == "__main__":
    d = load_clean("data.csv")
    print(d.lead_time.describe())
    print(pd.crosstab(d["Ship Mode"], d.lead_time).to_string())
    print("clipped:", d.lt_flag_clipped.sum(), "nan geo:", d.dest_lat.isna().sum())
    print(d.groupby("Factory").distance_km.mean().round(0))
