"""Scenario simulation + recommendation engine (shared by the batch build and the Streamlit app)."""
import json

import joblib
import numpy as np
import pandas as pd

from prep import FACTORIES

OUT = "artifacts"
LONG_HAUL_KM = 2000.0
DEFAULT_FREIGHT = 0.04  # USD per unit per 1,000 km  (ASSUMPTION - not in dataset)

GROUP = ["Product Name", "Region", "Ship Mode", "Current Factory", "Alt Factory"]


def build_cube(df, model):
    """Counterfactual: re-assign every historical order to every factory, re-predict lead time."""
    X = df[["Product Name", "Region", "Ship Mode", "distance_km", "Units"]].copy()
    cur_pred = model.predict(X)
    # observed distance support range per product (P1-P99 of its current-factory distances)
    lo = df.groupby("Product Name").distance_km.quantile(.01)
    hi = df.groupby("Product Name").distance_km.quantile(.99)
    parts = []
    for f in FACTORIES:
        Xa = X.copy()
        Xa["distance_km"] = df[f"dist_{f}"].values
        alt_pred = model.predict(Xa)
        d_alt = df[f"dist_{f}"].values
        t = pd.DataFrame({
            "Product Name": df["Product Name"].values, "Region": df.Region.values, "Ship Mode": df["Ship Mode"].values,
            "Current Factory": df.Factory.values, "Alt Factory": f,
            "n": 1, "units": df.Units.values, "sales": df.Sales.values, "gp": df["Gross Profit"].values,
            "sum_pred_cur": cur_pred, "sum_pred_alt": alt_pred,
            "sum_dist_cur": df.distance_km.values, "sum_dist_alt": d_alt,
            "sum_ud_cur": df.Units.values * df.distance_km.values, "sum_ud_alt": df.Units.values * d_alt,
            "n_closer": (d_alt <= df.distance_km.values).astype(int),
            "n_long_cur": (df.distance_km.values > LONG_HAUL_KM).astype(int),
            "n_long_alt": (d_alt > LONG_HAUL_KM).astype(int),
            "n_support": ((d_alt >= df["Product Name"].map(lo).values) &
                          (d_alt <= df["Product Name"].map(hi).values)).astype(int),
        })
        parts.append(t)
    cube = pd.concat(parts).groupby(GROUP, as_index=False).sum(numeric_only=True)
    return cube


def load_effect():
    return json.load(open(f"{OUT}/model_results.json"))["distance_effect"]


def score_scenarios(cube, regions=None, modes=None, products=None, freight=DEFAULT_FREIGHT,
                    w_speed=0.5, effect=None, include_current=False):
    """Aggregate the cube under the selected filters and score each (product, alt factory)."""
    c = cube
    if regions: c = c[c.Region.isin(regions)]
    if modes: c = c[c["Ship Mode"].isin(modes)]
    if products: c = c[c["Product Name"].isin(products)]
    if c.empty:
        return (pd.DataFrame(), pd.DataFrame()) if include_current else pd.DataFrame()
    effect = effect or load_effect()
    g = c.groupby(["Product Name", "Current Factory", "Alt Factory"], as_index=False).sum(numeric_only=True)
    n = g.n
    g["Orders"] = n
    g["lt_cur"] = g.sum_pred_cur / n
    g["lt_alt"] = g.sum_pred_alt / n
    g["lt_change_days"] = g.lt_alt - g.lt_cur
    g["lt_reduction_pct"] = (g.lt_cur - g.lt_alt) / g.lt_cur * 100
    g["dist_cur"] = g.sum_dist_cur / n
    g["dist_alt"] = g.sum_dist_alt / n
    g["dist_reduction_pct"] = (g.dist_cur - g.dist_alt) / g.dist_cur * 100
    g["freight_delta"] = freight * (g.sum_ud_alt - g.sum_ud_cur) / 1000.0
    g["profit_impact"] = -g.freight_delta
    g["profit_impact_pct"] = g.profit_impact / g.gp * 100
    g["profit_stability_pct"] = g.n_closer / n * 100
    g["long_haul_cur_pct"] = g.n_long_cur / n * 100
    g["long_haul_alt_pct"] = g.n_long_alt / n * 100
    g["risk_reduction_pp"] = g.long_haul_cur_pct - g.long_haul_alt_pct
    g["confidence"] = g.n_support / n * np.sqrt(np.minimum(n, 100) / 100) * 100
    dd = (g.dist_alt - g.dist_cur) / 1000.0
    b, se = effect["beta_days_per_1000km"], effect["se"]
    g["lt_ci_low"] = (b - 1.96 * se) * dd
    g["lt_ci_high"] = (b + 1.96 * se) * dd
    g["lt_ci_low"], g["lt_ci_high"] = (g[["lt_ci_low", "lt_ci_high"]].min(axis=1),
                                       g[["lt_ci_low", "lt_ci_high"]].max(axis=1))
    g["lt_significant"] = (g.lt_ci_low > 0) | (g.lt_ci_high < 0)

    rg = c.groupby(["Product Name", "Current Factory", "Alt Factory", "Region"], as_index=False).sum(numeric_only=True)
    rg["chg"] = (rg.sum_dist_alt / rg.sum_dist_cur - 1) * 100
    worst = rg.loc[rg.groupby(["Product Name", "Current Factory", "Alt Factory"]).chg.idxmax()]
    worst = worst.rename(columns={"Region": "worst_region", "chg": "worst_region_change_pct"})[
        ["Product Name", "Current Factory", "Alt Factory", "worst_region", "worst_region_change_pct"]]
    g = g.merge(worst, on=["Product Name", "Current Factory", "Alt Factory"], how="left")
    cand = g[g["Alt Factory"] != g["Current Factory"]].copy()
    if not cand.empty:
        def norm(s):
            m = s.abs().max()
            return s / m if m > 0 else s * 0
        cand["score"] = 100 * (0.2 * norm(cand.risk_reduction_pp) +
                               0.8 * (w_speed * norm(cand.lt_reduction_pct) +
                                      (1 - w_speed) * norm(cand.profit_impact)))
        cand["risk_level"] = np.select(
            [(cand.profit_impact < 0) | (cand.risk_reduction_pp < -5),
             (cand.profit_stability_pct < 60) | (cand.Orders < 100) | (cand.worst_region_change_pct > 50)],
            ["High", "Medium"], default="Low")
    if include_current:
        return g, cand
    return cand


def regional_tradeoff(cube, product, alt, modes=None, freight=DEFAULT_FREIGHT):
    c = cube[(cube["Product Name"] == product) & (cube["Alt Factory"] == alt)]
    if modes: c = c[c["Ship Mode"].isin(modes)]
    r = c.groupby("Region", as_index=False).sum(numeric_only=True)
    r["dist_cur"] = r.sum_dist_cur / r.n
    r["dist_alt"] = r.sum_dist_alt / r.n
    r["dist_change_pct"] = (r.dist_alt - r.dist_cur) / r.dist_cur * 100
    r["profit_impact"] = -freight * (r.sum_ud_alt - r.sum_ud_cur) / 1000.0
    return r[["Region", "n", "dist_cur", "dist_alt", "dist_change_pct", "profit_impact"]]


def recommend(cand, top_n=10, min_conf=0.0):
    """Top-N reassignments: positive profit impact + positive score, not high risk."""
    r = cand[(cand.profit_impact > 0) & (cand.score > 0) & (cand.risk_level != "High") &
             (cand.confidence >= min_conf)]
    return r.sort_values("score", ascending=False).head(top_n)


def kpis(cand, rec):
    products = cand["Product Name"].nunique()
    best = cand.sort_values("score", ascending=False).groupby("Product Name").head(1)
    rec_prod = rec["Product Name"].nunique()
    vol = cand.groupby("Product Name").Orders.max()
    cover_vol = vol[vol.index.isin(rec["Product Name"].unique())].sum() / vol.sum() * 100 if len(vol) else 0
    w = rec.Orders if len(rec) else pd.Series(dtype=float)
    return {
        "lead_time_reduction_pct": float((rec.lt_reduction_pct * w).sum() / w.sum()) if len(rec) else 0.0,
        "profit_stability_pct": float((rec.profit_stability_pct * w).sum() / w.sum()) if len(rec) else 0.0,
        "confidence": float((rec.confidence * w).sum() / w.sum()) if len(rec) else 0.0,
        "coverage_products_pct": rec_prod / products * 100 if products else 0.0,
        "coverage_volume_pct": float(cover_vol),
    }


def split_sourcing(cube, regions=None, modes=None, products=None, freight=DEFAULT_FREIGHT):
    """Upper-bound analysis: serve each product-region from its nearest factory (needs multi-site production)."""
    c = cube
    if regions: c = c[c.Region.isin(regions)]
    if modes: c = c[c["Ship Mode"].isin(modes)]
    if products: c = c[c["Product Name"].isin(products)]
    if c.empty: return pd.DataFrame(), pd.DataFrame()
    pr = c.groupby(["Product Name", "Current Factory", "Region", "Alt Factory"], as_index=False).sum(numeric_only=True)
    pr["md"] = pr.sum_dist_alt / pr.n
    pick = pr.loc[pr.groupby(["Product Name", "Region"]).md.idxmin()].copy()
    pick["profit_impact"] = -freight * (pick.sum_ud_alt - pick.sum_ud_cur) / 1000.0
    pick["lt_pred_cur"] = pick.sum_pred_cur
    prod = pick.groupby(["Product Name", "Current Factory"], as_index=False).agg(
        Orders=("n", "sum"), d_cur=("sum_dist_cur", "sum"), d_new=("sum_dist_alt", "sum"),
        profit_impact=("profit_impact", "sum"), gp=("gp", "sum"))
    prod["dist_cur"] = prod.d_cur / prod.Orders
    prod["dist_split"] = prod.d_new / prod.Orders
    prod["dist_reduction_pct"] = (1 - prod.dist_split / prod.dist_cur) * 100
    prod["profit_impact_pct"] = prod.profit_impact / prod.gp * 100
    return prod, pick[["Product Name", "Region", "Alt Factory", "n", "md"]].rename(
        columns={"Alt Factory": "Nearest factory", "n": "Orders", "md": "Avg distance (km)"})
