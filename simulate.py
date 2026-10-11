"""Batch build: scenario cube, route/product clustering, recommendation tables."""
import json

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import engine
from prep import FACTORIES

OUT = "artifacts"


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def clustering(df):
    d = df.copy()
    d["lt_adj"] = d.lead_time - d.groupby("Ship Mode").lead_time.transform("mean")

    # --- route = factory x state/province
    r = d.groupby(["Factory", "State/Province"]).agg(
        orders=("lead_time", "size"), lt_mean=("lead_time", "mean"), lt_std=("lead_time", "std"),
        lt_adj=("lt_adj", "mean"), lt_adj_sd=("lt_adj", "std"), dist=("distance_km", "mean"),
        margin=("margin_pct", "mean"), sales=("Sales", "sum")).reset_index()
    yr = d.assign(year=pd.to_datetime(d["Order Date"]).dt.year).pivot_table(
        index=["Factory", "State/Province"], columns="year", values="lt_adj", aggfunc="mean")
    yr.columns = [f"lt_adj_{c}" for c in yr.columns]
    r = r.merge(yr.reset_index(), on=["Factory", "State/Province"], how="left")
    r = r[r.orders >= 20].copy()
    r["consistent_slow"] = (r.lt_adj_2024 > 0) & (r.lt_adj_2025 > 0) & (r.lt_adj > 0)
    r["se"] = r.lt_adj_sd / np.sqrt(r.orders)
    r["t"] = r.lt_adj / r.se
    r["p"] = 2 * (1 - stats.t.cdf(np.abs(r.t), r.orders - 1))
    r["q_fdr"] = bh_fdr(r.p.values)
    r["slow_significant"] = (r.q_fdr < 0.10) & (r.lt_adj > 0)
    feats = ["lt_mean", "lt_std", "lt_adj", "dist", "margin"]
    Z = StandardScaler().fit_transform(r[feats].fillna(0))
    best = (None, -1, None)
    sil = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, n_init=20, random_state=42).fit(Z)
        s = silhouette_score(Z, km.labels_)
        sil[k] = float(s)
        if s > best[1]: best = (k, s, km)
    r["cluster"] = best[2].labels_
    prof = r.groupby("cluster").agg(routes=("orders", "size"), orders=("orders", "sum"), lt_mean=("lt_mean", "mean"),
                                    lt_adj=("lt_adj", "mean"), dist=("dist", "mean"), margin=("margin", "mean")).round(3)
    r.to_csv(f"{OUT}/route_clusters.csv", index=False)
    prof.to_csv(f"{OUT}/route_cluster_profile.csv")

    # --- congestion: region x product
    c = d.groupby(["Region", "Product Name"]).agg(
        orders=("lead_time", "size"), units=("Units", "sum"), lt_mean=("lead_time", "mean"),
        lt_adj=("lt_adj", "mean"), lt_adj_sd=("lt_adj", "std"), dist=("distance_km", "mean")).reset_index()
    c = c[c.orders >= 20].copy()
    c["se"] = c.lt_adj_sd / np.sqrt(c.orders)
    c["p"] = 2 * (1 - stats.t.cdf(np.abs(c.lt_adj / c.se), c.orders - 1))
    c["q_fdr"] = bh_fdr(c.p.values)
    vol_cut = c.orders.quantile(.75)
    c["high_volume"] = c.orders >= vol_cut
    c["congested"] = c.high_volume & (c.lt_adj > 0)
    c["slow_significant"] = (c.q_fdr < 0.10) & (c.lt_adj > 0)
    c.to_csv(f"{OUT}/congestion.csv", index=False)
    info = {"k": int(best[0]), "silhouette": float(best[1]), "silhouette_by_k": sil,
            "n_routes": int(len(r)), "n_slow_sig_routes": int(r.slow_significant.sum()),
            "n_slow_consistent": int((r.slow_significant & r.consistent_slow).sum()),
            "n_congested": int(c.congested.sum()), "n_slow_sig_combos": int(c.slow_significant.sum()),
            "n_combos": int(len(c))}
    return r, c, info


def main():
    df = pd.read_csv(f"{OUT}/clean.csv")
    model = joblib.load(f"{OUT}/model.joblib")
    cube = engine.build_cube(df, model)
    cube.to_csv(f"{OUT}/scenario_cube.csv", index=False)
    print("cube", cube.shape)

    g, cand = engine.score_scenarios(cube, w_speed=0.5, include_current=True)
    rec_all = engine.recommend(cand, top_n=999)
    rec = rec_all.head(15)
    best = rec_all.groupby("Product Name").head(1)
    cand.sort_values("score", ascending=False).to_csv(f"{OUT}/scenarios_all.csv", index=False)
    rec.to_csv(f"{OUT}/recommendations.csv", index=False)
    best.to_csv(f"{OUT}/best_per_product.csv", index=False)
    k = engine.kpis(cand, best)
    print(json.dumps(k, indent=1))
    show = ["Product Name", "Current Factory", "Alt Factory", "Orders", "dist_cur", "dist_alt", "dist_reduction_pct",
            "lt_reduction_pct", "lt_ci_low", "lt_ci_high", "profit_impact", "profit_impact_pct", "profit_stability_pct",
            "risk_reduction_pp", "confidence", "risk_level", "score"]
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(rec[show].round(3).to_string())

    r, c, info = clustering(df)
    print(info)
    print(pd.read_csv(f"{OUT}/route_cluster_profile.csv").to_string())
    print(r[r.slow_significant][["Factory", "State/Province", "orders", "lt_adj", "lt_adj_2024", "lt_adj_2025", "q_fdr", "consistent_slow"]])
    print(best[["Product Name", "Alt Factory", "Orders", "dist_reduction_pct", "profit_impact", "confidence", "risk_level"]].round(2).to_string())
    print(c.sort_values("lt_adj", ascending=False).head(8))
    json.dump({"kpis": k, "cluster": info}, open(f"{OUT}/sim_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
