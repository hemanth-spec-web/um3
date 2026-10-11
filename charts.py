"""Figures for the research paper (light mode, validated default palette slots 1-3)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from prep import FACTORIES, load_clean

A = "artifacts"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True,
    "font.size": 9, "legend.frameon": False, "figure.dpi": 150,
})

SHORT = {"Wonka Bar - Nutty Crunch Surprise": "Nutty Crunch", "Wonka Bar - Fudge Mallows": "Fudge Mallows",
         "Wonka Bar -Scrumdiddlyumptious": "Scrumdiddly.", "Wonka Bar - Milk Chocolate": "Milk Choc.",
         "Wonka Bar - Triple Dazzle Caramel": "Triple Dazzle"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    raw = load_clean("data.csv")
    df = pd.read_csv(f"{A}/clean.csv")
    res = json.load(open(f"{A}/model_results.json"))
    df["lt_adj"] = df.lead_time - df.groupby("Ship Mode").lead_time.transform("mean")

    # 1 offset correction
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].hist(raw.raw_gap, bins=60, color=BLUE)
    ax[0].set_title("Raw order-to-ship gap (days)")
    ax[0].set_xlabel("days"); ax[0].set_ylabel("orders")
    ax[1].hist(raw.lead_time, bins=np.arange(-.5, 12.5, 1), color=BLUE, rwidth=.85)
    ax[1].set_title("After offset correction (days)")
    ax[1].set_xlabel("days")
    save(fig, "fig1_offset")

    # 2 lead time by ship mode + region flatness
    order = ["Same Day", "First Class", "Second Class", "Standard Class"]
    m = df.groupby("Ship Mode").lead_time.agg(["mean", "std", "size"]).loc[order]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].bar(order, m["mean"], color=BLUE, width=.55)
    for i, v in enumerate(m["mean"]): ax[0].text(i, v + .08, f"{v:.2f}", ha="center", fontsize=8)
    ax[0].set_title("Mean lead time by ship mode (days)")
    ax[0].tick_params(axis="x", labelsize=8)
    reg = df.groupby("Region").lt_adj.agg(["mean", "std", "size"])
    reg["ci"] = 1.96 * reg["std"] / np.sqrt(reg["size"])
    ax[1].bar(reg.index, reg["mean"], yerr=reg["ci"], color=BLUE, width=.5, ecolor=INK2, capsize=3)
    ax[1].axhline(0, color=INK2, lw=.8)
    ax[1].set_title("Ship-mode-adjusted lead time by region (days, 95% CI)")
    ax[1].set_ylim(-.15, .15)
    save(fig, "fig2_mode_region")

    # 3 distance vs adjusted lead time (deciles)
    df["bin"] = pd.qcut(df.distance_km, 10)
    b = df.groupby("bin", observed=True).agg(x=("distance_km", "mean"), y=("lt_adj", "mean"),
                                             s=("lt_adj", "std"), n=("lt_adj", "size"))
    b["ci"] = 1.96 * b.s / np.sqrt(b.n)
    eff = res["distance_effect"]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.errorbar(b.x, b.y, yerr=b.ci, fmt="o", color=BLUE, ecolor=BLUE, capsize=3, ms=5)
    xs = np.linspace(b.x.min(), b.x.max(), 50)
    ax.plot(xs, eff["beta_days_per_1000km"] * (xs - df.distance_km.mean()) / 1000, color=ORANGE, lw=2)
    ax.axhline(0, color=INK2, lw=.8)
    ax.set_xlabel("distance factory to customer (km), decile means")
    ax.set_ylabel("adjusted lead time (days)")
    ax.set_title(f"Distance has no measurable effect (slope {eff['beta_days_per_1000km']:.3f} d/1000 km, p={eff['p_value']:.2f})")
    ax.title.set_fontsize(9.5)
    save(fig, "fig3_distance")

    # 4 model comparison + ablation
    names = ["Linear Regression", "Random Forest", "Gradient Boosting"]
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.2))
    rm = [res["models"][n]["cv"]["RMSE_mean"] for n in names]
    ax[0].bar(["Linear\nRegression", "Random\nForest", "Gradient\nBoosting"], rm, color=BLUE, width=.5)
    for i, v in enumerate(rm): ax[0].text(i, v + .01, f"{v:.3f}", ha="center", fontsize=8)
    ax[0].set_ylim(0, 1.25); ax[0].set_title("5-fold grouped CV RMSE (days)")
    ab = res["ablation"]
    keys = ["Ship mode only", "Ship mode + region", "Ship mode + product", "Ship mode + distance", "Full feature set"]
    y = np.arange(len(keys))[::-1]
    ax[1].scatter([ab[f"{k} | LR"]["R2"] for k in keys], y, color=BLUE, s=36, label="Linear Regression", zorder=3)
    ax[1].scatter([ab[f"{k} | GBR"]["R2"] for k in keys], y, color=ORANGE, s=36, label="Gradient Boosting", zorder=3)
    ax[1].set_yticks(y); ax[1].set_yticklabels(keys, fontsize=8)
    ax[1].set_xlim(.65, .665); ax[1].set_xlabel("CV R-squared")
    ax[1].set_title("Adding features does not help")
    ax[1].legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2, fontsize=8)
    save(fig, "fig4_models")

    # 5 mean distance if all orders shipped from each factory
    dm = pd.Series({f: df[f"dist_{f}"].mean() for f in FACTORIES}).sort_values()
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(dm.index[::-1], dm.values[::-1], color=BLUE, height=.55)
    for i, v in enumerate(dm.values[::-1]): ax.text(v + 15, i, f"{v:,.0f} km", va="center", fontsize=8)
    ax.set_xlim(0, dm.max() * 1.15)
    ax.set_title("Average customer distance if all orders shipped from each factory")
    ax.title.set_fontsize(9.5)
    save(fig, "fig5_factory_distance")

    # 6 routes scatter
    r = pd.read_csv(f"{A}/route_clusters.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ns = r[~r.slow_significant]; sg = r[r.slow_significant]
    ax.scatter(ns.dist, ns.lt_adj, s=np.sqrt(ns.orders) * 4, color=BLUE, alpha=.55, label="Not significant", edgecolors=SURF, linewidths=1)
    ax.scatter(sg.dist, sg.lt_adj, s=np.sqrt(sg.orders) * 4, color=ORANGE, label="Slow (FDR 10%)", edgecolors=SURF, linewidths=1)
    ax.axhline(0, color=INK2, lw=.8)
    for _, row in sg.iterrows():
        ax.annotate(f"{row['Factory'].split()[0].replace(chr(39), '')}-{row['State/Province']}", (row.dist, row.lt_adj), fontsize=7,
                    xytext=(4, 4), textcoords="offset points", color=INK2)
    ax.set_xlabel("average distance (km)"); ax.set_ylabel("adjusted lead time (days)")
    ax.set_title("Route performance (factory x state, n>=20); bubble = orders")
    ax.title.set_fontsize(9.5); ax.legend(fontsize=8)
    save(fig, "fig6_routes")

    # 7 best recommendations
    b = pd.read_csv(f"{A}/best_per_product.csv")
    b = b[b.Orders >= 100].copy()
    b["lbl"] = b["Product Name"].map(SHORT) + "\n-> " + b["Alt Factory"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].bar(b.lbl, b.dist_reduction_pct, color=BLUE, width=.55)
    for i, v in enumerate(b.dist_reduction_pct): ax[0].text(i, v + .5, f"{v:.0f}%", ha="center", fontsize=8)
    ax[0].set_title("Distance reduction"); ax[0].tick_params(axis="x", labelsize=7)
    ax[1].bar(b.lbl, b.profit_impact, color=AQUA, width=.55)
    for i, v in enumerate(b.profit_impact): ax[1].text(i, v + 3, f"${v:,.0f}", ha="center", fontsize=8)
    ax[1].set_title("Profit impact (USD, freight assumption)"); ax[1].tick_params(axis="x", labelsize=7)
    save(fig, "fig7_recs")
    print("figures done", os.listdir(FIG))


if __name__ == "__main__":
    main()
