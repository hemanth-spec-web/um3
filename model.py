"""Train and evaluate lead-time models, run ablations, save artifacts."""
import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prep import load_clean

OUT = "artifacts"
os.makedirs(OUT, exist_ok=True)
SEED = 42

CAT = ["Product Name", "Region", "Ship Mode"]
NUM = ["distance_km", "Units"]


def remove_outliers(df):
    """Extreme outliers: lead time beyond 3xIQR of its ship mode; Sales/Units beyond 4 robust z."""
    keep = pd.Series(True, index=df.index)
    for _, g in df.groupby("Ship Mode"):
        q1, q3 = g.lead_time.quantile([.25, .75])
        iqr = q3 - q1
        bad = (g.lead_time < q1 - 3 * iqr) | (g.lead_time > q3 + 3 * iqr)
        keep[g.index[bad]] = False
    n_lt = (~keep).sum()
    for col in ["Sales", "Units"]:
        x = np.log1p(df[col])
        med = x.median()
        mad = (x - med).abs().median() * 1.4826
        keep &= ((x - med).abs() / mad) < 4
    return df[keep].copy(), n_lt, int((~keep).sum() - n_lt)


def make_pipe(model, cat, num):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler(), num),
    ])
    return Pipeline([("pre", pre), ("m", model)])


def metrics(y, p):
    return dict(RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=float(mean_absolute_error(y, p)),
                R2=float(r2_score(y, p)))


def cv_eval(pipe_fn, X, y, groups, k=5):
    rows = []
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        p = pipe_fn()
        p.fit(X.iloc[tr], y.iloc[tr])
        rows.append(metrics(y.iloc[te], p.predict(X.iloc[te])))
    r = pd.DataFrame(rows)
    return {f"{c}_mean": float(r[c].mean()) for c in r} | {f"{c}_sd": float(r[c].std()) for c in r}


def main():
    raw = load_clean("data.csv")
    df, n_lt, n_val = remove_outliers(raw)
    print(f"rows {len(raw)} -> {len(df)} (lead-time outliers {n_lt}, sales/units outliers {n_val})")
    df.to_csv(f"{OUT}/clean.csv", index=False)

    X = df[CAT + NUM]
    y = df["lead_time"]
    g = df["Customer ID"]

    models = {
        "Linear Regression": lambda: make_pipe(LinearRegression(), CAT, NUM),
        "Random Forest": lambda: make_pipe(RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                                                                  random_state=SEED), CAT, NUM),
        "Gradient Boosting": lambda: make_pipe(GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                                          learning_rate=0.05, subsample=0.8,
                                                                          random_state=SEED), CAT, NUM),
    }

    # hold-out split grouped by customer
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, g))
    res = {}
    fitted = {}
    for name, fn in models.items():
        p = fn().fit(X.iloc[tr], y.iloc[tr])
        fitted[name] = p
        res[name] = {"holdout": metrics(y.iloc[te], p.predict(X.iloc[te])),
                     "cv": cv_eval(fn, X, y, g)}
        print(name, res[name]["holdout"], {k: round(v, 4) for k, v in res[name]["cv"].items() if "mean" in k})

    # naive benchmark: predict train mean
    res["Mean baseline"] = {"holdout": metrics(y.iloc[te], np.full(len(te), y.iloc[tr].mean()))}

    # ablations with Linear Regression and Gradient Boosting
    abl = {}
    sets = {
        "Ship mode only": (["Ship Mode"], []),
        "Ship mode + region": (["Ship Mode", "Region"], []),
        "Ship mode + product": (["Ship Mode", "Product Name"], []),
        "Ship mode + distance": (["Ship Mode"], ["distance_km"]),
        "Full feature set": (CAT, NUM),
    }
    for nm, (c, n) in sets.items():
        for mn, mk in [("LR", LinearRegression), ("GBR", lambda: GradientBoostingRegressor(
                n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED))]:
            r = cv_eval(lambda: make_pipe(mk(), c, n), df[c + n], y, g)
            abl[f"{nm} | {mn}"] = {"R2": r["R2_mean"], "RMSE": r["RMSE_mean"], "MAE": r["MAE_mean"]}
    print(pd.DataFrame(abl).T.round(4))

    # permutation importance on GBR (held-out)
    gb = fitted["Gradient Boosting"]
    pi = permutation_importance(gb, X.iloc[te], y.iloc[te], n_repeats=10, random_state=SEED,
                                scoring="neg_root_mean_squared_error")
    imp = pd.DataFrame({"feature": X.columns, "rmse_increase": pi.importances_mean,
                        "sd": pi.importances_std}).sort_values("rmse_increase", ascending=False)
    print(imp)

    # distance effect within ship mode: fixed-effects OLS (demeaned), cluster-robust by customer
    d = df.copy()
    d["lt_c"] = d.lead_time - d.groupby("Ship Mode").lead_time.transform("mean")
    d["dist_c"] = d.distance_km - d.groupby("Ship Mode").distance_km.transform("mean")
    d["dist_c"] /= 1000.0  # per 1000 km
    xx, yy = d.dist_c.values, d.lt_c.values
    beta = (xx * yy).sum() / (xx ** 2).sum()
    resid = yy - beta * xx
    # cluster-robust variance by customer
    clus = d["Customer ID"].values
    s = pd.Series(xx * resid).groupby(clus).sum()
    var = (s ** 2).sum() / ((xx ** 2).sum() ** 2)
    se = float(np.sqrt(var))
    pval = float(2 * (1 - stats.norm.cdf(abs(beta / se))))
    effect = dict(beta_days_per_1000km=float(beta), se=se, p_value=pval,
                  ci95=[float(beta - 1.96 * se), float(beta + 1.96 * se)], n=int(len(d)))
    print("distance effect:", effect)

    # per ship mode kruskal for region and factory
    tests = {}
    for m, gg in df.groupby("Ship Mode"):
        h = stats.kruskal(*[x.lead_time.values for _, x in gg.groupby("Region")])
        tests[m] = {"region_kruskal_p": float(h.pvalue)}
        f = stats.kruskal(*[x.lead_time.values for _, x in gg.groupby("Factory")])
        tests[m]["factory_kruskal_p"] = float(f.pvalue)
    print(tests)

    # selection: best accuracy, then simplest within tolerance
    cvr = {k: v["cv"]["RMSE_mean"] for k, v in res.items() if "cv" in v}
    best_acc = min(cvr, key=cvr.get)
    tol = 0.01  # within 1% RMSE of best
    cand = [k for k, v in cvr.items() if v <= cvr[best_acc] * (1 + tol)]
    order = ["Linear Regression", "Random Forest", "Gradient Boosting"]
    selected = next(k for k in order if k in cand)
    print("best accuracy:", best_acc, "selected:", selected)

    final = models[selected]().fit(X, y)
    joblib.dump(final, f"{OUT}/model.joblib")
    joblib.dump(fitted["Gradient Boosting"], f"{OUT}/model_gbr_holdout.joblib")

    json.dump({"models": res, "ablation": abl, "importance": imp.to_dict("records"),
               "distance_effect": effect, "group_tests": tests, "selected": selected,
               "best_accuracy": best_acc, "n_rows_raw": int(len(raw)), "n_rows_clean": int(len(df)),
               "outliers_leadtime": int(n_lt), "outliers_value": int(n_val),
               "clipped_negative_lt": int(raw.lt_flag_clipped.sum())},
              open(f"{OUT}/model_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
