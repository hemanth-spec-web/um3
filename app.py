"""Nassau Candy - Factory Reallocation & Shipping Optimization (Streamlit).

Run:  streamlit run app.py
Requires the artifacts/ folder produced by model.py and simulate.py.
"""
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import engine
from prep import FACTORIES

A = "artifacts"
BLUE, ORANGE, AQUA, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"

st.set_page_config(page_title="Nassau Candy - Factory Optimization", layout="wide")


@st.cache_data
def load():
    cube = pd.read_csv(f"{A}/scenario_cube.csv")
    clean = pd.read_csv(f"{A}/clean.csv")
    routes = pd.read_csv(f"{A}/route_clusters.csv")
    cong = pd.read_csv(f"{A}/congestion.csv")
    res = json.load(open(f"{A}/model_results.json"))
    return cube, clean, routes, cong, res


cube, clean, routes, cong, res = load()
eff = res["distance_effect"]
ALL_PRODUCTS = sorted(cube["Product Name"].unique())
REGIONS = sorted(cube.Region.unique())
MODES = ["Same Day", "First Class", "Second Class", "Standard Class"]

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Controls")
region_sel = st.sidebar.multiselect("Destination region", REGIONS, default=REGIONS)
mode_sel = st.sidebar.multiselect("Ship mode", MODES, default=MODES)
prio = st.sidebar.slider("Optimization priority", 0, 100, 50, help="0 = maximize profit, 100 = maximize speed")
freight = st.sidebar.slider("Freight cost assumption (USD per unit per 1,000 km)", 0.0, 0.20, engine.DEFAULT_FREIGHT, 0.01,
                            help="Not in the dataset. Profit impact scales linearly with this assumption.")
top_n = st.sidebar.slider("Top-N recommendations", 3, 30, 10)
min_conf = st.sidebar.slider("Minimum scenario confidence", 0, 100, 50)
st.sidebar.caption(f"Speed {prio}% / Profit {100 - prio}%")

if not region_sel or not mode_sel:
    st.warning("Select at least one region and one ship mode.")
    st.stop()

cand = engine.score_scenarios(cube, regions=region_sel, modes=mode_sel, freight=freight,
                              w_speed=prio / 100, effect=eff)
if cand.empty:
    st.warning("No orders match the selected region and ship mode combination.")
    st.stop()
rec_all = engine.recommend(cand, top_n=999, min_conf=min_conf)
rec = rec_all.head(top_n)
best = rec_all.groupby("Product Name").head(1)
k = engine.kpis(cand, best)

st.title("Factory Reallocation & Shipping Optimization")
st.caption("Nassau Candy Distributor - decision intelligence for product-to-factory assignment")

if not (eff["ci95"][0] > 0 or eff["ci95"][1] < 0):
    st.info(
        f"Evidence note: across {eff['n']:,} orders, distance explains no measurable lead-time variation once ship mode is "
        f"accounted for ({eff['beta_days_per_1000km']:+.3f} days per 1,000 km, 95% CI "
        f"{eff['ci95'][0]:+.3f} to {eff['ci95'][1]:+.3f}). Recommendations are therefore driven by distance, freight cost and "
        f"long-haul exposure, not by a proven lead-time gain.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Factory Optimization Simulator", "What-If Scenario Analysis",
                                        "Recommendation Dashboard", "Risk & Impact Panel", "Route Insights"])

# ------------------------------------------------------------------ tab 1
with tab1:
    prod = st.selectbox("Product", ALL_PRODUCTS, key="p1")
    g, c = engine.score_scenarios(cube, regions=region_sel, modes=mode_sel, products=[prod], freight=freight,
                                  w_speed=prio / 100, effect=eff, include_current=True)
    if g.empty:
        st.info("This product has no orders under the selected region and ship mode filters.")
        st.stop()
    cur = g["Current Factory"].iloc[0]
    st.write(f"Currently produced at: **{cur}**  |  orders in selection: **{int(g.Orders.iloc[0]):,}**")
    t = g.rename(columns={"Alt Factory": "Factory"}).copy()
    t["Status"] = t.Factory.apply(lambda x: "Current" if x == cur else "Alternative")
    c1, c2 = st.columns(2)
    fig = px.bar(t, x="Factory", y="dist_alt", color="Status",
                 color_discrete_map={"Current": BLUE, "Alternative": "#9fc3ee"},
                 labels={"dist_alt": "Average distance to customers (km)"}, title="Distance to customers")
    fig.update_layout(height=340, legend_title_text="")
    c1.plotly_chart(fig, width="stretch")
    fig = px.bar(t, x="Factory", y="lt_alt", color="Status",
                 color_discrete_map={"Current": BLUE, "Alternative": "#9fc3ee"},
                 labels={"lt_alt": "Predicted lead time (days)"}, title="Predicted lead time")
    fig.update_layout(height=340, legend_title_text="")
    fig.update_yaxes(rangemode="tozero")
    c2.plotly_chart(fig, width="stretch")
    show = t[["Factory", "Status", "dist_alt", "lt_alt", "lt_change_days", "lt_ci_low", "lt_ci_high", "profit_impact",
              "long_haul_alt_pct", "confidence"]].rename(columns={
        "dist_alt": "Avg distance (km)", "lt_alt": "Predicted lead time (d)", "lt_change_days": "Change vs current (d)",
        "lt_ci_low": "95% CI low (d)", "lt_ci_high": "95% CI high (d)", "profit_impact": "Profit impact (USD)",
        "long_haul_alt_pct": "Long-haul share (%)", "confidence": "Confidence"})
    st.dataframe(show.round(3), hide_index=True, width="stretch")
    st.caption("The confidence interval comes from the fitted distance effect; intervals that straddle zero mean the lead-time "
               "change is not distinguishable from no change.")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Current vs recommended assignment")
    if best.empty:
        st.warning("No reassignment passes the current filters, confidence threshold and risk screen.")
    else:
        b = best.sort_values("dist_reduction_pct", ascending=False)
        labels = b["Product Name"] + " -> " + b["Alt Factory"]
        fig = go.Figure()
        fig.add_bar(name="Current", x=b["Product Name"], y=b.dist_cur, marker_color=BLUE)
        fig.add_bar(name="Recommended", x=b["Product Name"], y=b.dist_alt, marker_color=AQUA)
        fig.update_layout(barmode="group", height=380, title="Average distance to customers (km)",
                          yaxis_title="km", legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        fig = go.Figure()
        fig.add_bar(name="Current", x=b["Product Name"], y=b.lt_cur, marker_color=BLUE)
        fig.add_bar(name="Recommended", x=b["Product Name"], y=b.lt_alt, marker_color=AQUA,
                    error_y=dict(type="data", symmetric=False, array=b.lt_ci_high - b.lt_change_days,
                                 arrayminus=b.lt_change_days - b.lt_ci_low, color=INK2))
        fig.update_layout(barmode="group", height=380, title="Predicted lead time (days) with 95% CI on the change",
                          yaxis_title="days", legend_title_text="")
        fig.update_yaxes(rangemode="tozero")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(b[["Product Name", "Current Factory", "Alt Factory", "Orders", "dist_reduction_pct",
                        "lt_reduction_pct", "profit_impact"]].rename(columns={
            "Alt Factory": "Recommended factory", "dist_reduction_pct": "Distance reduction (%)",
            "lt_reduction_pct": "Lead-time reduction (%)", "profit_impact": "Profit impact (USD)"}).round(3),
            hide_index=True, width="stretch")
    st.subheader("Split-sourcing potential (upper bound)")
    sp, sp_map = engine.split_sourcing(cube, regions=region_sel, modes=mode_sel, freight=freight)
    if not sp.empty:
        st.caption("Each product-region served from its nearest factory. This needs production capability at several sites, "
                   "which the dataset cannot confirm, so treat it as a ceiling rather than a plan.")
        st.dataframe(sp[["Product Name", "Current Factory", "Orders", "dist_cur", "dist_split", "dist_reduction_pct",
                         "profit_impact", "profit_impact_pct"]].rename(columns={
            "dist_cur": "Dist now (km)", "dist_split": "Dist split (km)", "dist_reduction_pct": "Distance -%",
            "profit_impact": "Profit (USD)", "profit_impact_pct": "Profit (% of GP)"}).round(2),
            hide_index=True, width="stretch")
        pv = sp_map.pivot(index="Product Name", columns="Region", values="Nearest factory")
        st.markdown("**Nearest factory by region**")
        st.dataframe(pv, width="stretch")
    fm = pd.DataFrame([{"Factory": f, "lat": v[0], "lon": v[1]} for f, v in FACTORIES.items()])
    fig = px.scatter_geo(fm, lat="lat", lon="lon", text="Factory", scope="north america")
    fig.update_traces(marker=dict(size=11, color=BLUE), textposition="top center")
    fig.update_layout(height=380, title="Factory locations", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ tab 3
with tab3:
    c = st.columns(4)
    c[0].metric("Lead-time reduction", f"{k['lead_time_reduction_pct']:.2f}%", help="Order-weighted, best option per product")
    c[1].metric("Profit impact stability", f"{k['profit_stability_pct']:.0f}%",
                help="Share of orders whose delivery distance does not increase under the recommendation")
    c[2].metric("Scenario confidence", f"{k['confidence']:.0f}/100",
                help="Data coverage: share of counterfactual routes inside the observed range, scaled by sample size")
    c[3].metric("Recommendation coverage", f"{k['coverage_products_pct']:.0f}% of products",
                f"{k['coverage_volume_pct']:.0f}% of volume", delta_color="off")
    st.subheader(f"Top {top_n} ranked reassignment suggestions")
    if rec.empty:
        st.warning("No recommendations under the current settings.")
    else:
        out = rec[["Product Name", "Current Factory", "Alt Factory", "Orders", "dist_cur", "dist_alt", "dist_reduction_pct",
                   "lt_reduction_pct", "profit_impact", "profit_impact_pct", "risk_reduction_pp", "confidence",
                   "worst_region", "worst_region_change_pct", "risk_level", "score"]].rename(columns={
            "Alt Factory": "Recommended", "dist_cur": "Dist now (km)", "dist_alt": "Dist new (km)",
            "dist_reduction_pct": "Distance -%", "lt_reduction_pct": "Lead time -%", "profit_impact": "Profit (USD)",
            "profit_impact_pct": "Profit (% of GP)", "risk_reduction_pp": "Long-haul share -pp",
            "confidence": "Confidence", "worst_region": "Worst region", "worst_region_change_pct": "Worst region dist +%",
            "risk_level": "Risk", "score": "Score"})
        out.insert(0, "Rank", range(1, len(out) + 1))
        st.dataframe(out.round(2), hide_index=True, width="stretch")
        st.download_button("Download recommendations (CSV)", out.to_csv(index=False), "recommendations.csv")
        fig = px.bar(rec.assign(Option=rec["Product Name"] + " -> " + rec["Alt Factory"]).sort_values("score"),
                     x="score", y="Option", orientation="h", color_discrete_sequence=[BLUE],
                     title="Composite score")
        fig.update_layout(height=60 + 34 * len(rec), yaxis_title="")
        st.plotly_chart(fig, width="stretch")
    st.caption("Score = 20% long-haul risk reduction + 80% x (speed weight x lead-time reduction + profit weight x profit impact), "
               "each scaled to the largest value among candidates.")

# ------------------------------------------------------------------ tab 4
with tab4:
    st.subheader("Profit impact alerts and high-risk warnings")
    allc = cand.copy()
    alerts = allc[(allc.profit_impact < 0) | (allc.risk_level == "High") | (allc.worst_region_change_pct > 50)].sort_values("profit_impact")
    n_hi = int((allc.risk_level == "High").sum())
    c = st.columns(3)
    c[0].metric("High-risk scenarios", n_hi, help="Profit decline or long-haul exposure up more than 5 points. Medium risk also covers a region whose distance rises more than 50%, low-sample scenarios and profit stability below 60%.")
    c[1].metric("Scenarios with profit decline", int((allc.profit_impact < 0).sum()))
    c[2].metric("Low-sample scenarios (<100 orders)", int((allc.Orders < 100).sum()))
    if alerts.empty:
        st.success("No profit-decline or high-risk scenarios under the current filters.")
    else:
        st.dataframe(alerts[["Product Name", "Current Factory", "Alt Factory", "Orders", "dist_reduction_pct", "profit_impact",
                             "risk_reduction_pp", "risk_level"]].rename(columns={
            "Alt Factory": "Proposed factory", "dist_reduction_pct": "Distance reduction (%)",
            "profit_impact": "Profit (USD)", "risk_reduction_pp": "Long-haul share -pp", "risk_level": "Risk"}).round(2).head(30),
            hide_index=True, width="stretch")
    st.subheader("Regional trade-off check")
    if rec.empty:
        st.info("No recommendation to inspect.")
    else:
        opt = (rec["Product Name"] + " -> " + rec["Alt Factory"]).tolist()
        pick = st.selectbox("Recommended reassignment", opt)
        p, a = pick.split(" -> ")
        rt = engine.regional_tradeoff(cube, p, a, modes=mode_sel, freight=freight)
        worse = rt[rt.dist_change_pct > 0]
        if not worse.empty:
            st.warning("Delivery distance increases for: " + ", ".join(
                f"{r.Region} (+{r.dist_change_pct:.0f}%)" for r in worse.itertuples()))
        else:
            st.success("Distance falls in every region.")
        fig = px.bar(rt, x="Region", y="dist_change_pct", color_discrete_sequence=[BLUE],
                     labels={"dist_change_pct": "Distance change (%)"})
        fig.update_layout(height=320)
        st.plotly_chart(fig, width="stretch")
    st.markdown(
        "**Standing caveats.** Factory capacity, equipment fit and per-factory production costs are not in the dataset, so "
        "profit impact reflects freight only. The lead-time effect of distance is statistically indistinguishable from zero.")

# ------------------------------------------------------------------ tab 5
with tab5:
    st.subheader("Route clusters and consistently slow routes")
    r = routes.copy()
    r["cluster"] = r.cluster.astype(str)
    fig = px.scatter(r, x="dist", y="lt_adj", size="orders", color="cluster", hover_data=["Factory", "State/Province", "orders"],
                     labels={"dist": "Average distance (km)", "lt_adj": "Ship-mode-adjusted lead time (days)"},
                     title="Factory x state routes (n >= 20), coloured by behavioural cluster")
    fig.add_hline(y=0, line_color=INK2)
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")
    slow = r[r.slow_significant].sort_values("lt_adj", ascending=False)
    st.markdown("**Statistically slow routes (false-discovery rate 10%)**")
    st.dataframe(slow[["Factory", "State/Province", "orders", "lt_adj", "lt_adj_2024", "lt_adj_2025", "q_fdr",
                       "consistent_slow"]].round(3), hide_index=True, width="stretch")
    st.markdown("**Congested region x product combinations** (top-quartile volume and slower than ship-mode norm)")
    st.dataframe(cong[cong.congested][["Region", "Product Name", "orders", "lt_adj", "q_fdr"]].round(3),
                 hide_index=True, width="stretch")
