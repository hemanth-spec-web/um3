# Nassau Candy: Factory Reallocation & Shipping Optimization

Decision-intelligence pipeline and Streamlit dashboard for reassigning products to factories.

## Run

```bash
pip install -r requirements.txt
python model.py       # cleans data, trains 3 models, ablation, distance-effect test  -> artifacts/
python simulate.py    # scenario cube, clustering, recommendations                    -> artifacts/
python charts.py      # report figures                                                -> figures/
streamlit run app.py  # dashboard
```

`data.csv` must sit next to the scripts (the Nassau Candy Distributor order file).
`artifacts/` is already populated, so `streamlit run app.py` works immediately.

## Files

| File | Purpose |
|---|---|
| `prep.py` | Ship-date offset correction, geocoding (zip centroids, Canadian cities), factory distances |
| `model.py` | Outlier removal, Linear / Random Forest / Gradient Boosting, ablation, distance-effect test |
| `engine.py` | Counterfactual scenario cube, scoring, ranking, KPIs, regional trade-off, split sourcing |
| `simulate.py` | Builds the cube, clusters routes, flags slow/congested routes |
| `charts.py` | Figures for the research paper |
| `app.py` | Streamlit dashboard (5 tabs) |

## Things to know before presenting this

1. **Ship-date correction.** Raw order-to-ship gaps are 904-1,642 days. They form three clusters whose
   bases (904 / 1,269 / 1,635 days) are removed in `prep.py`; residuals then match ship-mode service windows.
   This is an inference. Confirm with the data owner.
2. **Lead time is order-to-dispatch.** There is no delivery date, so the effect of distance on delivery time
   cannot be tested with this file.
3. **Freight rate is an assumption** (`engine.DEFAULT_FREIGHT` = $0.04 per unit per 1,000 km). All profit
   impact scales linearly with it; the dashboard has a slider.
4. **Factory is collinear with product** (one factory per product), so the model uses distance to the factory
   as the origin feature rather than a factory label.
