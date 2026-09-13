
# Dealer Positioning Map

Streamlit dashboard for six OptionsCharts-style CSV files:

1. DEX history
2. GEX history
3. IV history
4. Expected move
5. Net DEX exposure by expiration
6. Net GEX exposure by expiration

## Run

```bash
pip install -r requirements.txt
streamlit run dealer_positioning_app.py
```

## Important gamma levels

The supplied GEX-by-expiration file contains aggregate GEX plus call wall, put wall,
and gamma flip per expiry, not strike-by-strike gamma bars. Therefore the dashboard
derives "important gamma levels" from recurring wall/flip levels across the selected
near expirations, weighted by:

- time to expiry (near expiries receive more weight)
- absolute net GEX magnitude
- type of level (walls receive slightly more weight than gamma flip)

If you later add strike-level GEX data, the scoring function can be upgraded to use
actual gamma concentration by strike.
