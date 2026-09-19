# Cheap Convexity Discovery

Install and run:

```bash
pip install -r requirements.txt
streamlit run cheap_convexity_streamlit.py
```

Upload the six Barchart CSV exports. Default score:
- 40% IV/HV underpricing
- 30% low IV Rank / IV Percentile
- 15% IV acceleration
- 15% options liquidity

The app discovers cheap optionality. It intentionally leaves direction and strike selection to the calendar screener + GEX/DEX + chart workflow.
