
from __future__ import annotations

import io
import math
import re
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import yfinance as yf
except Exception:
    yf = None


st.set_page_config(page_title="Dealer Positioning Map", layout="wide")

st.title("Dealer Positioning + Expected Move Map")
st.caption(
    "Upload the six OptionsCharts-style CSVs. The app maps expected move, "
    "gamma flip, call/put walls and recurring high-importance gamma levels, "
    "then flags potential buy/sell areas of interest."
)


# -----------------------------
# Helpers
# -----------------------------
def read_csv(uploaded_file):
    if uploaded_file is None:
        return None
    return pd.read_csv(uploaded_file)


def infer_ticker(name: str, fallback="SPY") -> str:
    if not name:
        return fallback
    m = re.search(r"([A-Z]{1,6})(?:_|-)", name.upper())
    return m.group(1) if m else fallback


def parse_expiration(s):
    return pd.to_datetime(s, errors="coerce")


def nearest_increment(x: float, inc: float = 5.0) -> float:
    if pd.isna(x):
        return np.nan
    return round(float(x) / inc) * inc


def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    values, weights = values[mask], weights[mask]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cutoff = weights.sum() / 2.0
    return float(values[np.searchsorted(np.cumsum(weights), cutoff)])


def weighted_mode_level(values, weights, increment=5.0):
    rows = []
    for v, w in zip(values, weights):
        if pd.isna(v) or pd.isna(w):
            continue
        lvl = nearest_increment(float(v), increment)
        rows.append((lvl, float(w)))
    if not rows:
        return np.nan
    d = pd.DataFrame(rows, columns=["level", "weight"])
    g = d.groupby("level", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
    return float(g.iloc[0]["level"])


def build_level_scores(gex_exp: pd.DataFrame, spot: float, n_expiries=5, half_life_days=10, increment=5.0):
    d = gex_exp.copy()
    d["expiration"] = pd.to_datetime(d["expiration"], errors="coerce")
    d = d.dropna(subset=["expiration"]).sort_values("expiration")

    today = pd.Timestamp.today().normalize()
    d = d[d["expiration"] >= today].head(n_expiries)
    if d.empty:
        d = gex_exp.copy()
        d["expiration"] = pd.to_datetime(d["expiration"], errors="coerce")
        d = d.dropna(subset=["expiration"]).sort_values("expiration").head(n_expiries)

    d["days"] = (d["expiration"] - today).dt.days.clip(lower=0)
    d["time_weight"] = np.exp(-np.log(2) * d["days"] / max(half_life_days, 1))
    d["gex_weight"] = d["net_gex"].abs().fillna(0)
    scale = d["gex_weight"].max()
    if not np.isfinite(scale) or scale == 0:
        d["mag_weight"] = 1.0
    else:
        d["mag_weight"] = 0.25 + 0.75 * (d["gex_weight"] / scale)
    d["weight"] = d["time_weight"] * d["mag_weight"]

    # Composite structural levels:
    call_wall = weighted_mode_level(d["call_wall"], d["weight"], increment)
    put_wall = weighted_mode_level(d["put_wall"], d["weight"], increment)
    gamma_flip = weighted_median(d["gamma_flip"], d["weight"])

    # "Important gamma levels" from recurring wall/flip concentrations.
    rows = []
    for _, r in d.iterrows():
        for kind, factor in [("Call Wall", 1.0), ("Put Wall", 1.0), ("Gamma Flip", 0.65)]:
            col = {"Call Wall": "call_wall", "Put Wall": "put_wall", "Gamma Flip": "gamma_flip"}[kind]
            v = r.get(col)
            if pd.isna(v):
                continue
            lvl = nearest_increment(float(v), increment)
            rows.append({
                "Level": lvl,
                "Type": kind,
                "Score": float(r["weight"]) * factor,
                "Expiration": r["expiration"].date(),
            })

    detail = pd.DataFrame(rows)
    if detail.empty:
        imp = pd.DataFrame(columns=["Level", "Importance", "Occurrences", "Types"])
    else:
        imp = (
            detail.groupby("Level")
            .agg(
                Importance=("Score", "sum"),
                Occurrences=("Level", "size"),
                Types=("Type", lambda x: ", ".join(sorted(set(x)))),
            )
            .reset_index()
        )
        if imp["Importance"].max() > 0:
            imp["Importance"] = 100 * imp["Importance"] / imp["Importance"].max()
        imp = imp.sort_values(["Importance", "Occurrences"], ascending=False)

    return d, call_wall, put_wall, gamma_flip, imp


def select_expected_move(em: pd.DataFrame, expiry_date):
    d = em.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"]).sort_values("date")
    if d.empty:
        return None
    target = pd.Timestamp(expiry_date)
    idx = (d["date"] - target).abs().idxmin()
    return d.loc[idx]


def get_price_data(ticker, history_df, period="6mo"):
    # Prefer OHLC from Yahoo for candles.
    if yf is not None:
        try:
            px = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
            if not px.empty:
                if isinstance(px.columns, pd.MultiIndex):
                    px.columns = [c[0] for c in px.columns]
                px = px.reset_index()
                px = px.rename(columns={"Date": "date"})
                needed = {"Open", "High", "Low", "Close"}
                if needed.issubset(px.columns):
                    return px[["date", "Open", "High", "Low", "Close"]].dropna()
        except Exception:
            pass

    # Fallback: close-only synthetic OHLC so the app still renders.
    d = history_df.copy()
    d["date"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["date", "close_price"]).sort_values("date")
    d["Open"] = d["close_price"].shift(1).fillna(d["close_price"])
    d["Close"] = d["close_price"]
    d["High"] = d[["Open", "Close"]].max(axis=1)
    d["Low"] = d[["Open", "Close"]].min(axis=1)
    return d[["date", "Open", "High", "Low", "Close"]]


def add_hline(fig, y, text, dash="dash", width=1.5):
    if y is None or pd.isna(y):
        return
    fig.add_hline(
        y=float(y),
        line_dash=dash,
        line_width=width,
        annotation_text=text,
        annotation_position="right",
    )


# -----------------------------
# Uploads
# -----------------------------
with st.sidebar:
    st.header("Upload files")
    dex_hist_f = st.file_uploader("DEX history", type="csv")
    gex_hist_f = st.file_uploader("GEX history", type="csv")
    iv_hist_f = st.file_uploader("IV history", type="csv")
    em_f = st.file_uploader("Expected move", type="csv")
    dex_exp_f = st.file_uploader("DEX by expiration", type="csv")
    gex_exp_f = st.file_uploader("GEX by expiration", type="csv")

    st.divider()
    ticker_input = st.text_input("Ticker override", value="")
    n_exp = st.slider("Near expiries to combine", 1, 10, 5)
    half_life = st.slider("Expiry weight half-life (days)", 3, 30, 10)
    strike_inc = st.selectbox("Gamma-level grouping increment", [1.0, 2.5, 5.0, 10.0], index=2)
    chart_period = st.selectbox("Chart history", ["3mo", "6mo", "1y"], index=1)
    aoi_pct = st.slider("AOI tolerance (% of spot)", 0.10, 2.00, 0.50, 0.05)
    max_imp_levels = st.slider("Important gamma levels on chart", 2, 10, 5)


files = [dex_hist_f, gex_hist_f, iv_hist_f, em_f, dex_exp_f, gex_exp_f]
if not all(files):
    st.info("Upload all six CSVs to build the dealer-positioning map.")
    st.stop()

dex_hist = read_csv(dex_hist_f)
gex_hist = read_csv(gex_hist_f)
iv_hist = read_csv(iv_hist_f)
em = read_csv(em_f)
dex_exp = read_csv(dex_exp_f)
gex_exp = read_csv(gex_exp_f)

ticker = ticker_input.strip().upper() or infer_ticker(gex_hist_f.name, "SPY")

# Basic validation
required_gex = {"expiration", "net_gex", "call_gex", "put_gex", "call_wall", "put_wall", "gamma_flip"}
required_em = {"date", "expected_move_amt", "upper_price", "lower_price", "implied_volatility"}
required_hist = {"timestamp", "close_price"}

missing = []
if not required_gex.issubset(gex_exp.columns):
    missing.append(f"GEX by expiration missing: {sorted(required_gex - set(gex_exp.columns))}")
if not required_em.issubset(em.columns):
    missing.append(f"Expected move missing: {sorted(required_em - set(em.columns))}")
if not required_hist.issubset(gex_hist.columns):
    missing.append(f"GEX history missing: {sorted(required_hist - set(gex_hist.columns))}")

if missing:
    st.error("\n".join(missing))
    st.stop()

# Spot from latest close in GEX history
gex_hist["timestamp"] = pd.to_datetime(gex_hist["timestamp"], errors="coerce")
latest_hist = gex_hist.dropna(subset=["timestamp", "close_price"]).sort_values("timestamp")
spot = float(latest_hist.iloc[-1]["close_price"])
spot_date = latest_hist.iloc[-1]["timestamp"].date()

# Expiry selector based on expected move file
em_dates = pd.to_datetime(em["date"], errors="coerce").dropna().sort_values().unique()
default_idx = min(len(em_dates) - 1, 4) if len(em_dates) else 0
chosen_expiry = st.selectbox(
    "Expected-move horizon",
    options=[pd.Timestamp(x).date() for x in em_dates],
    index=default_idx,
)

em_row = select_expected_move(em, chosen_expiry)
selected_exp, call_wall, put_wall, gamma_flip, important = build_level_scores(
    gex_exp, spot, n_expiries=n_exp, half_life_days=half_life, increment=strike_inc
)

upper_em = float(em_row["upper_price"]) if em_row is not None else np.nan
lower_em = float(em_row["lower_price"]) if em_row is not None else np.nan
em_amt = float(em_row["expected_move_amt"]) if em_row is not None else np.nan
em_iv = float(em_row["implied_volatility"]) if em_row is not None else np.nan

# -----------------------------
# AOI construction
# -----------------------------
tol = spot * aoi_pct / 100.0

candidate_rows = [
    {"Level": lower_em, "Level Type": "Expected Move Lower", "Bias": "BUY", "Base Score": 92},
    {"Level": put_wall, "Level Type": "Put Wall", "Bias": "BUY", "Base Score": 100},
    {"Level": upper_em, "Level Type": "Expected Move Upper", "Bias": "SELL", "Base Score": 92},
    {"Level": call_wall, "Level Type": "Call Wall", "Bias": "SELL", "Base Score": 100},
    {"Level": gamma_flip, "Level Type": "Gamma Flip", "Bias": "REGIME", "Base Score": 85},
]

for _, r in important.head(max_imp_levels).iterrows():
    lvl = float(r["Level"])
    # Context-sensitive bias: below spot = support candidate, above spot = resistance candidate.
    bias = "BUY" if lvl < spot else ("SELL" if lvl > spot else "REGIME")
    candidate_rows.append({
        "Level": lvl,
        "Level Type": f"Important Gamma ({r['Types']})",
        "Bias": bias,
        "Base Score": min(100, float(r["Importance"])),
    })

levels = pd.DataFrame(candidate_rows).dropna(subset=["Level"])
levels["Distance $"] = levels["Level"] - spot
levels["Distance %"] = 100 * levels["Distance $"] / spot
levels["AOI Low"] = levels["Level"] - tol
levels["AOI High"] = levels["Level"] + tol
levels["In AOI Now"] = (spot >= levels["AOI Low"]) & (spot <= levels["AOI High"])

# Add confluence by clustering nearby levels.
confluence = []
for i, r in levels.iterrows():
    near = levels[(levels["Level"] - r["Level"]).abs() <= tol]
    confluence.append(len(near))
levels["Confluence"] = confluence
levels["AOI Score"] = (
    levels["Base Score"]
    + (levels["Confluence"] - 1) * 8
    - levels["Distance %"].abs().clip(upper=10) * 2
).clip(0, 100).round(1)

levels["Flag"] = np.where(
    levels["Bias"].eq("BUY"),
    "BUY AREA",
    np.where(levels["Bias"].eq("SELL"), "SELL AREA", "REGIME / PIVOT")
)
levels = levels.sort_values(["AOI Score", "Distance %"], ascending=[False, True])

# -----------------------------
# Headline metrics
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Spot", f"{spot:.2f}", f"{spot_date}")
c2.metric("Expected Move", f"±{em_amt:.2f}", f"IV {em_iv:.2f}%")
c3.metric("Gamma Flip", f"{gamma_flip:.2f}" if np.isfinite(gamma_flip) else "—")
c4.metric("Call Wall", f"{call_wall:.2f}" if np.isfinite(call_wall) else "—")
c5.metric("Put Wall", f"{put_wall:.2f}" if np.isfinite(put_wall) else "—")

# -----------------------------
# Candlestick chart
# -----------------------------
px = get_price_data(ticker, gex_hist, chart_period)

fig = go.Figure(
    data=[
        go.Candlestick(
            x=px["date"],
            open=px["Open"],
            high=px["High"],
            low=px["Low"],
            close=px["Close"],
            name=ticker,
        )
    ]
)

add_hline(fig, upper_em, f"Upper EM {upper_em:.2f}", "dot", 2)
add_hline(fig, lower_em, f"Lower EM {lower_em:.2f}", "dot", 2)
add_hline(fig, gamma_flip, f"Gamma Flip {gamma_flip:.2f}", "dash", 2)
add_hline(fig, call_wall, f"Call Wall {call_wall:.2f}", "solid", 2.5)
add_hline(fig, put_wall, f"Put Wall {put_wall:.2f}", "solid", 2.5)

core = {round(x, 4) for x in [upper_em, lower_em, gamma_flip, call_wall, put_wall] if np.isfinite(x)}
for _, r in important.head(max_imp_levels).iterrows():
    lvl = float(r["Level"])
    if round(lvl, 4) in core:
        continue
    add_hline(fig, lvl, f"Gamma {lvl:.2f} ({r['Importance']:.0f})", "dashdot", 1)

fig.update_layout(
    title=f"{ticker} — Dealer Positioning Map",
    xaxis_title="Date",
    yaxis_title="Price",
    height=700,
    xaxis_rangeslider_visible=False,
    legend_orientation="h",
)

st.plotly_chart(fig, use_container_width=True)

if yf is None:
    st.warning("yfinance is not installed. Chart is using close-only fallback candles.")
elif (px["High"] == px[["Open", "Close"]].max(axis=1)).all():
    st.warning("Yahoo OHLC data was unavailable, so the chart is using close-only fallback candles.")

# -----------------------------
# Main dealer-level table
# -----------------------------
st.subheader("Dealer levels and areas of interest")

display_cols = [
    "Flag", "Level Type", "Level", "AOI Low", "AOI High",
    "Distance $", "Distance %", "Confluence", "AOI Score", "In AOI Now"
]
tbl = levels[display_cols].copy()
for c in ["Level", "AOI Low", "AOI High", "Distance $", "Distance %", "AOI Score"]:
    tbl[c] = pd.to_numeric(tbl[c], errors="coerce").round(2)

def highlight_flag(row):
    # Use subtle dataframe highlighting; colors adapt reasonably in Streamlit.
    if row["Flag"] == "BUY AREA":
        return ["background-color: rgba(0,160,80,0.13)"] * len(row)
    if row["Flag"] == "SELL AREA":
        return ["background-color: rgba(220,60,60,0.13)"] * len(row)
    return ["background-color: rgba(120,120,120,0.08)"] * len(row)

st.dataframe(tbl.style.apply(highlight_flag, axis=1), use_container_width=True, hide_index=True)

# -----------------------------
# Important gamma level table
# -----------------------------
st.subheader("Important gamma levels")
if important.empty:
    st.info("No important gamma clusters could be derived.")
else:
    imp_show = important.head(15).copy()
    imp_show["Distance %"] = ((imp_show["Level"] - spot) / spot * 100).round(2)
    imp_show["Importance"] = imp_show["Importance"].round(1)
    imp_show["Side"] = np.where(
        imp_show["Level"] < spot, "Below spot / support candidate",
        np.where(imp_show["Level"] > spot, "Above spot / resistance candidate", "At spot")
    )
    st.dataframe(imp_show, use_container_width=True, hide_index=True)

# -----------------------------
# Per-expiration context
# -----------------------------
with st.expander("Selected expiration inputs"):
    exp_show = selected_exp[
        ["expiration", "net_gex", "call_gex", "put_gex", "call_wall", "put_wall", "gamma_flip", "weight"]
    ].copy()
    exp_show["expiration"] = exp_show["expiration"].dt.date
    exp_show["weight"] = exp_show["weight"].round(4)
    st.dataframe(exp_show, use_container_width=True, hide_index=True)

# -----------------------------
# Downloadable summary
# -----------------------------
summary_csv = tbl.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download dealer-level summary CSV",
    summary_csv,
    file_name=f"{ticker}_dealer_positioning_map.csv",
    mime="text/csv",
)

st.caption(
    "AOI flags are decision-support zones, not automatic trade signals. "
    "BUY areas are levels below/near spot associated with put-wall/lower-EM/gamma support; "
    "SELL areas are levels above/near spot associated with call-wall/upper-EM/gamma resistance. "
    "Gamma flip is treated as a regime/pivot level."
)
