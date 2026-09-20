from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Cheap Convexity Lab",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2.5rem;}
div[data-testid="stMetric"] {
    background: rgba(30,41,59,.30);
    border: 1px solid rgba(148,163,184,.18);
    padding: 13px; border-radius: 11px;
}
div[data-testid="stSidebar"] {border-right: 1px solid rgba(148,163,184,.16);}
.hero {font-size: 2rem; font-weight: 800; margin-bottom: -.25rem;}
.subtle {color: #94a3b8; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

EXPECTED = [
    "Symbol","Strike","Type","Exp Date","DTE","IV","Imp Vol","IV/HV","IV Rank",
    "1D IV Chg","5D IV Chg","5D/1M IV%","Delta","Gamma","Theta","Vega",
    "Th/Ga","Th/Ve","Strk Dist","Exp Move","Bid","Ask","Open Int","Volume","Premium"
]
PCT_COLS = {"IV","Imp Vol","IV Rank","1D IV Chg","5D IV Chg","5D/1M IV%"}
NUM_COLS = set(EXPECTED) - {"Symbol","Type","Exp Date"} - PCT_COLS

def to_num(s):
    return pd.to_numeric(
        s.astype(str).str.strip()
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("+", "", regex=False),
        errors="coerce"
    )

def read_csv(upload):
    raw = upload.getvalue()
    try:
        d = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except Exception:
        d = pd.read_csv(io.BytesIO(raw))
    d.columns = [str(c).strip() for c in d.columns]
    for c in d.columns:
        if c in PCT_COLS or c in NUM_COLS:
            d[c] = to_num(d[c])
    if "Symbol" in d:
        d["Symbol"] = d["Symbol"].astype(str).str.upper().str.strip()
    if "Type" in d:
        d["Type"] = d["Type"].astype(str).str.title().str.strip()
    if "Exp Date" in d:
        d["Exp Date"] = pd.to_datetime(d["Exp Date"], errors="coerce")
    return d

def abs_score(s, lo, hi, inverse=False, neutral=50):
    x = pd.to_numeric(s, errors="coerce")
    z = ((x-lo)/(hi-lo)*100).clip(0,100)
    if inverse:
        z = 100-z
    return z.fillna(neutral)

def percentile_score(s, higher=True):
    x = pd.to_numeric(s, errors="coerce")
    r = x.rank(pct=True)
    if not higher:
        r = 1-r
    return (100*r).fillna(50)

def enrich(d, weights):
    x = d.copy()
    for c in EXPECTED:
        if c not in x.columns:
            x[c] = np.nan

    # Derived execution and Greek-efficiency metrics
    x["Mid"] = (x["Bid"] + x["Ask"]) / 2
    x["Spread %"] = np.where(
        x["Mid"] > 0, (x["Ask"] - x["Bid"]) / x["Mid"] * 100, np.nan
    )
    theta_abs = x["Theta"].abs()
    x["Gamma/Theta"] = np.where(theta_abs > 0, x["Gamma"].abs()/theta_abs, np.nan)
    x["Vega/Theta"] = np.where(theta_abs > 0, x["Vega"].abs()/theta_abs, np.nan)

    # Cross-check native Barchart ratios where available.
    x["Calc Th/Ga"] = np.where(x["Gamma"].abs() > 0, theta_abs/x["Gamma"].abs(), np.nan)
    x["Calc Th/Ve"] = np.where(x["Vega"].abs() > 0, theta_abs/x["Vega"].abs(), np.nan)

    # Underlying volatility cheapness
    ivhv = abs_score(x["IV/HV"], .60, 1.10, inverse=True)
    ivrank = abs_score(x["IV Rank"], 0, 60, inverse=True)
    x["Cheapness Score"] = .60*ivhv + .40*ivrank

    # IV acceleration
    a1 = abs_score(x["1D IV Chg"], -3, 5)
    a5 = abs_score(x["5D IV Chg"], -5, 12)
    x["Acceleration Score"] = .35*a1 + .65*a5

    # Contract efficiency. Use raw Greeks now; cap via cross-sectional percentiles.
    gamma_eff = percentile_score(np.log1p(x["Gamma/Theta"].clip(lower=0)), True)
    vega_eff = percentile_score(np.log1p(x["Vega/Theta"].clip(lower=0)), True)

    # Penalize lottery-like deltas and extremely far strikes.
    abs_delta = x["Delta"].abs()
    delta_quality = (100 - ((abs_delta - .35).abs()/.35*100)).clip(0,100).fillna(40)
    strike_quality = abs_score(x["Strk Dist"].abs(), 0, 2.0, inverse=True)
    x["Greek Efficiency Score"] = (
        .35*gamma_eff + .25*vega_eff + .25*delta_quality + .15*strike_quality
    )

    # Execution/tradeability
    oi = percentile_score(np.log1p(x["Open Int"].clip(lower=0)), True)
    vol = percentile_score(np.log1p(x["Volume"].clip(lower=0)), True)
    spread = abs_score(x["Spread %"], 2, 25, inverse=True)
    x["Tradeability Score"] = .40*oi + .30*vol + .30*spread

    w1,w2,w3,w4 = weights
    den = max(sum(weights), 1)
    x["Contract Score"] = (
        w1*x["Cheapness Score"] +
        w2*x["Acceleration Score"] +
        w3*x["Greek Efficiency Score"] +
        w4*x["Tradeability Score"]
    ) / den

    # Expression engine: structure only; direction remains chart/GEX-DEX driven.
    cheap = x["IV/HV"] < 1.00
    deep = x["IV/HV"] <= .80
    lowrank = x["IV Rank"] <= 25
    accelerating = x["Acceleration Score"] >= 62
    very_accel = x["Acceleration Score"] >= 72
    gamma_good = gamma_eff >= 65
    vega_good = vega_eff >= 65
    liquid = x["Tradeability Score"] >= 45
    near_atm = x["Delta"].abs().between(.38,.62, inclusive="both")
    directional_delta = x["Delta"].abs().between(.18,.42, inclusive="both")

    x["Suggested Expression"] = "WAIT / NO EDGE"
    x["Expression Reason"] = "No strong cheapness + efficiency combination."

    m = deep & lowrank & very_accel & gamma_good & liquid
    x.loc[m, "Suggested Expression"] = "LONG OPTION / BACKSPREAD"
    x.loc[m, "Expression Reason"] = "Deep IV/HV discount, low IV Rank, strong IV acceleration and efficient gamma."

    m = cheap & directional_delta & liquid & ~very_accel
    x.loc[m, "Suggested Expression"] = "DIRECTIONAL OTM CALENDAR"
    x.loc[m, "Expression Reason"] = "IV below HV with a directional-delta contract; use GEX/DEX for destination and chart for trigger."

    m = lowrank & near_atm & vega_good & (x["IV/HV"].between(.85,1.15)) & (x["Acceleration Score"] < 58) & liquid
    x.loc[m, "Suggested Expression"] = "ATM VEGA CALENDAR"
    x.loc[m, "Expression Reason"] = "Historically cheap IV, near-ATM delta and efficient vega with subdued acceleration."

    m = (x["IV/HV"] >= 1.15) & near_atm & (x["Acceleration Score"] < 45) & liquid
    x.loc[m, "Suggested Expression"] = "ATM THETA CALENDAR"
    x.loc[m, "Expression Reason"] = "IV rich versus realized volatility with weak acceleration; theta structure may be more relevant than long convexity."

    # Helpful classifications
    x["Vol Regime"] = np.select(
        [x["IV/HV"] <= .80, x["IV/HV"] < 1.0, x["IV/HV"] >= 1.15],
        ["DEEP CHEAP", "CHEAP", "RICH"],
        default="NEUTRAL"
    )
    return x

def underlying_summary(x):
    # Best executable contract represents opportunity, while volatility metrics use medians
    # because they repeat across many contracts.
    rows = []
    for sym, g in x.groupby("Symbol", dropna=False):
        g2 = g.sort_values("Contract Score", ascending=False)
        best = g2.iloc[0]
        rows.append({
            "Symbol": sym,
            "Convexity Score": round(float(best["Contract Score"]),1),
            "IV/HV": g["IV/HV"].median(),
            "IV Rank": g["IV Rank"].median(),
            "1D IV Chg": g["1D IV Chg"].median(),
            "5D IV Chg": g["5D IV Chg"].median(),
            "5D/1M IV%": g["5D/1M IV%"].median(),
            "Best Expression": best["Suggested Expression"],
            "Best Type": best["Type"],
            "Best Strike": best["Strike"],
            "Best Exp": best["Exp Date"],
            "Best DTE": best["DTE"],
            "Best Delta": best["Delta"],
            "Best Γ/Θ": best["Gamma/Theta"],
            "Best Vega/Θ": best["Vega/Theta"],
            "Best Spread %": best["Spread %"],
            "Best OI": best["Open Int"],
            "Best Volume": best["Volume"],
        })
    return pd.DataFrame(rows).sort_values("Convexity Score", ascending=False)

def fmt_contract(row):
    exp = row["Exp Date"].date().isoformat() if pd.notna(row["Exp Date"]) else "NA"
    return f'{row["Symbol"]} {row["Type"]} {row["Strike"]} | {exp} | {int(row["DTE"]) if pd.notna(row["DTE"]) else "?"} DTE'

st.markdown('<div class="hero">⚡ Cheap Convexity Lab</div>', unsafe_allow_html=True)
st.caption("One Barchart Options Screener CSV → underlying discovery → contract efficiency → expression selection → individual option analysis")

with st.sidebar:
    st.header("Input")
    upload = st.file_uploader("Barchart Options Screener CSV", type=["csv"])
    st.divider()
    st.header("Score weights")
    wcheap = st.slider("Volatility cheapness", 0, 60, 30, 5)
    waccel = st.slider("Volatility acceleration", 0, 50, 20, 5)
    weff = st.slider("Greek efficiency", 0, 60, 30, 5)
    wtrade = st.slider("Tradeability", 0, 50, 20, 5)
    st.caption("Weights are normalized automatically.")
    st.divider()
    st.header("Contract filters")
    min_oi = st.number_input("Minimum OI", 0, 1000000, 250, 50)
    min_vol = st.number_input("Minimum volume", 0, 1000000, 25, 25)
    max_spread = st.slider("Maximum bid/ask spread %", 1, 100, 20)
    min_dte, max_dte = st.slider("DTE range", 0, 365, (7,90))
    min_score = st.slider("Minimum contract score", 0, 100, 45)
    st.link_button(
        "⚡ Cheap Convexity Screener",
        "YOUR_CHEAP_CONVEXITY_APP_URL",
        use_container_width=True
    )
    
    st.link_button(
        "🎯 Dealer Positioning",
        "https://dealerpositioning-n2m58uzu42afb44usojrwe.streamlit.app/",
        use_container_width=True
    )
    
    st.link_button(
        "📊 Option Contract Analysis",
        "https://dealerpositioning-zenvhgc3fs3dcsd9yunvct.streamlit.app/",
        use_container_width=True
    )
    
    st.link_button(
        "🔬 Calender Spread Regime Screening",
        "https://freedom-fuxffx4ohuuosfojdmffxl.streamlit.app/",
        use_container_width=True
    )


if upload is None:
    st.info("Upload the new 25-column Barchart export to begin.")
    st.code(", ".join(EXPECTED), language=None)
    st.stop()

df = read_csv(upload)
missing = [c for c in EXPECTED if c not in df.columns]
if missing:
    st.warning("Missing expected columns: " + ", ".join(missing))

x = enrich(df, (wcheap,waccel,weff,wtrade))

filtered = x[
    (x["Open Int"].fillna(0) >= min_oi) &
    (x["Volume"].fillna(0) >= min_vol) &
    (x["Spread %"].fillna(999) <= max_spread) &
    (x["DTE"].fillna(-1).between(min_dte,max_dte)) &
    (x["Contract Score"] >= min_score)
].copy()

u = underlying_summary(filtered) if not filtered.empty else pd.DataFrame()

m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Contracts loaded", f"{len(x):,}")
m2.metric("Tickers", f"{x['Symbol'].nunique():,}")
m3.metric("Qualified contracts", f"{len(filtered):,}")
m4.metric("Qualified tickers", f"{filtered['Symbol'].nunique():,}")
m5.metric("Median IV/HV", f"{x['IV/HV'].median():.2f}" if x["IV/HV"].notna().any() else "—")

tabs = st.tabs([
    "🏆 Convexity Screener",
    "🔎 Contract Explorer",
    "🧪 Individual Option Analysis",
    "📊 Regime Map",
    "🧾 Data Audit"
])

with tabs[0]:
    st.subheader("Underlying cheap-convexity ranking")
    st.caption("Each ticker is represented by its highest-scoring contract after your execution filters. Direction is not inferred here.")
    if u.empty:
        st.warning("No contracts pass the current filters.")
    else:
        show = u.copy()
        show["Best Exp"] = pd.to_datetime(show["Best Exp"]).dt.date
        st.dataframe(
            show,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Convexity Score": st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.1f"),
                "IV/HV": st.column_config.NumberColumn(format="%.2f"),
                "IV Rank": st.column_config.NumberColumn(format="%.1f%%"),
                "1D IV Chg": st.column_config.NumberColumn(format="%+.2f%%"),
                "5D IV Chg": st.column_config.NumberColumn(format="%+.2f%%"),
                "Best Γ/Θ": st.column_config.NumberColumn(format="%.2f"),
                "Best Vega/Θ": st.column_config.NumberColumn(format="%.2f"),
                "Best Spread %": st.column_config.NumberColumn(format="%.1f%%"),
            }
        )
        st.download_button(
            "Download underlying ranking",
            u.to_csv(index=False).encode(),
            "cheap_convexity_underlying_ranking.csv",
            "text/csv"
        )

with tabs[1]:
    st.subheader("Contract Explorer")
    if filtered.empty:
        st.warning("No contracts pass the current filters.")
    else:
        symbols = sorted(filtered["Symbol"].dropna().unique())
        sym = st.selectbox("Ticker", symbols, key="explorer_symbol")
        g = filtered[filtered["Symbol"]==sym].sort_values("Contract Score", ascending=False).copy()
        exp_values = sorted(g["Exp Date"].dropna().dt.date.unique())
        if exp_values:
            chosen_exp = st.multiselect("Expiration", exp_values, default=exp_values)
            g = g[g["Exp Date"].dt.date.isin(chosen_exp)]
        type_values = sorted(g["Type"].dropna().unique())
        if type_values:
            chosen_type = st.multiselect("Type", type_values, default=type_values)
            g = g[g["Type"].isin(chosen_type)]

        cols = [
            "Symbol","Type","Strike","Exp Date","DTE","Contract Score","Suggested Expression",
            "Delta","Gamma","Theta","Vega","Gamma/Theta","Vega/Theta","Th/Ga","Th/Ve",
            "Strk Dist","Exp Move","IV","IV/HV","IV Rank","1D IV Chg","5D IV Chg",
            "Bid","Ask","Mid","Spread %","Open Int","Volume"
        ]
        gg = g[[c for c in cols if c in g.columns]].copy()
        if "Exp Date" in gg: gg["Exp Date"] = gg["Exp Date"].dt.date
        st.dataframe(
            gg, use_container_width=True, hide_index=True,
            column_config={
                "Contract Score": st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.1f"),
                "Spread %": st.column_config.NumberColumn(format="%.1f%%"),
                "Gamma/Theta": st.column_config.NumberColumn(format="%.2f"),
                "Vega/Theta": st.column_config.NumberColumn(format="%.2f"),
                "IV": st.column_config.NumberColumn(format="%.2f%%"),
                "IV Rank": st.column_config.NumberColumn(format="%.2f%%"),
            }
        )

with tabs[2]:
    st.subheader("Individual Option Analysis")
    source = filtered if not filtered.empty else x
    if source.empty:
        st.warning("No contracts available.")
    else:
        syms = sorted(source["Symbol"].dropna().unique())
        sym2 = st.selectbox("Ticker", syms, key="analysis_symbol")
        g2 = source[source["Symbol"]==sym2].sort_values("Contract Score", ascending=False).copy()
        labels = {idx: fmt_contract(row) for idx,row in g2.iterrows()}
        chosen_idx = st.selectbox(
            "Contract",
            list(labels.keys()),
            format_func=lambda z: labels[z],
            key="analysis_contract"
        )
        r = g2.loc[chosen_idx]

        st.markdown(f"### {fmt_contract(r)}")
        a,b,c,d,e = st.columns(5)
        a.metric("Contract score", f'{r["Contract Score"]:.1f}')
        b.metric("Expression", str(r["Suggested Expression"]))
        c.metric("IV / HV", f'{r["IV/HV"]:.2f}' if pd.notna(r["IV/HV"]) else "—")
        d.metric("IV Rank", f'{r["IV Rank"]:.1f}%' if pd.notna(r["IV Rank"]) else "—")
        e.metric("Spread", f'{r["Spread %"]:.1f}%' if pd.notna(r["Spread %"]) else "—")

        st.markdown("#### Greeks & convexity efficiency")
        g1,g2c,g3,g4,g5,g6 = st.columns(6)
        g1.metric("Delta", f'{r["Delta"]:.3f}' if pd.notna(r["Delta"]) else "—")
        g2c.metric("Gamma", f'{r["Gamma"]:.4f}' if pd.notna(r["Gamma"]) else "—")
        g3.metric("Theta", f'{r["Theta"]:.4f}' if pd.notna(r["Theta"]) else "—")
        g4.metric("Vega", f'{r["Vega"]:.4f}' if pd.notna(r["Vega"]) else "—")
        g5.metric("Gamma / |Theta|", f'{r["Gamma/Theta"]:.2f}' if pd.notna(r["Gamma/Theta"]) else "—")
        g6.metric("Vega / |Theta|", f'{r["Vega/Theta"]:.2f}' if pd.notna(r["Vega/Theta"]) else "—")

        st.markdown("#### Volatility & location")
        v1,v2,v3,v4,v5,v6 = st.columns(6)
        v1.metric("Contract IV", f'{r["IV"]:.2f}%' if pd.notna(r["IV"]) else "—")
        v2.metric("Underlying IV", f'{r["Imp Vol"]:.2f}%' if pd.notna(r["Imp Vol"]) else "—")
        v3.metric("1D IV change", f'{r["1D IV Chg"]:+.2f}%' if pd.notna(r["1D IV Chg"]) else "—")
        v4.metric("5D IV change", f'{r["5D IV Chg"]:+.2f}%' if pd.notna(r["5D IV Chg"]) else "—")
        v5.metric("Strike distance", f'{r["Strk Dist"]:.2f}' if pd.notna(r["Strk Dist"]) else "—")
        v6.metric("Expected move", f'{r["Exp Move"]:.4f}' if pd.notna(r["Exp Move"]) else "—")

        st.markdown("#### Execution")
        q1,q2,q3,q4,q5,q6 = st.columns(6)
        q1.metric("Bid", f'${r["Bid"]:.2f}' if pd.notna(r["Bid"]) else "—")
        q2.metric("Ask", f'${r["Ask"]:.2f}' if pd.notna(r["Ask"]) else "—")
        q3.metric("Mid", f'${r["Mid"]:.2f}' if pd.notna(r["Mid"]) else "—")
        q4.metric("Open interest", f'{r["Open Int"]:,.0f}' if pd.notna(r["Open Int"]) else "—")
        q5.metric("Volume", f'{r["Volume"]:,.0f}' if pd.notna(r["Volume"]) else "—")
        q6.metric("Premium traded", f'${r["Premium"]:,.0f}' if pd.notna(r["Premium"]) else "—")

        st.info(str(r["Expression Reason"]))
        st.caption(
            "Expression is a structure classification, not a directional signal. "
            "Use your chart plus GEX/DEX model for bullish/bearish bias, trigger and destination."
        )

        score_frame = pd.DataFrame({
            "Component":["Cheapness","Acceleration","Greek efficiency","Tradeability"],
            "Score":[r["Cheapness Score"],r["Acceleration Score"],r["Greek Efficiency Score"],r["Tradeability Score"]]
        }).set_index("Component")
        st.bar_chart(score_frame)

        with st.expander("Barchart ratio cross-check"):
            st.write({
                "Barchart Th/Ga": None if pd.isna(r["Th/Ga"]) else round(float(r["Th/Ga"]),4),
                "Calculated |Theta|/|Gamma|": None if pd.isna(r["Calc Th/Ga"]) else round(float(r["Calc Th/Ga"]),4),
                "Barchart Th/Ve": None if pd.isna(r["Th/Ve"]) else round(float(r["Th/Ve"]),4),
                "Calculated |Theta|/|Vega|": None if pd.isna(r["Calc Th/Ve"]) else round(float(r["Calc Th/Ve"]),4),
            })

with tabs[3]:
    st.subheader("Volatility regime map")
    plot = x[["Symbol","IV/HV","IV Rank","Contract Score"]].dropna().copy()
    if plot.empty:
        st.info("Not enough data for the regime map.")
    else:
        # Aggregate repeated underlying values so each ticker appears once.
        plot = plot.groupby("Symbol", as_index=False).agg({
            "IV/HV":"median","IV Rank":"median","Contract Score":"max"
        })
        st.scatter_chart(plot, x="IV/HV", y="IV Rank", size="Contract Score")
        st.caption("Lower-left is the classic cheap-volatility quadrant: lower IV/HV and lower IV Rank.")

with tabs[4]:
    st.subheader("Data Audit")
    st.write(f"Rows: **{len(df):,}** · Columns: **{len(df.columns)}** · Symbols: **{df['Symbol'].nunique():,}**")
    audit = pd.DataFrame({
        "Column": EXPECTED,
        "Present": [c in df.columns for c in EXPECTED],
        "Non-null": [int(df[c].notna().sum()) if c in df.columns else 0 for c in EXPECTED]
    })
    st.dataframe(audit, use_container_width=True, hide_index=True)
    st.markdown("#### Raw uploaded data")
    st.dataframe(df.head(250), use_container_width=True, hide_index=True)
    st.caption(
        "Premium appears to be traded premium/turnover in this export, not the option purchase price. "
        "The app therefore uses Bid/Ask/Mid for contract price and keeps Premium as an informational activity field. "
        "Expected Move is displayed but not used in a dollar-gamma formula because its unit should be confirmed before mixing it with raw Gamma."
    )
