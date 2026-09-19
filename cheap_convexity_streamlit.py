import io, re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Cheap Convexity Discovery", page_icon="⚡", layout="wide")
st.title("⚡ Cheap Convexity Discovery")
st.caption("Barchart volatility screens → cheap optionality shortlist → calendar/GEX/chart workflow")

ALIASES = {
 "symbol":["symbol","ticker"], "name":["name","company name"],
 "iv":["imp vol","implied volatility","iv"],
 "hv30":["30d his vol","30d hist vol","30d hv","historical volatility"],
 "iv_hv":["iv/hv","iv hv","iv/hv ratio"],
 "iv_rank":["iv rank","implied volatility rank"],
 "iv_percentile":["iv percentile","iv pctile"],
 "iv_change":["iv chg","iv change"],
 "options_volume":["options vol","options volume","option volume"],
 "pc_volume":["p/c vol","put/call volume"], "earnings":["earnings","earnings date"],
 "last":["last","last price","price"]
}
def clean(x):
    return re.sub(r"\s+"," ",re.sub(r"[^a-z0-9]+"," ",str(x).lower())).strip()
def find_col(df,key):
    m={clean(c):c for c in df.columns}
    for a in ALIASES[key]:
        if clean(a) in m: return m[clean(a)]
    return None
def nums(s):
    x=s.astype(str).str.strip().str.replace(",","",regex=False).str.replace("$","",regex=False).str.replace("%","",regex=False)
    mult=pd.Series(1.0,index=x.index)
    mult[x.str.endswith("K",na=False)]=1e3; mult[x.str.endswith("M",na=False)]=1e6; mult[x.str.endswith("B",na=False)]=1e9
    return pd.to_numeric(x.str.replace(r"[KMB]$","",regex=True),errors="coerce")*mult
def load(up,source):
    raw=up.getvalue()
    try: df=pd.read_csv(io.BytesIO(raw),sep=None,engine="python")
    except: df=pd.read_csv(io.BytesIO(raw))
    out=pd.DataFrame()
    for k in ALIASES:
        c=find_col(df,k)
        if c is not None: out[k]=df[c]
    if "symbol" not in out: return pd.DataFrame()
    out["symbol"]=out["symbol"].astype(str).str.strip().str.upper()
    for c in ["iv","hv30","iv_hv","iv_rank","iv_percentile","iv_change","options_volume","pc_volume","last"]:
        if c in out: out[c]=nums(out[c])
    if "iv_hv" in out and len(out["iv_hv"].dropna()) and out["iv_hv"].dropna().median()>10: out["iv_hv"]/=100
    out["source"]=source
    return out
def first(s):
    z=s.dropna()
    return z.iloc[0] if len(z) else np.nan
def merge_frames(frames):
    x=pd.concat([f for f in frames if not f.empty],ignore_index=True,sort=False)
    num={"iv","hv30","iv_hv","iv_rank","iv_percentile","iv_change","options_volume","pc_volume","last"}
    agg={}
    for c in x.columns:
        if c=="symbol": continue
        if c=="source": agg[c]=lambda s:" | ".join(sorted(set(s.dropna().astype(str))))
        else: agg[c]=first
    d=x.groupby("symbol",as_index=False).agg(agg)
    if "iv" in d and "hv30" in d:
        calc=d["iv"]/d["hv30"].replace(0,np.nan)
        d["iv_hv"]=d.get("iv_hv",pd.Series(np.nan,index=d.index)).fillna(calc)
    return d
def prank(s,high=True):
    r=pd.to_numeric(s,errors="coerce").rank(pct=True)
    return ((r if high else 1-r)*100).fillna(0)
def score(d,w):
    d=d.copy()
    for c in ["iv","hv30","iv_hv","iv_rank","iv_percentile","iv_change","options_volume"]:
        if c not in d: d[c]=np.nan
    d["underpricing_score"]=prank((1/d["iv_hv"].replace(0,np.nan)).clip(upper=5))
    d["historical_score"]=pd.concat([(100-d["iv_rank"].clip(0,100)),(100-d["iv_percentile"].clip(0,100))],axis=1).mean(axis=1).fillna(50)
    d["acceleration_score"]=np.where(d["iv_change"].notna(),prank(d["iv_change"]),50)
    d["liquidity_score"]=prank(np.log1p(d["options_volume"].clip(lower=0)))
    den=max(sum(w),1)
    d["convexity_score"]=(w[0]*d["underpricing_score"]+w[1]*d["historical_score"]+w[2]*d["acceleration_score"]+w[3]*d["liquidity_score"])/den
    cheap=d["iv_hv"]<1; deep=d["iv_hv"]<=.8; low=(d["iv_rank"]<=25)|(d["iv_percentile"]<=25); rising=d["iv_change"]>0
    d["regime"]="NO EDGE"; d.loc[cheap,"regime"]="CONVEXITY"; d.loc[cheap&low,"regime"]="CHEAP CONVEXITY"; d.loc[deep&low,"regime"]="DEEP CHEAP CONVEXITY"; d.loc[d["iv_hv"]>=1.15,"regime"]="THETA / EXPENSIVE IV"
    d["setup"]=np.select([deep&low&rising,deep&low,cheap&rising,cheap],
      ["HV > IV + historically cheap + IV turning up","HV > IV + historically cheap","HV > IV + IV turning up","HV > IV"],default="No long-convexity trigger")

    # Best Expression Engine. This deliberately chooses structure, NOT direction.
    # Strong acceleration + deeply underpriced vol favors uncapped convexity.
    strong_accel = d["acceleration_score"] >= 70
    liquid = d["liquidity_score"] >= 35
    compressed = d["acceleration_score"] <= 45
    expensive = d["iv_hv"] >= 1.15
    hist_cheap = low

    d["best_expression"] = "WAIT"
    d["expression_reason"] = "No sufficiently clear volatility edge"

    m = deep & hist_cheap & strong_accel & liquid
    d.loc[m, "best_expression"] = "LONG OPTION / BACKSPREAD"
    d.loc[m, "expression_reason"] = "Deeply cheap IV vs HV + historically cheap vol + strong repricing; preserve convexity"

    m = cheap & ~strong_accel & liquid
    d.loc[m, "best_expression"] = "DIRECTIONAL OTM CALENDAR"
    d.loc[m, "expression_reason"] = "Movement is underpriced but not explosive; finance longer-dated optionality with front decay"

    m = hist_cheap & compressed & (d["iv_hv"] >= 0.85) & (d["iv_hv"] < 1.15)
    d.loc[m, "best_expression"] = "ATM VEGA CALENDAR"
    d.loc[m, "expression_reason"] = "Historically cheap IV with subdued acceleration; position for volatility repricing near spot"

    m = expensive & compressed
    d.loc[m, "best_expression"] = "ATM THETA CALENDAR"
    d.loc[m, "expression_reason"] = "IV is rich versus realized movement while volatility acceleration is weak"

    # Flag situations where a calendar may be the wrong tool because the move is already accelerating.
    d["calendar_warning"] = np.where(cheap & strong_accel,
        "Fast-move risk: compare outright option/backspread before using a calendar", "")
    return d.sort_values("convexity_score",ascending=False)

with st.sidebar:
    st.header("Score weights")
    wu=st.slider("IV/HV underpricing",0,60,40,5); wh=st.slider("Low IV Rank/Percentile",0,60,30,5)
    wa=st.slider("IV acceleration",0,40,15,5); wl=st.slider("Options liquidity",0,40,15,5)
    st.header("Filters")
    maxr=st.slider("Max IV/HV",.30,1.20,1.00,.05); maxrank=st.slider("Max IV Rank",0,100,40,5)
    minvol=st.number_input("Min options volume",0,step=5000,value=10000); mins=st.slider("Min score",0,100,50,5)
    topn=st.slider("Top candidates",5,50,20,5)

a,b,c=st.columns(3)
with a:
    lowrank=st.file_uploader("Low IV Rank / Percentile",type="csv"); highrank=st.file_uploader("High IV Rank / Percentile",type="csv")
with b:
    rising=st.file_uploader("Rising Volatility",type="csv"); falling=st.file_uploader("Falling Volatility",type="csv")
with c:
    lowvh=st.file_uploader("Low IV vs Realized Volatility",type="csv"); highvh=st.file_uploader("High IV vs Realized Volatility",type="csv")

ups={"LOW_IV_RANK":lowrank,"HIGH_IV_RANK":highrank,"RISING_VOL":rising,"FALLING_VOL":falling,"LOW_IV_VS_HV":lowvh,"HIGH_IV_VS_HV":highvh}
frames=[]; audit=[]
for label,u in ups.items():
    if u:
        try:
            f=load(u,label); frames.append(f); audit.append([label,len(f),", ".join(f.columns)])
        except Exception as e: st.error(f"{label}: {e}")
if not frames:
    st.info("Upload the six Barchart CSV exports."); st.stop()

master=merge_frames(frames); ranked=score(master,[wu,wh,wa,wl])
mask=(ranked["iv_hv"].le(maxr)|ranked["iv_hv"].isna()) & (ranked["iv_rank"].le(maxrank)|ranked["iv_rank"].isna()) & (ranked["options_volume"].ge(minvol)|ranked["options_volume"].isna()) & ranked["convexity_score"].ge(mins)
cand=ranked[mask].head(topn)

m1,m2,m3,m4=st.columns(4)
m1.metric("Unique symbols",len(master)); m2.metric("IV < HV",int((ranked["iv_hv"]<1).sum()))
m3.metric("Deep cheap",int(((ranked["iv_hv"]<=.8)&((ranked["iv_rank"]<=25)|(ranked["iv_percentile"]<=25))).sum())); m4.metric("Displayed",len(cand))

st.subheader("Structure mix")
expr_counts=cand["best_expression"].value_counts().rename_axis("Expression").reset_index(name="Candidates") if len(cand) else pd.DataFrame(columns=["Expression","Candidates"])
if len(expr_counts):
    st.dataframe(expr_counts,use_container_width=True,hide_index=True)

t1,t2,t3,t4=st.tabs(["🏆 Ranking","🔎 Inspector","🧭 Regime Map","🧪 Data Audit"])
with t1:
    cols=["symbol","convexity_score","regime","best_expression","iv","hv30","iv_hv","iv_rank","iv_percentile","iv_change","options_volume","underpricing_score","historical_score","acceleration_score","liquidity_score","setup","expression_reason","calendar_warning","source"]
    v=cand[[x for x in cols if x in cand]]
    st.dataframe(v,use_container_width=True,hide_index=True,height=620)
    st.download_button("Download ranked CSV",cand.to_csv(index=False).encode(),"cheap_convexity_ranked.csv","text/csv")
    st.caption("Score is relative to the uploaded universe and is not a directional signal.")
with t2:
    universe=cand["symbol"].tolist() or ranked["symbol"].tolist()
    s=st.selectbox("Ticker",universe); r=ranked[ranked["symbol"]==s].iloc[0]
    x1,x2,x3,x4=st.columns(4)
    x1.metric("Convexity Score",f"{r.convexity_score:.1f}"); x2.metric("IV/HV","—" if pd.isna(r.iv_hv) else f"{r.iv_hv:.2f}")
    x3.metric("IV Rank","—" if pd.isna(r.iv_rank) else f"{r.iv_rank:.1f}"); x4.metric("Options Volume","—" if pd.isna(r.options_volume) else f"{r.options_volume:,.0f}")
    st.write("**Regime:**",r.regime)
    st.success(f"Best expression: {r.best_expression}")
    st.write("**Why flagged:**",r.setup)
    st.write("**Why this structure:**",r.expression_reason)
    if r.calendar_warning:
        st.warning(r.calendar_warning)
    st.write("**Found in:**",r.source)
    comp=pd.DataFrame({"Score":[r.underpricing_score,r.historical_score,r.acceleration_score,r.liquidity_score]},index=["IV/HV underpricing","Historical cheapness","Acceleration","Liquidity"])
    st.bar_chart(comp)
    st.info("Direction is intentionally separate. Next: chart bias + GEX/DEX destination. For LONG OPTION/BACKSPREAD, compare uncapped convexity; for OTM CALENDAR, use the dealer destination as a strike candidate; for ATM calendars, prioritize spot/pin structure.")
with t3:
    p=ranked.dropna(subset=["iv_hv","iv_rank"])
    if len(p): st.scatter_chart(p,x="iv_hv",y="iv_rank",size="options_volume" if p["options_volume"].notna().any() else None)
    st.caption("Lower-left is especially interesting: IV below realized volatility while IV Rank is near historical lows.")
with t4:
    st.dataframe(pd.DataFrame(audit,columns=["Screen","Rows","Detected fields"]),use_container_width=True,hide_index=True)
    with st.expander("Merged universe"): st.dataframe(master,use_container_width=True,hide_index=True)
    st.markdown("**Logic:** 40% IV/HV underpricing + 30% historical cheapness + 15% IV acceleration + 15% options liquidity by default. Adjust weights in the sidebar.")
