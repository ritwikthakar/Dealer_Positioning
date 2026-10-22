from __future__ import annotations

import io, re
from collections import defaultdict
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Dealer Positioning Command Center', page_icon='🎯', layout='wide')

# ---------- Styling ----------
st.markdown('''
<style>
.block-container{padding-top:1.1rem;padding-bottom:2rem;max-width:1500px}
[data-testid="stMetric"]{background:rgba(127,127,127,.07);border:1px solid rgba(127,127,127,.18);padding:12px 14px;border-radius:12px}
.small-note{opacity:.72;font-size:.86rem}.pill{display:inline-block;padding:3px 9px;border-radius:999px;border:1px solid rgba(127,127,127,.3);margin-right:5px;font-size:.8rem}
/* Readable command-center cards: don't use st.metric for the top strip because
   Streamlit truncates long labels/values when six columns are squeezed. */
.dp-close{display:inline-block;margin:.15rem 0 .9rem 0;padding:.38rem .72rem;border-radius:12px;background:rgba(16,185,129,.12);color:#08783f;font-weight:700;font-size:1rem}
.dp-grid{display:grid;grid-template-columns:repeat(6,minmax(185px,1fr));gap:14px;margin:.2rem 0 1.15rem 0}
.dp-card{min-height:238px;border:1px solid rgba(127,127,127,.20);border-radius:18px;padding:20px 20px 16px 20px;box-shadow:0 1px 2px rgba(0,0,0,.02);overflow:visible}
.dp-card.regime{background:linear-gradient(135deg,rgba(16,185,129,.08),rgba(16,185,129,.025))}
.dp-card.gex{background:linear-gradient(135deg,rgba(59,130,246,.08),rgba(59,130,246,.025))}
.dp-card.dex{background:linear-gradient(135deg,rgba(124,58,237,.08),rgba(124,58,237,.025))}
.dp-card.iv{background:linear-gradient(135deg,rgba(245,158,11,.09),rgba(245,158,11,.025))}
.dp-card.flip{background:linear-gradient(135deg,rgba(239,68,68,.08),rgba(239,68,68,.025))}
.dp-card.em{background:linear-gradient(135deg,rgba(20,184,166,.08),rgba(20,184,166,.025))}
.dp-card.emreg{background:linear-gradient(135deg,rgba(99,102,241,.09),rgba(99,102,241,.025))}
.dp-label{font-size:1.02rem;font-weight:750;line-height:1.25;margin-bottom:18px;white-space:normal}
.dp-value{font-size:2.15rem;font-weight:760;line-height:1.08;letter-spacing:-.025em;margin-bottom:12px;white-space:normal;overflow-wrap:anywhere}
.dp-card.regime .dp-value{font-size:1.72rem;color:#08783f}.dp-card.gex .dp-value{color:#174f91}.dp-card.dex .dp-value{color:#4b2796}.dp-card.iv .dp-value{color:#a64d00}.dp-card.flip .dp-value{color:#b5121b}.dp-card.em .dp-value{color:#08783f;font-size:1.82rem}.dp-card.emreg .dp-value{color:#4338ca;font-size:1.72rem}
.dp-delta{display:inline-block;padding:6px 10px;border-radius:12px;background:rgba(34,197,94,.12);color:#08783f;font-weight:650;font-size:.93rem;margin-bottom:12px;white-space:normal}
.dp-sub{font-size:.94rem;line-height:1.5;color:rgba(49,61,82,.78);white-space:normal}
.dp-icon{margin-right:.35rem}
.emr-grid{display:grid;grid-template-columns:repeat(4,minmax(210px,1fr));gap:16px;margin:.5rem 0 1rem 0}
.emr-card{min-height:190px;border:1px solid rgba(127,127,127,.20);border-radius:18px;padding:20px;background:rgba(127,127,127,.045);overflow:visible}
.emr-label{font-size:1rem;font-weight:750;line-height:1.3;margin-bottom:18px;white-space:normal}
.emr-value{font-size:2.35rem;font-weight:760;line-height:1.08;letter-spacing:-.025em;margin-bottom:14px;white-space:normal;overflow-wrap:anywhere}
.emr-value.emr-regime{font-size:1.75rem}
.current-em-grid{grid-template-columns:repeat(auto-fit,minmax(min(100%,210px),1fr))}
.current-em-grid .emr-card{min-height:160px}
.current-em-grid .emr-value{white-space:normal;overflow:visible;text-overflow:clip;line-height:1.25}
.emr-sub{font-size:.9rem;line-height:1.45;color:rgba(49,61,82,.78);white-space:normal}
@media (max-width:1100px){.emr-grid{grid-template-columns:repeat(2,minmax(220px,1fr))}}
@media (max-width:650px){.emr-grid{grid-template-columns:1fr}.emr-card{min-height:auto}}
@media (max-width:1350px){.dp-grid{grid-template-columns:repeat(3,minmax(210px,1fr))}.dp-card{min-height:220px}}
@media (max-width:760px){.dp-grid{grid-template-columns:1fr}.dp-card{min-height:auto}.dp-value{font-size:2rem}}
/* Calendar Regime Engine cards */
.cre-grid{display:grid;grid-template-columns:repeat(4,minmax(220px,1fr));gap:16px;margin:.6rem 0 1.15rem 0}
.cre-card{min-height:180px;border:1px solid rgba(127,127,127,.20);border-radius:18px;padding:20px;background:rgba(127,127,127,.045);overflow:visible}
.cre-label{font-size:1rem;font-weight:750;line-height:1.3;margin-bottom:16px;white-space:normal}
.cre-value{font-size:1.72rem;font-weight:760;line-height:1.18;letter-spacing:-.02em;margin-bottom:12px;white-space:normal;overflow-wrap:anywhere;word-break:normal}
.cre-value.price{font-size:2rem}
.cre-sub{font-size:.9rem;line-height:1.45;color:rgba(49,61,82,.78);white-space:normal}
.cre-badge{display:inline-block;padding:5px 9px;border-radius:999px;background:rgba(34,197,94,.12);color:#08783f;font-weight:650;font-size:.88rem;margin-top:3px}
@media (max-width:1150px){.cre-grid{grid-template-columns:repeat(2,minmax(240px,1fr))}}
@media (max-width:650px){.cre-grid{grid-template-columns:1fr}.cre-card{min-height:auto}}
</style>''', unsafe_allow_html=True)

# ---------- Helpers ----------
def num(x):
    if pd.isna(x): return np.nan
    if isinstance(x,(int,float,np.number)): return float(x)
    s=str(x).strip().replace(',','').replace('$','').replace('%','')
    mult=1
    if s.endswith(('K','k')): mult=1e3; s=s[:-1]
    elif s.endswith(('M','m')): mult=1e6; s=s[:-1]
    elif s.endswith(('B','b')): mult=1e9; s=s[:-1]
    try:return float(s)*mult
    except:return np.nan

def pct(x):
    v=num(x); return v/100 if pd.notna(v) else np.nan

def fmt_money(x):
    if pd.isna(x): return '—'
    a=abs(x)
    if a>=1e9:return f'${x/1e9:,.2f}B'
    if a>=1e6:return f'${x/1e6:,.2f}M'
    if a>=1e3:return f'${x/1e3:,.1f}K'
    return f'${x:,.0f}'

def fmt_price(x): return '—' if pd.isna(x) else f'${x:,.2f}'

def zclip(v, lo=0, hi=100): return float(np.clip(v,lo,hi))

def first_valid(*xs):
    for x in xs:
        if pd.notna(x): return x
    return np.nan

# ---------- Robustness helpers (price/vol-scaled; ignore expired + negligible expirations) ----------
LEVEL_BAND=.35      # ignore dealer / max-pain levels further than 35% from spot (junk strikes)
MIN_EXP_SHARE=.02   # an expiration needs >=2% of gross exposure to drive headline levels

def weekly_unit(spot, iv30):
    """One-week 1-sigma move in $. Every distance rule scales off this, so the same logic works for a
    $100 stock and a $6,000 index (replaces hard-coded dollar floors that only suited high-priced tickers)."""
    base=spot*(iv30/100)/np.sqrt(52) if pd.notna(iv30) and iv30>0 else spot*.02
    return float(max(base, spot*.005))

def live_expirations(g, x, asof, min_share=MIN_EXP_SHARE):
    """Drop expirations on/before the analysis date (expired after the close) and expirations whose
    exposure is negligible (e.g. a $89-GEX expiry with 190 walls). Falls back rather than returning empty."""
    asof=pd.Timestamp(asof).normalize()
    def keep(df, cols):
        d=df[pd.to_datetime(df['expiration_dt']).dt.normalize()>asof].copy()
        if d.empty: return df.iloc[0:0].copy()  # Never resurrect expired expirations
        gross=sum(d[c].abs().fillna(0) for c in cols)
        k=d[gross/max(gross.sum(),1e-9)>=min_share]
        return (k if len(k) else d).sort_values('expiration_dt')
    return keep(g,['call_gex','put_gex']), keep(x,['call_dex','put_dex'])

def profile_flip(gxs, spot):
    """Zero-gamma price from the strike file's gex_profile (total net GEX at hypothetical spot prices).
    Returns the zero crossing nearest spot (linear interpolation), or NaN if unavailable."""
    if gxs is None or gxs.empty or 'gex_profile' not in gxs.columns or pd.isna(spot): return np.nan
    p=gxs[['strike','gex_profile']].copy()
    p['strike']=p['strike'].map(num); p['gex_profile']=p['gex_profile'].map(num)
    p=p.dropna().sort_values('strike')
    if len(p)<2: return np.nan
    s=p['strike'].to_numpy(float); v=p['gex_profile'].to_numpy(float)
    cross=[]
    for i in range(len(s)-1):
        if v[i]==0: cross.append(s[i])
        elif v[i]*v[i+1]<0: cross.append(s[i]+(s[i+1]-s[i])*(-v[i])/(v[i+1]-v[i]))
    if not cross: return np.nan
    cross=np.array(cross); return float(cross[np.argmin(np.abs(cross-spot))])

def aggregate_walls(gxs, spot):
    """Largest call-GEX strike above spot and largest put-GEX strike below spot across ALL expirations."""
    if gxs is None or gxs.empty or pd.isna(spot): return np.nan, np.nan
    z=gxs.copy()
    for c in ['strike','call_gex','put_gex']: z[c]=z[c].map(num)
    z=z.dropna(subset=['strike'])
    up=z[(z['strike']>spot)&(z['strike']<=spot*(1+LEVEL_BAND))].dropna(subset=['call_gex'])
    dn=z[(z['strike']<spot)&(z['strike']>=spot*(1-LEVEL_BAND))].dropna(subset=['put_gex'])
    cw=float(up.loc[up['call_gex'].idxmax(),'strike']) if len(up) and up['call_gex'].max()>0 else np.nan
    pw=float(dn.loc[dn['put_gex'].idxmin(),'strike']) if len(dn) and dn['put_gex'].min()<0 else np.nan
    return cw, pw

def exposure_weighted_flip(gg):
    """Fallback when no strike file: gross-gamma-weighted median of the per-expiration flips."""
    fl=gg.dropna(subset=['gamma_flip'])
    if fl.empty: return np.nan
    w=(fl['call_gex'].abs()+fl['put_gex'].abs()).to_numpy(float); fv=fl['gamma_flip'].to_numpy(float)
    o=np.argsort(fv); cum=np.cumsum(w[o])
    return float(fv[o][min(np.searchsorted(cum,cum[-1]/2),len(fv)-1)])

def flow_direction_bias(f):
    """Type-aware aggressor lean in [-1,1]: ask calls / bid puts = bullish; ask puts / bid calls = bearish.
    Mid prints are unclassified; multi-leg (ML*) prints are down-weighted, matching the level table."""
    side=f['Side'].astype(str).str.lower(); typ=f['Type'].astype(str).str.title()
    w=np.where(f['Code'].astype(str).str.upper().str.startswith('ML'),.45,1.0)
    prem=f['Premium'].fillna(0).clip(lower=0)*w
    bull=prem[((typ=='Call')&(side=='ask'))|((typ=='Put')&(side=='bid'))].sum()
    bear=prem[((typ=='Put')&(side=='ask'))|((typ=='Call')&(side=='bid'))].sum()
    return float((bull-bear)/max(bull+bear,1))

def usable_em(em, asof, min_dte=3):
    """Expected-move rows usable as levels/cards: drops 0-2 DTE rows (degenerate 300-500% IV artifacts)."""
    e=em.dropna(subset=['date']).sort_values('date').copy()
    dte=(e['date'].dt.normalize()-pd.Timestamp(asof).normalize()).dt.days
    iv=pd.to_numeric(e['implied_volatility'],errors='coerce')
    med=iv[dte>=min_dte].median()
    ok=(dte>=min_dte)
    if pd.notna(med): ok=ok&(iv<=3*med)
    out=e[ok]
    return out if len(out) else e

def classify(df):
    c=set(df.columns)
    if {'type','strike','expiration_date','bid','ask','implied_volatility'}.issubset(c): return 'option_chain'
    if {'strike','call_gex','put_gex','net_gex'}.issubset(c): return 'gex_strike'
    if {'Date','Close Price','GEX Net OI','DEX Net OI'}.issubset(c): return 'history'
    if {'expiration','net_gex','call_gex','put_gex','gamma_flip'}.issubset(c): return 'gex'
    if {'expiration','net_dex','call_dex','put_dex'}.issubset(c): return 'dex'
    if {'date','expected_move_amt','upper_price','lower_price'}.issubset(c): return 'em'
    if {'Symbol','Premium','Side','Code','Trade','Size'}.issubset(c): return 'flow'
    if {'Symbol','Vol/OI','Moneyness','Imp Vol','Open Int'}.issubset(c): return 'unusual'
    return None

def load_uploaded(files):
    out={}; errors=[]
    for f in files:
        try:
            d=pd.read_csv(f); k=classify(d)
            if k and k not in out: out[k]=d
            elif not k: errors.append(f'{f.name}: unrecognized columns')
            else: errors.append(f'{f.name}: duplicate {k} file ignored')
        except Exception as e: errors.append(f'{f.name}: {e}')
    return out, errors

def demo_data():
    paths={
      'history':'/mnt/data/option_history_coin(2).csv','gex':'/mnt/data/net_gex_exposure_by_expiration_COIN(4).csv',
      'dex':'/mnt/data/net_dex_exposure_by_expiration_COIN(1).csv','em':'/mnt/data/COIN_expected_move_2026-09-25_w(1).csv',
      'flow':'/mnt/data/coin-options-flow-09-20-2026(1).csv','unusual':'/mnt/data/unusual-options-activity-09-20-2026(1).csv'}
    try:return {k:pd.read_csv(v) for k,v in paths.items()}
    except:return {}

def prep(d):
    h=d['history'].copy(); h['Date']=pd.to_datetime(h['Date'],errors='coerce')
    hist_numeric=[c for c in h.columns if (c in ['Close Price','Option Volume Total','Option Volume Put-Call Ratio','OI Total','OI Put-Call Ratio','IV Rank','Total Option Contracts','Call Volume','Put Volume','Call OI','Put OI'] or c.startswith(('IV ','Max Pain','GEX ','DEX ')))]
    for c in hist_numeric: h[c]=h[c].map(num)
    h=h.sort_values('Date')
    g=d['gex'].copy(); x=d['dex'].copy()
    g['expiration_dt']=pd.to_datetime(g['expiration'],errors='coerce'); x['expiration_dt']=pd.to_datetime(x['expiration'],errors='coerce')
    for c in ['net_gex','call_gex','put_gex','call_wall','put_wall','gamma_flip']: g[c]=g[c].map(num)
    for c in ['net_dex','call_dex','put_dex','call_wall','put_wall']: x[c]=x[c].map(num)
    em=d['em'].copy(); em['date']=pd.to_datetime(em['date'],errors='coerce')
    for c in ['expected_move_amt','expected_move_percentage','upper_price','lower_price','implied_volatility']: em[c]=em[c].map(num)
    f=d['flow'].copy(); u=d['unusual'].copy()
    for c in ['Price~','Strike','DTE','Trade','Size','Premium','Volume','Open Int','Delta']:
        if c in f:f[c]=f[c].map(num)
    for c in ['Price~','Strike','DTE','Bid','Latest','Ask','Volume','Open Int','Vol/OI','Delta']:
        if c in u:u[c]=u[c].map(num)
    f['Exp Date']=pd.to_datetime(f['Exp Date'],errors='coerce'); u['Exp Date']=pd.to_datetime(u['Exp Date'],errors='coerce')
    return h,g,x,em,f,u

def expected_move_regime(h, verified_final_date=None, intraday=None):
    """Frozen prior-week IV30 bands; keep close and intraday evidence separate.

    A Friday row is NOT proof of a final settlement. A week completes only if a
    later week is observed, or the user explicitly verifies its last observation.
    intraday: optional {date, high, low, last} live snapshot; never a daily close.
    """
    empty = dict(streak=0, hit_rate=np.nan, utilization=np.nan,
                 label='Insufficient history', weekly=pd.DataFrame(), current=None,
                 basis='Daily closes only', intraday_status='Unknown',
                 closing_status='Unknown', reentry_status='Unknown')
    z=h.copy()
    z['Date']=pd.to_datetime(z['Date'],errors='coerce')
    for c in ['Close Price','IV 30d','High','Low']:
        if c in z: z[c]=pd.to_numeric(z[c],errors='coerce')
    z=z.dropna(subset=['Date','Close Price']).sort_values('Date').drop_duplicates('Date',keep='last')
    if z.empty: return empty
    z['week']=z['Date'].dt.to_period('W-FRI')
    groups=list(z.groupby('week',sort=True)); rows=[]
    for i in range(1,len(groups)):
        prev_week,prev=groups[i-1]; week,days=groups[i]
        if week.ordinal-prev_week.ordinal!=1:continue
        anchor=prev.iloc[-1]; iv=anchor['IV 30d']; center=anchor['Close Price']
        em=center*(iv/100)/np.sqrt(52)
        if not np.isfinite(em) or em<=0:continue
        upper=center+em;lower=center-em
        # All recorded daily closes are observations; latest Friday may be intraday.
        close_sides=np.where(days['Close Price']>upper,1,np.where(days['Close Price']<lower,-1,0))
        verified_date=pd.Timestamp(verified_final_date).date() if verified_final_date is not None else None
        is_final=i<len(groups)-1 or (verified_date is not None and days['Date'].iloc[-1].date()==verified_date)
        # For the current unfinished week, do not claim its last observation was a close.
        # An unverified last observation must not be counted as a closing price.
        close_valid=close_sides if is_final else close_sides[:-1]
        consecutive=0;last_side=0
        for side in close_valid:
            consecutive=consecutive+1 if side and side==last_side else (1 if side else 0)
            last_side=side
        has_hilo=('High' in days and 'Low' in days and days['High'].notna().all() and days['Low'].notna().all())
        high=float(days['High'].max()) if has_hilo else np.nan
        low=float(days['Low'].min()) if has_hilo else np.nan
        latest_snapshot=False; snap_last=np.nan
        if intraday and pd.Timestamp(intraday['date']).to_period('W-FRI')==week:
            latest_snapshot=True
            for field in ['high','low']:
                val=intraday.get(field,np.nan)
                if pd.notna(val):
                    if field=='high':high=max(high,val) if pd.notna(high) else float(val)
                    else:low=min(low,val) if pd.notna(low) else float(val)
            snap_last=intraday.get('last',np.nan)
        has_range=pd.notna(high) and pd.notna(low)
        # When only daily closes exist, show utilization as close-only, not verified range.
        observed_high=high if pd.notna(high) else float(days['Close Price'].max())
        observed_low=low if pd.notna(low) else float(days['Close Price'].min())
        up=max(0,(observed_high-center)/em);down=max(0,(center-observed_low)/em)
        breach=bool(high>upper or low<lower) if has_range else None
        closing_breach=bool(np.any(close_valid!=0)) if len(close_valid) else False
        reentry='Unknown'
        if latest_snapshot and pd.notna(snap_last) and breach is True:
            reentry='Yes' if lower<=snap_last<=upper else 'No'
        elif has_hilo and breach is True and is_final:
            reentry='Yes (close)' if lower<=days['Close Price'].iloc[-1]<=upper else 'No (close)'
        # Completed containment requires a genuine intraday range; otherwise close-only status.
        inside=bool((not breach) if breach is not None else not closing_breach)
        rows.append({'Week':week.end_time.normalize(),'Date':anchor['Date'],
                     'Close Price':center,'EM':em,'Upper EM':upper,'Lower EM':lower,
                     'Last Close':days['Close Price'].iloc[-1] if is_final else np.nan,
                     'Observed High':observed_high,'Observed Low':observed_low,
                     'EM Utilization':max(up,down), 'Close-to-Close Utilization':abs(days['Close Price'].iloc[-1]-center)/em,
                     'Close Breaches':int(np.sum(close_valid!=0)),
                     'Consecutive Close Breaches':consecutive,
                     'Intraday Breach':breach,'Closing Breach':closing_breach,
                     'Re-entry Inside EM':reentry,'Inside EM':inside,'Completed':is_final,
                     'Basis':'Daily high/low' if has_hilo else ('Manual intraday snapshot + closes' if latest_snapshot else 'Daily closes only')})
    if not rows:return empty
    w=pd.DataFrame(rows);completed=w[w.Completed];streak=0
    for inside in completed['Inside EM'].iloc[::-1]:
        if not inside:break
        streak+=1
    recent=completed.tail(8);hit_rate=recent['Inside EM'].mean() if len(recent) else np.nan
    current=w.iloc[-1].to_dict();util=current['EM Utilization']
    if current['Intraday Breach'] is True:
        label='Expansion warning — intraday EM breach'
    elif current['Closing Breach']:
        label='Expansion — closing EM breach'
    elif len(completed)<3:label='Insufficient history'
    else:
        util3=completed['EM Utilization'].tail(3).mean()
        if util3<.50 and hit_rate>=.75:label='Compression'
        elif util3<=.80 and hit_rate>=.625:label='Contained'
        elif util3<=1:label='Testing implied range'
        else:label='Expansion'
    intraday_status=('BREACH' if current['Intraday Breach'] else 'Inside observed range') if current['Intraday Breach'] is not None else 'Unknown — no high/low'
    closing_status='BREACH' if current['Closing Breach'] else ('No confirmed breach' if current['Completed'] else 'Pending verified close')
    return dict(streak=streak,hit_rate=hit_rate,utilization=util,label=label,
                weekly=w,current=current,basis=current['Basis'],
                intraday_status=intraday_status,closing_status=closing_status,
                reentry_status=current['Re-entry Inside EM'])

def regime_metrics(h,g,x,em,f,u,gxs=None):
    last=h.iloc[-1]; prev=h.iloc[-2] if len(h)>1 else last
    # Historical reference price: do not substitute an undated options-flow print.
    spot = last.get('Close Price', np.nan)
    gex=last.get('GEX Net OI',np.nan); dex=last.get('DEX Net OI',np.nan)
    dg=gex-prev.get('GEX Net OI',gex); dd=dex-prev.get('DEX Net OI',dex)
    iv=last.get('IV 30d',np.nan); ivr=last.get('IV Rank',np.nan)
    # Regime label: sign + recent change
    if gex>=0:
        reg='Positive Gamma — Strengthening' if dg>0 else 'Positive Gamma — Weakening'
    else:
        reg='Negative Gamma — Deepening' if dg<0 else 'Negative Gamma — Recovering'
    if np.sign(gex)!=np.sign(prev.get('GEX Net OI',gex)): reg='Gamma Regime Transition'
    # g/x are already filtered to live, material expirations (see live_expirations): no expired 0DTE chain.
    gg=g.sort_values('expiration_dt')
    gr=gg.iloc[0] if not gg.empty else pd.Series(dtype=float)
    # Aggregate gamma flip from the strike-level profile (zero crossing nearest spot); otherwise an
    # exposure-weighted flip across live expirations. Walls likewise come from the all-expiration strike file.
    flip=profile_flip(gxs,spot)
    flip_src='Strike-level GEX profile'
    if pd.isna(flip):
        flip=exposure_weighted_flip(gg); flip_src='Exposure-weighted across expirations'
    if pd.isna(flip): flip_src='Unavailable'
    agg_cw,agg_pw=aggregate_walls(gxs,spot)
    # Scores are descriptive, not directional recommendations
    g_scale=max(h['GEX Net OI'].abs().quantile(.75),1)
    d_scale=max(h['DEX Net OI'].abs().quantile(.75),1)
    stability=50 + 25*np.tanh(gex/g_scale) + 10*np.tanh(dg/g_scale)
    if pd.notna(flip) and pd.notna(spot): stability += 10*np.tanh((spot-flip)/(spot*.08))
    stability=zclip(stability)
    # directional pressure: positive means call/delta pressure; negative put pressure.
    # Flow is type-aware (ask puts are bearish, bid puts bullish) and down-weights multi-leg prints.
    flow_bias=flow_direction_bias(f)
    pressure=50 + 22*np.tanh(dex/d_scale) + 13*flow_bias
    pressure=zclip(pressure)
    return dict(spot=spot,gex=gex,dex=dex,dg=dg,dd=dd,iv=iv,ivr=ivr,regime=reg,stability=stability,pressure=pressure,
                gamma_flip=flip,flip_src=flip_src,flow_bias=flow_bias,
                call_wall=first_valid(agg_cw,gr.get('call_wall',np.nan)),put_wall=first_valid(agg_pw,gr.get('put_wall',np.nan)),
                maxpain=last.get('Max Pain 1d',np.nan))

def strike_gex_metrics(gxs, spot, unit=None):
    unit=unit if unit else weekly_unit(spot,np.nan)
    if gxs is None or gxs.empty or pd.isna(spot):
        return dict(near_magnet=np.nan, raw_magnet=np.nan, zone_low=np.nan, zone_high=np.nan, near_gex=np.nan, available=False)
    z=gxs.copy()
    for c in ['strike','call_gex','put_gex','net_gex']:
        z[c]=z[c].map(num)
    z=z.dropna(subset=['strike','net_gex'])
    pos=z[z['net_gex']>0].copy()
    if pos.empty:
        return dict(near_magnet=np.nan, raw_magnet=np.nan, zone_low=np.nan, zone_high=np.nan, near_gex=np.nan, available=True)
    raw=pos.loc[pos['net_gex'].idxmax()]
    # Near-spot magnet: positive GEX weighted by distance so a huge distant strike
    # does not overwhelm the dealer structure immediately relevant to spot.
    # Distances scale with the weekly 1-sigma move (price AND vol aware), not hard-coded dollar floors.
    window=1.5*unit
    near=pos[(pos['strike']-spot).abs()<=window].copy()
    if near.empty: near=pos.copy()
    decay=.5*unit
    near['magnet_score']=near['net_gex']*np.exp(-(near['strike']-spot).abs()/decay)
    nr=near.loc[near['magnet_score'].idxmax()]
    # Positive-GEX zone = material nearby bars (>=20% of strongest nearby +GEX).
    cutoff=max(near['net_gex'].max()*.20,0)
    zone=near[near['net_gex']>=cutoff]
    return dict(near_magnet=float(nr['strike']), raw_magnet=float(raw['strike']),
                zone_low=float(zone['strike'].min()) if len(zone) else np.nan,
                zone_high=float(zone['strike'].max()) if len(zone) else np.nan,
                near_gex=float(nr['net_gex']), available=True)

def calendar_regime_engine(h,g,x,m,emr,gxs=None):
    last=h.iloc[-1]; spot=m['spot']
    unit=weekly_unit(spot,m['iv'])
    near_spot=lambda v: pd.notna(v) and abs(float(v)/spot-1)<=LEVEL_BAND   # drop junk levels (e.g. 190 walls, 190 max pain)
    mp=[num(last.get(c)) for c in h.columns if c.startswith('Max Pain') and pd.notna(last.get(c))]
    gs=g.sort_values('expiration_dt').head(4); xs=x.sort_values('expiration_dt').head(4)
    lv=[]
    for _,r in gs.iterrows(): lv += [r.get('call_wall'),r.get('put_wall'),r.get('gamma_flip')]
    for _,r in xs.iterrows(): lv += [r.get('call_wall'),r.get('put_wall')]
    lv=np.array([float(v) for v in lv+mp if near_spot(v)])
    tol=max(.2*unit,spot*.004); magnet=np.nan; share=0
    if len(lv):
        cand=np.unique(np.round(lv*2)/2); counts=[np.sum(np.abs(lv-c)<=tol) for c in cand]; j=int(np.argmax(counts)); magnet=float(cand[j]); share=float(counts[j]/len(lv))
    # A defined range must have TWO boundaries that are genuinely distinct from
    # the central dealer magnet.  Previously a 7700 magnet could also be counted
    # as the lower boundary, incorrectly turning "7700 magnet + 7800 resistance"
    # into a double-calendar regime.
    boundary_levels=[]
    for df in (gs,xs):
        for _,r in df.iterrows():
            for c in ['call_wall','put_wall']:
                v=r.get(c,np.nan)
                if near_spot(v): boundary_levels.append(float(v))

    # Require each boundary to sit outside the magnet cluster.  0.50% of spot is
    # deliberately a little stricter than the 0.40% magnet-clustering tolerance.
    distinct_gap=max(spot*.005, tol*1.15)
    lows=[]; highs=[]
    for v in boundary_levels:
        anchor=magnet if pd.notna(magnet) else spot
        if v <= anchor-distinct_gap: lows.append(v)
        elif v >= anchor+distinct_gap: highs.append(v)

    # Use the median of repeated wall observations on each side; repeated
    # expirations agreeing on a level increase credibility rather than creating
    # a second boundary at the magnet itself.
    lower=np.nanmedian(lows) if lows else np.nan
    upper=np.nanmedian(highs) if highs else np.nan
    lower_n=len(lows); upper_n=len(highs)
    side_balance=min(lower_n,upper_n)
    bscore=min(100,25*side_balance) if side_balance else 0
    width=(upper-lower)/spot if pd.notna(lower) and pd.notna(upper) else np.nan

    # Double calendar requires: both distinct sides, meaningful total width, and
    # neither boundary overlapping the dealer magnet.
    distinct_from_magnet=(pd.notna(magnet) and pd.notna(lower) and pd.notna(upper)
                          and (magnet-lower)>=distinct_gap
                          and (upper-magnet)>=distinct_gap)
    two=bool(distinct_from_magnet and pd.notna(width) and width>=.012 and bscore>=50)
    # Short-dated IV buckets can be polluted by expiring/illiquid chains (IV 7d of 234% vs IV30 of 62%):
    # keep only readings within 0.4x-2.5x of IV30 before computing term-structure acceleration.
    iv_ref=m['iv']
    ok_iv=lambda v: pd.notna(v) and (pd.isna(iv_ref) or .4*iv_ref<=v<=2.5*iv_ref)
    vals=lambda names:[num(last.get(c)) for c in names if c in h.columns and ok_iv(num(last.get(c)))]
    short=vals(['IV 1d','IV 3d','IV 7d']); mid=vals(['IV 7d','IV 14d','IV 21d','IV 30d'])
    iva=(np.nanmean(short)/np.nanmean(mid)-1) if short and mid and np.nanmean(mid) else np.nan
    g5=m['gex']-h.iloc[-6].get('GEX Net OI',m['gex']) if len(h)>=6 else np.nan
    scale=max(h['GEX Net OI'].abs().quantile(.75),1); gm=g5/scale if pd.notna(g5) else np.nan
    util=emr.get('utilization',np.nan); ts=0
    ts += 25 if pd.notna(util) and util>=.8 else (12 if pd.notna(util) and util>=.6 else 0)
    ts += 20 if pd.notna(iva) and iva>.08 else (10 if pd.notna(iva) and iva>0 else 0)
    ts += 20 if pd.notna(gm) and gm<-.35 else (10 if pd.notna(gm) and gm<0 else 0); ts += 20 if m['gex']<0 else 0
    compression=emr.get('label') in ['Compression','Contained']; expansion=str(emr.get('label','')).startswith('Expansion') or (pd.notna(util) and util>1)
    if expansion: reg,strat='EXPANSION','OTM directional calendar / diagonal'
    elif ts>=50 or emr.get('label')=='Testing implied range': reg,strat='TRANSITION','Reduce / stop new calendars'
    elif compression and share>=.45 and not two: reg,strat='COMPRESSION — SINGLE MAGNET','ATM calendar'
    elif compression and two: reg,strat='COMPRESSION — DEFINED RANGE','Double calendar'
    elif compression: reg,strat='COMPRESSION — MIXED STRUCTURE','ATM calendar / wait for clearer walls'
    else: reg,strat='NEUTRAL / UNRESOLVED','Wait for confirmation'
    sx=strike_gex_metrics(gxs,spot,unit)
    return dict(regime=reg,strategy=strat,magnet=magnet,share=share,lower=lower,upper=upper,boundary=bscore,transition=min(100,ts),ivacc=iva,gexmom=gm,distinct_gap=distinct_gap,two_boundary=two,unit=unit,**sx)

def level_table(m,g,x,em,f,u):
    spot=m['spot']; rows=defaultdict(lambda:{'score':0,'evidence':[],'flow_premium':0,'unusual':0})
    def add(level, pts, label):
        if pd.isna(level): return
        k=round(float(level)*2)/2
        rows[k]['score']+=pts; rows[k]['evidence'].append(label)
    # nearest 4 expirations, diminishing weights
    gs=g.sort_values('expiration_dt').head(4)
    for i,r in gs.iterrows():
        add(r.call_wall,14,'GEX Call Wall'); add(r.put_wall,14,'GEX Put Wall')
    # One aggregate gamma flip (strike-level profile) instead of a different flip per expiration. It is the level
    # where dealer hedging changes sign, so it is weighted above a single wall; flow/unusual activity usually
    # sits on whole strikes, so without this the flip would sink below the strike-level rows.
    add(m['gamma_flip'],40,'Gamma Flip')
    for _,r in x.sort_values('expiration_dt').head(4).iterrows():
        add(r.call_wall,10,'DEX Call Wall'); add(r.put_wall,10,'DEX Put Wall')
    add(m['maxpain'],10,'Max Pain')
    # expected move: first 5 future dates, boundary points
    for _,r in em.sort_values('date').head(5).iterrows(): add(r.upper_price,12,'EM Upper'); add(r.lower_price,12,'EM Lower')
    # flow by strike, downweight mid and complex prints
    ff=f.copy(); ff['weight']=np.where(ff['Side'].astype(str).str.lower().eq('ask'),1.0,np.where(ff['Side'].astype(str).str.lower().eq('bid'),1.0,.35))
    ff['weight']*=np.where(ff['Code'].astype(str).str.upper().str.startswith('ML'),.45,1.0)
    agg=ff.groupby('Strike').apply(lambda z: pd.Series({'prem':(z.Premium.fillna(0)*z.weight).sum(),'vol':z.Volume.sum()}),include_groups=False).reset_index()
    if len(agg):
        p95=max(agg.prem.quantile(.95),1)
        for _,r in agg.iterrows():
            pts=min(22,22*np.log1p(r.prem)/np.log1p(p95)) if r.prem>0 else 0
            add(r.Strike,pts,'Flow'); k=round(float(r.Strike)*2)/2; rows[k]['flow_premium']+=r.prem
    uu=u.copy(); uu['voi']=uu['Vol/OI'].fillna(0).clip(lower=0)
    ua=uu.groupby('Strike').agg(max_voi=('voi','max'),contracts=('Volume','sum')).reset_index()
    if len(ua):
        for _,r in ua.iterrows():
            pts=min(20,5*np.log1p(r.max_voi)); add(r.Strike,pts,'Unusual'); rows[round(float(r.Strike)*2)/2]['unusual']=r.max_voi
    out=[]
    for lvl,v in rows.items():
        dist=(lvl-spot)/spot if spot else np.nan
        prox=max(0,8*(1-abs(dist)/.08)) if pd.notna(dist) else 0
        score=min(100,v['score']+prox)
        ev=list(dict.fromkeys(v['evidence']))
        if 'Gamma Flip' in ev: cls='Regime boundary'
        elif 'EM Upper' in ev: cls='Upper reaction / extension'
        elif 'EM Lower' in ev: cls='Lower reaction / extension'
        elif any('Wall' in e for e in ev) and lvl>spot: cls='Upper reaction / pivot'
        elif any('Wall' in e for e in ev) and lvl<spot: cls='Lower reaction / pivot'
        else: cls='Flow / positioning pivot'
        out.append([lvl,score,dist,v['flow_premium'],v['unusual'],', '.join(ev),cls])
    return pd.DataFrame(out,columns=['Level','Confluence','Distance','Weighted Flow Premium','Max Vol/OI','Evidence','Classification']).sort_values('Confluence',ascending=False)

# ---------- Sidebar / load ----------
st.title('🎯 Dealer Positioning Command Center')
st.caption('Regime → Positioning → Flow → Levels → Setup  •  Six required files + optional GEX-by-strike and option-chain enhancements')
with st.sidebar:
    st.header('Data')
    uploads=st.file_uploader('Drop 6 required CSVs + optional GEX-by-strike and option-chain CSVs',type='csv',accept_multiple_files=True,help='Files are identified automatically from their columns.')
    use_demo=st.toggle('Use bundled COIN files',value=not bool(uploads))
    st.link_button(
        "⚡ Convexity Screener",
        "https://convexity-bqhpmsmhg5lvruwjltk6bl.streamlit.app/",
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
        "🔬 Calendar Spread Regime Screening",
        "https://freedom-fuxffx4ohuuosfojdmffxl.streamlit.app/",
        use_container_width=True
    )
    
    st.link_button(
        "📅 Calendar Spread Screener",
        "https://freedom-hdf89xczpjdheb6kuq2qnz.streamlit.app/",
        use_container_width=True
    )
    
    st.link_button(
        "🚀 Option Flow Screener",
        "https://freedom-4rbsvk5mg32dybsiqwxwna.streamlit.app/",
        use_container_width=True
    )
    st.divider(); st.caption('Required: historical option data, GEX by expiration, DEX by expiration, expected move, options flow, unusual activity. Optional: GEX by strike for near-spot gamma magnet/zone detection.')

data=demo_data() if use_demo else load_uploaded(uploads)[0]
required={'history','gex','dex','em','flow','unusual'}
missing=required-set(data)
if missing:
    st.warning(f'Recognized required datasets: {', '.join(sorted(required & set(data))) or 'none'}. Missing: {', '.join(sorted(missing))}.')
    st.stop()

h,g,x,em,f,u=prep(data)
ticker=str(f['Symbol'].dropna().iloc[0]) if 'Symbol' in f and f['Symbol'].notna().any() else 'Ticker'

# Analysis-date selector. Historical regime/cards are calculated using only data
# available on or before the selected date. Dealer/flow files remain the uploaded
# positioning snapshot because they do not contain historical snapshots.
valid_dates=sorted(pd.to_datetime(h['Date'].dropna()).dt.date.unique())
if not valid_dates:
    st.error('No valid dates were found in the historical option data.')
    st.stop()
latest_date=valid_dates[-1]
header_left,header_right=st.columns([4,1])
with header_right:
    selected_date=st.date_input(
        'Analysis Date', value=latest_date, min_value=valid_dates[0], max_value=latest_date,
        help='Cards and historical calculations use data on or before this date.'
    )
# If a non-trading date is selected, use the latest available observation before it.
h_view=h[pd.to_datetime(h['Date']).dt.date<=selected_date].copy()
if h_view.empty:
    h_view=h.iloc[[0]].copy()
actual_date=pd.to_datetime(h_view['Date'].max()).date()
gxs=data.get('gex_strike')
# Live, material expirations only (no expired 0DTE chain, no negligible-exposure expiries) and usable EM rows.
g_live,x_live=live_expirations(g,x,actual_date)
em_live=usable_em(em,actual_date)
if g_live.empty or x_live.empty:
    st.error('No valid future GEX/DEX expirations for the selected analysis date. Upload a newer positioning snapshot; expired data will not be reused.')
    st.stop()
m=regime_metrics(h_view,g_live,x_live,em_live,f,u,gxs); levels=level_table(m,g_live,x_live,em_live,f,u)
# Optional verified intraday data. Never silently equate a last quote to a close.
with st.sidebar.expander('Weekly EM: intraday / verified close',expanded=False):
    st.caption('Optional current-session SPX observations. High/low are needed to confirm an intraday breach. A last quote is not a verified close.')
    use_intraday=st.checkbox('Supply intraday observations',value=False,key='em_manual_enable')
    manual_high=st.number_input('Session high',min_value=0.,value=0.,step=1.,key='em_manual_high',disabled=not use_intraday)
    manual_low=st.number_input('Session low',min_value=0.,value=0.,step=1.,key='em_manual_low',disabled=not use_intraday)
    manual_last=st.number_input('Latest traded SPX',min_value=0.,value=0.,step=1.,key='em_manual_last',disabled=not use_intraday)
    verify_close=st.checkbox('Latest historical row is a verified final session close',value=False,key='em_verified_close')
manual_snapshot=None
if use_intraday and manual_high>0 and manual_low>0 and manual_high>=manual_low:
    manual_snapshot={'date':actual_date,'high':manual_high,'low':manual_low,'last':manual_last if manual_last>0 else np.nan}
verified_final_date=actual_date if verify_close else None
emr=expected_move_regime(h_view,verified_final_date=verified_final_date,intraday=manual_snapshot)
cre=calendar_regime_engine(h_view,g_live,x_live,m,emr,gxs)

with st.expander('🔎 Data Integrity — source and date checks', expanded=False):
    st.write(f'History last observation: **{actual_date}** · Historical reference price: **{fmt_price(m["spot"])}**')
    st.write(f'GEX future expirations: **{len(g_live)}** · DEX future expirations: **{len(x_live)}** · Usable EM rows: **{len(em_live)}**')
    st.write(f'Flow rows: **{len(f):,}** · Unusual rows: **{len(u):,}** · GEX-by-strike: **{"loaded" if gxs is not None else "missing"}**')
    st.warning('GEX/DEX, flow, unusual, option-chain and GEX-by-strike files are uploaded snapshots, not historical time series. The Analysis Date selector cannot reconstruct their earlier states. Verify that all files were exported for the same session before relying on regime or BWB signals.')

with header_left:
    st.subheader(f'{ticker} • {actual_date}')
if actual_date != selected_date:
    st.caption(f'Using latest available market observation on {actual_date} for selected date {selected_date}.')
# Expected-move card: let the user pick a real, live expiration. Default = first expiration >=3 DTE, which
# skips the 0-2 DTE rows whose implied vol is a degenerate artifact (e.g. 535%).
em_by_date=em.assign(_d=em['date'].dt.date).drop_duplicates('_d').set_index('_d')
_dte=lambda d:(pd.Timestamp(d)-pd.Timestamp(actual_date)).days
em_dates=[d for d in g_live['expiration_dt'].dt.date.dropna().unique() if d in em_by_date.index and _dte(d)>=1]
if em_dates:
    with header_right:
        em_pick=st.selectbox('Expected move to', em_dates,
                             index=next((i for i,d in enumerate(em_dates) if _dte(d)>=3),0),
                             format_func=lambda d:f'{d:%b %d} ({_dte(d)} DTE)', key='em_target',
                             help='Expiration used for the Next Expected Move card.')
    firstem=em_by_date.loc[em_pick]
    em_tag=f'{em_pick:%b %d} · {_dte(em_pick)} DTE'
else:
    firstem=em_live.iloc[0]; em_tag=f'{firstem.date:%b %d}'
prev_close = h_view.iloc[-2]['Close Price'] if len(h_view)>1 else np.nan
close_chg = m['spot']-prev_close if pd.notna(m['spot']) and pd.notna(prev_close) else np.nan
close_pct = close_chg/prev_close if pd.notna(close_chg) and prev_close else np.nan
close_badge=f"Historical reference: {fmt_price(m['spot'])}"
if pd.notna(close_chg): close_badge += f" &nbsp; {'▲' if close_chg>=0 else '▼'} {close_chg:+.2f} ({close_pct:+.2%})"
st.markdown(f'<div class="dp-close">{close_badge}</div>',unsafe_allow_html=True)
flow_prices = pd.to_numeric(f.get('Price~',pd.Series(dtype=float)),errors='coerce').dropna()
if not flow_prices.empty:
    st.caption(f'Options-flow reference: {fmt_price(flow_prices.iloc[-1])} (print time not verified). Historical reference is sourced from option-history Close Price; it is not independently verified as an official session close.')

prev_gex=h_view.iloc[-2]['GEX Net OI'] if len(h_view)>1 else np.nan
prev_dex=h_view.iloc[-2]['DEX Net OI'] if len(h_view)>1 else np.nan
g5=(m['gex']-h_view.iloc[-6]['GEX Net OI']) if len(h_view)>=6 else np.nan
d5=(m['dex']-h_view.iloc[-6]['DEX Net OI']) if len(h_view)>=6 else np.nan
reg_parts=m['regime'].split(' — ',1)
reg_main=reg_parts[0]; reg_detail=reg_parts[1] if len(reg_parts)>1 else 'Transition'
em_amt=firstem.get('expected_move_amt',np.nan)
# Percent = amount / spot. (The old "if <1 then x100" heuristic turned a 0.91% move into 91%.)
em_pct=em_amt/m['spot']*100 if pd.notna(em_amt) and m['spot'] else firstem.get('expected_move_percentage',np.nan)

def _line(label,val):
    return f'{label}: {val}' if val not in (None,'') else ''

cards=f'''
<div class="dp-grid">
  <div class="dp-card regime"><div class="dp-label"><span class="dp-icon">▥</span>Dealer Regime</div><div class="dp-value">{reg_main}</div><div class="dp-delta">{reg_detail}</div><div class="dp-sub">{'Dealers likely stabilizing price with positive gamma exposure.' if m['gex']>=0 else 'Negative gamma can increase sensitivity to directional price moves.'}</div></div>
  <div class="dp-card gex"><div class="dp-label"><span class="dp-icon">↗</span>Net GEX</div><div class="dp-value">{fmt_money(m['gex'])}</div><div class="dp-delta">{'↑' if m['dg']>=0 else '↓'} {fmt_money(m['dg'])} (1D)</div><div class="dp-sub">Prev: {fmt_money(prev_gex)}<br>{'5D Change: '+fmt_money(g5) if pd.notna(g5) else ''}</div></div>
  <div class="dp-card dex"><div class="dp-label"><span class="dp-icon">▤</span>Net DEX</div><div class="dp-value">{fmt_money(m['dex'])}</div><div class="dp-delta">{'↑' if m['dd']>=0 else '↓'} {fmt_money(m['dd'])} (1D)</div><div class="dp-sub">Prev: {fmt_money(prev_dex)}<br>{'5D Change: '+fmt_money(d5) if pd.notna(d5) else ''}</div></div>
  <div class="dp-card iv"><div class="dp-label"><span class="dp-icon">%</span>IV30 / IV Rank</div><div class="dp-value">{m['iv']:.2f}%</div><div class="dp-delta">IV Rank: {m['ivr']:.2f}%</div><div class="dp-sub">Current 30-day implied volatility and its historical rank.</div></div>
  <div class="dp-card flip"><div class="dp-label"><span class="dp-icon">⊕</span>Aggregate Gamma Flip</div><div class="dp-value">{fmt_price(m['gamma_flip'])}</div><div class="dp-delta">Spot: {fmt_price(m['spot'])}</div><div class="dp-sub">Distance: {((m['spot']/m['gamma_flip'])-1):+.1%} ({m['spot']-m['gamma_flip']:+.2f})<br>{m['flip_src']}</div></div>
  <div class="dp-card em"><div class="dp-label"><span class="dp-icon">▣</span>Expected Move</div><div class="dp-value">{fmt_price(firstem.lower_price)} –<br>{fmt_price(firstem.upper_price)}</div><div class="dp-delta">{em_tag}</div><div class="dp-sub">Range: ± {fmt_price(em_amt).replace('$','')}<br>{f'({em_pct:.2f}%)' if pd.notna(em_pct) else ''}</div></div>
</div>'''
st.markdown(cards,unsafe_allow_html=True)

T1,T2,T3,T4,T5,T6,T7,T8=st.tabs(['🎯 Command Center','🧭 Calendar Regime Engine','↔ Expected Move Regime','📈 Historical Regime','🧲 Dealer Positioning','🌊 Flow Intelligence','🔬 Strike Explorer','🦋 Broken-Wing Butterfly'])

with T1:
    a,b,c=st.columns([1,1,2])
    a.metric('Stability Score',f"{m['stability']:.0f}/100",help='Higher = more stabilizing dealer backdrop based on GEX sign/trend and distance from gamma flip.')
    b.metric('Directional Pressure',f"{m['pressure']:.0f}/100",help='Descriptive pressure index. Above 50 = more positive delta/call-side pressure; below 50 = more negative/put-side pressure. Not a trade signal.')
    c.info(f"**Map:** Put wall {fmt_price(m['put_wall'])} • Call wall {fmt_price(m['call_wall'])} • Gamma flip {fmt_price(m['gamma_flip'])} • Max pain {fmt_price(m['maxpain'])}")

    st.markdown('#### Areas of Interest')
    ao=levels.copy()
    ao['Zone']=np.where(ao['Level'] < m['spot']-0.01,'Support',np.where(ao['Level'] > m['spot']+0.01,'Resistance','Pivot'))
    ao['Abs Distance']=ao['Distance'].abs()
    f1,f2,f3,f4=st.columns(4)
    zone_sel=f1.multiselect('Zone',['Support','Resistance','Pivot'],default=['Support','Resistance','Pivot'],key='aoi_zone')
    min_conf=f2.slider('Min Confluence',0,100,0,5,key='aoi_conf')
    max_dist=f3.slider('Max distance from spot',1,100,25,1,format='%d%%',key='aoi_dist')/100
    min_flow=f4.number_input('Min weighted flow premium',min_value=0.0,value=0.0,step=100000.0,key='aoi_flow')
    ao=ao[ao['Zone'].isin(zone_sel)&(ao['Confluence']>=min_conf)&(ao['Abs Distance']<=max_dist)&(ao['Weighted Flow Premium']>=min_flow)]
    show=ao.head(30).drop(columns=['Abs Distance']).copy()
    show['Confluence']=show['Confluence'].round(0).astype(int); show['Distance']=show['Distance'].map(lambda z:f'{z:+.1%}'); show['Weighted Flow Premium']=show['Weighted Flow Premium'].map(fmt_money); show['Max Vol/OI']=show['Max Vol/OI'].map(lambda z:f'{z:.1f}x' if z else '—')
    st.dataframe(show,use_container_width=True,hide_index=True,column_config={'Confluence':st.column_config.ProgressColumn('Confluence',min_value=0,max_value=100),'Zone':st.column_config.TextColumn('Zone',help='Below spot = Support; above spot = Resistance; at spot = Pivot.')})
    st.caption('Zone is relative to current spot: levels below spot are potential support; levels above spot are potential resistance. Price/TA confirmation is still required.')

    fig=go.Figure(); ee=em_live.sort_values('date').head(20)
    fig.add_trace(go.Scatter(x=ee.date,y=ee.upper_price,name='EM Upper',mode='lines'))
    fig.add_trace(go.Scatter(x=ee.date,y=ee.lower_price,name='EM Lower',mode='lines',fill='tonexty'))
    fig.add_hline(y=m['spot'],annotation_text=f'Spot {m["spot"]:.2f}')
    for y,name in [(m['gamma_flip'],'Gamma Flip'),(m['call_wall'],'Call Wall'),(m['put_wall'],'Put Wall')]:
        if pd.notna(y): fig.add_hline(y=y,line_dash='dot',annotation_text=name)
    fig.update_layout(height=420,title='Expected Move Envelope + Dealer Levels',xaxis_title='',yaxis_title='Price',legend_orientation='h')
    st.plotly_chart(fig,use_container_width=True)

with T2:
    st.markdown('#### 🧭 Calendar Regime Engine')
    st.caption('Classifies the dealer/volatility setup into the calendar regime playbook. When GEX-by-strike is uploaded, the engine separately identifies the structural magnet and the near-spot gamma magnet/positive-GEX zone.')
    regime_cards=f"""
    <div class="cre-grid">
      <div class="cre-card"><div class="cre-label">Current Regime</div><div class="cre-value">{cre['regime']}</div><div class="cre-sub">Calendar regime classification from EM, dealer clustering and transition inputs.</div></div>
      <div class="cre-card"><div class="cre-label">Preferred Structure</div><div class="cre-value">{cre['strategy']}</div><div class="cre-sub">Structure associated with the detected regime.</div></div>
      <div class="cre-card"><div class="cre-label">Structural Magnet</div><div class="cre-value price">{fmt_price(cre['magnet'])}</div><div class="cre-badge">↑ {cre['share']:.0%} cluster share</div><div class="cre-sub">GEX/DEX walls, gamma flip and max-pain clustering.</div></div>
      <div class="cre-card"><div class="cre-label">Transition Risk</div><div class="cre-value price">{cre['transition']:.0f}/100</div><div class="cre-sub">Higher readings indicate increasing risk of leaving the current regime.</div></div>
      <div class="cre-card"><div class="cre-label">Lower Boundary</div><div class="cre-value price">{fmt_price(cre['lower'])}</div><div class="cre-sub">Lower dealer-positioning boundary detected by the engine.</div></div>
      <div class="cre-card"><div class="cre-label">Upper Boundary</div><div class="cre-value price">{fmt_price(cre['upper'])}</div><div class="cre-sub">Upper dealer-positioning boundary detected by the engine.</div></div>
      <div class="cre-card"><div class="cre-label">Boundary Credibility</div><div class="cre-value price">{cre['boundary']:.0f}/100</div><div class="cre-sub">Strength of evidence supporting the detected range boundaries.</div></div>
      <div class="cre-card"><div class="cre-label">EM Regime</div><div class="cre-value">{emr['label']}</div><div class="cre-sub">Current expected-move containment classification.</div></div>
      <div class="cre-card"><div class="cre-label">Near-Spot GEX Magnet</div><div class="cre-value price">{fmt_price(cre['near_magnet'])}</div><div class="cre-sub">Distance-weighted positive GEX concentration near current spot.</div></div>
      <div class="cre-card"><div class="cre-label">Positive-GEX Zone</div><div class="cre-value price">{fmt_price(cre['zone_low'])} – {fmt_price(cre['zone_high'])}</div><div class="cre-sub">Material nearby positive-GEX strikes from the optional strike-level file.</div></div>
      <div class="cre-card"><div class="cre-label">Largest +GEX Strike</div><div class="cre-value price">{fmt_price(cre['raw_magnet'])}</div><div class="cre-sub">Largest positive net-GEX strike in the uploaded strike distribution.</div></div>
      <div class="cre-card"><div class="cre-label">GEX-by-Strike</div><div class="cre-value">{'Loaded' if cre['available'] else 'Not loaded'}</div><div class="cre-sub">Optional enhancement; regime engine still works with the six required files.</div></div>
    </div>
    """
    st.markdown(regime_cards,unsafe_allow_html=True)
    st.markdown('##### Regime Playbook')
    play=pd.DataFrame([['Compression — Single Magnet','ATM calendar','Dealer/max-pain levels cluster around one strike'],['Compression — Defined Range','Double calendar','Credible lower and upper dealer boundaries'],['Transition','Reduce / stop new calendars','EM/IV acceleration and weakening gamma'],['Expansion','OTM directional calendar / diagonal','Use price/TA to confirm direction']],columns=['Regime','Structure','Confirmation'])
    st.dataframe(play,use_container_width=True,hide_index=True)
    st.markdown('##### Diagnostics')
    ivtxt='—' if pd.isna(cre['ivacc']) else f"{cre['ivacc']:+.1%}"; gtxt='—' if pd.isna(cre['gexmom']) else f"{cre['gexmom']:+.2f}"
    diag=pd.DataFrame([['Magnet cluster share',f"{cre['share']:.0%}"],['IV short-vs-mid acceleration',ivtxt],['5D normalized GEX momentum',gtxt],['Latest EM utilization','—' if pd.isna(emr['utilization']) else f"{emr['utilization']:.2f}x"],['Near-spot GEX magnet',fmt_price(cre['near_magnet'])],['Positive-GEX zone',f"{fmt_price(cre['zone_low'])} – {fmt_price(cre['zone_high'])}"]],columns=['Input','Reading'])
    st.dataframe(diag,use_container_width=True,hide_index=True)

    if cre['available'] and gxs is not None:
        gz=gxs.copy()
        for cc in ['strike','net_gex']: gz[cc]=gz[cc].map(num)
        gz=gz.dropna(subset=['strike','net_gex'])
        w=max(2.5*cre['unit'],m['spot']*.05)
        gz=gz[(gz['strike']>=m['spot']-w)&(gz['strike']<=m['spot']+w)].sort_values('strike')
        if len(gz):
            fig=go.Figure(go.Bar(x=gz['strike'],y=gz['net_gex'],name='Net GEX'))
            fig.add_hline(y=0)
            fig.add_vline(x=m['spot'],line_dash='dash',annotation_text='Spot')
            if pd.notna(cre['magnet']): fig.add_vline(x=cre['magnet'],line_dash='dot',annotation_text='Structural Magnet')
            if pd.notna(cre['near_magnet']): fig.add_vline(x=cre['near_magnet'],line_dash='dot',annotation_text='Near-Spot GEX Magnet')
            fig.update_layout(height=380,title='Strike-Level Net GEX Near Spot',xaxis_title='Strike',yaxis_title='Net GEX')
            st.plotly_chart(fig,use_container_width=True)
    st.info('Expansion direction is intentionally left to price/TA confirmation before choosing call vs put expression.')

with T3:
    st.markdown('#### Expected Move Regime')
    st.caption('Weekly expected-move containment and utilization. Use this as volatility-regime context alongside GEX and the gamma flip—not as a breakout timer.')
    hit_txt=f"{emr['hit_rate']:.0%}" if pd.notna(emr['hit_rate']) else '—'
    util_txt=f"{emr['utilization']:.2f}x" if pd.notna(emr['utilization']) else '—'
    em_cards=f'''<div class="emr-grid">
      <div class="emr-card"><div class="emr-label">Consecutive Weeks Inside EM</div><div class="emr-value">{emr['streak']}</div><div class="emr-sub">Completed weeks inside frozen IV30 bands on the available observations.</div></div>
      <div class="emr-card"><div class="emr-label">8-Week Containment</div><div class="emr-value">{hit_txt}</div><div class="emr-sub">Share of the latest eight completed weeks contained on the available observations.</div></div>
      <div class="emr-card"><div class="emr-label">Latest EM Utilization</div><div class="emr-value">{util_txt}</div><div class="emr-sub">Maximum observed excursion from the frozen prior-week close ÷ expected move.</div></div>
      <div class="emr-card"><div class="emr-label">Expected Move Regime</div><div class="emr-value emr-regime">{emr['label']}</div><div class="emr-sub">Latest observed week included; a breach overrides historical compression.</div></div>
    </div>'''
    st.markdown(em_cards,unsafe_allow_html=True)
    st.caption('Bands freeze the prior trading week’s final close and IV30: close × IV30 / √52. These estimates can differ from broker/straddle expected moves. All status is as of the latest uploaded observation.')
    current=emr.get('current')
    if current:
        status='Completed' if current['Completed'] else 'In progress'
        st.markdown(f"**Latest week ({status}) — {current['Week']:%Y-%m-%d}**")
        observed_status=emr['intraday_status']
        st.markdown(f"""<div class="emr-grid current-em-grid">
          <div class="emr-card"><div class="emr-label">Frozen lower / upper</div><div class="emr-value emr-regime">{current['Lower EM']:,.1f}<br/>{current['Upper EM']:,.1f}</div></div>
          <div class="emr-card"><div class="emr-label">Daily closing breaches</div><div class="emr-value">{current['Close Breaches']}</div></div>
          <div class="emr-card"><div class="emr-label">Consecutive closing breaches</div><div class="emr-value">{current['Consecutive Close Breaches']}</div></div>
          <div class="emr-card"><div class="emr-label">Observed status</div><div class="emr-value emr-regime">{observed_status}</div></div>
        </div>""", unsafe_allow_html=True)
        st.caption(f"Anchor: {current['Date']:%Y-%m-%d}. Coverage: {current['Basis']}. Last verified close: {fmt_price(current['Last Close'])}.")
        import html as _html
        _em_statuses=[('Intraday EM breach',emr['intraday_status']),('Closing EM breach',emr['closing_status']),('Re-entry inside EM',emr['reentry_status'])]
        st.markdown('''<style>
        .em-status-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin:12px 0}
        .em-status-card{min-width:0;border:1px solid rgba(127,127,127,.22);border-radius:13px;padding:17px;background:rgba(127,127,127,.05)}
        .em-status-label{font-size:.9rem;line-height:1.4;margin-bottom:10px}
        .em-status-value{font-size:1.25rem;font-weight:650;line-height:1.4;white-space:normal;overflow-wrap:anywhere}
        @media(max-width:900px){.em-status-grid{grid-template-columns:1fr}}
        </style>''',unsafe_allow_html=True)
        st.markdown('<div class="em-status-grid">'+''.join(
            '<div class="em-status-card"><div class="em-status-label">'+_html.escape(str(label))+'</div>'
            '<div class="em-status-value">'+_html.escape(str(value))+'</div></div>'
            for label,value in _em_statuses)+'</div>',unsafe_allow_html=True)
        if not current['Completed']:st.info('Current week remains IN PROGRESS until a later week is observed or the latest final close is explicitly verified.')
        if current['Basis']=='Daily closes only':
            st.warning('No complete daily high/low data: intraday containment cannot be verified. Inside EM means daily closes only.')
    if len(emr['weekly']):
        ew=emr['weekly'].tail(12).copy()
        ew['Inside EM']=ew['Inside EM'].map({True:'Yes',False:'No'})
        st.dataframe(ew[['Week','Date','Close Price','Lower EM','Upper EM','EM',
                         'Observed High','Observed Low','EM Utilization',
                         'Close-to-Close Utilization','Close Breaches',
                         'Consecutive Close Breaches','Intraday Breach','Closing Breach','Re-entry Inside EM','Inside EM','Completed','Basis']],
                     use_container_width=True,hide_index=True,height=455)

with T4:
    days=st.segmented_control('Window',['20D','60D','All'],default='60D')
    hh=h_view.tail(20 if days=='20D' else 60 if days=='60D' else len(h_view))
    fig=go.Figure(); fig.add_trace(go.Scatter(x=hh.Date,y=hh['Close Price'],name='Close')); fig.add_trace(go.Scatter(x=hh.Date,y=hh['Max Pain 1d'],name='Max Pain',line=dict(dash='dot'))); fig.update_layout(height=350,title='Price vs Max Pain'); st.plotly_chart(fig,use_container_width=True)
    col1,col2=st.columns(2)
    for col,y,title in [(col1,'GEX Net OI','Net GEX History'),(col2,'DEX Net OI','Net DEX History')]:
        fig=go.Figure(go.Bar(x=hh.Date,y=hh[y],name=y)); fig.add_hline(y=0); fig.update_layout(height=330,title=title); col.plotly_chart(fig,use_container_width=True)
    fig=go.Figure(); fig.add_trace(go.Scatter(x=hh.Date,y=hh['IV 30d'],name='IV30')); fig.add_trace(go.Scatter(x=hh.Date,y=hh['IV Rank'],name='IV Rank')); fig.update_layout(height=330,title='Volatility Regime'); st.plotly_chart(fig,use_container_width=True)

with T5:
    pos=pd.merge(g_live,x_live,on='expiration_dt',how='outer',suffixes=('_gex','_dex')).sort_values('expiration_dt')
    pos['GEX Share']=pos.net_gex.abs()/max(pos.net_gex.abs().sum(),1); pos['DEX Share']=pos.net_dex.abs()/max(pos.net_dex.abs().sum(),1)
    analysis_date=pd.Timestamp(actual_date).normalize()
    pos['DTE']=(pd.to_datetime(pos['expiration_dt']).dt.normalize()-analysis_date).dt.days
    em_map=em_live.copy(); em_map['date']=pd.to_datetime(em_map['date']).dt.normalize()
    em_map['Expected Move']=em_map.apply(lambda r:f"{fmt_price(r['lower_price'])} – {fmt_price(r['upper_price'])} (±{fmt_price(r['expected_move_amt']).replace('$','')}, {r['expected_move_percentage']:.2f}%)",axis=1)
    pos=pos.merge(em_map[['date','Expected Move']],left_on=pd.to_datetime(pos['expiration_dt']).dt.normalize(),right_on='date',how='left').drop(columns=['date'])

    st.markdown('#### Dealer Positioning by Expiration')
    q1,q2,q3,q4=st.columns(4)
    max_dte=int(max(pos['DTE'].max(),1)); min_dte=q1.number_input('Min DTE',0,max_dte,0,key='pos_mindte'); max_dte_sel=q2.number_input('Max DTE',0,max_dte,max_dte,key='pos_maxdte')
    gsign=q3.multiselect('GEX sign',['Positive','Negative'],default=['Positive','Negative'],key='pos_gsign'); dsign=q4.multiselect('DEX sign',['Positive','Negative'],default=['Positive','Negative'],key='pos_dsign')
    filt=pos[(pos.DTE>=min_dte)&(pos.DTE<=max_dte_sel)].copy()
    filt=filt[((filt.net_gex>=0)&('Positive' in gsign))|((filt.net_gex<0)&('Negative' in gsign))]
    filt=filt[((filt.net_dex>=0)&('Positive' in dsign))|((filt.net_dex<0)&('Negative' in dsign))]
    # Expiration-specific dealer levels: linked to the Command Center's
    # "Expected move to" selection, but selectable independently here.
    # Use only live, filtered expirations; do not include expired 0DTE rows.
    st.markdown('#### Dealer Levels — Selected Expiration')
    eligible=filt[filt['DTE']>0].dropna(subset=['expiration_dt']).copy()
    if eligible.empty:
        st.info('No active expiration matches these filters. Adjust DTE or GEX/DEX sign filters to see dealer levels.')
    else:
        eligible=eligible.sort_values('expiration_dt').drop_duplicates('expiration_dt')
        available_dates=list(pd.to_datetime(eligible['expiration_dt']).dt.date)
        default_exp=em_pick if 'em_pick' in locals() and em_pick in available_dates else available_dates[0]
        chosen_exp=st.selectbox(
            'Dealer levels for expiration', available_dates,
            index=available_dates.index(default_exp),
            format_func=lambda d:f'{d:%b %d, %Y} ({_dte(d)} DTE)',
            key='dealer_levels_exp',
            help='Defaults to the Expected move to expiration when it passes the table filters. Select another active expiration to compare levels.'
        )
        chosen=eligible.loc[pd.to_datetime(eligible['expiration_dt']).dt.date==chosen_exp].iloc[0]
        wall_cols=st.columns(4)
        wall_cols[0].metric('Put Wall',fmt_price(chosen['put_wall_gex']))
        wall_cols[1].metric('Call Wall',fmt_price(chosen['call_wall_gex']))
        wall_cols[2].metric('Gamma Flip',fmt_price(chosen['gamma_flip']))
        wall_cols[3].metric('Expiration Net GEX',fmt_money(chosen['net_gex']))
        st.caption(f'Selected: {chosen_exp:%b %d, %Y} · {int(chosen["DTE"])} DTE. These are expiration-specific GEX levels, not the aggregate gamma flip on the Command Center. Source: uploaded positioning files.')
    disp=filt[['expiration_dt','DTE','net_gex','net_dex','call_gex','put_gex','call_dex','put_dex','call_wall_gex','put_wall_gex','gamma_flip','Expected Move','GEX Share','DEX Share']].copy()
    disp.columns=['Expiration','DTE','Net GEX','Net DEX','Call GEX','Put GEX','Call DEX','Put DEX','Call Wall','Put Wall','Gamma Flip','Expected Move','GEX Share','DEX Share']
    st.dataframe(disp,use_container_width=True,hide_index=True,column_config={'GEX Share':st.column_config.ProgressColumn(format='%.1%%',min_value=0,max_value=1),'DEX Share':st.column_config.ProgressColumn(format='%.1%%',min_value=0,max_value=1)})
    st.caption('Expected Move is matched to the expiration date when that date exists in the uploaded expected-move file; otherwise it is left blank.')
    col1,col2=st.columns(2)
    fig=go.Figure(go.Bar(x=filt.expiration_dt,y=filt.net_gex)); fig.add_hline(y=0); fig.update_layout(height=350,title='Net GEX by Expiration'); col1.plotly_chart(fig,use_container_width=True)
    fig=go.Figure(go.Bar(x=filt.expiration_dt,y=filt.net_dex)); fig.add_hline(y=0); fig.update_layout(height=350,title='Net DEX by Expiration'); col2.plotly_chart(fig,use_container_width=True)

with T6:
    c1,c2,c3,c4=st.columns(4)
    maxd=int(max(f.DTE.max(skipna=True),u.DTE.max(skipna=True),1)); dte=c1.slider('Max DTE',0,min(maxd,900),60)
    typ=c2.multiselect('Type',['Call','Put'],default=['Call','Put']); minprem=c3.number_input('Min premium $',0.0,value=0.0,step=10000.0); minvoi=c4.number_input('Min unusual Vol/OI',0.0,value=2.0,step=.5)
    ff=f[(f.DTE<=dte)&f.Type.isin(typ)&(f.Premium>=minprem)].copy(); uu=u[(u.DTE<=dte)&u.Type.isin(typ)&(u['Vol/OI']>=minvoi)].copy()
    st.markdown('#### Flow by Strike')
    fa=ff.groupby(['Strike','Type'],as_index=False).agg(Premium=('Premium','sum'),Volume=('Volume','sum'),Trades=('Strike','size'))
    fig=go.Figure()
    for t in typ:
        z=fa[fa.Type==t]; fig.add_trace(go.Bar(x=z.Strike,y=z.Premium,name=t))
    fig.add_vline(x=m['spot'],annotation_text='Spot'); fig.update_layout(height=380,title='Premium Concentration',barmode='group',xaxis_title='Strike',yaxis_title='Premium'); st.plotly_chart(fig,use_container_width=True)
    a,b=st.columns(2)
    a.markdown('**Largest flow prints**'); a.dataframe(ff.sort_values('Premium',ascending=False).head(30),use_container_width=True,hide_index=True)
    b.markdown('**Unusual / new-positioning candidates**'); b.dataframe(uu.sort_values('Vol/OI',ascending=False).head(30),use_container_width=True,hide_index=True)
    st.caption('Mid-market and multi-leg prints are intentionally treated with lower directional confidence in the confluence model.')

with T7:
    strikes=sorted(set(f.Strike.dropna()).union(set(u.Strike.dropna())))
    default=min(range(len(strikes)),key=lambda i:abs(strikes[i]-m['spot'])) if strikes else 0
    strike=st.selectbox('Strike',strikes,index=default if strikes else 0)
    sf=f[f.Strike==strike].copy(); su=u[u.Strike==strike].copy()
    a,b,c,d=st.columns(4); a.metric('Strike',fmt_price(strike)); b.metric('Distance from spot',f'{(strike/m["spot"]-1):+.2%}'); c.metric('Flow premium',fmt_money(sf.Premium.sum())); d.metric('Max Vol/OI',f'{su["Vol/OI"].max():.1f}x' if len(su) else '—')
    evid=levels[np.isclose(levels.Level,strike,atol=.26)]
    if len(evid): st.info(f"**Level context:** {evid.iloc[0].Evidence} • {evid.iloc[0].Classification} • Confluence {evid.iloc[0].Confluence:.0f}/100")
    left,right=st.columns(2); left.markdown('**Flow at strike**'); left.dataframe(sf.sort_values('Premium',ascending=False),use_container_width=True,hide_index=True); right.markdown('**Unusual activity at strike**'); right.dataframe(su.sort_values('Vol/OI',ascending=False),use_container_width=True,hide_index=True)

st.divider(); st.caption('Dealer positioning is an analytical map, not a prediction. Flow side and multi-leg prints can be ambiguous; use price/TA confirmation before interpreting a level as support or resistance.')
# ---------- Broken-wing butterfly research engine ----------
def bfly_chain(raw):
    q=raw.copy()
    q.columns=[str(c).strip().lower() for c in q.columns]
    required={'type','strike','expiration_date','bid','ask','implied_volatility'}
    if not required.issubset(q.columns):
        raise ValueError('Option chain missing: '+', '.join(sorted(required-set(q.columns))))
    for c in ['strike','bid','ask','mid','implied_volatility','open_interest','volume','delta','gamma','theta','vega']:
        if c in q: q[c]=q[c].map(num)
    q['type']=q['type'].astype(str).str.upper().str.strip()
    q['expiry']=pd.to_datetime(q['expiration_date'].astype(str).str.split(':').str[0],errors='coerce').dt.normalize()
    q=q[q['type'].isin(['PUT','CALL']) & q['expiry'].notna()].copy()
    q=q[q['strike'].notna() & q['bid'].notna() & q['ask'].notna() & (q['bid']>=0) & (q['ask']>=q['bid'])].copy()
    q['mid']=(q['bid']+q['ask'])/2
    q=q.drop_duplicates(['expiry','type','strike'],keep='first')
    return q

def bfly_payoff(spot,typ,klo,kmid,khi,debit):
    # A put BWB: +P(khi)-2P(kmid)+P(klo). Call BWB: +C(klo)-2C(kmid)+C(khi).
    s=np.asarray(spot,dtype=float)
    if typ=='PUT': intrinsic=np.maximum(khi-s,0)-2*np.maximum(kmid-s,0)+np.maximum(klo-s,0)
    else: intrinsic=np.maximum(s-klo,0)-2*np.maximum(s-kmid,0)+np.maximum(s-khi,0)
    return (intrinsic-debit)*100

def bfly_black_scholes(spot,strike,t,vol,typ,r=0.0):
    from math import erf
    s=np.asarray(spot,dtype=float)
    normal_cdf=lambda z: .5*(1+np.vectorize(erf,otypes=[float])(np.asarray(z)/np.sqrt(2)))
    t=max(float(t),0)
    if t<=0: return np.maximum((s-strike) if typ=='CALL' else (strike-s),0)
    vol=max(float(vol),.0001)
    root=np.sqrt(t); d1=(np.log(np.maximum(s,1e-9)/strike)+(r+.5*vol**2)*t)/(vol*root)
    d2=d1-vol*root
    if typ=='CALL': return s*normal_cdf(d1)-strike*np.exp(-r*t)*normal_cdf(d2)
    return strike*np.exp(-r*t)*normal_cdf(-d2)-s*normal_cdf(-d1)

def bfly_candidates(chain,expiry,spot,typ,targets,em_amt,gxs,wing_sizes,max_debit,spread_cap,min_oi,allow_credit=False):
    q=chain[(chain['expiry']==pd.Timestamp(expiry)) & (chain['type']==typ)].copy()
    if q.empty:return pd.DataFrame()
    q=q.set_index('strike',drop=False)
    q=q[~q.index.duplicated(keep='first')]
    strikes=np.array(sorted(q.index.to_numpy(dtype=float)))
    if gxs is not None and not gxs.empty:
        gz=gxs.copy()
        gz['strike']=gz['strike'].map(num);gz['net_gex']=gz['net_gex'].map(num)
        gz=gz.dropna(subset=['strike','net_gex']).groupby('strike',as_index=False)['net_gex'].sum().sort_values('strike')
    else: gz=pd.DataFrame(columns=['strike','net_gex'])
    rows=[]
    for body in targets:
        if body not in q.index:continue
        for lower_width in wing_sizes:
            for upper_width in wing_sizes:
                if lower_width==upper_width:continue
                klo=body-lower_width; khi=body+upper_width
                if klo not in q.index or khi not in q.index:continue
                # Restrict to the useful broken-wing orientation: overshoot on the intended directional side
                if typ=='PUT' and upper_width<=lower_width:continue
                if typ=='CALL' and lower_width<=upper_width:continue
                legs=[q.loc[klo],q.loc[body],q.loc[khi]]
                if any(z.get('open_interest',0)<min_oi for z in legs):continue
                if any(z['ask']<=0 or (z['ask']-z['bid'])/max(z['mid'],.01)>spread_cap for z in legs):continue
                debit_mid=float(legs[0]['mid']-2*legs[1]['mid']+legs[2]['mid'])
                debit_nat=float(legs[0]['ask']-2*legs[1]['bid']+legs[2]['ask'])
                if debit_nat*100>max_debit or (debit_nat<=0 and not allow_credit):continue
                # Worst-case max loss and max gain across all piecewise-linear knots and tails.
                test_spots=np.array([0.,klo,body,khi,max(spot,khi)*2])
                payoff=bfly_payoff(test_spots,typ,klo,body,khi,debit_nat)
                max_loss=float(max(0,-payoff.min())); max_profit=float(payoff.max())
                if max_loss<=0:continue
                path_low=min(spot,body);path_high=max(spot,body)
                path=gz[(gz.strike>=path_low)&(gz.strike<=path_high)]
                path_total=path.net_gex.sum() if len(path) else np.nan
                path_negative_share=float(path.loc[path.net_gex<0,'net_gex'].abs().sum()/path.net_gex.abs().sum()) if len(path) and path.net_gex.abs().sum()>0 else np.nan
                distance=abs(body-spot)
                reach=distance/em_amt if pd.notna(em_amt) and em_amt>0 else np.nan
                rows.append({'Direction':'Bearish' if typ=='PUT' else 'Bullish','Type':typ,'Long Lower':klo,'Short Body':body,'Long Upper':khi,
                             'Lower Wing':lower_width,'Upper Wing':upper_width,'Mid Debit $':debit_mid*100,'Natural Debit $':debit_nat*100,
                             'Max Loss $':max_loss,'Max Profit $':max_profit,'Reward / Risk':max_profit/max_loss,
                             'Body Distance':distance,'Distance / EM':reach,'Path Net GEX':path_total,'Path Negative Share':path_negative_share,
                             'Body Expiry P/L $':float(bfly_payoff(body,typ,klo,body,khi,debit_nat)),
                             'Overshoot P/L $':float(bfly_payoff(0 if typ=='PUT' else 2*khi,typ,klo,body,khi,debit_nat)),
                             'Quote Cost / Wing %':100*debit_nat/max(lower_width,upper_width),
                             'Leg IVs':tuple(float(z['implied_volatility']) for z in legs),
                             'Leg Mid':tuple(float(z['mid']) for z in legs)})
    return pd.DataFrame(rows)

def expiry_strike_gex(chain, expiry, spot):
    """OI-based snapshot proxy, signed calls + / puts -. NOT actual dealer inventory."""
    q=chain[chain['expiry']==pd.Timestamp(expiry)].copy()
    if q.empty or 'gamma' not in q or 'open_interest' not in q:return pd.DataFrame()
    q=q.dropna(subset=['strike','gamma','open_interest'])
    q=q[(q.gamma>=0)&(q.open_interest>=0)]
    if q.empty:return pd.DataFrame()
    q['exposure']=q.gamma*q.open_interest*100*spot**2*.01*np.where(q.type=='CALL',1,-1)
    a=q.pivot_table(index='strike',columns='type',values='exposure',aggfunc='sum',fill_value=0)
    for col in ('CALL','PUT'):
        if col not in a:a[col]=0.
    a=a.reset_index().rename(columns={'CALL':'call_gex','PUT':'put_gex'})
    a['net_gex']=a.call_gex+a.put_gex
    return a.sort_values('strike')

with T8:
    st.subheader('🦋 SPX Broken-Wing Butterfly Lab')
    st.caption('Research tool: IV/HV → dealer gamma path → reachable body strike → quoted 1:-2:1 butterfly → expiration and early-exit scenarios. Quotes are snapshots, not firm executable prices.')
    raw_chain=data.get('option_chain')
    if raw_chain is None:
        st.info('Upload the eighth CSV: SPX option chain with type, strike, expiration_date, bid, ask and implied_volatility. Your seven original datasets remain supported.')
    else:
        try:
            chain=bfly_chain(raw_chain)
            st.info('Option-chain prices are from the uploaded snapshot. Selecting an earlier analysis date does not reconstruct historical option quotes or historical strike-level GEX.')
            dates=sorted(pd.Timestamp(d).date() for d in chain.expiry.dropna().unique() if pd.Timestamp(d).date()>=actual_date)
            if not dates: st.warning('No option-chain expirations after the analysis date.')
            else:
                exp=st.selectbox('Butterfly expiration',dates,key='bwb_exp')
                days_left=(exp-actual_date).days
                q=chain[chain.expiry==pd.Timestamp(exp)]
                close=pd.to_numeric(h_view['Close Price'],errors='coerce').dropna()
                hv5=close.pct_change().tail(5).std(ddof=1)*np.sqrt(252)*100 if len(close)>=7 else np.nan
                hv10=close.pct_change().tail(10).std(ddof=1)*np.sqrt(252)*100 if len(close)>=12 else np.nan
                hv20=close.pct_change().tail(20).std(ddof=1)*np.sqrt(252)*100 if len(close)>=22 else np.nan
                iv7=num(h_view.iloc[-1].get('IV 7d',np.nan))
                iv7pct=iv7 if pd.notna(iv7) else np.nan
                ratio=iv7pct/hv5 if pd.notna(hv5) and hv5>0 else np.nan
                matched=g_live[pd.to_datetime(g_live.expiration_dt).dt.date==exp]
                expiry_gex=matched.net_gex.sum() if len(matched) else np.nan
                # Responsive cards avoid st.metric's ellipsis on the chain-coverage value.
                import html
                metric_cards=[('IV 7d',f'{iv7pct:.2f}%' if pd.notna(iv7pct) else '—'),
                              ('HV5',f'{hv5:.2f}%' if pd.notna(hv5) else '—'),
                              ('IV7 / HV5',f'{ratio:.2f}' if pd.notna(ratio) else '—'),
                              ('Expiry net GEX',fmt_money(expiry_gex)),
                              ('Chain coverage',f'{len(q):,} contracts')]
                st.markdown('''<style>
                .bwb-metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(185px,1fr));gap:12px;margin:12px 0}
                .bwb-metric{min-width:0;border:1px solid rgba(127,127,127,.22);background:rgba(127,127,127,.06);border-radius:12px;padding:15px 17px}
                .bwb-metric-label{font-size:.95rem;line-height:1.3;margin-bottom:9px;white-space:normal}
                .bwb-metric-value{font-size:1.8rem;line-height:1.2;font-weight:650;overflow-wrap:anywhere;white-space:normal}
                </style>''',unsafe_allow_html=True)
                st.markdown('<div class="bwb-metrics">'+''.join(
                    '<div class="bwb-metric"><div class="bwb-metric-label">'+html.escape(label)+'</div>'
                    '<div class="bwb-metric-value">'+html.escape(value)+'</div></div>'
                    for label,value in metric_cards)+'</div>',unsafe_allow_html=True)
                st.caption(f'HV10: {hv10:.2f}% • HV20: {hv20:.2f}% • {days_left} calendar days to expiry. IV7/HV5 is an approximate short-horizon comparison, not a forward volatility forecast.')
                e=em[(em.date.dt.date==exp)]
                em_amount=float(e.iloc[0].expected_move_amt) if len(e) else float(m['spot']*(iv7pct/100)*np.sqrt(days_left/365)) if pd.notna(iv7pct) else np.nan
                if days_left<=1:
                    st.warning('0–1DTE expected-move snapshots can be distorted near expiry. The distance/EM exclusion is disabled for this expiry; validate live expected move before trading.')
                a,b,c,d=st.columns(4)
                direction=a.selectbox('Direction',['Both','Bearish put BWB','Bullish call BWB'],key='bwb_dir')
                max_debit=b.number_input('Maximum natural debit ($)',min_value=50.,max_value=5000.,value=250.,step=25.,key='bwb_maxdebit')
                min_oi=c.number_input('Min OI per leg',min_value=0,max_value=5000,value=10,step=10,key='bwb_oi')
                max_spread=d.slider('Max leg spread / mid',5,100,40,5,key='bwb_spread')/100
                wing_choice=st.multiselect('Wing widths (SPX points)',[5,10,15,20,25,30,40,50,75,100],default=[5,10,15,20,25,30,40,50],key='bwb_wings')
                require_neg=st.checkbox('Require negative path GEX for expansion candidates',value=False,key='bwb_neg')
                only_reach=st.slider('Max body distance / expected move',.25,2.5,1.25,.05,key='bwb_reach')
                st.caption('Negative path GEX is a research filter based on static strike exposure, not a simulated dealer hedge response. Technical breakout confirmation is not supplied by these CSVs.')
                chain_strikes=sorted(set(q['strike'].to_numpy(float)))
                possible=[k for k in chain_strikes if abs(k-m['spot'])<=max(2*em_amount if pd.notna(em_amount) else 150,60)]
                # Expiry-specific exposure drives body selection; pricing only selects wings.
                expiry_profile=expiry_strike_gex(chain,pd.Timestamp(exp),m['spot'])
                if not expiry_profile.empty:
                    view=expiry_profile[(expiry_profile.strike>=m['spot']-150)&(expiry_profile.strike<=m['spot']+150)].copy()
                    st.markdown('#### GEX by strike — selected expiration')
                    st.caption('Estimated OI × gamma exposure from uploaded option chain; calls assumed positive and puts negative. This is a proxy, not verified dealer inventory. Aggregate strike GEX is NOT used as 0DTE GEX.')
                    fig_g=go.Figure()
                    fig_g.add_bar(x=view.strike,y=view.net_gex/1e6,marker_color=np.where(view.net_gex>=0,'#16a34a','#dc2626'),name='Net GEX')
                    fig_g.add_vline(x=m['spot'],line_dash='dash',annotation_text='Spot')
                    fig_g.update_layout(height=400,xaxis_title='SPX strike',yaxis_title='Estimated $M / 1% move',bargap=.15)
                    st.plotly_chart(fig_g,use_container_width=True)
                    candidate_profile=view[view.strike.isin(possible)].copy()
                    candidate_profile=candidate_profile[candidate_profile.net_gex>0].nlargest(12,'net_gex')
                    default_targets=candidate_profile.strike.tolist()
                else:
                    default_targets=[]
                    st.warning('Expiry-specific GEX unavailable: option chain requires gamma and open_interest. No aggregate GEX substitution.')
                body_mode=st.radio('BWB short-body selection',['Positive expiry GEX','Manual strikes'],horizontal=True,key='bwb_body_mode')
                if body_mode=='Positive expiry GEX':
                    targets=default_targets
                    st.caption('Candidate bodies are the strongest positive-GEX strikes within the selected search range.')
                else:
                    targets=st.multiselect('Manual short-body strikes',possible,default=default_targets[:3],key='bwb_manual_bodies')
                st.write('Selected BWB bodies:',', '.join(f'{v:,.0f}' for v in targets) if targets else 'None')
                modes=[]
                if direction!='Bullish call BWB':modes.append('PUT')
                if direction!='Bearish put BWB':modes.append('CALL')
                result=pd.concat([bfly_candidates(chain,pd.Timestamp(exp),m['spot'],typ,targets,em_amount,expiry_profile,wing_choice,max_debit,max_spread,min_oi,allow_credit=True) for typ in modes],ignore_index=True)
                if not result.empty:
                    if require_neg:result=result[result['Path Net GEX'].notna() & (result['Path Net GEX']<0)]
                    if pd.notna(em_amount) and days_left>1:result=result[result['Distance / EM']<=only_reach]
                    result=result[result['Max Profit $']>0].copy()
                if pd.notna(expiry_gex) and expiry_gex>=0:
                    st.warning('Selected expiration has nonnegative net GEX. Directional expansion needs stronger confirmation; do not treat this as a verified negative-gamma regime.')
                if pd.notna(ratio) and ratio>=1:
                    st.warning('IV7/HV5 is not below 1.0. Cheap-volatility qualification is not met under the current heuristic.')
                st.markdown('#### Weekly expected-move expansion alert')
                cur=emr.get('current')
                if cur is None:
                    st.info('EM expansion alert unavailable: insufficient weekly history.')
                else:
                    breach=cur.get('Intraday Breach')
                    if breach is None:
                        st.info('EM breach unverified: supply a session high/low in the sidebar or upload historical High/Low data.')
                    elif not breach:
                        st.info('No verified intraday EM breach. Monitor frozen weekly boundaries before treating this as expansion.')
                    else:
                        st.warning(f"Intraday weekly EM BREACH detected. Lower {cur['Lower EM']:,.2f} / upper {cur['Upper EM']:,.2f}. Closing status: {emr['closing_status']}. Re-entry: {emr['reentry_status']}.")
                        if result.empty:
                            st.info('No priced butterfly candidate currently passes negative path-GEX and other filters.')
                        else:
                            # Only candidate paths whose direction matches the actual EM breach qualify.
                            breached_down=cur['Observed Low']<cur['Lower EM']
                            breached_up=cur['Observed High']>cur['Upper EM']
                            aligned=result[((result['Type']=='PUT') & breached_down) | ((result['Type']=='CALL') & breached_up)]
                            aligned=aligned[aligned['Path Net GEX'].notna() & (aligned['Path Net GEX']<0)]
                            if len(aligned):
                                st.success(f'EM EXPANSION ALERT: {len(aligned):,} direction-aligned BWB candidates with negative path GEX. Verify live quotes and price-action confirmation.')
                            else:
                                st.info('EM breach confirmed, but no direction-aligned negative-path-GEX candidate passes current filters: WAIT.')
                if result.empty:
                    st.info('WAIT / REJECT: No candidate passes the selected pricing, gamma-path and reachability constraints. Try viewing candidates without the negative-path requirement to diagnose the bottleneck.')
                else:
                    result['Ranking Metric']=result['Reward / Risk'].clip(upper=15)/(1+result['Distance / EM'].fillna(2))
                    result=result.sort_values(['Ranking Metric','Natural Debit $'],ascending=[False,True]).reset_index(drop=True)
                    st.metric('Candidates passing',f'{len(result):,}')
                    cols=['Direction','Long Lower','Short Body','Long Upper','Natural Debit $','Mid Debit $','Max Loss $','Max Profit $','Reward / Risk','Distance / EM','Path Net GEX','Path Negative Share','Overshoot P/L $']
                    st.dataframe(result[cols].head(150),use_container_width=True,hide_index=True)
                    st.download_button('Download butterfly candidates',result.drop(columns=['Leg IVs','Leg Mid']).to_csv(index=False).encode(),'spx_broken_wing_candidates.csv','text/csv')
                    idx=st.selectbox('Inspect candidate',list(range(min(100,len(result)))),format_func=lambda i:f"{result.iloc[i]['Direction']} | {result.iloc[i]['Long Lower']:,.0f} / {result.iloc[i]['Short Body']:,.0f} / {result.iloc[i]['Long Upper']:,.0f} | ${result.iloc[i]['Natural Debit $']:,.0f} natural",key='bwb_pick')
                    row=result.iloc[idx];typ=row['Type'];kl=row['Long Lower'];kb=row['Short Body'];kh=row['Long Upper'];debit=row['Natural Debit $']/100
                    grid=np.linspace(min(kl,m['spot'])-max(20,em_amount*.4),max(kh,m['spot'])+max(20,em_amount*.4),251)
                    expiry_pnl=bfly_payoff(grid,typ,kl,kb,kh,debit)
                    fig=go.Figure(go.Scatter(x=grid,y=expiry_pnl,name='Expiration P/L'))
                    fig.add_hline(y=0,line_dash='dash');fig.add_vline(x=m['spot'],line_dash='dot',annotation_text='Spot');fig.add_vline(x=kb,line_dash='dot',annotation_text='Body')
                    fig.update_layout(height=390,title='Expiration P/L — conservative natural-entry debit',xaxis_title='SPX at expiry',yaxis_title='P/L ($)')
                    st.plotly_chart(fig,use_container_width=True)
                    st.markdown('#### Early-exit scenario (theoretical, not executable)')
                    t1,t2,t3=st.columns(3)
                    exit_days=t1.slider('Trading days held',0,max(1,min(5,days_left)),2,key='bwb_days')
                    iv_shift=t2.slider('Parallel IV change (vol points)',-8.,8.,0.,.5,key='bwb_ivshift')
                    annual_rate=t3.number_input('Risk-free annual rate (%)',min_value=0.,max_value=15.,value=4.,step=.25,key='bwb_rate')/100
                    rem_t=max((days_left-exit_days*7/5)/365,0)
                    vols=row['Leg IVs']
                    # Price at hypothetical exit is model value; actual exit fills will differ.
                    values=[bfly_black_scholes(grid,k,rem_t,max(v+iv_shift/100,.0001),typ,annual_rate) for k,v in zip((kl,kb,kh),vols)]
                    model_pnl=(values[0]-2*values[1]+values[2]-debit)*100
                    fig2=go.Figure(go.Scatter(x=grid,y=model_pnl,name='Model exit P/L'))
                    fig2.add_hline(y=0,line_dash='dash');fig2.add_vline(x=m['spot'],line_dash='dot',annotation_text='Spot');fig2.add_vline(x=kb,line_dash='dot',annotation_text='Body')
                    fig2.update_layout(height=390,title=f'Theoretical P/L after {exit_days} trading days | IV shift {iv_shift:+.1f} pts',xaxis_title='SPX at exit',yaxis_title='Modeled P/L ($)')
                    st.plotly_chart(fig2,use_container_width=True)
                    st.caption('Scenario uses per-leg snapshot IV held constant except for a parallel shift, Black–Scholes European pricing, a user-set risk-free rate, and an approximate trading-day/calendar-day conversion. It excludes closing slippage, fees, volatility skew changes, and evolving GEX. No modeled profit is guaranteed.')
        except Exception as exc:
            st.error(f'Broken-wing analysis could not run: {exc}')