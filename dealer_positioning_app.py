from __future__ import annotations

import io, html
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Dealer Positioning Command Center', page_icon='🎯', layout='wide')

# =====================================================================================
# UI KIT  — one card system, one CSS block, no media queries (auto-fit grid is responsive)
# =====================================================================================
st.markdown('''<style>
.block-container{padding-top:1rem;padding-bottom:2rem;max-width:1500px}
.card-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:.3rem 0 1rem 0}
.card{min-width:0;border:1px solid rgba(127,127,127,.22);border-left-width:4px;border-radius:14px;padding:14px 16px;background:rgba(127,127,127,.05)}
.card-label{font-size:.85rem;font-weight:700;opacity:.75;margin-bottom:6px}
.card-value{font-size:1.5rem;font-weight:750;line-height:1.2;letter-spacing:-.02em;overflow-wrap:anywhere}
.card-badge{display:inline-block;margin-top:7px;padding:2px 9px;border-radius:999px;background:rgba(127,127,127,.15);font-size:.82rem;font-weight:650}
.card-sub{margin-top:6px;font-size:.84rem;line-height:1.4;opacity:.7}
.card.green{border-left-color:#10b981}.card.blue{border-left-color:#3b82f6}.card.purple{border-left-color:#7c3aed}
.card.amber{border-left-color:#f59e0b}.card.red{border-left-color:#ef4444}.card.teal{border-left-color:#14b8a6}
.card.indigo{border-left-color:#6366f1}.card.gray{border-left-color:rgba(127,127,127,.5)}
</style>''', unsafe_allow_html=True)

def cards(items):
    """items: (label, value, tone, badge, sub). HTML is built on ONE line: indented HTML inside st.markdown
    is treated as a markdown code block, which is a classic way for card grids to break."""
    out=[]
    for it in items:
        label,value=it[0],it[1]; tone=it[2] if len(it)>2 else 'gray'
        badge=it[3] if len(it)>3 else ''; sub=it[4] if len(it)>4 else ''
        out.append(f'<div class="card {tone}"><div class="card-label">{label}</div><div class="card-value">{value}</div>'
                   + (f'<div class="card-badge">{badge}</div>' if badge else '')
                   + (f'<div class="card-sub">{sub}</div>' if sub else '') + '</div>')
    st.markdown('<div class="card-grid">'+''.join(out)+'</div>', unsafe_allow_html=True)

# =====================================================================================
# HELPERS
# =====================================================================================
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

LEVEL_BAND=.35
MIN_EXP_SHARE=.02

def weekly_unit(spot, iv30):
    base=spot*(iv30/100)/np.sqrt(52) if pd.notna(iv30) and iv30>0 else spot*.02
    return float(max(base, spot*.005))

def live_expirations(g, x, asof, min_share=MIN_EXP_SHARE):
    asof=pd.Timestamp(asof).normalize()
    def keep(df, cols):
        d=df[pd.to_datetime(df['expiration_dt']).dt.normalize()>asof].copy()
        if d.empty: return df.iloc[0:0].copy()
        gross=sum(d[c].abs().fillna(0) for c in cols)
        k=d[gross/max(gross.sum(),1e-9)>=min_share]
        return (k if len(k) else d).sort_values('expiration_dt')
    return keep(g,['call_gex','put_gex']), keep(x,['call_dex','put_dex'])

def profile_flip(gxs, spot):
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
    fl=gg.dropna(subset=['gamma_flip'])
    if fl.empty: return np.nan
    w=(fl['call_gex'].abs()+fl['put_gex'].abs()).to_numpy(float); fv=fl['gamma_flip'].to_numpy(float)
    o=np.argsort(fv); cum=np.cumsum(w[o])
    return float(fv[o][min(np.searchsorted(cum,cum[-1]/2),len(fv)-1)])

def flow_direction_bias(f):
    side=f['Side'].astype(str).str.lower(); typ=f['Type'].astype(str).str.title()
    w=np.where(f['Code'].astype(str).str.upper().str.startswith('ML'),.45,1.0)
    prem=f['Premium'].fillna(0).clip(lower=0)*w
    bull=prem[((typ=='Call')&(side=='ask'))|((typ=='Put')&(side=='bid'))].sum()
    bear=prem[((typ=='Put')&(side=='ask'))|((typ=='Call')&(side=='bid'))].sum()
    return float((bull-bear)/max(bull+bear,1))

def usable_em(em, asof, min_dte=3):
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

@st.cache_data(show_spinner=False)
def read_csv_bytes(b: bytes):
    return pd.read_csv(io.BytesIO(b))

def load_uploaded(files):
    out={}; errors=[]
    for f in files:
        try:
            d=read_csv_bytes(f.getvalue()); k=classify(d)
            if k and k not in out: out[k]=d
            elif not k: errors.append(f'{f.name}: unrecognized columns')
            else: errors.append(f'{f.name}: duplicate {k} file ignored')
        except Exception as e: errors.append(f'{f.name}: {e}')
    return out, errors

@st.cache_data(show_spinner=False)
def demo_data():
    paths={
      'history':'/mnt/data/option_history_coin(2).csv','gex':'/mnt/data/net_gex_exposure_by_expiration_COIN(4).csv',
      'dex':'/mnt/data/net_dex_exposure_by_expiration_COIN(1).csv','em':'/mnt/data/COIN_expected_move_2026-09-25_w(1).csv',
      'flow':'/mnt/data/coin-options-flow-09-20-2026(1).csv','unusual':'/mnt/data/unusual-options-activity-09-20-2026(1).csv'}
    try:return {k:pd.read_csv(v) for k,v in paths.items()}
    except:return {}

@st.cache_data(show_spinner='Preparing data…')
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

# =====================================================================================
# ENGINES (logic unchanged)
# =====================================================================================
def expected_move_regime(h, verified_final_date=None, intraday=None):
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
        close_sides=np.where(days['Close Price']>upper,1,np.where(days['Close Price']<lower,-1,0))
        verified_date=pd.Timestamp(verified_final_date).date() if verified_final_date is not None else None
        is_final=i<len(groups)-1 or (verified_date is not None and days['Date'].iloc[-1].date()==verified_date)
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
    spot = last.get('Close Price', np.nan)
    gex=last.get('GEX Net OI',np.nan); dex=last.get('DEX Net OI',np.nan)
    dg=gex-prev.get('GEX Net OI',gex); dd=dex-prev.get('DEX Net OI',dex)
    iv=last.get('IV 30d',np.nan); ivr=last.get('IV Rank',np.nan)
    if gex>=0:
        reg='Positive Gamma — Strengthening' if dg>0 else 'Positive Gamma — Weakening'
    else:
        reg='Negative Gamma — Deepening' if dg<0 else 'Negative Gamma — Recovering'
    if np.sign(gex)!=np.sign(prev.get('GEX Net OI',gex)): reg='Gamma Regime Transition'
    gg=g.sort_values('expiration_dt')
    gr=gg.iloc[0] if not gg.empty else pd.Series(dtype=float)
    flip=profile_flip(gxs,spot)
    flip_src='Strike-level GEX profile'
    if pd.isna(flip):
        flip=exposure_weighted_flip(gg); flip_src='Exposure-weighted across expirations'
    if pd.isna(flip): flip_src='Unavailable'
    agg_cw,agg_pw=aggregate_walls(gxs,spot)
    g_scale=max(h['GEX Net OI'].abs().quantile(.75),1)
    d_scale=max(h['DEX Net OI'].abs().quantile(.75),1)
    stability=50 + 25*np.tanh(gex/g_scale) + 10*np.tanh(dg/g_scale)
    if pd.notna(flip) and pd.notna(spot): stability += 10*np.tanh((spot-flip)/(spot*.08))
    stability=zclip(stability)
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
    window=1.5*unit
    near=pos[(pos['strike']-spot).abs()<=window].copy()
    if near.empty: near=pos.copy()
    decay=.5*unit
    near['magnet_score']=near['net_gex']*np.exp(-(near['strike']-spot).abs()/decay)
    nr=near.loc[near['magnet_score'].idxmax()]
    cutoff=max(near['net_gex'].max()*.20,0)
    zone=near[near['net_gex']>=cutoff]
    return dict(near_magnet=float(nr['strike']), raw_magnet=float(raw['strike']),
                zone_low=float(zone['strike'].min()) if len(zone) else np.nan,
                zone_high=float(zone['strike'].max()) if len(zone) else np.nan,
                near_gex=float(nr['net_gex']), available=True)

def calendar_regime_engine(h,g,x,m,emr,gxs=None):
    last=h.iloc[-1]; spot=m['spot']
    unit=weekly_unit(spot,m['iv'])
    near_spot=lambda v: pd.notna(v) and abs(float(v)/spot-1)<=LEVEL_BAND
    mp=[num(last.get(c)) for c in h.columns if c.startswith('Max Pain') and pd.notna(last.get(c))]
    gs=g.sort_values('expiration_dt').head(4); xs=x.sort_values('expiration_dt').head(4)
    lv=[]
    for _,r in gs.iterrows(): lv += [r.get('call_wall'),r.get('put_wall'),r.get('gamma_flip')]
    for _,r in xs.iterrows(): lv += [r.get('call_wall'),r.get('put_wall')]
    lv=np.array([float(v) for v in lv+mp if near_spot(v)])
    tol=max(.2*unit,spot*.004); magnet=np.nan; share=0
    if len(lv):
        cand=np.unique(np.round(lv*2)/2); counts=[np.sum(np.abs(lv-c)<=tol) for c in cand]; j=int(np.argmax(counts)); magnet=float(cand[j]); share=float(counts[j]/len(lv))
    boundary_levels=[]
    for df in (gs,xs):
        for _,r in df.iterrows():
            for c in ['call_wall','put_wall']:
                v=r.get(c,np.nan)
                if near_spot(v): boundary_levels.append(float(v))
    distinct_gap=max(spot*.005, tol*1.15)
    lows=[]; highs=[]
    for v in boundary_levels:
        anchor=magnet if pd.notna(magnet) else spot
        if v <= anchor-distinct_gap: lows.append(v)
        elif v >= anchor+distinct_gap: highs.append(v)
    lower=np.nanmedian(lows) if lows else np.nan
    upper=np.nanmedian(highs) if highs else np.nan
    lower_n=len(lows); upper_n=len(highs)
    side_balance=min(lower_n,upper_n)
    bscore=min(100,25*side_balance) if side_balance else 0
    width=(upper-lower)/spot if pd.notna(lower) and pd.notna(upper) else np.nan
    distinct_from_magnet=(pd.notna(magnet) and pd.notna(lower) and pd.notna(upper)
                          and (magnet-lower)>=distinct_gap
                          and (upper-magnet)>=distinct_gap)
    two=bool(distinct_from_magnet and pd.notna(width) and width>=.012 and bscore>=50)
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

@st.cache_data(show_spinner=False)
def level_table(m,g,x,em,f,u):
    spot=m['spot']; rows=defaultdict(lambda:{'score':0,'evidence':[],'flow_premium':0,'unusual':0})
    def add(level, pts, label):
        if pd.isna(level): return
        k=round(float(level)*2)/2
        rows[k]['score']+=pts; rows[k]['evidence'].append(label)
    gs=g.sort_values('expiration_dt').head(4)
    for i,r in gs.iterrows():
        add(r.call_wall,14,'GEX Call Wall'); add(r.put_wall,14,'GEX Put Wall')
    add(m['gamma_flip'],40,'Gamma Flip')
    for _,r in x.sort_values('expiration_dt').head(4).iterrows():
        add(r.call_wall,10,'DEX Call Wall'); add(r.put_wall,10,'DEX Put Wall')
    add(m['maxpain'],10,'Max Pain')
    for _,r in em.sort_values('date').head(5).iterrows(): add(r.upper_price,12,'EM Upper'); add(r.lower_price,12,'EM Lower')
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

# ---------- Broken-wing butterfly engine (logic unchanged; chain parsing + candidate search cached) ----------
@st.cache_data(show_spinner=False)
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

@st.cache_data(show_spinner='Scanning butterflies…')
def bfly_candidates(chain,expiry,spot,typ,targets,em_amt,gxs,wing_sizes,max_debit,spread_cap,min_oi,allow_credit=False):
    q=chain[(chain['expiry']==pd.Timestamp(expiry)) & (chain['type']==typ)].copy()
    if q.empty:return pd.DataFrame()
    q=q.set_index('strike',drop=False)
    q=q[~q.index.duplicated(keep='first')]
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
                if typ=='PUT' and upper_width<=lower_width:continue
                if typ=='CALL' and lower_width<=upper_width:continue
                legs=[q.loc[klo],q.loc[body],q.loc[khi]]
                if any(z.get('open_interest',0)<min_oi for z in legs):continue
                if any(z['ask']<=0 or (z['ask']-z['bid'])/max(z['mid'],.01)>spread_cap for z in legs):continue
                debit_mid=float(legs[0]['mid']-2*legs[1]['mid']+legs[2]['mid'])
                debit_nat=float(legs[0]['ask']-2*legs[1]['bid']+legs[2]['ask'])
                if debit_nat*100>max_debit or (debit_nat<=0 and not allow_credit):continue
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

@st.cache_data(show_spinner=False)
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

# =====================================================================================
# SIDEBAR — all inputs live here; main page is for reading
# =====================================================================================
st.title('🎯 Dealer Positioning Command Center')
with st.sidebar:
    st.header('Data')
    uploads=st.file_uploader('Drop CSVs (6 required + optional strike GEX / option chain)',type='csv',accept_multiple_files=True,
                             help='Files are identified automatically from their columns.')
    use_demo=st.toggle('Use bundled COIN files',value=not bool(uploads))
    if use_demo: data=demo_data()
    else:
        data,_errs=load_uploaded(uploads)
        for _e in _errs: st.caption(f'⚠️ {_e}')
    st.caption('Required: history, GEX/DEX by expiration, expected move, flow, unusual activity.')

required={'history','gex','dex','em','flow','unusual'}
missing=required-set(data)
if missing:
    have=', '.join(sorted(required & set(data))) or 'none'
    st.warning(f'Recognized: {have}. Missing: {", ".join(sorted(missing))}.')
    st.stop()

h,g,x,em,f,u=prep(data)
ticker=str(f['Symbol'].dropna().iloc[0]) if 'Symbol' in f and f['Symbol'].notna().any() else 'Ticker'

valid_dates=sorted(pd.to_datetime(h['Date'].dropna()).dt.date.unique())
if not valid_dates:
    st.error('No valid dates were found in the historical option data.'); st.stop()
latest_date=valid_dates[-1]

with st.sidebar:
    st.header('Analysis')
    selected_date=st.date_input('Analysis date',value=latest_date,min_value=valid_dates[0],max_value=latest_date,
                                help='Cards and history use data on or before this date. Uploaded positioning files are single snapshots.')

h_view=h[pd.to_datetime(h['Date']).dt.date<=selected_date].copy()
if h_view.empty: h_view=h.iloc[[0]].copy()
actual_date=pd.to_datetime(h_view['Date'].max()).date()
gxs=data.get('gex_strike')
g_live,x_live=live_expirations(g,x,actual_date)
em_live=usable_em(em,actual_date)
if g_live.empty or x_live.empty:
    st.error('No valid future GEX/DEX expirations for the selected analysis date. Upload a newer positioning snapshot.'); st.stop()

em_by_date=em.assign(_d=em['date'].dt.date).drop_duplicates('_d').set_index('_d')
_dte=lambda d:(pd.Timestamp(d)-pd.Timestamp(actual_date)).days
em_dates=[d for d in g_live['expiration_dt'].dt.date.dropna().unique() if d in em_by_date.index and _dte(d)>=1]
em_pick=None
with st.sidebar:
    if em_dates:
        em_pick=st.selectbox('Expected move to',em_dates,index=next((i for i,d in enumerate(em_dates) if _dte(d)>=3),0),
                             format_func=lambda d:f'{d:%b %d} ({_dte(d)} DTE)',key='em_target',
                             help='Drives the Expected Move card, dealer levels and the butterfly lab. Defaults to the first expiry ≥3 DTE.')
    with st.expander('Weekly EM: intraday / verified close'):
        st.caption('Optional live-session values. High/low confirm an intraday breach; a last quote is not a verified close.')
        use_intraday=st.checkbox('Supply intraday observations',value=False,key='em_manual_enable')
        manual_high=st.number_input('Session high',min_value=0.,value=0.,step=1.,key='em_manual_high',disabled=not use_intraday)
        manual_low=st.number_input('Session low',min_value=0.,value=0.,step=1.,key='em_manual_low',disabled=not use_intraday)
        manual_last=st.number_input('Latest traded price',min_value=0.,value=0.,step=1.,key='em_manual_last',disabled=not use_intraday)
        verify_close=st.checkbox('Latest history row is a verified final close',value=False,key='em_verified_close')
    with st.expander('🔗 Other tools'):
        for label,url in [('⚡ Convexity Screener','https://convexity-bqhpmsmhg5lvruwjltk6bl.streamlit.app/'),
                          ('🎯 Dealer Positioning','https://dealerpositioning-n2m58uzu42afb44usojrwe.streamlit.app/'),
                          ('📊 Option Contract Analysis','https://dealerpositioning-zenvhgc3fs3dcsd9yunvct.streamlit.app/'),
                          ('🔬 Calendar Regime Screening','https://freedom-fuxffx4ohuuosfojdmffxl.streamlit.app/'),
                          ('📅 Calendar Spread Screener','https://freedom-hdf89xczpjdheb6kuq2qnz.streamlit.app/'),
                          ('🚀 Option Flow Screener','https://freedom-4rbsvk5mg32dybsiqwxwna.streamlit.app/')]:
            st.link_button(label,url,use_container_width=True)

manual_snapshot=None
if use_intraday and manual_high>0 and manual_low>0 and manual_high>=manual_low:
    manual_snapshot={'date':actual_date,'high':manual_high,'low':manual_low,'last':manual_last if manual_last>0 else np.nan}
verified_final_date=actual_date if verify_close else None

m=regime_metrics(h_view,g_live,x_live,em_live,f,u,gxs); levels=level_table(m,g_live,x_live,em_live,f,u)
emr=expected_move_regime(h_view,verified_final_date=verified_final_date,intraday=manual_snapshot)
cre=calendar_regime_engine(h_view,g_live,x_live,m,emr,gxs)

if em_pick is not None:
    firstem=em_by_date.loc[em_pick]; em_tag=f'{em_pick:%b %d} · {_dte(em_pick)} DTE'
else:
    firstem=em_live.iloc[0]; em_tag=f'{firstem.date:%b %d}'

# =====================================================================================
# HEADER — one line + one KPI strip + one collapsed data-check panel
# =====================================================================================
prev_close=h_view.iloc[-2]['Close Price'] if len(h_view)>1 else np.nan
close_chg=m['spot']-prev_close if pd.notna(m['spot']) and pd.notna(prev_close) else np.nan
chg_txt=''
if pd.notna(close_chg) and prev_close:
    chg_txt=f"  {'▲' if close_chg>=0 else '▼'} {close_chg:+.2f} ({close_chg/prev_close:+.2%})"
st.subheader(f'{ticker} · {actual_date} · {fmt_price(m["spot"])}{chg_txt}')
if actual_date!=selected_date:
    st.caption(f'Selected {selected_date} is not a trading day in the data; using {actual_date}.')

with st.expander('🔎 Data checks',expanded=False):
    st.write(f'History last observation **{actual_date}** · reference price **{fmt_price(m["spot"])}** (option-history Close Price, not independently verified) · '
             f'GEX expirations **{len(g_live)}** · DEX **{len(x_live)}** · EM rows **{len(em_live)}** · flow rows **{len(f):,}** · unusual rows **{len(u):,}** · '
             f'strike GEX **{"loaded" if gxs is not None else "missing"}** · option chain **{"loaded" if data.get("option_chain") is not None else "missing"}**')
    fp=pd.to_numeric(f.get('Price~',pd.Series(dtype=float)),errors='coerce').dropna()
    if not fp.empty: st.write(f'Options-flow reference price: **{fmt_price(fp.iloc[-1])}** (print time not verified).')
    st.warning('GEX/DEX, flow, unusual, option-chain and strike-GEX files are single snapshots; the Analysis date cannot reconstruct their earlier states. Make sure every file was exported for the same session.')

prev_gex=h_view.iloc[-2]['GEX Net OI'] if len(h_view)>1 else np.nan
prev_dex=h_view.iloc[-2]['DEX Net OI'] if len(h_view)>1 else np.nan
g5=(m['gex']-h_view.iloc[-6]['GEX Net OI']) if len(h_view)>=6 else np.nan
d5=(m['dex']-h_view.iloc[-6]['DEX Net OI']) if len(h_view)>=6 else np.nan
reg_main,_,reg_detail=m['regime'].partition(' — ')
em_amt=firstem.get('expected_move_amt',np.nan)
em_pct=em_amt/m['spot']*100 if pd.notna(em_amt) and m['spot'] else firstem.get('expected_move_percentage',np.nan)
arrow=lambda v:'↑' if v>=0 else '↓'
flip_dist=f"{(m['spot']/m['gamma_flip']-1):+.1%} from spot" if pd.notna(m['gamma_flip']) else '—'

cards([
 ('Dealer regime',reg_main,'green' if m['gex']>=0 else 'red',reg_detail or 'Transition',
  'Dealers likely stabilizing price.' if m['gex']>=0 else 'Moves can accelerate.'),
 ('Net GEX',fmt_money(m['gex']),'blue',f"{arrow(m['dg'])} {fmt_money(m['dg'])} 1D",
  f"Prev {fmt_money(prev_gex)}"+(f" · 5D {fmt_money(g5)}" if pd.notna(g5) else '')),
 ('Net DEX',fmt_money(m['dex']),'purple',f"{arrow(m['dd'])} {fmt_money(m['dd'])} 1D",
  f"Prev {fmt_money(prev_dex)}"+(f" · 5D {fmt_money(d5)}" if pd.notna(d5) else '')),
 ('IV30 / IV Rank',f"{m['iv']:.2f}%",'amber',f"Rank {m['ivr']:.1f}%",'30-day implied vol and its rank.'),
 ('Aggregate gamma flip',fmt_price(m['gamma_flip']),'red',flip_dist,m['flip_src']),
 (f'Expected move · {em_tag}',f"{fmt_price(firstem.lower_price)} – {fmt_price(firstem.upper_price)}",'teal',
  f"± {fmt_price(em_amt).replace('$','')}"+(f" ({em_pct:.2f}%)" if pd.notna(em_pct) else '')),
])

T1,T2,T3,T4,T5=st.tabs(['🎯 Overview','🧭 Regime','🧲 Positioning','🌊 Flow','🦋 Butterfly Lab'])

# =====================================================================================
# 1 · OVERVIEW
# =====================================================================================
with T1:
    a,b,c=st.columns([1,1,3])
    a.metric('Stability',f"{m['stability']:.0f}/100",help='Higher = more stabilizing dealer backdrop (GEX sign/trend and distance from gamma flip).')
    b.metric('Directional pressure',f"{m['pressure']:.0f}/100",help='Above 50 = call/positive-delta pressure; below 50 = put-side. Descriptive, not a signal.')
    c.info(f"**Map** · Put wall {fmt_price(m['put_wall'])} · Call wall {fmt_price(m['call_wall'])} · Flip {fmt_price(m['gamma_flip'])} · Max pain {fmt_price(m['maxpain'])}")

    fig=go.Figure(); ee=em_live.sort_values('date').head(20)
    fig.add_trace(go.Scatter(x=ee.date,y=ee.upper_price,name='EM upper',mode='lines'))
    fig.add_trace(go.Scatter(x=ee.date,y=ee.lower_price,name='EM lower',mode='lines',fill='tonexty'))
    fig.add_hline(y=m['spot'],annotation_text=f'Spot {m["spot"]:.2f}')
    for y,name in [(m['gamma_flip'],'Gamma flip'),(m['call_wall'],'Call wall'),(m['put_wall'],'Put wall')]:
        if pd.notna(y): fig.add_hline(y=y,line_dash='dot',annotation_text=name)
    fig.update_layout(height=380,title='Expected-move envelope + dealer levels',yaxis_title='Price',legend_orientation='h',margin=dict(t=50,b=10))
    st.plotly_chart(fig,use_container_width=True)

    st.markdown('#### Areas of interest')
    ao=levels.copy()
    ao['Zone']=np.where(ao['Level']<m['spot']-0.01,'Support',np.where(ao['Level']>m['spot']+0.01,'Resistance','Pivot'))
    with st.expander('Filters'):
        f1,f2,f3,f4=st.columns(4)
        zone_sel=f1.multiselect('Zone',['Support','Resistance','Pivot'],default=['Support','Resistance','Pivot'],key='aoi_zone')
        min_conf=f2.slider('Min confluence',0,100,0,5,key='aoi_conf')
        max_dist=f3.slider('Max distance from spot',1,100,25,1,format='%d%%',key='aoi_dist')/100
        min_flow=f4.number_input('Min weighted flow premium',min_value=0.0,value=0.0,step=100000.0,key='aoi_flow')
    ao=ao[ao['Zone'].isin(zone_sel)&(ao['Confluence']>=min_conf)&(ao['Distance'].abs()<=max_dist)&(ao['Weighted Flow Premium']>=min_flow)]
    show=ao.head(15).copy()
    show['Confluence']=show['Confluence'].round(0).astype(int)
    show['Distance']=show['Distance'].map(lambda z:f'{z:+.1%}')
    show['Weighted Flow Premium']=show['Weighted Flow Premium'].map(fmt_money)
    show['Max Vol/OI']=show['Max Vol/OI'].map(lambda z:f'{z:.1f}x' if z else '—')
    st.dataframe(show[['Level','Zone','Confluence','Distance','Evidence','Classification','Weighted Flow Premium','Max Vol/OI']],
                 use_container_width=True,hide_index=True,
                 column_config={'Confluence':st.column_config.ProgressColumn('Confluence',min_value=0,max_value=100,format='%d')})
    st.caption('Below spot = potential support, above = potential resistance. Price/TA confirmation is still required.')

# =====================================================================================
# 2 · REGIME  (calendar engine + weekly EM regime)
# =====================================================================================
with T2:
    st.markdown('#### Calendar regime')
    tone='green' if cre['regime'].startswith('COMPRESSION') else ('red' if cre['regime'] in ('EXPANSION','TRANSITION') else 'gray')
    cards([
      ('Current regime',cre['regime'],tone,'',cre['strategy']),
      ('Structural magnet',fmt_price(cre['magnet']),'blue',f"{cre['share']:.0%} cluster share"),
      ('Transition risk',f"{cre['transition']:.0f}/100",'amber','','Higher = more risk of leaving the regime.'),
      ('EM regime',emr['label'],'indigo'),
      ('Lower boundary',fmt_price(cre['lower']),'gray'),
      ('Upper boundary',fmt_price(cre['upper']),'gray'),
      ('Boundary credibility',f"{cre['boundary']:.0f}/100",'gray'),
      ('Near-spot +GEX zone',f"{fmt_price(cre['zone_low'])} – {fmt_price(cre['zone_high'])}",'teal','',
       f"Near-spot magnet {fmt_price(cre['near_magnet'])} · largest +GEX {fmt_price(cre['raw_magnet'])}"),
    ])
    ivtxt='—' if pd.isna(cre['ivacc']) else f"{cre['ivacc']:+.1%}"; gtxt='—' if pd.isna(cre['gexmom']) else f"{cre['gexmom']:+.2f}"
    with st.expander('Diagnostics & playbook'):
        st.dataframe(pd.DataFrame([['Magnet cluster share',f"{cre['share']:.0%}"],['IV short-vs-mid acceleration',ivtxt],
                                   ['5D normalized GEX momentum',gtxt],
                                   ['Latest EM utilization','—' if pd.isna(emr['utilization']) else f"{emr['utilization']:.2f}x"]],
                                  columns=['Input','Reading']),use_container_width=True,hide_index=True)
        st.dataframe(pd.DataFrame([['Compression — Single Magnet','ATM calendar','Levels cluster around one strike'],
                                   ['Compression — Defined Range','Double calendar','Credible lower and upper boundaries'],
                                   ['Transition','Reduce / stop new calendars','EM/IV acceleration and weakening gamma'],
                                   ['Expansion','OTM directional calendar / diagonal','Use price/TA to confirm direction']],
                                  columns=['Regime','Structure','Confirmation']),use_container_width=True,hide_index=True)
        st.caption('Expansion direction is left to price/TA confirmation.')
    if cre['available'] and gxs is not None:
        gz=gxs.copy()
        for cc in ['strike','net_gex']: gz[cc]=gz[cc].map(num)
        gz=gz.dropna(subset=['strike','net_gex'])
        w=max(2.5*cre['unit'],m['spot']*.05)
        gz=gz[(gz['strike']>=m['spot']-w)&(gz['strike']<=m['spot']+w)].sort_values('strike')
        if len(gz):
            fig=go.Figure(go.Bar(x=gz['strike'],y=gz['net_gex'],name='Net GEX'))
            fig.add_hline(y=0); fig.add_vline(x=m['spot'],line_dash='dash',annotation_text='Spot')
            if pd.notna(cre['magnet']): fig.add_vline(x=cre['magnet'],line_dash='dot',annotation_text='Structural magnet')
            if pd.notna(cre['near_magnet']): fig.add_vline(x=cre['near_magnet'],line_dash='dot',annotation_text='Near-spot magnet')
            fig.update_layout(height=340,title='Strike-level net GEX near spot',xaxis_title='Strike',yaxis_title='Net GEX',margin=dict(t=50,b=10))
            st.plotly_chart(fig,use_container_width=True)

    st.divider()
    st.markdown('#### Weekly expected-move regime')
    hit_txt=f"{emr['hit_rate']:.0%}" if pd.notna(emr['hit_rate']) else '—'
    util_txt=f"{emr['utilization']:.2f}x" if pd.notna(emr['utilization']) else '—'
    cards([('Weeks inside EM (streak)',str(emr['streak']),'gray'),('8-week containment',hit_txt,'gray'),
           ('Latest EM utilization',util_txt,'gray'),('EM regime',emr['label'],'indigo')])
    current=emr.get('current')
    if current:
        status='Completed' if current['Completed'] else 'In progress'
        st.markdown(f"**Latest week ({status}) — {current['Week']:%Y-%m-%d}**")
        cards([('Frozen lower / upper',f"{current['Lower EM']:,.1f} / {current['Upper EM']:,.1f}",'gray'),
               ('Closing breaches (consecutive)',f"{current['Close Breaches']} ({current['Consecutive Close Breaches']})",'gray'),
               ('Intraday EM breach',emr['intraday_status'],'red' if emr['intraday_status']=='BREACH' else 'gray'),
               ('Closing EM breach',emr['closing_status'],'red' if emr['closing_status']=='BREACH' else 'gray'),
               ('Re-entry inside EM',emr['reentry_status'],'gray')])
        st.caption(f"Bands = prior-week final close × IV30 / √52, anchored {current['Date']:%Y-%m-%d}. Coverage: {current['Basis']}. Last verified close: {fmt_price(current['Last Close'])}. "
                   "These can differ from broker/straddle expected moves.")
        if not current['Completed']: st.info('Current week stays IN PROGRESS until a later week is observed or the final close is verified in the sidebar.')
        if current['Basis']=='Daily closes only': st.warning('No daily high/low: intraday containment cannot be verified; "inside EM" means daily closes only.')
    if len(emr['weekly']):
        with st.expander('Weekly history (last 12)'):
            ew=emr['weekly'].tail(12).copy(); ew['Inside EM']=ew['Inside EM'].map({True:'Yes',False:'No'})
            st.dataframe(ew[['Week','Close Price','Lower EM','Upper EM','Observed High','Observed Low','EM Utilization','Close Breaches',
                             'Intraday Breach','Closing Breach','Re-entry Inside EM','Inside EM','Completed']],
                         use_container_width=True,hide_index=True)

# =====================================================================================
# 3 · POSITIONING  (by expiration + history)
# =====================================================================================
with T3:
    pos=pd.merge(g_live,x_live,on='expiration_dt',how='outer',suffixes=('_gex','_dex')).sort_values('expiration_dt')
    pos['GEX Share %']=pos.net_gex.abs()/max(pos.net_gex.abs().sum(),1)*100
    pos['DEX Share %']=pos.net_dex.abs()/max(pos.net_dex.abs().sum(),1)*100
    pos['DTE']=(pd.to_datetime(pos['expiration_dt']).dt.normalize()-pd.Timestamp(actual_date).normalize()).dt.days
    em_map=em_live.copy(); em_map['date']=pd.to_datetime(em_map['date']).dt.normalize()
    em_map['Expected Move']=em_map.apply(lambda r:f"{fmt_price(r['lower_price'])} – {fmt_price(r['upper_price'])} (±{fmt_price(r['expected_move_amt']).replace('$','')}, {r['expected_move_percentage']:.2f}%)",axis=1)
    pos=pos.merge(em_map[['date','Expected Move']],left_on=pd.to_datetime(pos['expiration_dt']).dt.normalize(),right_on='date',how='left').drop(columns=['date'])

    with st.expander('Filters'):
        q1,q2,q3,q4=st.columns(4)
        max_dte=int(max(pos['DTE'].max(),1))
        min_dte=q1.number_input('Min DTE',0,max_dte,0,key='pos_mindte'); max_dte_sel=q2.number_input('Max DTE',0,max_dte,max_dte,key='pos_maxdte')
        gsign=q3.multiselect('GEX sign',['Positive','Negative'],default=['Positive','Negative'],key='pos_gsign')
        dsign=q4.multiselect('DEX sign',['Positive','Negative'],default=['Positive','Negative'],key='pos_dsign')
    filt=pos[(pos.DTE>=min_dte)&(pos.DTE<=max_dte_sel)].copy()
    filt=filt[((filt.net_gex>=0)&('Positive' in gsign))|((filt.net_gex<0)&('Negative' in gsign))]
    filt=filt[((filt.net_dex>=0)&('Positive' in dsign))|((filt.net_dex<0)&('Negative' in dsign))]

    st.markdown('#### Dealer levels — selected expiration')
    eligible=filt[filt['DTE']>0].dropna(subset=['expiration_dt']).sort_values('expiration_dt').drop_duplicates('expiration_dt')
    if eligible.empty:
        st.info('No active expiration matches these filters.')
    else:
        available_dates=list(pd.to_datetime(eligible['expiration_dt']).dt.date)
        default_exp=em_pick if em_pick in available_dates else available_dates[0]
        chosen_exp=st.selectbox('Expiration',available_dates,index=available_dates.index(default_exp),
                                format_func=lambda d:f'{d:%b %d, %Y} ({_dte(d)} DTE)',key='dealer_levels_exp')
        chosen=eligible.loc[pd.to_datetime(eligible['expiration_dt']).dt.date==chosen_exp].iloc[0]
        cards([('Put wall',fmt_price(chosen['put_wall_gex']),'red'),('Call wall',fmt_price(chosen['call_wall_gex']),'green'),
               ('Gamma flip',fmt_price(chosen['gamma_flip']),'amber'),('Expiration net GEX',fmt_money(chosen['net_gex']),'blue')])
        st.caption(f'{int(chosen["DTE"])} DTE · expiration-specific levels, not the aggregate flip in the header.')

    col1,col2=st.columns(2)
    fig=go.Figure(go.Bar(x=filt.expiration_dt,y=filt.net_gex)); fig.add_hline(y=0); fig.update_layout(height=300,title='Net GEX by expiration',margin=dict(t=50,b=10)); col1.plotly_chart(fig,use_container_width=True)
    fig=go.Figure(go.Bar(x=filt.expiration_dt,y=filt.net_dex)); fig.add_hline(y=0); fig.update_layout(height=300,title='Net DEX by expiration',margin=dict(t=50,b=10)); col2.plotly_chart(fig,use_container_width=True)

    with st.expander('Full expiration table'):
        disp=filt[['expiration_dt','DTE','net_gex','net_dex','call_wall_gex','put_wall_gex','gamma_flip','Expected Move','GEX Share %','DEX Share %']].copy()
        disp.columns=['Expiration','DTE','Net GEX','Net DEX','Call Wall','Put Wall','Gamma Flip','Expected Move','GEX Share %','DEX Share %']
        st.dataframe(disp,use_container_width=True,hide_index=True,
                     column_config={'GEX Share %':st.column_config.ProgressColumn(format='%.1f%%',min_value=0,max_value=100),
                                    'DEX Share %':st.column_config.ProgressColumn(format='%.1f%%',min_value=0,max_value=100)})

    with st.expander('Historical charts'):
        days=st.segmented_control('Window',['20D','60D','All'],default='60D')
        hh=h_view.tail(20 if days=='20D' else 60 if days=='60D' else len(h_view))
        r1,r2=st.columns(2)
        fig=go.Figure(); fig.add_trace(go.Scatter(x=hh.Date,y=hh['Close Price'],name='Close')); fig.add_trace(go.Scatter(x=hh.Date,y=hh['Max Pain 1d'],name='Max pain',line=dict(dash='dot')))
        fig.update_layout(height=300,title='Price vs max pain',margin=dict(t=50,b=10)); r1.plotly_chart(fig,use_container_width=True)
        fig=go.Figure(); fig.add_trace(go.Scatter(x=hh.Date,y=hh['IV 30d'],name='IV30')); fig.add_trace(go.Scatter(x=hh.Date,y=hh['IV Rank'],name='IV Rank'))
        fig.update_layout(height=300,title='Volatility regime',margin=dict(t=50,b=10)); r2.plotly_chart(fig,use_container_width=True)
        for col,y,title in [(r1,'GEX Net OI','Net GEX history'),(r2,'DEX Net OI','Net DEX history')]:
            fig=go.Figure(go.Bar(x=hh.Date,y=hh[y])); fig.add_hline(y=0); fig.update_layout(height=300,title=title,margin=dict(t=50,b=10)); col.plotly_chart(fig,use_container_width=True)

# =====================================================================================
# 4 · FLOW  (flow intelligence + strike explorer)
# =====================================================================================
with T4:
    c1,c2,c3,c4=st.columns(4)
    maxd=int(max(f.DTE.max(skipna=True),u.DTE.max(skipna=True),1))
    dte=c1.slider('Max DTE',0,min(maxd,900),min(60,min(maxd,900)))
    typ=c2.multiselect('Type',['Call','Put'],default=['Call','Put'])
    minprem=c3.number_input('Min premium $',0.0,value=0.0,step=10000.0)
    minvoi=c4.number_input('Min unusual Vol/OI',0.0,value=2.0,step=.5)
    ff=f[(f.DTE<=dte)&f.Type.isin(typ)&(f.Premium>=minprem)].copy(); uu=u[(u.DTE<=dte)&u.Type.isin(typ)&(u['Vol/OI']>=minvoi)].copy()
    fa=ff.groupby(['Strike','Type'],as_index=False).agg(Premium=('Premium','sum'),Volume=('Volume','sum'),Trades=('Strike','size'))
    fig=go.Figure()
    for t in typ:
        z=fa[fa.Type==t]; fig.add_trace(go.Bar(x=z.Strike,y=z.Premium,name=t))
    fig.add_vline(x=m['spot'],annotation_text='Spot')
    fig.update_layout(height=340,title='Premium by strike',barmode='group',xaxis_title='Strike',yaxis_title='Premium',margin=dict(t=50,b=10))
    st.plotly_chart(fig,use_container_width=True)
    a,b=st.columns(2)
    a.markdown('**Largest flow prints**'); a.dataframe(ff.sort_values('Premium',ascending=False).head(25),use_container_width=True,hide_index=True)
    b.markdown('**Unusual / new-positioning candidates**'); b.dataframe(uu.sort_values('Vol/OI',ascending=False).head(25),use_container_width=True,hide_index=True)
    st.caption('Mid-market and multi-leg prints carry lower directional confidence in the confluence model.')

    with st.expander('🔬 Strike explorer'):
        strikes=sorted(set(f.Strike.dropna()).union(set(u.Strike.dropna())))
        if strikes:
            default=min(range(len(strikes)),key=lambda i:abs(strikes[i]-m['spot']))
            strike=st.selectbox('Strike',strikes,index=default)
            sf=f[f.Strike==strike]; su=u[u.Strike==strike]
            cards([('Strike',fmt_price(strike),'gray'),('Distance from spot',f'{(strike/m["spot"]-1):+.2%}','gray'),
                   ('Flow premium',fmt_money(sf.Premium.sum()),'gray'),('Max Vol/OI',f'{su["Vol/OI"].max():.1f}x' if len(su) else '—','gray')])
            evid=levels[np.isclose(levels.Level,strike,atol=.26)]
            if len(evid): st.info(f"**Level context:** {evid.iloc[0].Evidence} · {evid.iloc[0].Classification} · Confluence {evid.iloc[0].Confluence:.0f}/100")
            left,right=st.columns(2)
            left.markdown('**Flow at strike**'); left.dataframe(sf.sort_values('Premium',ascending=False),use_container_width=True,hide_index=True)
            right.markdown('**Unusual activity at strike**'); right.dataframe(su.sort_values('Vol/OI',ascending=False),use_container_width=True,hide_index=True)

# =====================================================================================
# 5 · BUTTERFLY LAB
# =====================================================================================
with T5:
    st.caption('IV/HV → dealer gamma path → reachable body strike → quoted 1:-2:1 butterfly → expiration and early-exit scenarios. Quotes are snapshots, not executable prices.')
    raw_chain=data.get('option_chain')
    if raw_chain is None:
        st.info('Upload an option-chain CSV (type, strike, expiration_date, bid, ask, implied_volatility; gamma and open_interest for GEX) to enable this tab.')
    else:
        try:
            chain=bfly_chain(raw_chain)
            dates=sorted(pd.Timestamp(d).date() for d in chain.expiry.dropna().unique() if pd.Timestamp(d).date()>=actual_date)
            if not dates: st.warning('No option-chain expirations after the analysis date.')
            else:
                if em_pick is not None and em_pick in dates:
                    exp=em_pick
                    st.caption(f'Linked to sidebar “Expected move to”: **{exp:%b %d, %Y}** ({(exp-actual_date).days} DTE).')
                else:
                    exp=st.selectbox('Butterfly expiration (fallback)',dates,key='bwb_exp',
                                     help='The “Expected move to” expiration is not in the uploaded option chain, so this selector is not synchronized.')
                days_left=(exp-actual_date).days
                q=chain[chain.expiry==pd.Timestamp(exp)]
                close=pd.to_numeric(h_view['Close Price'],errors='coerce').dropna()
                hv=lambda n:close.pct_change().tail(n).std(ddof=1)*np.sqrt(252)*100 if len(close)>=n+2 else np.nan
                hv5,hv10,hv20=hv(5),hv(10),hv(20)
                iv7pct=num(h_view.iloc[-1].get('IV 7d',np.nan))
                ratio=iv7pct/hv5 if pd.notna(iv7pct) and pd.notna(hv5) and hv5>0 else np.nan
                matched=g_live[pd.to_datetime(g_live.expiration_dt).dt.date==exp]
                expiry_gex=matched.net_gex.sum() if len(matched) else np.nan
                cards([('IV 7d',f'{iv7pct:.2f}%' if pd.notna(iv7pct) else '—','amber'),
                       ('HV5',f'{hv5:.2f}%' if pd.notna(hv5) else '—','gray','',f'HV10 {hv10:.1f}% · HV20 {hv20:.1f}%' if pd.notna(hv10) and pd.notna(hv20) else ''),
                       ('IV7 / HV5',f'{ratio:.2f}' if pd.notna(ratio) else '—','green' if pd.notna(ratio) and ratio<1 else 'red'),
                       ('Expiry net GEX',fmt_money(expiry_gex),'blue'),
                       ('Chain coverage',f'{len(q):,} contracts','gray',f'{days_left} DTE')])
                e=em[(em.date.dt.date==exp)]
                em_amount=float(e.iloc[0].expected_move_amt) if len(e) else float(m['spot']*(iv7pct/100)*np.sqrt(days_left/365)) if pd.notna(iv7pct) else np.nan

                notes=[]
                if days_left<=1: notes.append('0–1DTE expected-move snapshots can be distorted; the distance/EM filter is disabled for this expiry.')
                if pd.notna(expiry_gex) and expiry_gex>=0: notes.append('Selected expiration has non-negative net GEX; directional expansion needs stronger confirmation.')
                if pd.notna(ratio) and ratio>=1: notes.append('IV7/HV5 is not below 1.0: cheap-volatility test not met.')
                if notes: st.warning('\n\n'.join('• '+n for n in notes))

                a,b,c,d=st.columns(4)
                direction=a.selectbox('Direction',['Both','Bearish put BWB','Bullish call BWB'],key='bwb_dir')
                max_debit=b.number_input('Max natural debit ($)',min_value=50.,max_value=5000.,value=250.,step=25.,key='bwb_maxdebit')
                min_oi=c.number_input('Min OI per leg',min_value=0,max_value=5000,value=10,step=10,key='bwb_oi')
                max_spread=d.slider('Max leg spread / mid',5,100,40,5,key='bwb_spread')/100
                with st.expander('Advanced filters'):
                    wing_choice=st.multiselect('Wing widths (points)',[5,10,15,20,25,30,40,50,75,100],default=[5,10,15,20,25,30,40,50],key='bwb_wings')
                    require_neg=st.checkbox('Require negative path GEX',value=False,key='bwb_neg')
                    only_reach=st.slider('Max body distance / expected move',.25,2.5,1.25,.05,key='bwb_reach')
                    st.caption('Path GEX is a static-exposure filter, not a simulated hedging response. Technical breakout confirmation is not in these CSVs.')

                chain_strikes=sorted(set(q['strike'].to_numpy(float)))
                possible=[k for k in chain_strikes if abs(k-m['spot'])<=max(2*em_amount if pd.notna(em_amount) else 150,60)]
                expiry_profile=expiry_strike_gex(chain,pd.Timestamp(exp),m['spot'])
                if not expiry_profile.empty:
                    all_profiles=[expiry_strike_gex(chain,d_,m['spot']) for d_ in chain.expiry.dropna().unique()]
                    all_profiles=[z for z in all_profiles if not z.empty]
                    aggregate_chain_net=sum(float(z.net_gex.sum()) for z in all_profiles) if all_profiles else np.nan
                    selected_chain_net=float(expiry_profile.net_gex.sum())
                    gross_all=sum(float(z.net_gex.abs().sum()) for z in all_profiles) if all_profiles else np.nan
                    dominance=abs(selected_chain_net)/gross_all if pd.notna(gross_all) and gross_all>0 else np.nan
                    if pd.notna(aggregate_chain_net):
                        agreement='Same sign' if np.sign(selected_chain_net)==np.sign(aggregate_chain_net) else 'Opposite signs'
                        cards([('All-chain net GEX proxy',fmt_money(aggregate_chain_net),'blue'),
                               (f'{days_left}DTE net GEX proxy',fmt_money(selected_chain_net),'blue'),
                               ('Selected / all gross',f'{dominance:.1%}' if pd.notna(dominance) else '—','gray'),
                               ('Sign comparison',agreement,'gray')])
                        st.caption('Both proxies come from the SAME chain snapshot (calls +, puts −), not verified dealer inventory; do not compare them with the separately uploaded GEX series.')
                    view=expiry_profile[(expiry_profile.strike>=m['spot']-150)&(expiry_profile.strike<=m['spot']+150)].copy()
                    fig_g=go.Figure()
                    fig_g.add_bar(x=view.strike,y=view.net_gex/1e6,marker_color=np.where(view.net_gex>=0,'#16a34a','#dc2626'),name='Net GEX')
                    fig_g.add_vline(x=m['spot'],line_dash='dash',annotation_text='Spot')
                    if pd.notna(em_amount) and em_amount>0:
                        fig_g.add_vline(x=m['spot']-em_amount,line_dash='dot',line_color='#eab308',annotation_text='EM low')
                        fig_g.add_vline(x=m['spot']+em_amount,line_dash='dot',line_color='#eab308',annotation_text='EM high')
                    fig_g.update_layout(height=360,title=f'GEX by strike — {exp:%b %d} (estimated OI × gamma)',xaxis_title='Strike',yaxis_title='$M / 1% move',bargap=.15,margin=dict(t=50,b=10))
                    st.plotly_chart(fig_g,use_container_width=True)
                    cand_prof=view[view.strike.isin(possible)]; cand_prof=cand_prof[cand_prof.net_gex>0].nlargest(12,'net_gex')
                    default_targets=cand_prof.strike.tolist()
                else:
                    default_targets=[]
                    st.warning('Expiry-specific GEX unavailable: the chain needs gamma and open_interest. No aggregate substitution is made.')

                body_mode=st.radio('Short-body selection',['Positive expiry GEX','Manual strikes'],horizontal=True,key='bwb_body_mode')
                if body_mode=='Positive expiry GEX': targets=default_targets
                else: targets=st.multiselect('Manual short-body strikes',possible,default=default_targets[:3],key='bwb_manual_bodies')
                st.caption('Bodies: '+(', '.join(f'{v:,.0f}' for v in targets) if targets else 'none'))

                modes=[]
                if direction!='Bullish call BWB':modes.append('PUT')
                if direction!='Bearish put BWB':modes.append('CALL')
                parts=[bfly_candidates(chain,pd.Timestamp(exp),m['spot'],t_,targets,em_amount,expiry_profile,wing_choice,max_debit,max_spread,min_oi,True) for t_ in modes]
                parts=[p for p in parts if not p.empty]
                result=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
                if not result.empty:
                    if require_neg:result=result[result['Path Net GEX'].notna() & (result['Path Net GEX']<0)]
                    if pd.notna(em_amount) and days_left>1:result=result[result['Distance / EM']<=only_reach]
                    result=result[result['Max Profit $']>0].copy()

                cur=emr.get('current')
                if cur is not None and cur.get('Intraday Breach'):
                    st.warning(f"Weekly EM intraday BREACH · lower {cur['Lower EM']:,.2f} / upper {cur['Upper EM']:,.2f} · closing: {emr['closing_status']} · re-entry: {emr['reentry_status']}")
                    if not result.empty:
                        bd=cur['Observed Low']<cur['Lower EM']; bu=cur['Observed High']>cur['Upper EM']
                        al=result[((result['Type']=='PUT')&bd)|((result['Type']=='CALL')&bu)]
                        al=al[al['Path Net GEX'].notna()&(al['Path Net GEX']<0)]
                        if len(al): st.success(f'EM EXPANSION ALERT: {len(al):,} direction-aligned candidates with negative path GEX. Verify live quotes and price action.')
                        else: st.info('EM breach confirmed but no direction-aligned negative-path-GEX candidate passes: WAIT.')
                elif cur is not None and cur.get('Intraday Breach') is None:
                    st.caption('EM breach unverified — supply a session high/low in the sidebar or include High/Low in the history file.')

                if result.empty:
                    st.info('WAIT / REJECT: no candidate passes the pricing, gamma-path and reachability filters. Try relaxing the negative-path requirement to find the bottleneck.')
                else:
                    result['Ranking Metric']=result['Reward / Risk'].clip(upper=15)/(1+result['Distance / EM'].fillna(2))
                    result=result.sort_values(['Ranking Metric','Natural Debit $'],ascending=[False,True]).reset_index(drop=True)
                    st.markdown(f'#### {len(result):,} candidates')
                    cols=['Direction','Long Lower','Short Body','Long Upper','Natural Debit $','Max Loss $','Max Profit $','Reward / Risk','Distance / EM','Path Net GEX','Overshoot P/L $']
                    st.dataframe(result[cols].head(100),use_container_width=True,hide_index=True)
                    st.download_button('Download candidates',result.drop(columns=['Leg IVs','Leg Mid']).to_csv(index=False).encode(),'broken_wing_candidates.csv','text/csv')
                    idx=st.selectbox('Inspect candidate',list(range(min(100,len(result)))),key='bwb_pick',
                                     format_func=lambda i:f"{result.iloc[i]['Direction']} | {result.iloc[i]['Long Lower']:,.0f} / {result.iloc[i]['Short Body']:,.0f} / {result.iloc[i]['Long Upper']:,.0f} | ${result.iloc[i]['Natural Debit $']:,.0f}")
                    row=result.iloc[idx];typ=row['Type'];kl=row['Long Lower'];kb=row['Short Body'];kh=row['Long Upper'];debit=row['Natural Debit $']/100
                    pad=max(20,(em_amount if pd.notna(em_amount) else 50)*.4)
                    grid=np.linspace(min(kl,m['spot'])-pad,max(kh,m['spot'])+pad,251)
                    fig=go.Figure(go.Scatter(x=grid,y=bfly_payoff(grid,typ,kl,kb,kh,debit),name='Expiration P/L'))
                    fig.add_hline(y=0,line_dash='dash');fig.add_vline(x=m['spot'],line_dash='dot',annotation_text='Spot');fig.add_vline(x=kb,line_dash='dot',annotation_text='Body')
                    fig.update_layout(height=340,title='Expiration P/L (natural-entry debit)',xaxis_title='Price at expiry',yaxis_title='P/L ($)',margin=dict(t=50,b=10))
                    st.plotly_chart(fig,use_container_width=True)

                    with st.expander('Early-exit scenario (theoretical, not executable)'):
                        t1,t2,t3=st.columns(3)
                        exit_days=t1.slider('Trading days held',0,max(1,min(5,days_left)),2,key='bwb_days')
                        iv_shift=t2.slider('Parallel IV change (vol pts)',-8.,8.,0.,.5,key='bwb_ivshift')
                        annual_rate=t3.number_input('Risk-free rate (%)',min_value=0.,max_value=15.,value=4.,step=.25,key='bwb_rate')/100
                        rem_t=max((days_left-exit_days*7/5)/365,0)
                        values=[bfly_black_scholes(grid,k,rem_t,max(v+iv_shift/100,.0001),typ,annual_rate) for k,v in zip((kl,kb,kh),row['Leg IVs'])]
                        model_pnl=(values[0]-2*values[1]+values[2]-debit)*100
                        fig2=go.Figure(go.Scatter(x=grid,y=model_pnl,name='Model exit P/L'))
                        fig2.add_hline(y=0,line_dash='dash');fig2.add_vline(x=m['spot'],line_dash='dot',annotation_text='Spot');fig2.add_vline(x=kb,line_dash='dot',annotation_text='Body')
                        fig2.update_layout(height=340,title=f'Modeled P/L after {exit_days} trading days · IV {iv_shift:+.1f} pts',xaxis_title='Price at exit',yaxis_title='P/L ($)',margin=dict(t=50,b=10))
                        st.plotly_chart(fig2,use_container_width=True)
                        st.caption('Per-leg snapshot IV (parallel shift only), Black–Scholes European pricing, approximate trading→calendar day conversion. Excludes slippage, fees, skew changes and evolving GEX.')
        except Exception as exc:
            st.error(f'Butterfly analysis could not run: {exc}')

st.divider()
st.caption('Dealer positioning is an analytical map, not a prediction. Flow side and multi-leg prints can be ambiguous; confirm with price/TA before treating a level as support or resistance.')
