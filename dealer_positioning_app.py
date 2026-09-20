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
.dp-label{font-size:1.02rem;font-weight:750;line-height:1.25;margin-bottom:18px;white-space:normal}
.dp-value{font-size:2.15rem;font-weight:760;line-height:1.08;letter-spacing:-.025em;margin-bottom:12px;white-space:normal;overflow-wrap:anywhere}
.dp-card.regime .dp-value{font-size:1.72rem;color:#08783f}.dp-card.gex .dp-value{color:#174f91}.dp-card.dex .dp-value{color:#4b2796}.dp-card.iv .dp-value{color:#a64d00}.dp-card.flip .dp-value{color:#b5121b}.dp-card.em .dp-value{color:#08783f;font-size:1.82rem}
.dp-delta{display:inline-block;padding:6px 10px;border-radius:12px;background:rgba(34,197,94,.12);color:#08783f;font-weight:650;font-size:.93rem;margin-bottom:12px;white-space:normal}
.dp-sub{font-size:.94rem;line-height:1.5;color:rgba(49,61,82,.78);white-space:normal}
.dp-icon{margin-right:.35rem}
@media (max-width:1350px){.dp-grid{grid-template-columns:repeat(3,minmax(210px,1fr))}.dp-card{min-height:220px}}
@media (max-width:760px){.dp-grid{grid-template-columns:1fr}.dp-card{min-height:auto}.dp-value{font-size:2rem}}
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

def classify(df):
    c=set(df.columns)
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
    for c in ['Close Price','Option Volume Total','Option Volume Put-Call Ratio','OI Total','OI Put-Call Ratio','IV 30d','IV Rank','Max Pain 1d','GEX Net OI','DEX Net OI']:
        if c in h: h[c]=h[c].map(num)
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

def regime_metrics(h,g,x,em,f,u):
    last=h.iloc[-1]; prev=h.iloc[-2] if len(h)>1 else last
    spot=first_valid(last.get('Close Price'), f['Price~'].dropna().iloc[-1] if f['Price~'].notna().any() else np.nan)
    gex=last.get('GEX Net OI',np.nan); dex=last.get('DEX Net OI',np.nan)
    dg=gex-prev.get('GEX Net OI',gex); dd=dex-prev.get('DEX Net OI',dex)
    iv=last.get('IV 30d',np.nan); ivr=last.get('IV Rank',np.nan)
    # Regime label: sign + recent change
    if gex>=0:
        reg='Positive Gamma — Strengthening' if dg>0 else 'Positive Gamma — Weakening'
    else:
        reg='Negative Gamma — Deepening' if dg<0 else 'Negative Gamma — Recovering'
    if np.sign(gex)!=np.sign(prev.get('GEX Net OI',gex)): reg='Gamma Regime Transition'
    # weighted nearest expiration levels
    gg=g[g['expiration_dt']>=h['Date'].max()].sort_values('expiration_dt')
    if gg.empty: gg=g.sort_values('expiration_dt')
    gr=gg.iloc[0]
    # Scores are descriptive, not directional recommendations
    g_scale=max(h['GEX Net OI'].abs().quantile(.75),1)
    d_scale=max(h['DEX Net OI'].abs().quantile(.75),1)
    stability=50 + 25*np.tanh(gex/g_scale) + 10*np.tanh(dg/g_scale)
    if pd.notna(gr.gamma_flip) and pd.notna(spot): stability += 10*np.tanh((spot-gr.gamma_flip)/(spot*.08))
    stability=zclip(stability)
    # directional pressure: positive means call/delta pressure; negative put pressure
    ask=f[f['Side'].astype(str).str.lower().eq('ask')]['Premium'].sum()
    bid=f[f['Side'].astype(str).str.lower().eq('bid')]['Premium'].sum()
    flow_bias=(ask-bid)/max(ask+bid,1)
    pressure=50 + 22*np.tanh(dex/d_scale) + 13*flow_bias
    pressure=zclip(pressure)
    return dict(spot=spot,gex=gex,dex=dex,dg=dg,dd=dd,iv=iv,ivr=ivr,regime=reg,stability=stability,pressure=pressure,
                gamma_flip=gr.get('gamma_flip',np.nan),call_wall=gr.get('call_wall',np.nan),put_wall=gr.get('put_wall',np.nan),maxpain=last.get('Max Pain 1d',np.nan))

def level_table(m,g,x,em,f,u):
    spot=m['spot']; rows=defaultdict(lambda:{'score':0,'evidence':[],'flow_premium':0,'unusual':0})
    def add(level, pts, label):
        if pd.isna(level): return
        k=round(float(level)*2)/2
        rows[k]['score']+=pts; rows[k]['evidence'].append(label)
    # nearest 4 expirations, diminishing weights
    gs=g.sort_values('expiration_dt').head(4)
    for i,r in gs.iterrows():
        w=18 if len(rows)==0 else 12
        add(r.call_wall,14,'GEX Call Wall'); add(r.put_wall,14,'GEX Put Wall'); add(r.gamma_flip,18,'Gamma Flip')
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
st.caption('Regime → Positioning → Flow → Levels → Setup  •  Six-file dealer/flow model')
with st.sidebar:
    st.header('Data')
    uploads=st.file_uploader('Drop all 6 CSV files',type='csv',accept_multiple_files=True,help='Files are identified automatically from their columns.')
    use_demo=st.toggle('Use bundled COIN files',value=not bool(uploads))
    st.divider(); st.caption('Required: historical option data, GEX by expiration, DEX by expiration, expected move, options flow, unusual activity.')

data=demo_data() if use_demo else load_uploaded(uploads)[0]
if len(data)<6:
    st.warning(f'Recognized {len(data)}/6 datasets: {", ".join(data.keys()) or "none"}. Upload the remaining files.')
    st.stop()

h,g,x,em,f,u=prep(data); m=regime_metrics(h,g,x,em,f,u); levels=level_table(m,g,x,em,f,u)
ticker=str(f['Symbol'].dropna().iloc[0]) if 'Symbol' in f and f['Symbol'].notna().any() else 'Ticker'
latest_date=h['Date'].max().date() if h['Date'].notna().any() else ''

st.subheader(f'{ticker} • {latest_date}')
firstem=em.sort_values('date').iloc[0]
prev_close=h.iloc[-2]['Close Price'] if len(h)>1 else np.nan
close_chg=m['spot']-prev_close if pd.notna(prev_close) else np.nan
close_pct=(close_chg/prev_close) if pd.notna(prev_close) and prev_close else np.nan
close_badge=f"Close: {fmt_price(m['spot'])}"
if pd.notna(close_chg): close_badge += f" &nbsp; {'▲' if close_chg>=0 else '▼'} {close_chg:+.2f} ({close_pct:+.2%})"
st.markdown(f'<div class="dp-close">{close_badge}</div>',unsafe_allow_html=True)

prev_gex=h.iloc[-2]['GEX Net OI'] if len(h)>1 else np.nan
prev_dex=h.iloc[-2]['DEX Net OI'] if len(h)>1 else np.nan
g5=(m['gex']-h.iloc[-6]['GEX Net OI']) if len(h)>=6 else np.nan
d5=(m['dex']-h.iloc[-6]['DEX Net OI']) if len(h)>=6 else np.nan
reg_parts=m['regime'].split(' — ',1)
reg_main=reg_parts[0]; reg_detail=reg_parts[1] if len(reg_parts)>1 else 'Transition'
em_amt=firstem.get('expected_move_amt',np.nan)
em_pct=firstem.get('expected_move_percentage',np.nan)
if pd.notna(em_pct) and abs(em_pct)<1: em_pct*=100

def _line(label,val):
    return f'{label}: {val}' if val not in (None,'') else ''

cards=f'''
<div class="dp-grid">
  <div class="dp-card regime"><div class="dp-label"><span class="dp-icon">▥</span>Dealer Regime</div><div class="dp-value">{reg_main}</div><div class="dp-delta">{reg_detail}</div><div class="dp-sub">{'Dealers likely stabilizing price with positive gamma exposure.' if m['gex']>=0 else 'Negative gamma can increase sensitivity to directional price moves.'}</div></div>
  <div class="dp-card gex"><div class="dp-label"><span class="dp-icon">↗</span>Net GEX</div><div class="dp-value">{fmt_money(m['gex'])}</div><div class="dp-delta">{'↑' if m['dg']>=0 else '↓'} {fmt_money(m['dg'])} (1D)</div><div class="dp-sub">Prev: {fmt_money(prev_gex)}<br>{'5D Change: '+fmt_money(g5) if pd.notna(g5) else ''}</div></div>
  <div class="dp-card dex"><div class="dp-label"><span class="dp-icon">▤</span>Net DEX</div><div class="dp-value">{fmt_money(m['dex'])}</div><div class="dp-delta">{'↑' if m['dd']>=0 else '↓'} {fmt_money(m['dd'])} (1D)</div><div class="dp-sub">Prev: {fmt_money(prev_dex)}<br>{'5D Change: '+fmt_money(d5) if pd.notna(d5) else ''}</div></div>
  <div class="dp-card iv"><div class="dp-label"><span class="dp-icon">%</span>IV30 / IV Rank</div><div class="dp-value">{m['iv']:.2f}%</div><div class="dp-delta">IV Rank: {m['ivr']:.2f}%</div><div class="dp-sub">Current 30-day implied volatility and its historical rank.</div></div>
  <div class="dp-card flip"><div class="dp-label"><span class="dp-icon">⊕</span>Gamma Flip</div><div class="dp-value">{fmt_price(m['gamma_flip'])}</div><div class="dp-delta">Spot: {fmt_price(m['spot'])}</div><div class="dp-sub">Distance: {((m['spot']/m['gamma_flip'])-1):+.1%} ({m['spot']-m['gamma_flip']:+.2f})</div></div>
  <div class="dp-card em"><div class="dp-label"><span class="dp-icon">▣</span>Next Expected Move</div><div class="dp-value">{fmt_price(firstem.lower_price)} –<br>{fmt_price(firstem.upper_price)}</div><div class="dp-delta">{firstem.date.date()}</div><div class="dp-sub">Range: ± {fmt_price(em_amt).replace('$','')}<br>{f'({em_pct:.2f}%)' if pd.notna(em_pct) else ''}</div></div>
</div>'''
st.markdown(cards,unsafe_allow_html=True)

T1,T2,T3,T4,T5=st.tabs(['🎯 Command Center','📈 Historical Regime','🧲 Dealer Positioning','🌊 Flow Intelligence','🔬 Strike Explorer'])

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

    fig=go.Figure(); ee=em.sort_values('date').head(20)
    fig.add_trace(go.Scatter(x=ee.date,y=ee.upper_price,name='EM Upper',mode='lines'))
    fig.add_trace(go.Scatter(x=ee.date,y=ee.lower_price,name='EM Lower',mode='lines',fill='tonexty'))
    fig.add_hline(y=m['spot'],annotation_text=f'Spot {m["spot"]:.2f}')
    for y,name in [(m['gamma_flip'],'Gamma Flip'),(m['call_wall'],'Call Wall'),(m['put_wall'],'Put Wall')]:
        if pd.notna(y): fig.add_hline(y=y,line_dash='dot',annotation_text=name)
    fig.update_layout(height=420,title='Expected Move Envelope + Dealer Levels',xaxis_title='',yaxis_title='Price',legend_orientation='h')
    st.plotly_chart(fig,use_container_width=True)

with T2:
    days=st.segmented_control('Window',['20D','60D','All'],default='60D')
    hh=h.tail(20 if days=='20D' else 60 if days=='60D' else len(h))
    fig=go.Figure(); fig.add_trace(go.Scatter(x=hh.Date,y=hh['Close Price'],name='Close')); fig.add_trace(go.Scatter(x=hh.Date,y=hh['Max Pain 1d'],name='Max Pain',line=dict(dash='dot'))); fig.update_layout(height=350,title='Price vs Max Pain'); st.plotly_chart(fig,use_container_width=True)
    col1,col2=st.columns(2)
    for col,y,title in [(col1,'GEX Net OI','Net GEX History'),(col2,'DEX Net OI','Net DEX History')]:
        fig=go.Figure(go.Bar(x=hh.Date,y=hh[y],name=y)); fig.add_hline(y=0); fig.update_layout(height=330,title=title); col.plotly_chart(fig,use_container_width=True)
    fig=go.Figure(); fig.add_trace(go.Scatter(x=hh.Date,y=hh['IV 30d'],name='IV30')); fig.add_trace(go.Scatter(x=hh.Date,y=hh['IV Rank'],name='IV Rank')); fig.update_layout(height=330,title='Volatility Regime'); st.plotly_chart(fig,use_container_width=True)

with T3:
    pos=pd.merge(g,x,on='expiration_dt',how='outer',suffixes=('_gex','_dex')).sort_values('expiration_dt')
    pos['GEX Share']=pos.net_gex.abs()/max(pos.net_gex.abs().sum(),1); pos['DEX Share']=pos.net_dex.abs()/max(pos.net_dex.abs().sum(),1)
    analysis_date=pd.to_datetime(h['Date'].max()).normalize()
    pos['DTE']=(pd.to_datetime(pos['expiration_dt']).dt.normalize()-analysis_date).dt.days
    em_map=em.copy(); em_map['date']=pd.to_datetime(em_map['date']).dt.normalize()
    em_map['Expected Move']=em_map.apply(lambda r:f"{fmt_price(r['lower_price'])} – {fmt_price(r['upper_price'])} (±{fmt_price(r['expected_move_amt']).replace('$','')}, {r['expected_move_percentage']:.2f}%)",axis=1)
    pos=pos.merge(em_map[['date','Expected Move']],left_on=pd.to_datetime(pos['expiration_dt']).dt.normalize(),right_on='date',how='left').drop(columns=['date'])

    st.markdown('#### Dealer Positioning by Expiration')
    q1,q2,q3,q4=st.columns(4)
    max_dte=int(max(pos['DTE'].max(),1)); min_dte=q1.number_input('Min DTE',0,max_dte,0,key='pos_mindte'); max_dte_sel=q2.number_input('Max DTE',0,max_dte,max_dte,key='pos_maxdte')
    gsign=q3.multiselect('GEX sign',['Positive','Negative'],default=['Positive','Negative'],key='pos_gsign'); dsign=q4.multiselect('DEX sign',['Positive','Negative'],default=['Positive','Negative'],key='pos_dsign')
    filt=pos[(pos.DTE>=min_dte)&(pos.DTE<=max_dte_sel)].copy()
    filt=filt[((filt.net_gex>=0)&('Positive' in gsign))|((filt.net_gex<0)&('Negative' in gsign))]
    filt=filt[((filt.net_dex>=0)&('Positive' in dsign))|((filt.net_dex<0)&('Negative' in dsign))]
    disp=filt[['expiration_dt','DTE','net_gex','net_dex','call_gex','put_gex','call_dex','put_dex','call_wall_gex','put_wall_gex','gamma_flip','Expected Move','GEX Share','DEX Share']].copy()
    disp.columns=['Expiration','DTE','Net GEX','Net DEX','Call GEX','Put GEX','Call DEX','Put DEX','Call Wall','Put Wall','Gamma Flip','Expected Move','GEX Share','DEX Share']
    st.dataframe(disp,use_container_width=True,hide_index=True,column_config={'GEX Share':st.column_config.ProgressColumn(format='%.1%%',min_value=0,max_value=1),'DEX Share':st.column_config.ProgressColumn(format='%.1%%',min_value=0,max_value=1)})
    st.caption('Expected Move is matched to the expiration date when that date exists in the uploaded expected-move file; otherwise it is left blank.')
    col1,col2=st.columns(2)
    fig=go.Figure(go.Bar(x=filt.expiration_dt,y=filt.net_gex)); fig.add_hline(y=0); fig.update_layout(height=350,title='Net GEX by Expiration'); col1.plotly_chart(fig,use_container_width=True)
    fig=go.Figure(go.Bar(x=filt.expiration_dt,y=filt.net_dex)); fig.add_hline(y=0); fig.update_layout(height=350,title='Net DEX by Expiration'); col2.plotly_chart(fig,use_container_width=True)

with T4:
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

with T5:
    strikes=sorted(set(f.Strike.dropna()).union(set(u.Strike.dropna())))
    default=min(range(len(strikes)),key=lambda i:abs(strikes[i]-m['spot'])) if strikes else 0
    strike=st.selectbox('Strike',strikes,index=default if strikes else 0)
    sf=f[f.Strike==strike].copy(); su=u[u.Strike==strike].copy()
    a,b,c,d=st.columns(4); a.metric('Strike',fmt_price(strike)); b.metric('Distance from spot',f'{(strike/m["spot"]-1):+.2%}'); c.metric('Flow premium',fmt_money(sf.Premium.sum())); d.metric('Max Vol/OI',f'{su["Vol/OI"].max():.1f}x' if len(su) else '—')
    evid=levels[np.isclose(levels.Level,strike,atol=.26)]
    if len(evid): st.info(f"**Level context:** {evid.iloc[0].Evidence} • {evid.iloc[0].Classification} • Confluence {evid.iloc[0].Confluence:.0f}/100")
    left,right=st.columns(2); left.markdown('**Flow at strike**'); left.dataframe(sf.sort_values('Premium',ascending=False),use_container_width=True,hide_index=True); right.markdown('**Unusual activity at strike**'); right.dataframe(su.sort_values('Vol/OI',ascending=False),use_container_width=True,hide_index=True)

st.divider(); st.caption('Dealer positioning is an analytical map, not a prediction. Flow side and multi-leg prints can be ambiguous; use price/TA confirmation before interpreting a level as support or resistance.')
