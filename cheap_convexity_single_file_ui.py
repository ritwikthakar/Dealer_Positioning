from __future__ import annotations
import io, re
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Options Convexity Screener', page_icon='⚡', layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
.block-container{padding-top:1.1rem;padding-bottom:2rem}.stMetric{background:rgba(30,41,59,.32);border:1px solid rgba(148,163,184,.18);padding:14px;border-radius:10px}
div[data-testid="stSidebar"]{border-right:1px solid rgba(148,163,184,.16)}
.small{color:#94a3b8;font-size:.88rem}.hero{font-size:2rem;font-weight:800;margin-bottom:-.35rem}.badge{display:inline-block;padding:.18rem .55rem;border-radius:.45rem;background:#1e3a5f;font-weight:700}
</style>''', unsafe_allow_html=True)

PCT_COLS={'IV','ITM Prob','OTM Prob','5D/1M IV%','IV Rank','Imp Vol','5D IV Chg','1D IV Chg'}
NUM_COLS={'Th/Ga','Dl/Th','Th/Ve','$Vega','Strk Dist','Premium','Open Int','Theor.','IV/HV','Exp Move','Vol/OI Ratio','5D P/C OI','1M P/C OI','P/C OI'}

def num(s):
    x=s.astype(str).str.strip().str.replace(',','',regex=False).str.replace('$','',regex=False).str.replace('%','',regex=False)
    return pd.to_numeric(x,errors='coerce')

def load_csv(up):
    raw=up.getvalue()
    try: d=pd.read_csv(io.BytesIO(raw),sep=None,engine='python')
    except Exception: d=pd.read_csv(io.BytesIO(raw))
    d.columns=[str(c).strip() for c in d.columns]
    for c in d.columns:
        if c in PCT_COLS or c in NUM_COLS: d[c]=num(d[c])
    if 'Symbol' in d: d['Symbol']=d['Symbol'].astype(str).str.upper().str.strip()
    if 'Exp Date' in d: d['Exp Date']=pd.to_datetime(d['Exp Date'],errors='coerce')
    return d

def pct_rank(s, higher=True):
    r=pd.to_numeric(s,errors='coerce').rank(pct=True)
    return ((r if higher else 1-r)*100).fillna(50)

def clip_score(s, lo, hi, inverse=False):
    x=pd.to_numeric(s,errors='coerce')
    z=((x-lo)/(hi-lo)*100).clip(0,100)
    if inverse: z=100-z
    return z.fillna(50)

def enrich(d, weights):
    x=d.copy()
    for c in ['IV/HV','IV Rank','5D IV Chg','1D IV Chg','Th/Ga','Th/Ve','Open Int','Vol/OI Ratio','$Vega','Strk Dist','Exp Move']:
        if c not in x: x[c]=np.nan
    # Absolute cheapness avoids a score that changes too much with universe composition.
    ivhv=clip_score(x['IV/HV'],.60,1.10,inverse=True)
    rank=clip_score(x['IV Rank'],0,60,inverse=True)
    x['Cheapness Score']=(ivhv*.6+rank*.4)
    accel5=clip_score(x['5D IV Chg'],-5,12)
    accel1=clip_score(x['1D IV Chg'],-3,5)
    x['Acceleration Score']=accel5*.7+accel1*.3
    # Barchart Th/Ga and Th/Ve: lower magnitude = less theta paid per gamma/vega unit.
    x['Gamma Efficiency']=clip_score(x['Th/Ga'].abs(),0.02,1.00,inverse=True)
    x['Vega Efficiency']=clip_score(x['Th/Ve'].abs(),0.10,2.50,inverse=True)
    x['Efficiency Score']=x['Gamma Efficiency']*.55+x['Vega Efficiency']*.45
    oi=pct_rank(np.log1p(x['Open Int'].clip(lower=0)))
    voi=pct_rank(x['Vol/OI Ratio'].clip(lower=0))
    x['Tradeability Score']=oi*.7+voi*.3
    w1,w2,w3,w4=weights; den=max(sum(weights),1)
    x['Contract Score']=(w1*x['Cheapness Score']+w2*x['Acceleration Score']+w3*x['Efficiency Score']+w4*x['Tradeability Score'])/den

    cheap=x['IV/HV']<1; deep=x['IV/HV']<=.80; lowrank=x['IV Rank']<=25
    strong=x['Acceleration Score']>=68; gamma_good=x['Gamma Efficiency']>=65; vega_good=x['Vega Efficiency']>=65
    x['Suggested Expression']='WAIT / NO EDGE'
    x['Expression Reason']='No strong combination of cheapness, acceleration and option efficiency.'
    m=deep & lowrank & strong & gamma_good
    x.loc[m,'Suggested Expression']='LONG OPTION / BACKSPREAD'; x.loc[m,'Expression Reason']='Deep IV/HV discount + low IV Rank + acceleration + efficient gamma.'
    m=cheap & (~strong | ~gamma_good) & (x['Tradeability Score']>=35)
    x.loc[m,'Suggested Expression']='DIRECTIONAL OTM CALENDAR'; x.loc[m,'Expression Reason']='IV below HV with moderate acceleration; use chart/GEX-DEX to choose direction and destination.'
    m=lowrank & vega_good & (x['IV/HV']>=.85) & (x['IV/HV']<1.15) & (x['Acceleration Score']<55)
    x.loc[m,'Suggested Expression']='ATM VEGA CALENDAR'; x.loc[m,'Expression Reason']='Historically cheap IV + efficient vega + subdued near-term acceleration.'
    m=(x['IV/HV']>=1.15)&(x['Acceleration Score']<45)
    x.loc[m,'Suggested Expression']='ATM THETA CALENDAR'; x.loc[m,'Expression Reason']='IV rich versus realized movement with weak acceleration.'
    return x

def underlying_view(x):
    rows=[]
    for sym,g in x.groupby('Symbol',sort=False):
        best=g.sort_values('Contract Score',ascending=False).iloc[0]
        rows.append({
            'Ticker':sym,'Name':best.get('Name',''),'Convexity Score':best['Contract Score'],
            'Expression':best['Suggested Expression'],'IV/HV':best['IV/HV'],'IV Rank':best['IV Rank'],
            '5D IV Chg':best['5D IV Chg'],'Th/Ga':best['Th/Ga'],'Th/Ve':best['Th/Ve'],
            'Best Expiration':best.get('Exp Date',pd.NaT),'Strk Dist':best['Strk Dist'],'Open Int':best['Open Int'],
            'Vol/OI':best['Vol/OI Ratio'],'Reason':best['Expression Reason']})
    return pd.DataFrame(rows).sort_values('Convexity Score',ascending=False)

st.markdown('<div class="hero">⚡ Options Convexity Screener</div>',unsafe_allow_html=True)
st.caption('Identify underpriced optionality → choose the expression → analyze the individual contract.')

with st.sidebar:
    st.header('Data Input')
    up=st.file_uploader('Upload Barchart Options Screener CSV',type='csv')
    st.caption('Expected fields include IV/HV, IV Rank, IV changes, Th/Ga, Th/Ve, Strk Dist, Exp Date, OI and Vol/OI.')
    st.divider(); st.subheader('Scoring Settings')
    wc=st.slider('Volatility Cheapness',0,60,30,5); wa=st.slider('Volatility Acceleration',0,60,25,5)
    we=st.slider('Contract Efficiency',0,60,30,5); wt=st.slider('Tradeability',0,40,15,5)
    st.divider(); st.subheader('Filters')
    min_oi=st.number_input('Min Open Interest',0,step=100,value=100)
    min_voi=st.number_input('Min Volume/OI Ratio',0.0,10.0,0.0,.05)
    max_ivhv=st.slider('Max IV/HV',.40,2.00,1.10,.05)
    max_rank=st.slider('Max IV Rank',0,100,50,5)
    topn=st.slider('Top underlying opportunities',10,100,40,5)

if up is None:
    st.info('Upload the single Barchart Options Screener CSV to start. Your 09-19-2026 export is compatible with this layout.')
    st.stop()

try: raw=load_csv(up)
except Exception as e: st.error(f'Could not read CSV: {e}'); st.stop()
if 'Symbol' not in raw: st.error('The CSV needs a Symbol column.'); st.stop()

data=enrich(raw,[wc,wa,we,wt])
filtered=data[(data['Open Int'].fillna(0)>=min_oi)&(data['Vol/OI Ratio'].fillna(0)>=min_voi)&(data['IV/HV'].fillna(99)<=max_ivhv)&(data['IV Rank'].fillna(99)<=max_rank)].copy()
uv=underlying_view(filtered).head(topn) if len(filtered) else pd.DataFrame()

m1,m2,m3,m4,m5=st.columns(5)
m1.metric('Option rows',f'{len(raw):,}'); m2.metric('Unique tickers',raw['Symbol'].nunique())
m3.metric('Cheap IV/HV',int((raw['IV/HV']<1).sum())); m4.metric('Long-option candidates',int((data['Suggested Expression']=='LONG OPTION / BACKSPREAD').sum()))
m5.metric('Directional calendars',int((data['Suggested Expression']=='DIRECTIONAL OTM CALENDAR').sum()))

t1,t2,t3,t4,t5=st.tabs(['1. Convexity Screener','2. Contract Explorer','3. Individual Option Analysis','4. Visualizations','5. Data Overview'])

with t1:
    st.subheader('Top Cheap Convexity Opportunities — underlying level')
    st.caption('Each ticker is represented by its highest-scoring contract after the sidebar filters.')
    if uv.empty: st.warning('No rows meet the current filters.')
    else:
        show=uv.copy(); show['Convexity Score']=show['Convexity Score'].round(1); show['IV/HV']=show['IV/HV'].round(2); show['IV Rank']=show['IV Rank'].round(1); show['5D IV Chg']=show['5D IV Chg'].round(2); show['Th/Ga']=show['Th/Ga'].round(3); show['Th/Ve']=show['Th/Ve'].round(3); show['Strk Dist']=show['Strk Dist'].round(2)
        st.dataframe(show,use_container_width=True,hide_index=True,height=610)
        st.download_button('Download ranked opportunities',show.to_csv(index=False).encode(),'cheap_convexity_underlyings.csv','text/csv')
    st.info('Direction is intentionally not inferred from cheap volatility alone. For DIRECTIONAL OTM CALENDAR candidates, use your chart + GEX/DEX model for bullish/bearish bias and destination strike.')

with t2:
    st.subheader('Contract Explorer')
    syms=sorted(filtered['Symbol'].unique()) if len(filtered) else sorted(data['Symbol'].unique())
    sym=st.selectbox('Ticker',syms,key='explorer_symbol')
    g=(filtered if len(filtered) else data); g=g[g['Symbol']==sym].sort_values('Contract Score',ascending=False).copy()
    exp_choices=['All']+[d.strftime('%Y-%m-%d') for d in sorted(g['Exp Date'].dropna().unique())]
    exp=st.selectbox('Expiration',exp_choices)
    if exp!='All': g=g[g['Exp Date']==pd.Timestamp(exp)]
    cols=['Exp Date','Strk Dist','IV','IV/HV','IV Rank','5D IV Chg','1D IV Chg','Th/Ga','Th/Ve','$Vega','Exp Move','Open Int','Vol/OI Ratio','Contract Score','Suggested Expression','Expression Reason']
    st.dataframe(g[[c for c in cols if c in g]].sort_values('Contract Score',ascending=False),use_container_width=True,hide_index=True,height=590)
    st.caption('Lower |Th/Ga| = less theta per unit of gamma; lower |Th/Ve| = less theta per unit of vega. Compare contracts within similar expirations/moneyness rather than treating either ratio as a standalone signal.')

with t3:
    st.subheader('Individual Option Analysis')
    syms=sorted(data['Symbol'].unique()); sym2=st.selectbox('Select ticker',syms,key='analysis_symbol')
    gg=data[data['Symbol']==sym2].sort_values('Contract Score',ascending=False).reset_index(drop=True)
    labels=[]
    for i,r in gg.iterrows():
        ed=r['Exp Date'].strftime('%Y-%m-%d') if pd.notna(r.get('Exp Date')) else 'No expiry'
        labels.append(f"#{i+1} · {ed} · Strk Dist {r.get('Strk Dist',np.nan):.2f} · score {r['Contract Score']:.1f}")
    pick=st.selectbox('Select contract row',range(len(labels)),format_func=lambda i:labels[i])
    r=gg.iloc[pick]
    a,b,c,d,e=st.columns(5)
    a.metric('Contract Score',f"{r['Contract Score']:.1f}"); b.metric('IV/HV',f"{r['IV/HV']:.2f}" if pd.notna(r['IV/HV']) else '—'); c.metric('IV Rank',f"{r['IV Rank']:.1f}" if pd.notna(r['IV Rank']) else '—'); d.metric('Th/Ga',f"{r['Th/Ga']:.3f}" if pd.notna(r['Th/Ga']) else '—'); e.metric('Th/Ve',f"{r['Th/Ve']:.3f}" if pd.notna(r['Th/Ve']) else '—')
    st.success(f"Suggested expression: {r['Suggested Expression']}")
    st.write('**Why:**',r['Expression Reason'])
    left,right=st.columns([1,1])
    with left:
        st.markdown('#### Contract / volatility snapshot')
        details=pd.DataFrame({'Metric':['Expiration','Strike distance','IV','Implied vol','Expected move','Open interest','Vol/OI','$Vega','5D/1M IV%','5D IV change','1D IV change'],
        'Value':[r.get('Exp Date'),r.get('Strk Dist'),r.get('IV'),r.get('Imp Vol'),r.get('Exp Move'),r.get('Open Int'),r.get('Vol/OI Ratio'),r.get('$Vega'),r.get('5D/1M IV%'),r.get('5D IV Chg'),r.get('1D IV Chg')]})
        st.dataframe(details,use_container_width=True,hide_index=True)
    with right:
        st.markdown('#### Score components')
        comp=pd.DataFrame({'Score':[r['Cheapness Score'],r['Acceleration Score'],r['Efficiency Score'],r['Tradeability Score'],r['Gamma Efficiency'],r['Vega Efficiency']]},index=['Cheapness','Acceleration','Combined efficiency','Tradeability','Gamma efficiency','Vega efficiency'])
        st.bar_chart(comp)
    st.warning('This export does not include an explicit strike price, call/put type, bid/ask, gamma, theta or vega columns. Strk Dist and Barchart ratio fields can rank rows, but exact option construction should use a richer Options Screener export if those fields are available.')

with t4:
    st.subheader('Visualizations')
    c1,c2=st.columns(2)
    with c1:
        st.markdown('#### IV/HV vs IV Rank')
        p=data.dropna(subset=['IV/HV','IV Rank']).copy()
        st.scatter_chart(p,x='IV/HV',y='IV Rank',size='Open Int' if p['Open Int'].notna().any() else None)
        st.caption('Lower-left = IV below realized volatility and relatively low IV Rank.')
    with c2:
        st.markdown('#### Expression breakdown')
        ec=data['Suggested Expression'].value_counts().rename_axis('Expression').to_frame('Contracts')
        st.bar_chart(ec)
    st.markdown('#### Top contract scores')
    top=data.nlargest(30,'Contract Score')[['Symbol','Contract Score']].set_index('Symbol')
    st.bar_chart(top)

with t5:
    st.subheader('Data Overview')
    q1,q2,q3,q4=st.columns(4)
    q1.metric('Rows',f'{len(raw):,}'); q2.metric('Columns',len(raw.columns)); q3.metric('Tickers',raw['Symbol'].nunique()); q4.metric('Expirations',raw['Exp Date'].nunique() if 'Exp Date' in raw else 0)
    st.markdown('#### Columns detected')
    st.write(' · '.join(raw.columns))
    st.markdown('#### Raw uploaded data')
    st.dataframe(raw,use_container_width=True,hide_index=True,height=520)
