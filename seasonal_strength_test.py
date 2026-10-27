"""Strength test (lam scales the SEAS5 log-weights; CRU TS obs). Leave-one-year-out test (1993/94-2025/26): weight the outlook's 33 member years by the ECMWF SEAS5 rainfall
forecast (likelihood of each year's observed rain over the first K forecast months given the forecast and its
skill r), against no weighting. Results in EXPERIMENTS.md (25 Oct 2026)."""
# Does weighting the 33 member years by the SEAS5 rainfall forecast help? Leave-one-year-out, 1993/94-2025/26.
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/claude/flag_predictor/src'); sys.path.insert(0,'/home/claude/flag_predictor')
from pathlib import Path
from flag_predictor import termly as T
from verify_termly import bucket
ROOT=Path('/home/claude/flag_predictor')
q=T.load_flow(ROOT)
import argparse
ap=argparse.ArgumentParser(); ap.add_argument('--centre',default='ecmwf'); ap.add_argument('--last',type=int,default=2025)
ARGS=ap.parse_args()
s=pd.read_csv(ROOT/'data/seasonal/seas5_upper_thames.csv')
if 'centre' not in s: s['centre']='ecmwf'
def em_for(c):
    e=s[s.centre==c].groupby(['init','lead']).rain_mm.mean().unstack()
    # standardise each start month and lead across years, so centres can be averaged
    st=e.index.str[5:]
    return e.groupby(st).transform(lambda x:(x-x.mean())/x.std())
if ARGS.centre=='mean':
    a,b=em_for('ecmwf'),em_for('ukmo'); common=a.index.intersection(b.index)
    EM=(a.loc[common]+b.loc[common])/2
else:
    EM=em_for(ARGS.centre)
era=pd.read_csv(ROOT/'data/termly/cru_monthly_rain.csv',index_col=0,parse_dates=True).iloc[:,0]
era.index=era.index.to_period('M')
def run_for(t0):
    p=pd.Period(t0,'M')
    return p if t0.day>=6 else p-1
def weights(t0, Y, years, K, lam=1.0):
    init=run_for(t0); m=init.month
    # same start month, other academic years
    inits=[i for i in EM.index if int(i[5:])==m]
    def acad(i): y,mm=int(i[:4]),int(i[5:]); return y if mm>=9 else y-1
    fc={i:EM.loc[i,1:K].sum() for i in inits}
    ob={}
    for i in inits:
        p=pd.Period(i,'M'); vals=[era.get(p+k) for k in range(K)]
        if all(v is not None and np.isfinite(v) for v in vals): ob[i]=sum(vals)
    train=[i for i in inits if acad(i)!=Y and i in ob and i!=str(init)]
    f=np.array([fc[i] for i in train]); o=np.array([ob[i] for i in train])
    r=max(0.0,np.corrcoef(f,o)[0,1])
    zf=(fc[str(init)]-f.mean())/f.std()
    # member year y: observed rain over the same calendar months in that year
    zy=[]
    for y in years:
        p0=pd.Period(f'{init.year-(T.academic_year(t0)-y)}-{init.month:02d}','M')
        vals=[era.get(p0+k) for k in range(K)]
        zy.append(sum(vals) if all(v is not None and np.isfinite(v) for v in vals) else np.nan)
    zy=np.array(zy); ok=np.isfinite(zy)
    zy=(zy-o.mean())/o.std()
    if r<=0: return np.ones(len(years))/len(years), r, zf
    lw=lam*(-0.5*(zy-r*zf)**2/(1-r*r)+0.5*zy**2)
    w=np.where(ok,np.exp(lw-np.nanmax(lw)),np.nan); w=np.where(ok,w,np.nanmean(w))
    return w/w.sum(), r, zf
ISS=[(m,d) for m in (9,10,11,12,1,2,3,4) for d in (1,15)]
rows=[]
for Y in range(T.FIRST_YEAR,ARGS.last+1):
    model=T.Outlook(q,exclude=Y)
    for (mm,dd) in ISS:
        t0=pd.Timestamp(Y if mm>=9 else Y+1,mm,dd)
        if run_for(t0).strftime('%Y-%m') not in EM.index: continue
        e=model.ensemble(t0); yrs=np.array(e.years); prev=float(q[t0])
        W={f'K{K}L{l}':weights(t0,Y,yrs,K,l) for K in (1,2) for l in (0.25,0.5,1.0)}
        allm=np.concatenate([e.members,e.climatology])
        for p in T.PERIODS:
            w0,w1=T.term_window(p,Y)
            if w1<=t0: continue
            s_=max(w0,t0+pd.Timedelta(days=1)); oq=q.loc[s_-pd.Timedelta(days=1):w1].to_numpy()
            bw=T.blend_weight(max(0,(w0-t0).days))
            for loc in ('isis',):
                o=T.days_at_flags(oq[1:],pd.date_range(s_,w1),s_,w1,loc,prev_flow=oq[0])[0]
                fx=T.days_at_flags(allm,e.days,s_,w1,loc,prev_flow=prev)
                for k,fl in ((0,'green'),(3,'red')):
                    rec=dict(Y=Y,issue=f'{mm:02d}-{dd:02d}',p=p,flag=fl,lead=bucket((w0-t0).days))
                    for name,yw in [('none',np.ones(len(yrs))/len(yrs))]+[(k,W[k][0]) for k in W]:
                        ww=np.r_[bw*yw,(1-bw)*yw]
                        rec[name]=T.wcrps(fx[:,k],ww,o[k])
                    rows.append(rec)
        race=T.race_days('torpids',Y)
        if race[0]>t0 and Y!=2020:
            ob=float(q.reindex(race).mean()>=T.TORPIDS_FLOW); bw=T.blend_weight((race[0]-t0).days)
            hit=(T.race_flow(allm,e.days,race)>=T.TORPIDS_FLOW)
            rec=dict(Y=Y,issue=f'{mm:02d}-{dd:02d}',p='torpids',flag='',lead='last week' if (race[0]-t0).days<=8 else bucket((race[0]-t0).days))
            for name,yw in [('none',np.ones(len(yrs))/len(yrs))]+[(k,W[k][0]) for k in W]:
                pr=np.sum(np.r_[bw*yw,(1-bw)*yw]*hit); rec[name]=(pr-ob)**2
            rows.append(rec)
df=pd.DataFrame(rows); pass
KEYS=[f'K{K}L{l}' for K in (1,2) for l in (0.25,0.5,1.0)]
def tab(g): return pd.Series({k:round(100*(1-g[k].mean()/g['none'].mean()),1) for k in KEYS}|{'n':len(g)})
df.to_csv('/tmp/claude-0/-home-claude/eba52f28-d89b-5668-9554-615ed79c2b78/scratchpad/strength.csv',index=False)
print(df.groupby(['lead']).apply(tab).to_string())
print('by issue date, Torpids:'); print(df[df.p=='torpids'].groupby('issue').apply(tab).to_string())
print('overall terms:', tab(df[df.p!='torpids']).to_dict())
