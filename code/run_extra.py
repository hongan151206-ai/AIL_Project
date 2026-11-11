"""
run_extra.py
(1) multi-seed MAE for GradBoost, RandForest, Ridge to fully fill the accuracy table.
(2) ablation of Robust Anchored Stacking: full vs no-augmentation vs no-anchor vs
    +XGBoost, each on accuracy, robustness, and explanation fidelity.
"""
import json, time, numpy as np, exp_core as E
import shap
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import nnls
from scipy.stats import kendalltau
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
def log(s): open(E.out_path("logs/prog_extra.txt"),"a").write(s+"\n")

df=E.load_df(); FN=E.feature_names(df)
X=df[FN].values; y=df[E.TARGET].values; g=df[E.GROUP].values
SEEDS=[42,7,13,21,99]
def norm_pct(v):
    v=np.abs(np.asarray(v,float)); s=v.sum(); return v/s*100 if s>0 else np.full(len(v),100/len(v))
def prep(seed):
    tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=seed).split(X,y,groups=g))
    sc=StandardScaler().fit(X[tr])
    return sc.transform(X[tr]),sc.transform(X[te]),y[tr],y[te],g[tr]

# -------- (1) multiseed for slow/simple models --------
def mk(nm,seed):
    if nm=="GradBoost": return GradientBoostingRegressor(n_estimators=300,learning_rate=0.05,max_depth=5,subsample=0.8,random_state=seed)
    if nm=="RandForest": return RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=1,random_state=seed)
    if nm=="Ridge": return Ridge(alpha=1.0)
ms={}
for nm in ["Ridge","GradBoost","RandForest"]:
    vals=[]
    for sd in SEEDS:
        Xtr,Xte,ytr,yte,_=prep(sd); m=mk(nm,sd); m.fit(Xtr,ytr); vals.append(float(mean_absolute_error(yte,m.predict(Xte))))
        log(f"{nm} seed{sd} MAE={vals[-1]:.3f}")
        ms[nm]=dict(vals=vals,mean=float(np.mean(vals)),std=float(np.std(vals)))
        json.dump(ms,open(E.out_path("ms_extra.json"),"w"),indent=1)
log("MULTISEED DONE")

# -------- (2) ablation of the proposed ensemble (seed 42) --------
SEED=42; Xtr,Xte,ytr,yte,gtr=prep(SEED)
GT=norm_pct(Ridge(alpha=1.0).fit(Xtr,ytr).coef_)
def bases(names,seed=SEED):
    d={"cat":CatBoostRegressor(iterations=500,learning_rate=0.05,depth=6,loss_function="RMSE",random_seed=seed,verbose=0),
       "lgb":LGBMRegressor(n_estimators=600,learning_rate=0.05,num_leaves=63,subsample=0.8,colsample_bytree=0.8,n_jobs=1,random_state=seed,verbose=-1),
       "xgb":XGBRegressor(n_estimators=600,learning_rate=0.05,max_depth=6,subsample=0.8,colsample_bytree=0.8,n_jobs=1,random_state=seed,tree_method="hist"),
       "ridge":Ridge(alpha=1.0)}
    return [(n,d[n]) for n in names]
def aug(A,b,eps,seed=2):
    if eps<=0: return A,b
    r=np.random.RandomState(seed); return np.vstack([A,A+eps*r.randn(*A.shape)]),np.concatenate([b,b])
def fit_blend(names,eps):
    itr,ival=next(GroupShuffleSplit(1,test_size=0.25,random_state=SEED).split(Xtr,ytr,groups=gtr))
    Xi,yi=aug(Xtr[itr],ytr[itr],eps); P=[]
    for n,m in bases(names): m.fit(Xi,yi); P.append(m.predict(Xtr[ival]))
    w,_=nnls(np.vstack(P).T,ytr[ival]); w=w/(w.sum() if w.sum()>0 else 1)
    Xf,yf=aug(Xtr,ytr,eps); mods=[]
    for n,m in bases(names): m.fit(Xf,yf); mods.append((n,m))
    return mods,w
def predict(mods,w,Xin): return np.vstack([m.predict(Xin) for _,m in mods]).T@w
def evaluate(names,eps):
    mods,w=fit_blend(names,eps); pred=predict(mods,w,Xte)
    mae=mean_absolute_error(yte,pred)
    rob=[mae]+[float(np.mean([mean_absolute_error(yte,predict(mods,w,Xte+e*np.random.RandomState(70+s).randn(*Xte.shape))) for s in range(5)])) for e in [0.05,0.10,0.20]]
    d20=100*(rob[-1]-rob[0])/rob[0]
    Xs=Xte[np.random.RandomState(SEED).choice(Xte.shape[0],500,replace=False)]
    attrs=[]
    for (n,m),wi in zip(mods,w):
        a=norm_pct(m.coef_) if n=="ridge" else norm_pct(np.abs(shap.TreeExplainer(m).shap_values(Xs)).mean(0))
        attrs.append(wi*a)
    tau,_=kendalltau(norm_pct(np.sum(attrs,0)),GT)
    return dict(MAE=float(mae),d20=float(d20),tau=float(tau),w=dict(zip(names,np.round(w,3).tolist())))
abl={}
for tag,names,eps in [("Full (Cat+LGB+Ridge, aug)",["cat","lgb","ridge"],0.10),
                      ("without augmentation",["cat","lgb","ridge"],0.0),
                      ("without Ridge anchor",["cat","lgb"],0.10),
                      ("with XGBoost base",["cat","lgb","xgb","ridge"],0.10)]:
    t=time.time(); abl[tag]=evaluate(names,eps)
    log(f"{tag:30s} MAE={abl[tag]['MAE']:.3f} d20={abl[tag]['d20']:+.1f}% tau={abl[tag]['tau']:.3f} ({time.time()-t:.0f}s)")
    json.dump(abl,open(E.out_path("ablation.json"),"w"),indent=1)
log("ABLATION DONE")
