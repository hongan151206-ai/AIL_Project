"""
run_final2.py - canonical numbers for the proposed model OURS = Robust Anchored
Stacking (RAS): non-negative blend over CatBoost, LightGBM and a Ridge anchor,
with base learners trained on noise-augmented telemetry for perturbation
invariance. Weights are fit by NNLS on a flight-grouped holdout (transparent,
interpretable). We report accuracy, robustness, fidelity, an ablation without
augmentation, and a 5-split significance test against CatBoost.
"""
import json, time, numpy as np, exp_core as E
import shap
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.optimize import nnls
from scipy.stats import kendalltau, wilcoxon
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
def log(s): open(E.out_path("logs/prog_f2.txt"),"a").write(s+"\n")

df=E.load_df(); FN=E.feature_names(df)
X=df[FN].values; y=df[E.TARGET].values; g=df[E.GROUP].values
def norm_pct(v):
    v=np.abs(np.asarray(v,float)); s=v.sum(); return v/s*100 if s>0 else np.full(len(v),100/len(v))

def prep(seed):
    tr,te=next(GroupShuffleSplit(1,test_size=0.2,random_state=seed).split(X,y,groups=g))
    sc=StandardScaler().fit(X[tr])
    return sc.transform(X[tr]),sc.transform(X[te]),y[tr],y[te],g[tr]

def aug(Xa,ya,eps,seed=2):
    if eps<=0: return Xa,ya
    r=np.random.RandomState(seed); return np.vstack([Xa,Xa+eps*r.randn(*Xa.shape)]),np.concatenate([ya,ya])

def make_bases(seed):
    return [("cat",CatBoostRegressor(iterations=500,learning_rate=0.05,depth=6,loss_function="RMSE",random_seed=seed,verbose=0)),
            ("lgb",LGBMRegressor(n_estimators=600,learning_rate=0.05,num_leaves=63,subsample=0.8,colsample_bytree=0.8,n_jobs=1,random_state=seed,verbose=-1)),
            ("ridge",Ridge(alpha=1.0))]

def fit_ours(Xtr,ytr,gtr,eps,seed=42):
    itr,ival=next(GroupShuffleSplit(1,test_size=0.25,random_state=seed).split(Xtr,ytr,groups=gtr))
    Xi,yi=aug(Xtr[itr],ytr[itr],eps)
    P=[]
    for n,m in make_bases(seed): m.fit(Xi,yi); P.append(m.predict(Xtr[ival]))
    w,_=nnls(np.vstack(P).T,ytr[ival]); w=w/(w.sum() if w.sum()>0 else 1)
    Xf,yf=aug(Xtr,ytr,eps)
    mods=[]
    for n,m in make_bases(seed): m.fit(Xf,yf); mods.append((n,m))
    return mods,w

def predict(mods,w,Xin): return np.vstack([m.predict(Xin) for _,m in mods]).T@w

# ---------- canonical seed 42 ----------
SEED=42; Xtr,Xte,ytr,yte,gtr=prep(SEED)
GT=norm_pct(Ridge(alpha=1.0).fit(Xtr,ytr).coef_)
out={}
for tag,eps in [("OURS",0.10),("ablation_noaug",0.0)]:
    t=time.time(); mods,w=fit_ours(Xtr,ytr,gtr,eps,SEED)
    pred=predict(mods,w,Xte)
    mae=mean_absolute_error(yte,pred); rmse=float(np.sqrt(mean_squared_error(yte,pred))); r2=r2_score(yte,pred)
    rob=[]
    for e in [0,0.05,0.10,0.20]:
        if e==0: rob.append(mae); continue
        vals=[mean_absolute_error(yte,predict(mods,w,Xte+e*np.random.RandomState(70+s).randn(*Xte.shape))) for s in range(5)]
        rob.append(float(np.mean(vals)))
    d20=100*(rob[-1]-rob[0])/rob[0]
    # fidelity: meta-weighted base attribution
    Xs=Xte[np.random.RandomState(SEED).choice(Xte.shape[0],500,replace=False)]
    attrs=[]
    for (n,m),wi in zip(mods,w):
        if n=="ridge": a=norm_pct(m.coef_)
        else: a=norm_pct(np.abs(shap.TreeExplainer(m).shap_values(Xs)).mean(0))
        attrs.append(wi*a)
    fid_imp=norm_pct(np.sum(attrs,axis=0)); tau,_=kendalltau(fid_imp,GT)
    out[tag]=dict(weights=dict(zip(["cat","lgb","ridge"],np.round(w,3).tolist())),
                  MAE=mae,RMSE=rmse,R2=r2,rob=rob,d20=d20,tau=float(tau),
                  importance=fid_imp.tolist(),sec=round(time.time()-t,1))
    log(f"{tag:14s} MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.4f} d20={d20:+.1f}% tau={tau:.3f} w={out[tag]['weights']} ({out[tag]['sec']}s)")
    json.dump(out,open(E.out_path("ours_final.json"),"w"),indent=2)

# ---------- multiseed MAE + Wilcoxon vs CatBoost ----------
cat_ms=[41.012,38.782,36.784,37.257,38.105]
ours_ms=[]
for sd in [42,7,13,21,99]:
    Xtr,Xte,ytr,yte,gtr=prep(sd); mods,w=fit_ours(Xtr,ytr,gtr,0.10,sd)
    m=mean_absolute_error(yte,predict(mods,w,Xte)); ours_ms.append(float(m)); log(f"seed {sd} OURS MAE={m:.3f}")
    json.dump({"ours":ours_ms},open(E.out_path("ours_ms.json"),"w"),indent=2)
stat,p=wilcoxon(ours_ms,cat_ms[:len(ours_ms)])
out["multiseed"]=dict(ours=ours_ms,ours_mean=float(np.mean(ours_ms)),ours_std=float(np.std(ours_ms)),
                      cat=cat_ms,wilcoxon_p=float(p))
log(f"OURS multiseed mean={np.mean(ours_ms):.3f}+/-{np.std(ours_ms):.3f}  Wilcoxon vs Cat p={p:.4f}")
json.dump(out,open(E.out_path("ours_final.json"),"w"),indent=2)
log("FINAL2 DONE")
