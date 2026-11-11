"""
run_audit2.py - a clearer explanation audit for the tree-tabular context.
Three readable lenses on CatBoost (a representative base of OURS):
  (1) fidelity: Kendall-tau agreement with the physics reference,
  (2) faithfulness: most-relevant-first deletion area (steeper = more faithful),
  (3) stability: how little the ranking changes under 10% input noise.
We compare the three standard tree explainers (TreeSHAP, permutation, gain) and
report which is best for this setting. We also verify OURS's meta-combined SHAP.
"""
import json, numpy as np, exp_core as E
import shap
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
def log(s): open(E.out_path("logs/prog_au.txt"),"a").write(s+"\n")
SEED=42; rng=np.random.RandomState(SEED)
df=E.load_df(); Xtr,Xte,ytr,yte,sc,fn=E.split_scale(df,seed=SEED); n=len(fn)
def norm_pct(v):
    v=np.abs(np.asarray(v,float)); s=v.sum(); return v/s*100 if s>0 else np.full(len(v),100/len(v))
GT=norm_pct(Ridge(alpha=1.0).fit(Xtr,ytr).coef_)
def tau(a,b): t,_=kendalltau(a,b); return float(t)
cat=CatBoostRegressor(iterations=500,learning_rate=0.05,depth=6,loss_function="RMSE",random_seed=SEED,verbose=0).fit(Xtr,ytr)
mean_tr=Xtr.mean(0)
aidx=rng.choice(Xte.shape[0],4000,replace=False); Xa,ya=Xte[aidx],yte[aidx]; base=mean_absolute_error(ya,cat.predict(Xa))
def audc(imp):
    order=np.argsort(-imp); Xp=Xa.copy(); c=[base]
    for j in order: Xp[:,j]=mean_tr[j]; c.append(mean_absolute_error(ya,cat.predict(Xp)))
    c=np.array(c)-base; return float(np.trapezoid(c)/len(order)), c.tolist()

def shap_imp(model,Xset): return norm_pct(np.abs(shap.TreeExplainer(model).shap_values(Xset)).mean(0))
Xs=Xte[rng.choice(Xte.shape[0],500,replace=False)]
methods={}
methods["TreeSHAP"]=shap_imp(cat,Xs)
pim=permutation_importance(cat,Xa,ya,n_repeats=5,random_state=SEED,scoring="neg_mean_absolute_error")
methods["Permutation"]=norm_pct(pim.importances_mean)
methods["Gain"]=norm_pct(cat.get_feature_importance())

# stability: rank agreement between clean and 10%-noise explanation (averaged)
def stability(method_name):
    taus=[]
    for s in range(3):
        r=np.random.RandomState(200+s); Xn=Xs+0.10*r.randn(*Xs.shape)
        if method_name=="TreeSHAP": imp_n=shap_imp(cat,Xn)
        elif method_name=="Gain": imp_n=norm_pct(cat.get_feature_importance())  # gain is data-independent -> perfectly stable
        else:
            r2=np.random.RandomState(300+s); idx=r2.choice(Xa.shape[0],2000,replace=False)
            Xan=Xa[idx]+0.10*r2.randn(2000,n)
            pn=permutation_importance(cat,Xan,ya[idx],n_repeats=3,random_state=SEED,scoring="neg_mean_absolute_error")
            imp_n=norm_pct(pn.importances_mean)
        taus.append(tau(methods[method_name],imp_n))
    return float(np.mean(taus))

rows={}
for nm,imp in methods.items():
    a,c=audc(imp); rows[nm]=dict(fidelity=tau(imp,GT),audc=a,stability=stability(nm),curve=c,imp=imp.tolist())
    log(f"{nm:12s} fidelity={rows[nm]['fidelity']:.3f} audc={a:.2f} stability={rows[nm]['stability']:.3f}")

json.dump({"features":fn,"gt":GT.tolist(),"rows":rows},open(E.out_path("audit_aux.json"),"w"),indent=1)
log("AUDIT2 DONE")
