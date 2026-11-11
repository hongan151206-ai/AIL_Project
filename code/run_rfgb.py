import json, time, numpy as np, exp_core as E
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
def log(s): open(E.out_path("logs/prog_rfgb.txt"),"a").write(s+"\n")
df=E.load_df(); Xtr,Xte,ytr,yte,sc,fn=E.split_scale(df,seed=42)
res={}
for nm,mdl in [("GradBoost",GradientBoostingRegressor(n_estimators=300,learning_rate=0.05,max_depth=5,subsample=0.8,random_state=42)),
               ("RandForest",RandomForestRegressor(n_estimators=300,min_samples_leaf=5,n_jobs=1,random_state=42))]:
    t=time.time(); mdl.fit(Xtr,ytr); mae,rmse,r2=E.metrics(yte,mdl.predict(Xte))
    res[nm]=dict(MAE=mae,RMSE=rmse,R2=r2,sec=round(time.time()-t,1))
    log(f"{nm} MAE={mae:.3f} RMSE={rmse:.3f} R2={r2:.4f} ({res[nm]['sec']}s)")
    json.dump(res,open(E.out_path("rfgb.json"),"w"),indent=2)
log("RFGB DONE")
