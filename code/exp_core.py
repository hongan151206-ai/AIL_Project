"""
exp_core.py  -  reproducible experiment core.
Mirrors the original dataset.py preprocessing exactly, drops MLP, and adds
stronger learners (XGBoost, LightGBM, HistGB) plus a stacking ensemble.
All numbers downstream come from THIS code on the real telemetry.
"""
import time, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, StackingRegressor)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]          # repo root (code/ sits one level below)
CSV = ROOT / "real_data" / "flights_ready_for_eda.csv"
RESULTS = ROOT / "results"                          # every script writes its JSON / logs here

def out_path(name):
    """results/<name>; parent folders are created on demand (name may be 'logs/x.txt')."""
    p = RESULTS / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
FEATURE_COLS = ["payload","speed","altitude","wind_speed","velocity_mag",
                "accel_mag","angular_mag","pitch","headwind"]
TARGET, GROUP, REGIME = "Power", "flight", "regime"

def load_df():
    df = pd.read_csv(CSV)
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    dummies = pd.get_dummies(df[REGIME], prefix=REGIME, drop_first=False).astype(int)
    df = pd.concat([df.drop(columns=[REGIME]), dummies], axis=1)
    return df

def feature_names(df):
    ohe = [c for c in df.columns if c.startswith(REGIME + "_")]
    return [c for c in FEATURE_COLS + ohe if c in df.columns]

def split_scale(df, seed=42, test_size=0.2):
    fn = feature_names(df)
    X, y, g = df[fn].values, df[TARGET].values, df[GROUP].values
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(X, y, groups=g))
    sc = StandardScaler().fit(X[tr])
    return sc.transform(X[tr]), sc.transform(X[te]), y[tr], y[te], sc, fn

def make_models(seed=42):
    m = {}
    m["Ridge"] = Ridge(alpha=1.0)
    m["RandomForest"] = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                              n_jobs=-1, random_state=seed)
    m["GradBoost"] = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                               max_depth=5, subsample=0.8, random_state=seed)
    m["CatBoost"] = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                      loss_function="RMSE", eval_metric="MAE",
                                      random_seed=seed, verbose=0)
    m["XGBoost"] = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                                subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                                random_state=seed, tree_method="hist")
    m["LightGBM"] = LGBMRegressor(n_estimators=600, learning_rate=0.05, max_depth=-1,
                                  num_leaves=63, subsample=0.8, colsample_bytree=0.8,
                                  n_jobs=-1, random_state=seed, verbose=-1)
    m["HistGB"] = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                                max_depth=None, random_state=seed)
    return m

def make_stack(seed=42):
    # Heterogeneous stacking: strong boosters as bases, Ridge meta-learner.
    bases = [
        ("cat", CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                                  loss_function="RMSE", random_seed=seed, verbose=0)),
        ("xgb", XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                             random_state=seed, tree_method="hist")),
        ("lgb", LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                              random_state=seed, verbose=-1)),
    ]
    return StackingRegressor(estimators=bases, final_estimator=Ridge(alpha=1.0),
                             cv=5, n_jobs=1, passthrough=False)

def metrics(y, p):
    return (mean_absolute_error(y, p),
            float(np.sqrt(mean_squared_error(y, p))),
            r2_score(y, p))

if __name__ == "__main__":
    t0 = time.time()
    df = load_df()
    Xtr, Xte, ytr, yte, sc, fn = split_scale(df, seed=42)
    print(f"data {df.shape} | feats={len(fn)} | train {Xtr.shape[0]} test {Xte.shape[0]}")
    res = {}
    mods = make_models(42)
    mods["Stacking"] = make_stack(42)
    for name, mdl in mods.items():
        t = time.time()
        mdl.fit(Xtr, ytr)
        mae, rmse, r2 = metrics(yte, mdl.predict(Xte))
        res[name] = (mae, rmse, r2)
        print(f"{name:13s} MAE={mae:7.3f}  RMSE={rmse:7.3f}  R2={r2:.4f}  ({time.time()-t:5.1f}s)")
    print(f"TOTAL {time.time()-t0:.1f}s")
