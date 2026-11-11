"""
run_rob.py - three-axis audit of the six baseline regressors on the canonical split.

For Ridge, RandForest, GradBoost, CatBoost, XGBoost and LightGBM it measures
  (1) accuracy    MAE / RMSE / R2 on held-out flights (seed 42),
  (2) robustness  MAE under additive Gaussian noise at 5 / 10 / 20 % of one standard
                  deviation of the standardised features (5 draws), and the relative
                  MAE increase at 20 % noise ("d20"),
  (3) fidelity    Kendall tau between the model's global attribution and the physics
                  reference (normalised |Ridge coefficient|),
and, for the three boosters, the MAE over five leakage-free (flight-grouped) splits.

It is the counterpart of run_final2.py, which does the same for RAS, and it uses the SAME
split, the SAME noise draws (RandomState(70+s)) and the SAME attribution definition, so
the baselines and RAS are directly comparable.

Attribution = mean |SHAP value| over 500 held-out rows (seed-42 sample, as in run_final2):
exact TreeSHAP for the tree models, exact linear SHAP for Ridge. CatBoost / XGBoost /
LightGBM use their built-in exact TreeSHAP (same algorithm as shap.TreeExplainer, and no
dependency on `shap`/numba); RandForest and GradBoost use shap.TreeExplainer. If `shap`
cannot be imported, those two get tau = null and everything else still runs.
NOTE: RandForest's TreeSHAP is slow (~2 s per row measured on a laptop, i.e. ~20 min for
500 rows; every other model takes seconds). Use --shap-n 100 for a quick pass.

XGBoost's output depends on the number of threads, so XGBoost and LightGBM are pinned to
n_jobs=1 to make the numbers identical on any machine.

Run from anywhere:
    python code/run_rob.py                  # all six models + multi-seed  -> results/baselines_rob.json
    python code/run_rob.py --no-multiseed   # canonical split only (much faster)
    python code/run_rob.py --models Ridge CatBoost XGBoost LightGBM
"""
import argparse, json, time
import numpy as np
import exp_core as E
from scipy.stats import kendalltau
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

ALL_MODELS = ["Ridge", "RandForest", "GradBoost", "CatBoost", "XGBoost", "LightGBM"]
MS_MODELS = ["CatBoost", "XGBoost", "LightGBM"]      # the models multiseed_summary.json used to hold
SEED = 42
SEEDS = [42, 7, 13, 21, 99]
NOISE = [0.05, 0.10, 0.20]                           # fraction of one std (features are standardised)
N_DRAWS, NOISE_SEED0 = 5, 70                         # identical to run_final2.py / run_extra.py
EXP_CORE_KEY = {"RandForest": "RandomForest"}        # exp_core.make_models uses the long name


def log(s):
    print(s, flush=True)


def norm_pct(v):
    v = np.abs(np.asarray(v, float)); s = v.sum()
    return v / s * 100 if s > 0 else np.full(len(v), 100 / len(v))


def build(name, seed):
    m = E.make_models(seed)[EXP_CORE_KEY.get(name, name)]
    if name in ("XGBoost", "LightGBM"):
        m.set_params(n_jobs=1)                       # thread-count independent results
    return m


def shap_abs(name, model, X, train_mean):
    """Mean |SHAP value| per feature (exact)."""
    if name == "Ridge":                              # linear SHAP: coef * (x - E[x])
        return np.abs(model.coef_) * np.abs(X - train_mean).mean(0)
    if name == "CatBoost":
        from catboost import Pool
        return np.abs(model.get_feature_importance(Pool(X), type="ShapValues")[:, :-1]).mean(0)
    if name == "XGBoost":
        import xgboost as xgb
        return np.abs(model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]).mean(0)
    if name == "LightGBM":
        return np.abs(model.predict(X, pred_contrib=True)[:, :-1]).mean(0)
    import shap                                      # RandForest / GradBoost (sklearn)
    return np.abs(shap.TreeExplainer(model).shap_values(X)).mean(0)


def versions():
    import sklearn, xgboost, lightgbm, catboost
    return dict(numpy=np.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                lightgbm=lightgbm.__version__, catboost=catboost.__version__)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--no-multiseed", action="store_true", help="skip the 5-split MAE")
    ap.add_argument("--shap-n", type=int, default=500, help="rows used for the attribution (default 500)")
    ap.add_argument("--out", default=str(E.out_path("baselines_rob.json")))
    args = ap.parse_args()

    t_all = time.time()
    df = E.load_df()
    Xtr, Xte, ytr, yte, sc, fn = E.split_scale(df, seed=SEED)
    log(f"data {df.shape} | feats={len(fn)} | train {Xtr.shape[0]} test {Xte.shape[0]}")

    GT = norm_pct(Ridge(alpha=1.0).fit(Xtr, ytr).coef_)                      # physics reference
    noise = [np.random.RandomState(NOISE_SEED0 + s).randn(*Xte.shape) for s in range(N_DRAWS)]
    Xs = Xte[np.random.RandomState(SEED).choice(Xte.shape[0], args.shap_n, replace=False)]
    train_mean = Xtr.mean(0)

    out = dict(meta=dict(seed=SEED, noise_levels=NOISE, n_draws=N_DRAWS, noise_seed0=NOISE_SEED0,
                         shap_rows=args.shap_n, features=fn, reference=GT.tolist(),
                         n_jobs_pinned_to_1=["XGBoost", "LightGBM"], versions=versions()),
               models={})

    def save():
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)

    # ---------------- canonical split: accuracy + robustness + fidelity ----------------
    for name in args.models:
        t = time.time()
        m = build(name, SEED).fit(Xtr, ytr)
        mae, rmse, r2 = E.metrics(yte, m.predict(Xte))
        rob = [float(mae)] + [float(np.mean([mean_absolute_error(yte, m.predict(Xte + e * z)) for z in noise]))
                              for e in NOISE]
        rec = dict(MAE=float(mae), RMSE=float(rmse), R2=float(r2), rob=rob,
                   d20=float(100 * (rob[-1] - rob[0]) / rob[0]))
        if name == "RandForest":
            log(f"  RandForest TreeSHAP on {len(Xs)} rows is slow (~2 s/row); please wait...")
        try:
            imp = norm_pct(shap_abs(name, m, Xs, train_mean))
            rec.update(tau=float(kendalltau(imp, GT)[0]), importance=imp.tolist())
        except ImportError as e:                      # only the `shap` library path can land here
            rec.update(tau=None, importance=None)
            log(f"  ! {name}: tau skipped, shap not importable ({e})")
        rec["sec"] = round(time.time() - t, 1)
        out["models"][name] = rec
        tau_s = "  n/a" if rec["tau"] is None else f"{rec['tau']:.3f}"
        log(f"{name:11s} MAE={rec['MAE']:7.3f} RMSE={rec['RMSE']:7.3f} R2={rec['R2']:.4f} "
            f"d20={rec['d20']:+5.1f}% tau={tau_s} ({rec['sec']}s)")
        save()

    # ---------------- five leakage-free splits (MAE only) ----------------
    if not args.no_multiseed:
        ms_models = [n for n in MS_MODELS if n in args.models]
        ms = {n: [] for n in ms_models}
        for sd in SEEDS:
            if sd == SEED:                            # already computed above
                for n in ms_models:
                    ms[n].append(out["models"][n]["MAE"])
                continue
            a, b, c, d, _, _ = E.split_scale(df, seed=sd)
            for n in ms_models:
                p = build(n, sd).fit(a, c).predict(b)
                ms[n].append(float(mean_absolute_error(d, p)))
            log(f"seed {sd}: " + "  ".join(f"{n}={ms[n][-1]:.3f}" for n in ms_models))
        out["multiseed"] = {n: dict(vals=v, mean=float(np.mean(v)), std=float(np.std(v))) for n, v in ms.items()}
        save()

    log(f"DONE in {time.time() - t_all:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
