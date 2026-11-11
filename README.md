# Auditing Accuracy, Explanation Fidelity, and Robustness for UAV Power Prediction

Code and an interactive decision tool for a UAV power-prediction project. We argue that
selecting a UAV power-draw estimator on prediction error alone is unsafe for a
safety-relevant deployment, and we reframe model selection as a transparent
**three-axis audit** of predictive accuracy, explanation fidelity, and robustness.

> Every number reported by this project is produced by executed code on real flight
> telemetry; no value is hand-entered.


## Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── code/                  # experiment and figure-generation scripts
│   ├── exp_core.py        # data loading, leakage-free split, metrics
│   ├── run_final2.py      # RAS training, canonical run, multi-seed
│   ├── run_audit2.py      # explanation-method audit (TreeSHAP / permutation / gain)
│   ├── run_extra.py       # Ridge/GB/RF multi-seed + RAS ablation
│   ├── run_rob.py         # baselines: accuracy / robustness / fidelity (+ 5-split MAE)
│   ├── run_rfgb.py        # Random Forest / Gradient Boosting baselines
│   ├── analysis2.py       # aggregates results into the reported numbers
│   ├── charts.py          # result figures (accuracy, robustness, audit, UTS, ...)
│   ├── make_diagrams.py   # architecture and deployment diagrams
│   ├── shoot_app.py       # screenshot of app/index.html -> figures/fig_tool.pdf
│   └── style.py           # shared Matplotlib style
├── real_data/             # flights_ready_for_eda.csv (prepared telemetry table)
├── results/               # JSON outputs of the experiments (cached)
├── figures/               # generated figures (created by the scripts)
└── app/                   # runnable operator-facing decision tool
```

## Dataset

We use the public **DJI Matrice 100** in-flight telemetry data set of Rodrigues et al.,
*Scientific Data* **8**:155 (2021), DOI `10.1038/s41597-021-00930-x`. The prepared table is
`real_data/flights_ready_for_eda.csv` (177,464 rows, 188 flights; target
`Power = battery_voltage * battery_current`).

## Reproduce

```bash
# 1. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. experiments (write JSON into results/; can be run from any directory)
python code/run_final2.py        # RAS + multi-seed
python code/run_rob.py           # baselines: accuracy / robustness / fidelity (+ 5-split MAE)
python code/run_rfgb.py          # RF / GB baselines
python code/run_extra.py         # Ridge/GB/RF multi-seed + ablation
python code/run_audit2.py        # explainer audit

# 3. figures (write into figures/; can be run from any directory)
python code/charts.py            # accuracy / robustness / audit / UTS / importance
python code/make_diagrams.py     # architecture + deployment diagrams
# Tool screenshot needs: playwright install chromium && python code/shoot_app.py

# 4. decision tool
# open app/index.html in a browser (static page, no server needed)
```

## Key results (canonical leakage-free split)

| Model | MAE (W) | Fidelity (Kendall tau) | dMAE @20% noise | Pareto |
|-------|:-------:|:----------------------:|:---------------:|:------:|
| **RAS (ours)** | **40.82** | 0.636 | **14.2%** | yes |
| CatBoost | 41.01 | 0.606 | 16.4% | |
| LightGBM | 41.19 | 0.606 | 26.7% | |
| XGBoost | 41.27 | 0.636 | 44.3% | |
| Gradient Boosting | 41.68 | 0.576 | 17.0% | |
| Random Forest | 42.05 | 0.515 | 18.9% | |
| Ridge | 47.90 | 0.970 | 2.8% | yes |

RAS has the lowest MAE in every one of five seeds against CatBoost (Wilcoxon p = 0.0625),
is the most robust of all accurate models, and ties for the most faithful black box.
TreeSHAP is the recommended explainer for this tabular setting.

## License

Code is released under the MIT License (see `LICENSE`). The DJI Matrice 100 data set is
the property of its original authors and is governed by its own license.
