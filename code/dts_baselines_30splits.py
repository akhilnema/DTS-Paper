# ===================== DTS paper — baseline comparison on the SAME 30 splits =====================
# Paste this whole block into ONE Jupyter cell and run.
# Same protocol as the re-tuning run: ShuffleSplit(30, test=0.20, seed=0), MinMaxScaler fitted on
# each split's training data only. Models: Linear, Ridge (alpha chosen by CV on training data
# only), Random Forest, and the four default ensembles (same settings as your _BL_PARAMS).
# It then merges the tuned XGB results from results_raw.csv and tests every model against
# WOA-XGB and Default XGB with the Nadeau-Bengio corrected resampled t-test (Holm-adjusted).
# Expected run time on the workstation: a few minutes.

import os, time, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from scipy import stats
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
warnings.filterwarnings("ignore")

DATA_PATH = r"C:\Users\AHMAD\Desktop\AKHIL\DTSDATAJULY.xlsx"
OUT_DIR   = r"C:\Users\AHMAD\Desktop\AKHIL\retune_outputs"
RETUNE_RAW = os.path.join(OUT_DIR, "results_raw.csv")      # from the re-tuning run
N_JOBS = -1
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_excel(DATA_PATH)
df = df.loc[:, ~df.columns.str.startswith("Unnamed")].dropna(axis=1, how="all").dropna()
X = df.iloc[:, :-1].values.astype(float); y = df.iloc[:, -1].values.astype(float)
splits = list(ShuffleSplit(n_splits=30, test_size=0.20, random_state=0).split(X))
print(f"Data: {X.shape[0]} rows x {X.shape[1]} features | 30 splits")

MODELS = {
    "Linear":        lambda: LinearRegression(),
    "Ridge":         lambda: RidgeCV(alphas=np.logspace(-3, 3, 25)),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1),
    "Extra Trees":   lambda: ExtraTreesRegressor(n_estimators=300, max_depth=15, min_samples_split=5,
                                                 min_samples_leaf=2, bootstrap=True, random_state=42, n_jobs=1),
    "GBM":           lambda: GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                                       subsample=0.8, random_state=42),
    "CatBoost":      lambda: CatBoostRegressor(iterations=300, learning_rate=0.05, depth=6,
                                               random_seed=42, verbose=0, thread_count=1),
    "Default XGB":   lambda: XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8,
                                          colsample_bytree=0.8, reg_lambda=1.0, base_score=0.5,
                                          random_state=42, verbosity=0, n_jobs=1),
}

def run(k, name):
    tr, te = splits[k]
    sc = MinMaxScaler().fit(X[tr]); Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    m = MODELS[name](); m.fit(Xtr, y[tr]); pt, pe = m.predict(Xtr), m.predict(Xte)
    return dict(split=k + 1, method=name, train_R2=r2_score(y[tr], pt), test_R2=r2_score(y[te], pe),
                test_RMSE=float(np.sqrt(mean_squared_error(y[te], pe))), test_MAE=mean_absolute_error(y[te], pe),
                n_train=len(tr), n_test=len(te))

tasks = [(k, n) for n in MODELS for k in range(30)]
t0 = time.time()
gen = Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(delayed(run)(k, n) for k, n in tasks)
rows = [r for r in tqdm(gen, total=len(tasks), desc="model x split",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed {elapsed}, left {remaining}]")]
base = pd.DataFrame(rows)
base.to_csv(os.path.join(OUT_DIR, "baselines_raw.csv"), index=False)

# merge tuned XGB results from the re-tuning run
rt = pd.read_csv(RETUNE_RAW)
rt = rt[rt.method != "Default"].copy(); rt["method"] = rt.method + "-XGB"
allr = pd.concat([base, rt[base.columns]], ignore_index=True)
order = list(MODELS) + ["GTO-XGB", "PSO-XGB", "GWO-XGB", "WOA-XGB"]
piv = allr.pivot(index="split", columns="method", values="test_R2")[order]
n_tr, n_te = int(allr.n_train.iloc[0]), int(allr.n_test.iloc[0])

def ctest(a, b):
    d = np.asarray(a) - np.asarray(b); J = len(d); v = np.var(d, ddof=1)
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * v)
    return d.mean(), t, 2 * stats.t.sf(abs(t), J - 1)

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run_ = 0
    for r_, i in enumerate(o):
        run_ = max(run_, (len(p) - r_) * p[i]); adj[i] = min(run_, 1)
    return adj

summ = allr.groupby("method").agg(test_R2_mean=("test_R2", "mean"), test_R2_sd=("test_R2", "std"),
                                  test_RMSE_mean=("test_RMSE", "mean"), test_MAE_mean=("test_MAE", "mean"),
                                  train_R2_mean=("train_R2", "mean")).reindex(order)
summ["gap_mean"] = summ.train_R2_mean - summ.test_R2_mean

out = {}
for ref in ["WOA-XGB", "Default XGB"]:
    rr = [dict(model=m, ref=ref, **dict(zip(["mean_diff", "corrected_t", "corrected_p"], ctest(piv[m], piv[ref]))))
          for m in order if m != ref]
    rr = pd.DataFrame(rr); rr["p_holm"] = holm(rr.corrected_p); rr["significant_0.05"] = rr.p_holm < 0.05
    out[ref] = rr

xlsx = os.path.join(OUT_DIR, "baselines_summary.xlsx")
with pd.ExcelWriter(xlsx) as xw:
    summ.to_excel(xw, sheet_name="Summary")
    out["WOA-XGB"].to_excel(xw, sheet_name="vs_WOA-XGB", index=False)
    out["Default XGB"].to_excel(xw, sheet_name="vs_Default_XGB", index=False)
    piv.to_excel(xw, sheet_name="Test_R2_per_split")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.boxplot([piv[m] for m in order])
ax.set_xticks(range(1, len(order) + 1), order, rotation=35, ha="right")
ax.set_ylabel("Test R² (30 repeated holdout splits)")
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "boxplot_all_models.png"), dpi=300); plt.close(fig)

pd.set_option("display.width", 200)
print("\n=== Summary (30 splits) ===\n", summ.round(4).to_string())
print("\n=== vs WOA-XGB (corrected t-test, Holm) ===\n", out["WOA-XGB"].round(4).to_string(index=False))
print("\n=== vs Default XGB ===\n", out["Default XGB"].round(4).to_string(index=False))
print(f"\nSaved in {OUT_DIR}: baselines_raw.csv, baselines_summary.xlsx, boxplot_all_models.png"
      f"  | total time {time.time() - t0:.0f} s")
