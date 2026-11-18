"""
DTS paper — leakage-free 30-split repeated holdout (re-tuning INSIDE each split)
Uses YOUR pipeline exactly: same data file, same 30 splits (ShuffleSplit seed=0),
same per-split MinMaxScaler, same Default XGB (_BL_PARAMS), same search space
(LB/UB + decode_and_build), same fitness (7 holdout seeds 11..77 x 20% val),
same N_AGENTS=40, MAX_ITER=75, same re-injection, same optimizer code (GTO, PSO,
GWO, WOA copied from your v8.3 script; only the fitness is now passed in).

ONLY CHANGE vs your section [C]: for every split, each optimizer is re-run
using that split's TRAINING data only, instead of reusing the seed-42 best_pos.

Stats: Nadeau-Bengio corrected resampled t-test (main), Wilcoxon (reference),
Holm correction, Friedman, pairwise optimizer comparisons.

Resumable: every finished (split, method) is appended to results_raw.csv;
re-run the script to continue after any stop.

Install : pip install numpy pandas scikit-learn xgboost scipy joblib tqdm matplotlib openpyxl
Run     : python dts_retune_30splits_v2.py
"""

import os, time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy import stats
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ============================== CONFIG ==============================
DATA_PATH = r"C:\Users\AHMAD\Desktop\AKHIL\DTSDATAJULY.xlsx"
OUT_DIR   = r"C:\Users\AHMAD\Desktop\AKHIL\retune_outputs"
N_JOBS    = -1        # parallel workers (-1 = all cores). Each worker = 1 split x 1 optimizer
# ====================================================================

N_SPLITS       = 30
N_AGENTS       = 40
MAX_ITER       = 75
REINJECT_EVERY = 15
REINJECT_FRAC  = 0.25
OPT_SEED       = 42
_HOLDOUT_SEEDS = [11, 22, 33, 44, 55, 66, 77]

DIM = 8
LB  = np.array([200, 4, 0.01, 0.25, 0.4, 0.1, 1, 0.0], dtype=float)
UB  = np.array([800, 8, 0.20, 1.0, 1.0, 5.0, 10, 0.5], dtype=float)
PARAM_LABELS = ["n_estimators", "max_depth", "learning_rate", "subsample",
                "colsample_bytree", "reg_lambda", "min_child_weight", "gamma"]

DEFAULT_XGB = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                   base_score=0.5, random_state=42, verbosity=0, n_jobs=1)

os.makedirs(OUT_DIR, exist_ok=True)
RAW = os.path.join(OUT_DIR, "results_raw.csv")


# ---------------------------- data ----------------------------
def load_data():
    df_raw = pd.read_excel(DATA_PATH)
    df = df_raw.loc[:, ~df_raw.columns.str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)
    return X, y, list(df.columns[:-1])


def decode_and_build(position):
    p = np.clip(np.array(position, dtype=float), LB, UB)
    return XGBRegressor(
        n_estimators=max(50, int(round(p[0]))),
        max_depth=max(3, int(round(p[1]))),
        learning_rate=float(np.clip(p[2], 0.005, 0.3)),
        subsample=float(np.clip(p[3], 0.25, 1.0)),
        colsample_bytree=float(np.clip(p[4], 0.4, 1.0)),
        reg_lambda=float(np.clip(p[5], 0.0, 5.0)),
        min_child_weight=max(1, int(round(p[6]))),
        gamma=float(np.clip(p[7], 0.0, 0.5)),
        base_score=0.5, reg_alpha=0.0, random_state=42, verbosity=0, n_jobs=1)


def make_fitness(X_train, y_train):
    def fitness(position):
        try:
            sc = []
            for s in _HOLDOUT_SEEDS:
                Xa, Xb, ya, yb = train_test_split(X_train, y_train, test_size=0.20, random_state=s)
                m = decode_and_build(position).fit(Xa, ya)
                sc.append(r2_score(yb, m.predict(Xb)))
            v = float(np.mean(sc))
            return v if np.isfinite(v) else -999.0
        except Exception:
            return -999.0
    return fitness


def reinject_diversity(fitness, pop, fits, rng, best_pos, best_score):
    n_re = max(1, int(len(pop) * REINJECT_FRAC))
    for idx in np.argsort(fits)[:n_re]:
        pop[idx] = LB + rng.random(DIM) * (UB - LB)
        fits[idx] = fitness(pop[idx])
        if fits[idx] > best_score:
            best_pos, best_score = pop[idx].copy(), fits[idx]
    return pop, fits, best_pos, best_score


# ------------------ optimizers (from your v8.3, prints removed) ------------------
def gto(fitness, n_agents=N_AGENTS, max_iter=MAX_ITER, seed=OPT_SEED):
    rng = np.random.default_rng(seed)
    pop = LB + rng.random((n_agents, DIM)) * (UB - LB)
    fits = np.array([fitness(p) for p in pop])
    bi = np.argmax(fits); silverback = pop[bi].copy(); best_score = fits[bi]
    p_exploit = 0.03; beta = 3
    for t in range(max_iter):
        w = 2 * (1 - t / max_iter)
        for i in range(n_agents):
            if rng.random() >= 0.5:
                if rng.random() < p_exploit:
                    X_new = (UB - LB) * rng.random(DIM) + LB
                else:
                    j = rng.integers(0, n_agents)
                    X_new = (pop[i] - rng.random(DIM) * (pop[i] - pop[j])
                             + rng.random(DIM) * (silverback - pop[i]))
            else:
                if abs(w) < 1:
                    X_new = (np.abs(rng.random(DIM) * silverback - pop[i])
                             * (beta * silverback - pop[i]) + silverback)
                else:
                    L = rng.choice([-1, 1], DIM) * rng.random(DIM)
                    X_new = silverback - (silverback * L - pop[i] * L) * rng.random(DIM)
            X_new = np.clip(X_new, LB, UB)
            f = fitness(X_new)
            if f > fits[i]: pop[i] = X_new; fits[i] = f
            if f > best_score: silverback = X_new.copy(); best_score = f
        bi = np.argmax(fits)
        if fits[bi] > best_score: silverback = pop[bi].copy(); best_score = fits[bi]
        if (t + 1) % REINJECT_EVERY == 0:
            pop, fits, silverback, best_score = reinject_diversity(
                fitness, pop, fits, rng, silverback, best_score)
    return silverback, best_score


def pso(fitness, n_agents=N_AGENTS, max_iter=MAX_ITER, seed=OPT_SEED):
    rng = np.random.default_rng(seed)
    w_max, w_min, c1, c2 = 0.9, 0.4, 2.05, 2.05
    pos = LB + rng.random((n_agents, DIM)) * (UB - LB)
    vel = np.zeros((n_agents, DIM)); v_max = 0.2 * (UB - LB); v_min = -v_max
    fits = np.array([fitness(p) for p in pos])
    pbest = pos.copy(); pbest_fits = fits.copy()
    bi = np.argmax(fits); gbest = pos[bi].copy(); best_score = fits[bi]
    for t in range(max_iter):
        w = w_max - (w_max - w_min) * t / max_iter
        for i in range(n_agents):
            r1 = rng.random(DIM); r2 = rng.random(DIM)
            vel[i] = np.clip(w * vel[i] + c1 * r1 * (pbest[i] - pos[i])
                             + c2 * r2 * (gbest - pos[i]), v_min, v_max)
            pos[i] = np.clip(pos[i] + vel[i], LB, UB)
            f = fitness(pos[i]); fits[i] = f
            if f > pbest_fits[i]: pbest[i] = pos[i].copy(); pbest_fits[i] = f
            if f > best_score: gbest = pos[i].copy(); best_score = f
        if (t + 1) % REINJECT_EVERY == 0:
            for idx in np.argsort(fits)[:max(1, int(n_agents * REINJECT_FRAC))]:
                pos[idx] = LB + rng.random(DIM) * (UB - LB); vel[idx] = np.zeros(DIM)
                fits[idx] = fitness(pos[idx]); pbest[idx] = pos[idx].copy()
                pbest_fits[idx] = fits[idx]
                if fits[idx] > best_score: gbest = pos[idx].copy(); best_score = fits[idx]
    return gbest, best_score


def gwo(fitness, n_agents=N_AGENTS, max_iter=MAX_ITER, seed=OPT_SEED):
    rng = np.random.default_rng(seed)
    wolves = LB + rng.random((n_agents, DIM)) * (UB - LB)
    fits = np.array([fitness(w) for w in wolves])
    idx = np.argsort(-fits)
    alpha, beta_w, delta = wolves[idx[0]].copy(), wolves[idx[1]].copy(), wolves[idx[2]].copy()
    best_score = fits[idx[0]]
    for t in range(max_iter):
        a = 2.0 - 2.0 * t / max_iter
        for i in range(n_agents):
            new_pos = np.zeros(DIM)
            for leader in (alpha, beta_w, delta):
                r1 = rng.random(DIM); r2 = rng.random(DIM)
                A = 2 * a * r1 - a; C = 2 * r2
                new_pos += leader - A * np.abs(C * leader - wolves[i])
            wolves[i] = np.clip(new_pos / 3.0, LB, UB)
            fits[i] = fitness(wolves[i])
        idx = np.argsort(-fits)
        alpha, beta_w, delta = wolves[idx[0]].copy(), wolves[idx[1]].copy(), wolves[idx[2]].copy()
        best_score = fits[idx[0]]
        if (t + 1) % REINJECT_EVERY == 0:
            wolves, fits, alpha, best_score = reinject_diversity(
                fitness, wolves, fits, rng, alpha, best_score)
            idx = np.argsort(-fits)
            alpha, beta_w, delta = wolves[idx[0]].copy(), wolves[idx[1]].copy(), wolves[idx[2]].copy()
            best_score = fits[idx[0]]
    return alpha, best_score


def woa(fitness, n_agents=N_AGENTS, max_iter=MAX_ITER, seed=OPT_SEED):
    rng = np.random.default_rng(seed)
    pop = LB + rng.random((n_agents, DIM)) * (UB - LB)
    fits = np.array([fitness(p) for p in pop])
    bi = np.argmax(fits); best_pos = pop[bi].copy(); best_score = fits[bi]
    b = 1.0
    for t in range(max_iter):
        a = 2.0 - 2.0 * t / max_iter; a2 = -1.0 - t / max_iter
        for i in range(n_agents):
            r1 = rng.random(); r2 = rng.random()
            A = 2 * a * r1 - a; C = 2 * r2
            p = rng.random(); l = (a2 - 1) * rng.random() + 1
            if p < 0.5:
                if abs(A) < 1:
                    new_pos = best_pos - A * np.abs(C * best_pos - pop[i])
                else:
                    ri = rng.integers(0, n_agents)
                    new_pos = pop[ri] - A * np.abs(C * pop[ri] - pop[i])
            else:
                D = np.abs(best_pos - pop[i])
                new_pos = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_pos
            new_pos = np.clip(new_pos, LB, UB)
            f = fitness(new_pos)
            if f > fits[i]: pop[i] = new_pos; fits[i] = f
            if f > best_score: best_pos = new_pos.copy(); best_score = f
        if (t + 1) % REINJECT_EVERY == 0:
            pop, fits, best_pos, best_score = reinject_diversity(
                fitness, pop, fits, rng, best_pos, best_score)
    return best_pos, best_score


OPTIMIZERS = {"GTO": gto, "PSO": pso, "GWO": gwo, "WOA": woa}
METHODS = ["Default"] + list(OPTIMIZERS)


# ---------------------------- one task ----------------------------
def metrics(y, p):
    return dict(R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=mean_absolute_error(y, p))


def run_task(X, y, split_idx, tr_idx, te_idx, method):
    t0 = time.time()
    sc = MinMaxScaler(feature_range=(0, 1))
    Xtr = sc.fit_transform(X[tr_idx]); Xte = sc.transform(X[te_idx])
    ytr, yte = y[tr_idx], y[te_idx]
    row = dict(split=split_idx + 1, method=method, fitness=np.nan)
    row.update({n: np.nan for n in PARAM_LABELS})   # fixed column order for the CSV
    if method == "Default":
        m = XGBRegressor(**DEFAULT_XGB); row["fitness"] = np.nan
    else:
        best_pos, best_fit = OPTIMIZERS[method](make_fitness(Xtr, ytr))
        m = decode_and_build(best_pos); row["fitness"] = best_fit
        p = np.clip(best_pos, LB, UB)
        for n, v in zip(PARAM_LABELS, p):
            row[n] = int(round(v)) if n in ("n_estimators", "max_depth", "min_child_weight") else v
    m.fit(Xtr, ytr)
    row.update({f"train_{k}": v for k, v in metrics(ytr, m.predict(Xtr)).items()})
    row.update({f"test_{k}": v for k, v in metrics(yte, m.predict(Xte)).items()})
    row.update(n_train=len(tr_idx), n_test=len(te_idx), minutes=(time.time() - t0) / 60)
    return row


# ---------------------------- statistics ----------------------------
def corrected_ttest(a, b, n_tr, n_te):
    """Nadeau & Bengio (2003) corrected resampled t-test, two-sided."""
    d = np.asarray(a) - np.asarray(b); J = len(d); var = np.var(d, ddof=1)
    if var == 0: return np.nan, np.nan
    t = d.mean() / np.sqrt((1 / J + n_te / n_tr) * var)
    return t, 2 * stats.t.sf(abs(t), df=J - 1)


def holm(p):
    p = np.asarray(p, float); order = np.argsort(p); adj = np.empty_like(p); run = 0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i]); adj[i] = min(run, 1.0)
    return adj


def analyse():
    r = pd.read_csv(RAW)
    piv = r.pivot(index="split", columns="method", values="test_R2").dropna()[METHODS]
    J = len(piv)
    if J < N_SPLITS:
        print(f"\nOnly {J}/{N_SPLITS} complete splits so far — statistics are provisional.")
    n_tr, n_te = int(r.n_train.iloc[0]), int(r.n_test.iloc[0])

    s = r.groupby("method").agg(
        test_R2_mean=("test_R2", "mean"), test_R2_sd=("test_R2", "std"),
        test_RMSE_mean=("test_RMSE", "mean"), test_MAE_mean=("test_MAE", "mean"),
        train_R2_mean=("train_R2", "mean"), fitness_mean=("fitness", "mean")).reindex(METHODS)
    s["gap_mean"] = s.train_R2_mean - s.test_R2_mean

    vs = []
    for o in OPTIMIZERS:
        d = piv[o] - piv["Default"]
        t, p = corrected_ttest(piv[o], piv["Default"], n_tr, n_te)
        w = stats.wilcoxon(piv[o], piv["Default"])
        vs.append(dict(optimizer=o, mean_dR2=d.mean(), sd_dR2=d.std(ddof=1),
                       splits_improved=f"{int((d > 0).sum())}/{J}",
                       corrected_t=t, corrected_p=p, wilcoxon_p=w.pvalue))
    vs = pd.DataFrame(vs)
    vs["corrected_p_holm"] = holm(vs.corrected_p)
    vs["significant_0.05"] = vs.corrected_p_holm < 0.05

    names = list(OPTIMIZERS); pw = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            t, p = corrected_ttest(piv[a], piv[b], n_tr, n_te)
            pw.append(dict(A=a, B=b, mean_diff=(piv[a] - piv[b]).mean(),
                           corrected_t=t, corrected_p=p))
    pw = pd.DataFrame(pw); pw["corrected_p_holm"] = holm(pw.corrected_p)

    fr = stats.friedmanchisquare(*[piv[m] for m in METHODS])
    hp = r[r.method != "Default"].groupby("method")[PARAM_LABELS].agg(["mean", "std"])

    xlsx = os.path.join(OUT_DIR, "retune_summary.xlsx")
    with pd.ExcelWriter(xlsx) as xw:
        s.to_excel(xw, sheet_name="Summary")
        vs.to_excel(xw, sheet_name="Opt_vs_Default", index=False)
        pw.to_excel(xw, sheet_name="Pairwise", index=False)
        piv.to_excel(xw, sheet_name="Test_R2_per_split")
        hp.to_excel(xw, sheet_name="Tuned_params_mean_sd")
        pd.DataFrame([dict(Friedman_chi2=fr.statistic, p=fr.pvalue, splits=J)]).to_excel(
            xw, sheet_name="Friedman", index=False)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([piv[m] for m in METHODS])
    ax.set_xticks(range(1, len(METHODS) + 1), ["Default XGB"] + [f"{o}-XGB" for o in names])
    ax.axhline(piv["Default"].mean(), ls="--", lw=1, color="grey")
    ax.set_ylabel("Test R² (30 splits, tuned within each split)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "boxplot_test_R2.png"), dpi=300)

    pd.set_option("display.width", 200)
    print("\n=== Summary ===\n", s.round(4).to_string())
    print("\n=== Optimizer vs Default (corrected t-test, Holm) ===\n", vs.round(4).to_string(index=False))
    print("\n=== Pairwise ===\n", pw.round(4).to_string(index=False))
    print(f"\nFriedman chi2 = {fr.statistic:.3f}, p = {fr.pvalue:.5f}  (splits = {J})")
    print(f"\nSaved: {xlsx}, results_raw.csv, boxplot_test_R2.png in {OUT_DIR}")


# ---------------------------- main ----------------------------
def main():
    X, y, feats = load_data()
    print(f"Data: {X.shape[0]} rows x {X.shape[1]} features")
    splits = list(ShuffleSplit(n_splits=N_SPLITS, test_size=0.20, random_state=0).split(X))
    done = set()
    if os.path.exists(RAW):
        prev = pd.read_csv(RAW); done = set(zip(prev.split, prev.method))
    tasks = [(k, m) for m in list(OPTIMIZERS) + ["Default"] for k in range(N_SPLITS)
             if (k + 1, m) not in done]
    total = N_SPLITS * len(METHODS)
    print(f"Tasks remaining: {len(tasks)} / {total}   "
          f"(each optimizer task ~20-28 min on one core; runs in parallel)")
    if tasks:
        gen = Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(
            delayed(run_task)(X, y, k, splits[k][0], splits[k][1], m) for k, m in tasks)
        bar = tqdm(gen, total=len(tasks), desc="split x method", initial=0,
                   bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed {elapsed}, left {remaining}]")
        for row in bar:
            pd.DataFrame([row]).to_csv(RAW, mode="a", index=False, header=not os.path.exists(RAW))
            bar.set_postfix_str(f"split {row['split']} {row['method']} R2={row['test_R2']:.3f}")
    analyse()


if __name__ == "__main__":
    main()
