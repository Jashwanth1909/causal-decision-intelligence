#!/usr/bin/env python3
"""
Causal AI-based data-driven decision intelligence for prediction and optimal
intervention selection.

One script reproduces every table, figure and number in the paper:
    python causal_di.py            # full run (10 replications per setting)
    python causal_di.py --quick    # smoke test (3 replications)

Settings
  Synthetic : confounded simulation with known potential outcomes (5 covariates).
  IHDP      : semi-synthetic benchmark (Hill, 2011), 10 standard replications.
  Venture   : semi-synthetic digital-entrepreneurship benchmark built on the real
              covariates of 19,814 ventures (Crunchbase export, Dec 2015); the
              baseline exit model is fitted to real exits, the intervention effect
              is simulated so that both potential outcomes are known.
Learners    : naive association, S-, T-, X-, DR- and R-learners.
Decision    : cost-aware policy pi(x) = 1[tau_hat(x) > c]; evaluated by true policy
              value and regret, and selected WITHOUT ground truth by direct-method,
              IPW and doubly robust (DR) off-policy value estimates.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / "data", ROOT / "results"
SEED = 42
IHDP_URL = ("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/"
            "9081f863e24ce21bd34c8d6a41bf0edc7d1b65dd/datasets/IHDP/csv/ihdp_npci_{}.csv")
CB_URL = ("https://raw.githubusercontent.com/notpeter/crunchbase-data/"
          "be8711fd4ee6b81110d243eb887ccc6bc8513601/{}")
CB_FILES = ("companies.csv", "rounds.csv", "investments.csv", "acquisitions.csv")
COST = {"Synthetic": 0.10, "IHDP": None, "Venture": 0.03}  # intervention cost per unit (outcome units);
# IHDP replications differ in effect scale by an order of magnitude, so its cost is set per replication
# to the median true CATE of the learning sample (half of the units benefit by more than the cost).
CLIP = (0.05, 0.95)                                          # propensity clipping
LEARNERS = ["Naive", "S-learner", "T-learner", "X-learner", "DR-learner", "R-learner"]
POLICIES = LEARNERS + ["Treat all", "Treat none"]
SETTINGS = ["Synthetic", "IHDP", "Venture"]
# validated categorical palette (slots 1-6), fixed order
PAL = dict(zip(["Synthetic", "IHDP", "Venture"], ["#2a78d6", "#eb6834", "#1baf7a"]))
LPAL = dict(zip(LEARNERS + ["Treat all", "Oracle"],
                ["#a3a29b", "#2a78d6", "#eb6834", "#eda100", "#1baf7a", "#6250d6", "#e34948", "#0b0b0b"]))


# ============================================================================ data
def fetch(url: str, dest: Path):
    if not dest.exists():
        DATA.mkdir(exist_ok=True)
        print(f"  downloading {dest.name}")
        try:
            urllib.request.urlretrieve(url, dest)
        except OSError as e:
            raise SystemExit(f"Download of {url} failed ({e}); place the file in {DATA} manually.")


def synthetic(rep: int, n: int = 15000):
    """Confounded DGP of the conference version (Cheruku): 5 N(0,1) covariates."""
    rng = np.random.default_rng(1000 + rep)
    X = rng.normal(size=(n, 5))
    e = expit(0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.4 * X[:, 2])
    b = -0.5 + 0.9 * X[:, 0] - 0.7 * X[:, 1] + 0.4 * X[:, 2] + 0.25 * X[:, 3]
    h = 0.8 + 0.7 * X[:, 0] - 0.8 * X[:, 2] + 0.4 * X[:, 4]
    mu0, mu1 = expit(b), expit(b + h)
    T = rng.binomial(1, e)
    Y = rng.binomial(1, np.where(T == 1, mu1, mu0))
    return X, T, Y.astype(float), mu0, mu1, True


def ihdp(rep: int):
    f = DATA / f"ihdp_npci_{rep + 1}.csv"
    fetch(IHDP_URL.format(rep + 1), f)
    d = np.loadtxt(f, delimiter=",")
    T, Y, mu0, mu1, X = d[:, 0].astype(int), d[:, 1], d[:, 3], d[:, 4], d[:, 5:]
    return X, T, Y, mu0, mu1, False


_VENTURE = {}


def venture_covariates() -> pd.DataFrame:
    """Real covariates of ventures at their first financing round (2003-2010),
    computed only from events dated on or before that round (no look-ahead)."""
    if "X" in _VENTURE:
        return _VENTURE["X"]
    for f in CB_FILES:
        fetch(CB_URL.format(f), DATA / f)
    comp = pd.read_csv(DATA / "companies.csv", low_memory=False)
    rnd = pd.read_csv(DATA / "rounds.csv", low_memory=False)
    inv = pd.read_csv(DATA / "investments.csv", low_memory=False)
    acq = pd.read_csv(DATA / "acquisitions.csv", low_memory=False)
    rnd["funded_at"] = pd.to_datetime(rnd.funded_at, errors="coerce")
    inv["funded_at"] = pd.to_datetime(inv.funded_at, errors="coerce")
    rnd, inv = rnd.dropna(subset=["funded_at"]), inv.dropna(subset=["funded_at", "investor_permalink"])
    acq_date = pd.to_datetime(acq.acquired_at, errors="coerce").groupby(acq.company_permalink).min()
    t0 = rnd.groupby("company_permalink").funded_at.min().rename("t0")
    first = rnd.join(t0, on="company_permalink").query("funded_at == t0")
    g = first.groupby("company_permalink")
    df = pd.DataFrame({"t0": g.t0.first(), "usd": g.raised_amount_usd.sum(min_count=1),
                       "types": g.funding_round_type.agg(set)})
    df = df.join(comp.set_index("permalink")[["status", "category_list", "country_code", "region", "founded_at"]],
                 how="inner")
    mature = {"private_equity", "post_ipo_equity", "post_ipo_debt", "secondary_market"}
    df = df[df.t0.dt.year.between(2003, 2010) & ~df.types.apply(lambda s: bool(s & mature)) & df.status.notna()]
    df = df[~(df.index.map(acq_date).to_series(index=df.index) <= df.t0).fillna(False)]
    pair = inv.groupby(["investor_permalink", "company_permalink"]).funded_at.min().reset_index()
    pair["exit_at"] = pair.company_permalink.map(acq_date)
    deals = {k: np.sort(v.funded_at.values) for k, v in pair.groupby("investor_permalink")}
    ex = pair[pair.exit_at.notna() & (pair.funded_at <= pair.exit_at)]
    exits = {k: np.sort(v.exit_at.values) for k, v in ex.groupby("investor_permalink")}
    fi = inv.join(t0, on="company_permalink").query("funded_at == t0")
    synd = fi.groupby("company_permalink").investor_permalink.agg(lambda s: sorted(set(s)))
    synd = synd.reindex(df.index).apply(lambda x: x if isinstance(x, list) else [])

    def prior(events, members, t):
        return max([int(np.searchsorted(events[a], t)) for a in members if a in events] or [0])

    has = lambda *ts: df.types.apply(lambda s: float(bool(s & set(ts))))
    cats = df.category_list.fillna("").str.lower().str.split("|")
    dig = r"software|mobile|web|internet|online|saas|app|analytics|data|cloud|digital|commerce|social|search|fintech|payments|games|platform"
    X = pd.DataFrame(index=df.index)
    X["log_round"] = np.log10(df.usd.where(df.usd > 0))
    X["round_missing"] = X.log_round.isna().astype(float)
    X["log_round"] = X.log_round.fillna(X.log_round.median())
    X["n_investors"] = synd.str.len().astype(float)
    X["inv_experience"] = np.log1p([prior(deals, s, t) for s, t in zip(synd, df.t0.values)])
    X["inv_track_record"] = np.log1p([prior(exits, s, t) for s, t in zip(synd, df.t0.values)])
    X["venture_round"] = has("venture")
    X["seed_angel"] = has("seed", "angel")
    X["tier1_hub"] = df.region.isin({"SF Bay Area", "New York City", "Boston"}).astype(float)
    X["usa"] = (df.country_code == "USA").astype(float)
    rx = re.compile(dig)
    X["digital"] = cats.apply(lambda l: np.mean([bool(rx.search(c)) for c in l if c]) if any(l) else 0.0)
    X["life_science"] = cats.apply(lambda l: float(any(k in c for c in l for k in ("biotech", "pharma", "medical", "health"))))
    age = ((df.t0 - pd.to_datetime(df.founded_at, errors="coerce")).dt.days / 365.25).clip(0, 50)
    X["firm_age"] = age.fillna(age.median())
    X["real_exit"] = df.status.isin(["acquired", "ipo"]).astype(int)
    _VENTURE["X"] = X
    return X


def venture(rep: int):
    """Semi-synthetic intervention on real venture covariates.
    mu0: logistic exit model fitted to REAL exits (baseline surface grounded in data).
    T  : a digital-transformation support programme, confounded selection.
    tau: heterogeneous effect, larger for digital ventures with less experienced investors."""
    D = venture_covariates()
    feats = [c for c in D.columns if c != "real_exit"]
    if "b" not in _VENTURE:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(D[feats], D.real_exit)
        _VENTURE["b"] = m.decision_function(D[feats])
        _VENTURE["Z"] = StandardScaler().fit_transform(D[feats])
        _VENTURE["auc_real"] = roc_auc_score(D.real_exit, _VENTURE["b"])
    b, Z = _VENTURE["b"], pd.DataFrame(_VENTURE["Z"], columns=feats)
    rng = np.random.default_rng(2000 + rep)
    e = expit(-0.4 + 0.6 * Z.digital + 0.5 * Z.tier1_hub + 0.5 * Z.inv_experience + 0.3 * Z.log_round)
    h = 0.25 + 0.45 * Z.digital - 0.45 * Z.inv_experience - 0.25 * Z.tier1_hub + 0.2 * Z.seed_angel
    mu0, mu1 = expit(b), expit(b + h.to_numpy())
    T = rng.binomial(1, e)
    Y = rng.binomial(1, np.where(T == 1, mu1, mu0))
    return D[feats].to_numpy(), T, Y.astype(float), mu0, mu1, True


LOADERS = {"Synthetic": synthetic, "IHDP": ihdp, "Venture": venture}


# ============================================================================ learners
def outcome_model(binary: bool):
    kw = dict(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
              l2_regularization=1.0, random_state=SEED)
    return HistGradientBoostingClassifier(**kw) if binary else HistGradientBoostingRegressor(**kw)


def effect_model():
    return RandomForestRegressor(n_estimators=300, min_samples_leaf=20, max_features=0.5,
                                 n_jobs=-1, random_state=SEED)


def predict(m, X, binary):
    return m.predict_proba(X)[:, 1] if binary else m.predict(X)


def propensity(X, T):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, T)


def ps(m, X):
    return np.clip(m.predict_proba(X)[:, 1], *CLIP)


def crossfit_nuisance(X, T, Y, binary, k=3):
    """Out-of-fold mu0, mu1, m (=E[Y|X]) and e for every training unit."""
    n = len(Y)
    mu0, mu1, m, e = (np.zeros(n) for _ in range(4))
    for tr, te in KFold(k, shuffle=True, random_state=SEED).split(X):
        Xt, Tt, Yt = X[tr], T[tr], Y[tr]
        mu0[te] = predict(outcome_model(binary).fit(Xt[Tt == 0], Yt[Tt == 0]), X[te], binary)
        mu1[te] = predict(outcome_model(binary).fit(Xt[Tt == 1], Yt[Tt == 1]), X[te], binary)
        m[te] = predict(outcome_model(binary).fit(Xt, Yt), X[te], binary)
        e[te] = ps(propensity(Xt, Tt), X[te])
    return mu0, mu1, m, e


def fit_learners(X, T, Y, binary):
    """Return {learner: tau_hat function} trained on (X, T, Y), plus a factual outcome model."""
    out = {}
    ate = Y[T == 1].mean() - Y[T == 0].mean()
    out["Naive"] = lambda Xn, a=ate: np.full(len(Xn), a)
    s = outcome_model(binary).fit(np.column_stack([X, T]), Y)
    out["S-learner"] = lambda Xn, s=s: (predict(s, np.column_stack([Xn, np.ones(len(Xn))]), binary)
                                        - predict(s, np.column_stack([Xn, np.zeros(len(Xn))]), binary))
    m0 = outcome_model(binary).fit(X[T == 0], Y[T == 0])
    m1 = outcome_model(binary).fit(X[T == 1], Y[T == 1])
    out["T-learner"] = lambda Xn: predict(m1, Xn, binary) - predict(m0, Xn, binary)
    pm = propensity(X, T)
    d1 = Y[T == 1] - predict(m0, X[T == 1], binary)
    d0 = predict(m1, X[T == 0], binary) - Y[T == 0]
    t1, t0 = effect_model().fit(X[T == 1], d1), effect_model().fit(X[T == 0], d0)
    out["X-learner"] = lambda Xn: ps(pm, Xn) * t0.predict(Xn) + (1 - ps(pm, Xn)) * t1.predict(Xn)
    mu0, mu1, m, e = crossfit_nuisance(X, T, Y, binary)
    psi = mu1 - mu0 + T * (Y - mu1) / e - (1 - T) * (Y - mu0) / (1 - e)
    dr = effect_model().fit(X, psi)
    out["DR-learner"] = dr.predict
    rt = T - e
    rl = effect_model().fit(X, (Y - m) / rt, sample_weight=rt ** 2)
    out["R-learner"] = rl.predict
    return out, s


# ============================================================================ evaluation
def true_value(pi, mu0, mu1, c):
    return float(np.mean(np.where(pi == 1, mu1, mu0) - c * pi))


def ope(policies: dict, X, T, Y, binary, c):
    """Ground-truth-free policy values on the test fold (2-fold cross-fitted nuisances)."""
    mu0, mu1, _, e = crossfit_nuisance(X, T, Y, binary)
    p_obs = np.where(T == 1, e, 1 - e)
    mu_obs = np.where(T == 1, mu1, mu0)
    est = {}
    for k, pi in policies.items():
        mu_pi = np.where(pi == 1, mu1, mu0)
        match = (T == pi).astype(float)
        est[k] = {"DM": float(np.mean(mu_pi - c * pi)),
                  "IPW": float(np.mean(match * Y / p_obs - c * pi)),
                  "DR": float(np.mean(mu_pi + match * (Y - mu_obs) / p_obs - c * pi))}
    return est


def run_rep(setting: str, rep: int):
    X, T, Y, mu0, mu1, binary = LOADERS[setting](rep)
    strat = T * 2 + (Y if binary else 0)
    idx_tr, idx_te = train_test_split(np.arange(len(T)), test_size=0.3, random_state=SEED + rep, stratify=strat)
    c = COST[setting] if COST[setting] is not None else float(np.median((mu1 - mu0)[idx_tr]))
    taus, s_model = fit_learners(X[idx_tr], T[idx_tr], Y[idx_tr], binary)
    Xte, Tte, Yte = X[idx_te], T[idx_te], Y[idx_te]
    m0, m1 = mu0[idx_te], mu1[idx_te]
    tau = m1 - m0
    oracle = (tau > c).astype(int)
    v_none, v_star = true_value(np.zeros(len(tau), int), m0, m1, c), true_value(oracle, m0, m1, c)
    rows, policies, tau_hat = [], {}, {}
    for L, f in taus.items():
        th = f(Xte)
        tau_hat[L] = th
        pi = (th > c).astype(int)
        policies[L] = pi
    policies["Treat all"] = np.ones(len(tau), int)
    policies["Treat none"] = np.zeros(len(tau), int)
    for k, pi in policies.items():
        v = true_value(pi, m0, m1, c)
        th = tau_hat.get(k)
        rows.append({"setting": setting, "rep": rep, "policy": k,
                     "PEHE": float(np.sqrt(np.mean((th - tau) ** 2))) if th is not None else np.nan,
                     "ATE_err": float(abs(th.mean() - tau.mean())) if th is not None else np.nan,
                     "value": v, "regret": v_star - v, "gain": (v - v_none) / (v_star - v_none),
                     "treat_rate": float(pi.mean())})
    est = ope(policies, Xte, Tte, Yte, binary, c)
    sel = []
    true_v = {r["policy"]: r["value"] for r in rows}
    for crit in ("DM", "IPW", "DR"):
        e_v = {k: est[k][crit] for k in policies}
        pick = max(e_v, key=e_v.get)
        best = max(true_v, key=true_v.get)
        sel.append({"setting": setting, "rep": rep, "criterion": crit,
                    "rho": float(spearmanr([e_v[k] for k in policies], [true_v[k] for k in policies])[0]),
                    "mae": float(np.mean([abs(e_v[k] - true_v[k]) for k in policies])),
                    "selected": pick, "sel_regret": v_star - true_v[pick],
                    "picked_best": float(true_v[pick] >= true_v[best] - 1e-12),
                    "oracle_v": v_star})
    fact = predict(s_model, np.column_stack([Xte, Tte]), binary)
    meta = {"setting": setting, "rep": rep, "n": int(len(T)), "n_test": int(len(idx_te)),
            "true_ate": float(tau.mean()), "naive_ate": float(tau_hat["Naive"][0]), "cost": c,
            "oracle_treat_rate": float(oracle.mean()), "v_none": v_none, "v_all": true_value(np.ones(len(tau), int), m0, m1, c),
            "v_oracle": v_star, "treated_share": float(T.mean()),
            "factual_auc" if binary else "factual_r2": float(roc_auc_score(Yte, fact) if binary else r2_score(Yte, fact))}
    # cost sensitivity and OPE scatter material (only what the figures need)
    curves = {}
    if setting == "Venture":
        grid = np.round(np.linspace(0, 0.10, 21), 4)
        for k in ["S-learner", "T-learner", "X-learner", "DR-learner", "R-learner"]:
            curves[k] = [true_value((tau_hat[k] > g).astype(int), m0, m1, g) - true_value(np.zeros(len(tau), int), m0, m1, g) for g in grid]
        curves["Treat all"] = [true_value(np.ones(len(tau), int), m0, m1, g) - true_value(np.zeros(len(tau), int), m0, m1, g) for g in grid]
        curves["Oracle"] = [true_value((tau > g).astype(int), m0, m1, g) - true_value(np.zeros(len(tau), int), m0, m1, g) for g in grid]
        curves["grid"] = grid.tolist()
    scatter = [{"setting": setting, "rep": rep, "policy": k, "true": true_v[k], **est[k]} for k in policies]
    sample = None
    if rep == 0:
        sub = np.random.default_rng(SEED).choice(len(tau), min(1500, len(tau)), replace=False)
        sample = {"tau": tau[sub].tolist(), "dr": tau_hat["DR-learner"][sub].tolist()}
    return rows, sel, meta, curves, scatter, sample


# ============================================================================ figures
def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e",
        "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
        "grid.color": "#e6e5e0", "grid.linewidth": 0.5, "legend.frameon": False, "legend.fontsize": 7,
        "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def figure1(res: pd.DataFrame, sdsd: dict):
    """(a) relative PEHE and (b) share of oracle gain captured, per learner and setting."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    learners = LEARNERS
    off = {"Synthetic": -0.22, "IHDP": 0.0, "Venture": 0.22}
    for s in SETTINGS:
        d = res[(res.setting == s) & res.policy.isin(learners)].copy()
        d["rel_pehe"] = d.PEHE / d.rep.map(sdsd[s])
        g = d.groupby("policy").agg(pm=("rel_pehe", "mean"), ps=("rel_pehe", "std"),
                                    gm=("gain", "mean"), gs=("gain", "std")).reindex(learners)
        y = np.arange(len(learners)) + off[s]
        a.errorbar(g.pm, y, xerr=g.ps, fmt="o", ms=4.5, color=PAL[s], ecolor=PAL[s], elinewidth=1, capsize=2,
                   label=s, mec="white", mew=0.6)
        b.errorbar(100 * g.gm, y, xerr=100 * g.gs, fmt="o", ms=4.5, color=PAL[s], ecolor=PAL[s], elinewidth=1,
                   capsize=2, label=s, mec="white", mew=0.6)
    for ax in (a, b):
        ax.set_yticks(range(len(learners)), learners)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    a.set(xlabel="PEHE / SD of true effect (lower is better)", title="(a) Effect-estimation error")
    a.set_xlim(left=0)
    b.axvline(100, color="#52514e", lw=0.8, ls=(0, (3, 2)))
    b.axvline(0, color="#a3a29b", lw=0.6)
    b.set(xlabel="Share of oracle improvement captured (%)", title="(b) Decision quality at the base cost")
    b.set_yticklabels([])
    h, l = b.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06), frameon=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig1_estimation_vs_decision")


def figure2(curves: list, scatter: pd.DataFrame):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    grid = np.array(curves[0]["grid"])
    styles = {"Oracle": ((0, (4, 2)), 1.4), "X-learner": ("-", 2.0), "DR-learner": ("-", 1.2),
              "S-learner": ("-.", 1.2), "T-learner": (":", 1.4), "Treat all": ("--", 1.2)}
    for k in ["Oracle", "X-learner", "S-learner", "T-learner", "DR-learner", "Treat all"]:
        v = 1000 * np.mean([c[k] for c in curves], axis=0)
        a.plot(100 * grid, v, color=LPAL[k], ls=styles[k][0], lw=styles[k][1], label=k)
    a.axhline(0, color="#a3a29b", lw=0.6)
    a.axvline(100 * COST["Venture"], color="#c3c2b7", lw=0.6)
    a.set(xlabel="Intervention cost c (percentage points of exit probability)",
          ylabel="Net gain over treat-none (exits per 1,000 ventures)",
          title="(a) Venture: value of targeting vs. cost")
    a.legend(loc="upper right", ncol=2)
    d = scatter[scatter.setting == "Venture"]
    lim = [1000 * min(d[["true", "DM", "DR"]].min()), 1000 * max(d[["true", "DM", "DR"]].max())]
    b.plot(lim, lim, color="#a3a29b", lw=0.8, ls=(0, (2, 2)))
    b.scatter(1000 * d.true, 1000 * d.DM, s=12, color="#eb6834", alpha=0.75, label="Plug-in (DM) estimate",
              edgecolors="white", linewidths=0.3, marker="s")
    b.scatter(1000 * d.true, 1000 * d.DR, s=14, color="#2a78d6", alpha=0.85, label="Doubly robust estimate",
              edgecolors="white", linewidths=0.3)
    b.set(xlabel="True policy value (per 1,000 ventures)", ylabel="Estimated value without ground truth",
          title="(b) Venture: off-policy value estimates")
    b.legend(loc="upper left")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_cost_and_selection")


def graphical_abstract(summary: dict):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    boxes = [(0.2, "Observational\ndata", "Synthetic, IHDP and\n19,814 real ventures\n(X, T, Y)"),
             (2.7, "Outcome\nprediction", "What is likely to\nhappen: P(Y | X, T)"),
             (5.2, "Causal effect\nlearning", "S-, T-, X-, DR- and\nR-learners estimate\ntau(x) = E[Y(1)-Y(0) | x]"),
             (7.7, "Decision\nintelligence", "Treat if tau(x) > cost;\nselect the policy by its\noff-policy value estimate")]
    for x, head, body in boxes:
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, 1.5), 2.1, 2.2, boxstyle="round,pad=0.02,rounding_size=0.12",
                                                       fc="#f3f7fd", ec="#2a78d6", lw=1.4))
        ax.text(x + 1.05, 3.2, head, ha="center", va="center", fontsize=11, weight="bold", color="#0d366b")
        ax.text(x + 1.05, 2.15, body, ha="center", va="center", fontsize=9.5, color="#0b0b0b", linespacing=1.45)
    for x in (2.33, 4.83, 7.33):
        ax.annotate("", xy=(x + 0.34, 2.6), xytext=(x, 2.6), arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=1.5))
    ax.text(5.0, 0.85, summary["ga_line"], ha="center", va="center", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.5", fc="#ffffff", ec="#c3c2b7"))
    ax.text(5.0, 0.3, "From 'what is likely to happen' to 'which intervention is expected to help'",
            ha="center", va="center", fontsize=9, style="italic", color="#52514e")
    fig.savefig(OUT / "graphical_abstract.png", dpi=200)
    fig.savefig(OUT / "graphical_abstract.pdf")
    plt.close(fig)


# ============================================================================ main
def main(quick: bool):
    t_start = time.time()
    OUT.mkdir(exist_ok=True)
    style()
    reps = 3 if quick else 10
    rows, sels, metas, curves, scat, samples = [], [], [], [], [], {}
    for s in SETTINGS:
        print(f"[{s}] {reps} replications")
        for r in range(reps):
            ro, se, me, cu, sc, sa = run_rep(s, r)
            rows += ro; sels += se; metas.append(me); scat += sc
            if cu: curves.append(cu)
            if sa: samples[s] = sa
    res, sel, meta, scat = pd.DataFrame(rows), pd.DataFrame(sels), pd.DataFrame(metas), pd.DataFrame(scat)
    res.to_csv(OUT / "replications_policies.csv", index=False)
    sel.to_csv(OUT / "replications_selection.csv", index=False)
    meta.to_csv(OUT / "replications_meta.csv", index=False)

    # SD of the true effect per replication (to normalise PEHE across settings)
    sdsd = {}
    for s in SETTINGS:
        sdsd[s] = {}
        for r in range(reps):
            X, T, Y, mu0, mu1, binary = LOADERS[s](r)
            _, te = train_test_split(np.arange(len(T)), test_size=0.3, random_state=SEED + r,
                                     stratify=T * 2 + (Y if binary else 0))
            sdsd[s][r] = float(np.std(mu1[te] - mu0[te]))

    # ---- Table 1: causal and decision results (mean +- SD over replications)
    t1 = []
    res["nPEHE"] = res.PEHE / res.apply(lambda r: sdsd[r.setting][r.rep], axis=1)
    res["nATE"] = res.ATE_err / res.apply(lambda r: sdsd[r.setting][r.rep], axis=1)
    for s in SETTINGS:
        d = res[res.setting == s]
        for p in POLICIES:
            q = d[d.policy == p]
            f = lambda c, k=1: (f"{q[c].mean() * k:.3f} ± {q[c].std() * k:.3f}" if q[c].notna().any() else "–")
            t1.append({"Setting": s, "Policy": p, "PEHE / SD(tau)": f("nPEHE"), "|ATE error| / SD(tau)": f("nATE"),
                       "PEHE (raw)": f("PEHE"), "Policy value": f("value"), "Regret": f("regret"),
                       "Oracle gain captured (%)": f"{100 * q.gain.mean():.1f} ± {100 * q.gain.std():.1f}",
                       "Treat rate (%)": f"{100 * q.treat_rate.mean():.1f}"})
        m = meta[meta.setting == s]
        t1.append({"Setting": s, "Policy": "Oracle", "PEHE / SD(tau)": "0", "|ATE error| / SD(tau)": "0", "PEHE (raw)": "0",
                   "Policy value": f"{m.v_oracle.mean():.3f} ± {m.v_oracle.std():.3f}", "Regret": "0",
                   "Oracle gain captured (%)": "100.0", "Treat rate (%)": f"{100 * m.oracle_treat_rate.mean():.1f}"})
    pd.DataFrame(t1).to_csv(OUT / "table1_causal_decision.csv", index=False)
    res.to_csv(OUT / "replications_policies.csv", index=False)

    # ---- Table 2: ground-truth-free policy selection
    t2 = []
    gap = meta.set_index(["setting", "rep"]).eval("v_oracle - v_none")
    sel["sel_regret_pct"] = 100 * sel.sel_regret / [gap[(a, r)] for a, r in zip(sel.setting, sel.rep)]
    lp = res[res.policy.isin(LEARNERS[1:])].groupby(["setting", "rep"]).regret.mean()
    avg_learner = {s_: float(np.mean([100 * lp[(s_, r)] / gap[(s_, r)] for r in range(reps)])) for s_ in SETTINGS}
    for s in SETTINGS:
        for crit in ("DM", "IPW", "DR"):
            q = sel[(sel.setting == s) & (sel.criterion == crit)]
            t2.append({"Setting": s, "Criterion": crit,
                       "Spearman rho (est. vs true value)": f"{q.rho.mean():.3f} ± {q.rho.std():.3f}",
                       "Mean abs. value error": f"{q.mae.mean():.4f}",
                       "Selection regret (% of oracle gain)": f"{q.sel_regret_pct.mean():.1f} ± {q.sel_regret_pct.std():.1f}",
                       "Best policy selected (%)": f"{100 * q.picked_best.mean():.0f}",
                       "Most frequent choice": q.selected.mode().iloc[0]})
        t2.append({"Setting": s, "Criterion": "Reference: average causal learner",
                   "Spearman rho (est. vs true value)": "–", "Mean abs. value error": "–",
                   "Selection regret (% of oracle gain)": f"{avg_learner[s]:.1f}",
                   "Best policy selected (%)": "–", "Most frequent choice": "–"})
    pd.DataFrame(t2).to_csv(OUT / "table2_policy_selection.csv", index=False)

    figure1(res, sdsd)
    figure2(curves, scat)

    agg = lambda s, p, c: float(res[(res.setting == s) & (res.policy == p)][c].mean())
    summary = {
        "replications": reps, "costs": COST, "runtime_s": None,
        "meta": meta.groupby("setting").mean(numeric_only=True).drop(columns="rep").to_dict(orient="index"),
        "meta_sd": meta.groupby("setting").std(numeric_only=True).drop(columns="rep").to_dict(orient="index"),
        "policies": {f"{k[0]}|{k[1]}": {c: {"mean": float(g[c].mean()), "sd": float(g[c].std())}
                                         for c in ["PEHE", "ATE_err", "nPEHE", "nATE", "value", "regret", "gain", "treat_rate"]}
                     for k, g in res.groupby(["setting", "policy"])},
        "selection": sel.groupby(["setting", "criterion"]).agg(
            rho=("rho", "mean"), rho_sd=("rho", "std"), mae=("mae", "mean"), regret=("sel_regret", "mean"),
            regret_sd=("sel_regret", "std"), regret_pct=("sel_regret_pct", "mean"),
            regret_pct_sd=("sel_regret_pct", "std"), best=("picked_best", "mean")).reset_index().to_dict(orient="records"),
        "avg_learner_regret_pct": avg_learner,
        "selected_counts": sel.groupby(["setting", "criterion"]).selected.value_counts().rename("n").reset_index()
                              .to_dict(orient="records"),
        "venture_real_exit_auc": float(_VENTURE.get("auc_real", np.nan)),
        "venture_n": int(len(_VENTURE["X"])) if "X" in _VENTURE else None,
        "venture_real_exit_rate": float(_VENTURE["X"].real_exit.mean()) if "X" in _VENTURE else None,
        "dr_cate_corr": {s: float(np.corrcoef(v["tau"], v["dr"])[0, 1]) for s, v in samples.items()},
        "venture_curve_at_2x_cost": {k: float(np.mean([c[k][list(np.round(c["grid"], 4)).index(0.06)] for c in curves]))
                                     for k in ["Oracle", "X-learner", "DR-learner", "S-learner", "T-learner", "R-learner", "Treat all"]},
    }
    best_dr = max(LEARNERS[1:], key=lambda p: agg("Venture", p, "gain"))
    summary["ga_line"] = (f"Venture data: naive ATE {meta[meta.setting == 'Venture'].naive_ate.mean():.3f} vs true "
                          f"{meta[meta.setting == 'Venture'].true_ate.mean():.3f}  |  {best_dr} captures "
                          f"{100 * agg('Venture', best_dr, 'gain'):.0f}% of oracle gain vs "
                          f"{100 * agg('Venture', 'Treat all', 'gain'):.0f}% for treat-all")
    graphical_abstract(summary)
    summary["runtime_s"] = round(time.time() - t_start, 1)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    print(pd.DataFrame(t1).to_string(index=False))
    print(pd.DataFrame(t2).to_string(index=False))
    print(f"done in {summary['runtime_s']} s -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Causal AI decision intelligence experiments")
    ap.add_argument("--quick", action="store_true", help="3 replications per setting")
    main(ap.parse_args().quick)
