"""
Payroll-to-wins efficiency frontier for MLB teams, 1985-2023.

Two standard frontier estimators, written from scratch so every step is visible:

  * Stochastic frontier analysis (normal / half-normal, Aigner-Lovell-Schmidt 1977)
        ln(win%) = a + b ln(relative payroll) + v - u,   v ~ N(0, sv^2),  u ~ |N(0, su^2)|
    Team-season efficiency is the Battese-Coelli estimator E[exp(-u) | v - u].
  * Data envelopment analysis (output-oriented, variable returns to scale), run
    season by season with one input (relative payroll) and one output (win %).

A frontier fitted to noisy outcomes will always find "inefficiency", because luck
looks like inefficiency. So the key step is a parametric null: simulate seasons in
which every team is equally efficient and wins are pure binomial luck around the
payroll-implied win probability, then compare observed dispersion and persistence
of efficiency to that null.

Run:  python python/frontier.py            (full run, about 2 minutes)
      python python/frontier.py --fast     (fewer simulations)
"""
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog, minimize
from statsmodels.tools.numdiff import approx_hess

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"


# --------------------------------------------------------------------------- SFA
def _negll(theta, y, X):
    k = X.shape[1]
    b, sv, su = theta[:k], np.exp(theta[k]), np.exp(theta[k + 1])
    e = y - X @ b
    s2 = sv**2 + su**2
    s = np.sqrt(s2)
    lam = su / sv
    ll = 0.5 * np.log(2 / np.pi) - np.log(s) + stats.norm.logcdf(-e * lam / s) - e**2 / (2 * s2)
    return -ll.sum()


def sfa_halfnormal(y, X):
    """Production frontier, y = X b + v - u. Returns dict with estimates and SEs."""
    k = X.shape[1]
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    e0 = y - X @ b0
    # method-of-moments start: shift the intercept up by the expected size of u
    sd0 = e0.std()
    start = np.r_[b0, np.log(sd0 * 0.7), np.log(sd0 * 0.7)]
    start[0] += np.sqrt(2 / np.pi) * sd0 * 0.7
    best = None
    for s0 in (start, start + np.r_[np.zeros(k), -0.5, 0.5], start + np.r_[np.zeros(k), 0.5, -0.5]):
        r = minimize(_negll, s0, args=(y, X), method="BFGS")
        if best is None or r.fun < best.fun:
            best = r
    th = best.x
    H = approx_hess(th, _negll, args=(y, X))
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.full_like(H, np.nan)
    sv, su = np.exp(th[k]), np.exp(th[k + 1])
    # delta method for sigma_v, sigma_u standard errors
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    return {"beta": th[:k], "beta_se": se[:k], "sigma_v": sv, "sigma_u": su,
            "sigma_v_se": sv * se[k], "sigma_u_se": su * se[k + 1],
            "loglik": -best.fun, "converged": best.success}


def sfa_efficiency(y, X, fit):
    """Battese-Coelli E[exp(-u)|e] and Jondrow et al. E[u|e]."""
    e = y - X @ fit["beta"]
    sv, su = fit["sigma_v"], fit["sigma_u"]
    s2 = sv**2 + su**2
    mu = -e * su**2 / s2
    sig = su * sv / np.sqrt(s2)
    a = mu / sig
    te = np.exp(-mu + 0.5 * sig**2) * stats.norm.cdf(a - sig) / stats.norm.cdf(a)
    eu = sig * (stats.norm.pdf(a) / stats.norm.cdf(a) + a)
    return te, eu


# --------------------------------------------------------------------------- DEA
def dea_vrs_output_lp(x, y):
    """Output-oriented VRS DEA by linear programming. Returns efficiency in (0, 1]."""
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        # variables: [phi, lambda_1..lambda_n]; maximise phi
        c = np.r_[-1.0, np.zeros(n)]
        A_ub = np.vstack([np.r_[y[i], -y],        # phi*y_i - sum(l*y) <= 0
                          np.r_[0.0, x]])         # sum(l*x) <= x_i
        b_ub = np.r_[0.0, x[i]]
        A_eq = np.r_[0.0, np.ones(n)][None, :]
        r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                    bounds=[(1, None)] + [(0, None)] * n, method="highs")
        out[i] = 1.0 / r.x[0]
    return out


def dea_vrs_output_hull(x, y):
    """Same estimator in closed form for one input and one output: the
    free-disposal concave envelope. Used inside the simulations for speed and
    checked against the LP version on the real data."""
    order = np.lexsort((-y, x))
    xs, ys = x[order], y[order]
    # running maximum gives free disposal in the input
    hull = []
    for px, py in zip(xs, ys):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (px - x1) <= (py - y1) * (x2 - x1):
                hull.pop()          # middle point lies under the chord: not on the concave hull
            else:
                break
        hull.append((px, py))
    hx, hy = np.array([h[0] for h in hull]), np.array([h[1] for h in hull])
    top = hy.argmax()
    hx, hy = hx[: top + 1], hy[: top + 1]
    front = np.interp(x, hx, hy, right=hy[-1])
    front = np.maximum(front, y)
    return y / front


def dea_by_year(df, method=dea_vrs_output_hull):
    eff = pd.Series(index=df.index, dtype=float)
    for _, g in df.groupby("year"):
        eff.loc[g.index] = method(g.rel.to_numpy(), g.winpct.to_numpy())
    return eff


# ---------------------------------------------------------------- persistence
def persistence_stats(df, col):
    """Lag-1 autocorrelation within team, and share of variance that is
    between teams (one-way ANOVA ICC)."""
    d = df.sort_values(["team", "year"]).copy()
    d["lag"] = d.groupby("team")[col].shift(1)
    r1 = d[[col, "lag"]].dropna().corr().iloc[0, 1]
    acf = {}
    for L in range(1, 9):
        d[f"lag{L}"] = d.groupby("team")[col].shift(L)
        acf[f"acf{L}"] = d[[col, f"lag{L}"]].dropna().corr().iloc[0, 1]
    r5 = acf["acf5"]
    g = d.groupby("team")[col]
    n_i = g.size()
    k = len(n_i)
    n0 = (n_i.sum() - (n_i**2).sum() / n_i.sum()) / (k - 1)
    msb = (n_i * (g.mean() - d[col].mean()) ** 2).sum() / (k - 1)
    msw = ((d[col] - g.transform("mean")) ** 2).sum() / (n_i.sum() - k)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    return {"lag1_autocorr": r1, "lag5_autocorr": r5, "icc_team": icc,
            "sd": d[col].std(), "mean": d[col].mean(), **acf}


# ------------------------------------------------------------------- main
def main(n_sims):
    d = pd.read_csv(ROOT / "data" / "team_panel_1985_2023.csv")
    d["rel"] = d.payroll / d.groupby("year").payroll.transform("mean")
    d = d.sort_values(["team", "year"]).reset_index(drop=True)
    y = np.log(d.winpct.to_numpy())
    X = np.column_stack([np.ones(len(d)), np.log(d.rel.to_numpy())])

    fit = sfa_halfnormal(y, X)
    d["sfa_te"], d["sfa_u"] = sfa_efficiency(y, X, fit)
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ ols
    summ = pd.DataFrame({
        "term": ["const", "ln_rel_payroll", "sigma_v", "sigma_u", "loglik", "ols_resid_skew"],
        "estimate": [*fit["beta"], fit["sigma_v"], fit["sigma_u"], fit["loglik"], stats.skew(resid)],
        "se": [*fit["beta_se"], fit["sigma_v_se"], fit["sigma_u_se"], np.nan, np.nan]})
    summ["lambda_su_over_sv"] = fit["sigma_u"] / fit["sigma_v"]
    summ.to_csv(RES / "sfa_estimates.csv", index=False)
    print(summ.round(4).to_string(index=False))

    # DEA: LP and closed-form agree?
    d["dea_te"] = dea_by_year(d, dea_vrs_output_hull)
    chk = dea_by_year(d[d.year.isin([1990, 2005, 2019])], dea_vrs_output_lp)
    assert np.allclose(chk, d.loc[chk.index, "dea_te"], atol=1e-8), "DEA LP and hull disagree"
    print("DEA LP vs hull check passed")

    # --- parametric null: equal efficiency, binomial luck ---
    # true win probability = smooth function of relative payroll (OLS fit, no team effects),
    # re-centred each season so the league averages 0.5.
    p_hat = np.exp(X @ ols)
    obs = {c: persistence_stats(d, c) for c in ("sfa_te", "dea_te")}
    games = d.games.to_numpy().astype(int)
    rng = np.random.default_rng(488)
    sims = []
    for s in range(n_sims):
        p = p_hat / pd.Series(p_hat).groupby(d.year).transform("mean").to_numpy() * 0.5
        p = np.clip(p, 0.05, 0.95)
        w = rng.binomial(games, p)
        ds = d[["team", "year", "rel", "games"]].copy()
        ds["winpct"] = w / games
        ys = np.log(np.clip(ds.winpct.to_numpy(), 1e-3, None))
        f = sfa_halfnormal(ys, X)
        ds["sfa_te"], _ = sfa_efficiency(ys, X, f)
        ds["dea_te"] = dea_by_year(ds, dea_vrs_output_hull)
        row = {"sim": s, "sigma_u": f["sigma_u"], "sigma_v": f["sigma_v"]}
        for c in ("sfa_te", "dea_te"):
            ps = persistence_stats(ds, c)
            row.update({f"{c}_{k}": v for k, v in ps.items()})
        sims.append(row)
    S = pd.DataFrame(sims)
    S.to_csv(RES / "null_simulations.csv", index=False)

    rows = []
    for c in ("sfa_te", "dea_te"):
        for k in ("mean", "sd", "icc_team", *[f"acf{L}" for L in range(1, 9)]):
            sim_vals = S[f"{c}_{k}"]
            o = obs[c][k]
            rows.append({"measure": c, "statistic": k, "observed": o,
                         "null_mean": sim_vals.mean(), "null_p5": sim_vals.quantile(.05),
                         "null_p95": sim_vals.quantile(.95),
                         "share_of_sims_ge_observed": (sim_vals >= o).mean()})
    T = pd.DataFrame(rows)
    T.to_csv(RES / "observed_vs_null.csv", index=False)
    sig_u_null = S.sigma_u
    print("\nObserved vs equal-efficiency null:\n", T.round(4).to_string(index=False))
    print(f"\nsigma_u observed {fit['sigma_u']:.4f}; null mean {sig_u_null.mean():.4f} "
          f"(95th pct {sig_u_null.quantile(.95):.4f})")

    # --- team table with luck band ---
    team = d.groupby("team").agg(seasons=("year", "size"), mean_sfa_te=("sfa_te", "mean"),
                                 mean_dea_te=("dea_te", "mean"),
                                 mean_log_resid=("sfa_u", lambda s: np.nan)).drop(columns="mean_log_resid")
    d["log_resid"] = y - X @ ols
    team["mean_resid"] = d.groupby("team").log_resid.mean()
    # luck band: within-team residual noise implied by the null residual sd
    # binomial luck on the log scale is about 1/sqrt(games) per season
    d["luck_var"] = 1.0 / d.games
    team["luck_se"] = np.sqrt(d.groupby("team").luck_var.sum()) / team.seasons
    # luck-only SEs ignore the 2-3 year persistence found above, so also report a
    # heteroskedasticity-and-autocorrelation-robust SE of each team's mean residual
    import statsmodels.api as sm
    team["hac_se"] = pd.Series({t: sm.OLS(g.log_resid.to_numpy(), np.ones(len(g))).fit(
        cov_type="HAC", cov_kwds={"maxlags": 4}).bse[0] for t, g in d.groupby("team")})
    team["z_vs_luck"] = team.mean_resid / team.luck_se
    team["z_hac"] = team.mean_resid / team.hac_se
    team = team.sort_values("mean_resid", ascending=False)
    team.round(4).to_csv(RES / "team_efficiency_table.csv")
    print("\nTop and bottom teams by mean residual (log wins above payroll-implied):")
    print(team.head(5).round(3).to_string()); print(team.tail(5).round(3).to_string())

    # --- by era ---
    era = pd.cut(d.year, [1984, 2002, 2011, 2016, 2023], labels=["1985-02", "2003-11", "2012-16", "2017-23"])
    E = d.groupby(era).agg(mean_sfa_te=("sfa_te", "mean"), sd_sfa_te=("sfa_te", "std"),
                           mean_dea_te=("dea_te", "mean"), sd_log_resid=("log_resid", "std"))
    d["short"] = d.year.isin([1994, 1995, 2020])
    E["sd_log_resid_ex_short_seasons"] = d[~d.short].groupby(era[~d.short]).log_resid.std()
    E.round(4).to_csv(RES / "efficiency_by_era.csv")
    yr = d.groupby("year").agg(sd_resid=("log_resid", "std"), games=("games", "mean"))
    yr["luck_sd"] = np.sqrt(0.25 / yr.games) / 0.5
    yr.round(4).to_csv(RES / "residual_dispersion_by_year.csv")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(yr.index, yr.sd_resid, color="#1f4e79", label="sd of wins unexplained by payroll")
    ax.plot(yr.index, yr.luck_sd, color="#c00000", ls="--", label="binomial luck alone")
    ax.set_xlabel("season"); ax.set_ylabel("sd of log win%"); ax.legend()
    ax.set_title("Unexplained variation in wins, by season")
    fig.tight_layout(); fig.savefig(FIG / "residual_dispersion_by_year.png", dpi=160)
    print("\nBy era:\n", E.round(4))
    d[["team", "year", "winpct", "rel", "sfa_te", "dea_te", "log_resid"]].to_csv(
        RES / "team_season_efficiency.csv", index=False)

    # --- figures ---
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].scatter(d.rel, d.winpct, s=6, alpha=.35, color="#7f7f7f")
    gx = np.linspace(d.rel.min(), d.rel.max(), 200)
    ax[0].plot(gx, np.exp(ols[0] + ols[1] * np.log(gx)), color="#1f4e79", label="average (OLS)")
    ax[0].plot(gx, np.exp(fit["beta"][0] + fit["beta"][1] * np.log(gx)), color="#c00000", label="SFA frontier")
    ax[0].set_xlabel("payroll / league average"); ax[0].set_ylabel("win %"); ax[0].legend()
    ax[0].set_title("Frontier vs average relationship")
    ax[1].hist(d.sfa_te, bins=30, color="#1f4e79", alpha=.85)
    ax[1].set_title("SFA efficiency, team-seasons"); ax[1].set_xlabel("efficiency (1 = on frontier)")
    for c, col in (("sfa_te", "#c00000"), ("dea_te", "#1f4e79")):
        ax[2].hist(S[f"{c}_lag1_autocorr"], bins=20, alpha=.5, color=col, label=f"null {c}")
        ax[2].axvline(obs[c]["lag1_autocorr"], color=col, lw=2)
    ax[2].set_title("Persistence: observed (line) vs luck-only null")
    ax[2].set_xlabel("lag-1 autocorrelation of efficiency"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "frontier_overview.png", dpi=160)

    fig, ax = plt.subplots(figsize=(7, 8))
    t = team.sort_values("mean_resid")
    ax.barh(t.index, t.mean_resid, xerr=1.96 * t.hac_se, color="#1f4e79", ecolor="#999999")
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("average log win% above payroll-implied (bars: 95% HAC interval)")
    ax.set_title("Which teams beat their payroll, persistently?")
    fig.tight_layout(); fig.savefig(FIG / "team_residuals.png", dpi=160)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    main(40 if a.fast else 300)
