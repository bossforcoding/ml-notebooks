"""Small helpers for checking linear regression models fitted with statsmodels."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _smooth(ax, x, y, color="tab:red"):
    fitted = lowess(y, x, frac=2 / 3)
    ax.plot(fitted[:, 0], fitted[:, 1], color=color, lw=1.5)


def diagnostic_plots(model, title=None, simulate=20, seed=0):
    """Draw the four standard residual plots of a fitted OLS model.

    - Residuals vs fitted (Tukey-Anscombe): checks that the errors have mean zero.
    - Normal Q-Q plot: checks that the errors are normally distributed.
    - Scale-location: checks that the error variance is constant.
    - Residuals vs leverage: spots influential observations (Cook's distance).

    In the first and third plot the grey lines are smoothers fitted on residuals
    simulated from a correct model; if the red smoother stays within the grey
    band, the pattern can be explained by chance alone.
    """
    influence = model.get_influence()
    fitted = np.asarray(model.fittedvalues)
    resid = np.asarray(model.resid)
    std_resid = influence.resid_studentized_internal
    leverage = influence.hat_matrix_diag
    cooks = influence.cooks_distance[0]
    sigma = np.sqrt(model.scale)
    rng = np.random.default_rng(seed)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    for _ in range(simulate):
        _smooth(ax, fitted, rng.normal(0, sigma, len(fitted)), color="lightgrey")
    ax.scatter(fitted, resid, s=25, alpha=0.7)
    _smooth(ax, fitted, resid)
    ax.axhline(0, color="black", ls=":", lw=1)
    ax.set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs fitted")

    ax = axes[0, 1]
    sm.qqplot(std_resid, line="45", ax=ax, alpha=0.7)
    ax.set_title("Normal Q-Q")

    ax = axes[1, 0]
    for _ in range(simulate):
        sim = np.sqrt(np.abs(rng.normal(0, 1, len(fitted))))
        _smooth(ax, fitted, sim, color="lightgrey")
    ax.scatter(fitted, np.sqrt(np.abs(std_resid)), s=25, alpha=0.7)
    _smooth(ax, fitted, np.sqrt(np.abs(std_resid)))
    ax.set(xlabel="Fitted values", ylabel=r"$\sqrt{|\mathrm{standardized\ residuals}|}$",
           title="Scale-location")

    ax = axes[1, 1]
    ax.scatter(leverage, std_resid, s=25, alpha=0.7)
    p = len(model.params)
    lev = np.linspace(max(leverage.min(), 1e-3), leverage.max() * 1.05, 100)
    for d in (0.5, 1):
        bound = np.sqrt(d * p * (1 - lev) / lev)
        ax.plot(lev, bound, "--", color="tab:red", lw=1)
        ax.plot(lev, -bound, "--", color="tab:red", lw=1)
    ax.set_ylim(min(std_resid.min(), -3) * 1.1, max(std_resid.max(), 3) * 1.1)
    for i in np.argsort(cooks)[-3:]:
        ax.annotate(str(model.resid.index[i]), (leverage[i], std_resid[i]), fontsize=9)
    ax.axhline(0, color="black", ls=":", lw=1)
    ax.set(xlabel="Leverage", ylabel="Standardized residuals",
           title="Residuals vs leverage (Cook's distance 0.5 and 1)")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    plt.show()


def vif_table(X):
    """Variance inflation factor of every predictor in the design matrix X."""
    X = sm.add_constant(X)
    values = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
    return pd.Series(values, index=X.columns[1:], name="VIF").round(2)


def _criterion(data, response, terms, criterion):
    formula = f"{response} ~ {' + '.join(terms) if terms else '1'}"
    return getattr(smf.ols(formula, data=data).fit(), criterion)


def backward_selection(data, response, terms, criterion="aic", verbose=True):
    """Remove one term at a time while the information criterion improves.

    `terms` are formula terms, so a factor such as "C(region)" is kept or
    removed as a whole and transformations like "np.log(x)" are allowed.
    """
    selected = list(terms)
    current = _criterion(data, response, selected, criterion)
    while selected:
        trials = {t: _criterion(data, response, [s for s in selected if s != t], criterion) for t in selected}
        term, best = min(trials.items(), key=lambda kv: kv[1])
        if best >= current:
            break
        if verbose:
            print(f"drop {term:<24} {criterion.upper()} {current:.2f} -> {best:.2f}")
        selected.remove(term)
        current = best
    return selected


def forward_selection(data, response, terms, criterion="aic", verbose=True):
    """Add one term at a time, starting from the empty model, while the criterion improves."""
    selected, remaining = [], list(terms)
    current = _criterion(data, response, selected, criterion)
    while remaining:
        trials = {t: _criterion(data, response, selected + [t], criterion) for t in remaining}
        term, best = min(trials.items(), key=lambda kv: kv[1])
        if best >= current:
            break
        if verbose:
            print(f"add  {term:<24} {criterion.upper()} {current:.2f} -> {best:.2f}")
        selected.append(term)
        remaining.remove(term)
        current = best
    return selected
