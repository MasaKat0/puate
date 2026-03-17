
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde


METHOD_LABELS = {
    "ipw": "IPW",
    "dm": "DM",
    "efficient": "Efficient",
    "ipw_true_g": "IPW (true $g_0$)",
    "dm_true_g": "DM (true $g_0$)",
    "efficient_true_g": "Efficient (true $g_0$)",
    "ipw_true_e": "IPW (true $e_0$)",
    "dm_true_e": "DM (true $e_0$)",
    "efficient_true_e": "Efficient (true $e_0$)",
}

LINE_STYLES = ['--', '-.', '-', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]


def summarize_trials(
    estimates: pd.DataFrame,
    variances: pd.DataFrame,
    *,
    true_ate: float,
    ci_sample_size: int,
    method_labels: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Compute MSE, bias, and coverage ratio for each method."""
    method_labels = METHOD_LABELS if method_labels is None else dict(method_labels)
    rows = {}
    mse = ((estimates - true_ate) ** 2).mean(axis=0)
    bias = (estimates - true_ate).mean(axis=0)

    coverage = {}
    for col in estimates.columns:
        lower = estimates[col] - 1.96 * np.sqrt(variances[col] / ci_sample_size)
        upper = estimates[col] + 1.96 * np.sqrt(variances[col] / ci_sample_size)
        coverage[col] = ((lower <= true_ate) & (true_ate <= upper)).mean()

    summary = pd.DataFrame([mse, bias, pd.Series(coverage)], index=["MSE", "Bias", "Cov. ratio"])
    summary = summary.rename(columns=method_labels)
    return summary


def combine_summaries(
    censoring_summary: pd.DataFrame,
    case_control_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Combine left/right paper tables into one dataframe with grouped columns."""
    return pd.concat(
        {"Censoring": censoring_summary, "Case-control": case_control_summary},
        axis=1,
    )


def plot_density_estimates(
    estimates: pd.DataFrame,
    *,
    true_ate: float,
    ax,
    title: str,
    method_labels: Mapping[str, str] | None = None,
    x_range: np.ndarray | None = None,
):
    method_labels = METHOD_LABELS if method_labels is None else dict(method_labels)
    columns = list(estimates.columns)
    if x_range is None:
        low = float(np.quantile(estimates.values, 0.01))
        high = float(np.quantile(estimates.values, 0.99))
        pad = 0.1 * max(high - low, 1e-6)
        x_range = np.linspace(low - pad, high + pad, 500)

    for idx, col in enumerate(columns):
        values = estimates[col].to_numpy()
        label = method_labels.get(col, col)
        linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
        if len(values) < 2 or np.allclose(values, values[0]):
            ax.axvline(float(values.mean()), label=label, linestyle=linestyle, linewidth=2)
            continue
        kde = gaussian_kde(values)
        density = kde(x_range)
        ax.plot(
            x_range,
            density,
            label=label,
            linestyle=linestyle,
            linewidth=2,
        )

    ax.axvline(true_ate, color="red", linestyle="--", label="True ATE")
    ax.set_title(title)
    ax.set_xlabel("ATE")
    ax.set_ylabel("Empirical density")
    ax.grid(True)


def plot_two_setting_density(
    censoring_estimates: pd.DataFrame,
    case_control_estimates: pd.DataFrame,
    *,
    true_ate: float,
    figure_title: str | None = None,
    method_labels: Mapping[str, str] | None = None,
    x_range: np.ndarray | None = None,
):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=False)

    plot_density_estimates(
        censoring_estimates,
        true_ate=true_ate,
        ax=axes[0],
        title="Estimates of the ATE (censoring setting)",
        method_labels=method_labels,
        x_range=x_range,
    )
    plot_density_estimates(
        case_control_estimates,
        true_ate=true_ate,
        ax=axes[1],
        title="Estimates of the ATE (case-control setting)",
        method_labels=method_labels,
        x_range=x_range,
    )

    axes[0].legend()
    axes[1].legend()
    if figure_title:
        fig.suptitle(figure_title, y=0.995)
    fig.tight_layout()
    return fig
