import numpy as np
from typing import List
from looloo.utils import dict_flatten

import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(
    df_actual,
    df_pred,
    pk_cols: List[str] = ["CustomerId", "ProductId", "year-month"],
    target_col: str = "Volume",
    high_volume_threshold: int = 20,
):
    dfx = df_actual[pk_cols + [target_col]].merge(
        df_pred[pk_cols + [target_col]], on=pk_cols, suffixes=("_actual", "_pred")
    )

    dfx["error"] = dfx[f"{target_col}_pred"] - dfx[f"{target_col}_actual"]
    dfx["abs_error"] = np.abs(dfx["error"])
    dfx["pct_error"] = dfx["error"] / dfx[f"{target_col}_actual"]
    dfx["pct_error"] = dfx["pct_error"].replace([np.inf, -np.inf], np.nan)
    dfx["abs_pct_error"] = np.abs(dfx["pct_error"])

    cond = dfx[f"{target_col}_actual"] >= high_volume_threshold
    mae_low_volume = dfx.loc[~cond, "abs_error"].mean()
    mape_high_volume = dfx.loc[cond, "abs_pct_error"].mean()

    metrics = dfx.describe().to_dict()
    metrics = dict_flatten(metrics)
    metrics.update({"mae_low_volume": mae_low_volume, "mape_high_volume": mape_high_volume})

    return metrics, dfx


def plot_hist(df, col_name, title=None, num_bins=100, figsize=(12, 8)):
    x = df[col_name]

    fig, ax = plt.subplots(figsize=figsize)

    sns.histplot(x, bins=num_bins, ax=ax)

    ax2 = ax.twinx()
    sns.kdeplot(x, ax=ax2)

    plt.xlabel(f"{col_name}")
    plt.grid()

    if title is None:
        title = f"Histrogram ({col_name})"

    plt.title(title, fontweight="bold")

    plt.tight_layout()
    plt.show()
