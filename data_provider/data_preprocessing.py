import os
import joblib
import pandas as pd
from typing import List, Union
from sklearn.preprocessing import StandardScaler


def preprocess_multi_ts(
    df_raw: pd.DataFrame,
    seq_len: int = 3,
    pred_len: int = 1,
    pk_cols: List[str] = ["CustomerId", "ProductId", "year-month"],
    ts_cols: List[str] = ["CustomerId", "ProductId"],
    target_col: str = "Volume",
    cat_cols: List[str] = ["Region", "TradeMark"],
    scale_features: bool = False,
    scaler_path: Union[str, None] = None,
    y_actual_nonzero: bool = False,
):
    """
    Preprocess the raw data for multi-time series forecasting.

    Args:
        df_raw (pd.DataFrame): Raw data.
        seq_len (int): Sequence length.
        pred_len (int): Prediction length.
        pk_cols (List[str]): Primary key columns.
        ts_cols (List[str]): Time series columns.
        target_col (str): Target column.
        cat_cols (List[str]): Categorical columns.
        scale_features (bool): Scale features.
        scaler_path (Union[str, None]): Scaler path.
        y_actual_nonzero (bool): Y actual non-zero.
    """

    # Sort the raw data
    df_raw = df_raw.sort_values(pk_cols, ascending=len(pk_cols) * [True])

    # One-hot encode the categorical columns
    if cat_cols:
        df = pd.get_dummies(df_raw, columns=cat_cols, drop_first=True)
    else:
        df = df_raw.copy()

    # Sort the columns
    feature_cols = [target_col] + sorted(list(set(df.columns) - set(pk_cols) - set([target_col])))
    df = df[pk_cols + feature_cols].copy()
    df[f"{target_col}_raw"] = df[target_col].copy()

    # Scale the features if required and save the scaler
    if scale_features:
        if (scaler_path is not None) and (os.path.exists(scaler_path)):
            scaler = joblib.load(scaler_path)
            df.loc[:, feature_cols] = scaler.transform(df.loc[:, feature_cols])
        else:
            scaler = StandardScaler()
            df.loc[:, feature_cols] = scaler.fit_transform(df.loc[:, feature_cols])

            joblib.dump(scaler, scaler_path) if scaler_path is not None else joblib.dump(scaler, "scaler.pkl")

    # Create the lag features
    seq_feature_cols = []
    pred_feature_cols = feature_cols.copy()
    for i in range(1, seq_len + pred_len):
        for col in feature_cols:
            df[f"{col}_lag_{i:02d}"] = df.groupby(ts_cols)[col].shift(i)
            if i < pred_len:
                pred_feature_cols.append(f"{col}_lag_{i:02d}")
            else:
                seq_feature_cols.append(f"{col}_lag_{i:02d}")

    # Drop the rows with missing values
    dfx = df.dropna().copy()

    # Filter the rows with non-zero target values if required
    if y_actual_nonzero:
        dfx = dfx[dfx[f"{target_col}_raw"] > 0].copy()
    dfx = dfx.drop(columns=[f"{target_col}_raw"]).copy()

    # Create the X and y arrays
    X = dfx[seq_feature_cols].to_numpy().reshape(-1, seq_len, len(feature_cols))[:, ::-1, :].copy()
    y = dfx[pred_feature_cols].to_numpy().reshape(-1, pred_len, len(feature_cols))[:, ::-1, :].copy()

    return X, y, dfx, seq_feature_cols, pred_feature_cols
