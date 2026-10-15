"""Isolation Forest anomaly detection on daily sales, per SKU.

Features are computed from the sales series itself (level + rolling
statistics), so the model needs no labels at detection time. The injected
labels in sales.csv are used only by src/evaluate.py.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

CONTAMINATION = 0.045  # expected share of anomalies per SKU
ROLL_WINDOW = 7


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    roll = df["sales"].rolling(ROLL_WINDOW, min_periods=1)
    df["roll_mean"] = roll.mean()
    df["roll_std"] = roll.std().fillna(0)
    df["ratio_to_mean"] = df["sales"] / df["roll_mean"].replace(0, np.nan).fillna(1)
    roll_median = df["sales"].rolling(ROLL_WINDOW, min_periods=1).median()
    df["dev_from_median"] = (df["sales"] - roll_median).abs() / roll_median.replace(
        0, np.nan
    ).fillna(1)
    df["pct_change"] = df["sales"].pct_change().abs().fillna(0)
    df["dow"] = pd.to_datetime(df["date"]).dt.dayofweek
    return df


def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a boolean `flagged` column (True = anomaly)."""
    frames = []
    feats = [
        "sales", "roll_mean", "roll_std", "ratio_to_mean",
        "dev_from_median", "pct_change", "dow",
    ]
    for _, grp in df.groupby("sku"):
        g = add_features(grp)
        model = IsolationForest(
            n_estimators=300, contamination=CONTAMINATION, random_state=42
        )
        pred = model.fit_predict(g[feats])  # 1 = normal, -1 = anomaly
        g["flagged"] = pred == -1
        g["anomaly_score"] = -model.score_samples(g[feats])  # higher = weirder
        frames.append(g)
    return pd.concat(frames).sort_values(["sku", "date"]).reset_index(drop=True)
