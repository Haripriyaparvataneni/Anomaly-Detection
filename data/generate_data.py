"""Generate synthetic daily sales data for several SKUs with injected anomalies.

Anomaly types (labeled in the `is_anomaly` / `anomaly_type` columns):
  - spike:     sudden demand surge (e.g., flash promotion)
  - dip:       sudden one-day drop (e.g., stockout, site outage)
  - flatline:  several consecutive days of zero sales (e.g., SKU delisted)

The RNG is seeded for reproducibility. Writes sales.csv into this directory.
"""
import numpy as np
import pandas as pd

SEED = 42
N_DAYS = 180
SKUS = ["SKU-A100", "SKU-B200", "SKU-C300", "SKU-D400"]


def make_series(rng, base, trend, weekly_amp, days):
    t = np.arange(days)
    dow = t % 7
    weekly = 1 + weekly_amp * (dow >= 5).astype(float)  # weekend bump
    noise = rng.normal(1.0, 0.12, days)
    sales = base * (1 + trend * t / days) * weekly * noise
    return np.maximum(sales.round(), 0).astype(int)


def inject_anomalies(rng, df):
    df = df.copy()
    df["is_anomaly"] = 0
    df["anomaly_type"] = "none"
    for sku in df["sku"].unique():
        idx = df[df["sku"] == sku].index.to_numpy()
        # 2 demand spikes
        for i in rng.choice(idx, size=2, replace=False):
            df.loc[i, "sales"] *= int(rng.integers(6, 11))
            df.loc[i, "is_anomaly"] = 1
            df.loc[i, "anomaly_type"] = "spike"
        # 2 one-day dips
        for i in rng.choice(idx, size=2, replace=False):
            df.loc[i, "sales"] = max(int(df.loc[i, "sales"] * 0.05), 0)
            df.loc[i, "is_anomaly"] = 1
            df.loc[i, "anomaly_type"] = "dip"
        # 1 flatline run (3-4 consecutive zero-sales days)
        start = int(rng.choice(idx[:-4], size=1)[0])
        for i in range(start, start + int(rng.integers(3, 5))):
            df.loc[i, "sales"] = 0
            df.loc[i, "is_anomaly"] = 1
            df.loc[i, "anomaly_type"] = "flatline"
    return df


def main():
    rng = np.random.default_rng(SEED)
    frames = []
    params = [(120, 0.25, 0.4), (85, -0.1, 0.6), (200, 0.0, 0.25), (45, 0.4, 0.5)]
    start = pd.Timestamp("2026-01-01")
    dates = pd.date_range(start, periods=N_DAYS, freq="D")
    for sku, (base, trend, amp) in zip(SKUS, params):
        frames.append(
            pd.DataFrame(
                {"date": dates, "sku": sku, "sales": make_series(rng, base, trend, amp, N_DAYS)}
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df = inject_anomalies(rng, df)
    out = "data/sales.csv"
    df.to_csv(out, index=False)
    n = int(df["is_anomaly"].sum())
    print(f"wrote {out}: {len(df)} rows, {n} injected anomalies ({100*n/len(df):.1f}%)")


if __name__ == "__main__":
    main()
