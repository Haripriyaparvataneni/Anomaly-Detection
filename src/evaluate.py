"""Real evaluation: precision / recall / F1 of the Isolation Forest detector
against the injected anomaly labels in data/sales.csv.

Also saves per-SKU plots to reports/ (flagged points highlighted).
Run:  python data/generate_data.py   # from the repo root (or from data/)
      python src/evaluate.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from detect import flag_anomalies


def main():
    df = pd.read_csv("data/sales.csv", parse_dates=["date"])
    flagged = flag_anomalies(df)

    y_true = flagged["is_anomaly"].astype(int)
    y_pred = flagged["flagged"].astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    print(f"rows evaluated : {len(flagged)}")
    print(f"true anomalies : {int(y_true.sum())}")
    print(f"flagged        : {int(y_pred.sum())}")
    print(f"precision      : {precision:.3f}")
    print(f"recall         : {recall:.3f}")
    print(f"F1             : {f1:.3f}")

    import os

    os.makedirs("reports", exist_ok=True)
    for sku, g in flagged.groupby("sku"):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(g["date"], g["sales"], color="steelblue", lw=0.8, label="sales")
        hits = g[g["flagged"]]
        ax.scatter(hits["date"], hits["sales"], color="red", s=18, label="flagged")
        ax.set_title(f"{sku} — detected anomalies")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"reports/{sku}.png")
        plt.close(fig)
    print("saved per-SKU plots to reports/")


if __name__ == "__main__":
    main()
