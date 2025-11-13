import os

import pandas as pd
import matplotlib.pyplot as plt

METRICS_CSV = os.path.join("results", "metrics.csv")
PLOTS_DIR = os.path.join("results", "plots")


def load_metrics():
    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(f"Could not find metrics CSV at {METRICS_CSV}")
    df = pd.read_csv(METRICS_CSV)
    return df


def summarize_best_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each run_name, pick the test row (split='test') with the best F1.
    Also compute average epoch_time_sec for that run.
    Returns a summary DataFrame.
    """
    test = df[df["split"] == "test"].copy()

    # index of row with max F1 for each run
    idx = test.groupby("run_name")["f1"].idxmax()
    best = test.loc[idx].copy()

    # average epoch time per run (based on test rows)
    avg_time = (
        test.groupby("run_name")["epoch_time_sec"]
        .mean()
        .rename("avg_epoch_time")
        .reset_index()
    )

    best = best.merge(avg_time, on="run_name")

    cols = [
        "run_name",
        "architecture",
        "activation",
        "optimizer",
        "seq_len",
        "grad_clipping",
        "accuracy",
        "f1",
        "avg_epoch_time",
    ]
    best = best[cols].sort_values("f1", ascending=False).reset_index(drop=True)
    return best


def plot_seq_length_effects(summary: pd.DataFrame):
    """
    Plot Accuracy/F1 vs Sequence Length for LSTM + ReLU + Adam + grad clipping runs.
    This matches the assignment's sequence-length comparison. :contentReference[oaicite:1]{index=1}
    """
    mask = (
        (summary["architecture"] == "lstm")
        & (summary["activation"] == "relu")
        & (summary["optimizer"].str.lower() == "adam")
        & (summary["grad_clipping"] == True)
    )
    subset = summary[mask].copy()
    if subset.empty:
        print("No LSTM+ReLU+Adam+clip runs found; skipping seq-length plot.")
        return

    subset = subset.sort_values("seq_len")

    plt.figure()
    plt.plot(subset["seq_len"], subset["accuracy"], marker="o", label="Accuracy")
    plt.plot(subset["seq_len"], subset["f1"], marker="o", label="F1 (macro)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Score")
    plt.title("Accuracy and F1 vs Sequence Length (LSTM, ReLU, Adam, clip)")
    plt.legend()
    plt.grid(True)
    out_path = os.path.join(PLOTS_DIR, "accuracy_f1_vs_seq_len.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_loss_curves(df: pd.DataFrame, summary: pd.DataFrame):
    """
    Plot Training Loss vs. Epochs for:
      - Best model (highest F1)
      - Worst model (lowest F1)
    Assignment wants 'Training Loss vs Epochs (for best and worst models)'. :contentReference[oaicite:2]{index=2}
    """
    # best & worst by F1
    best_row = summary.iloc[0]
    worst_row = summary.iloc[-1]

    for model_type, row in [("best", best_row), ("worst", worst_row)]:
        run_name = row["run_name"]
        print(f"{model_type.capitalize()} model run_name: {run_name}")

        # training split only
        run_train = df[(df["run_name"] == run_name) & (df["split"] == "train")].copy()
        run_train = run_train.sort_values("epoch")

        plt.figure()
        plt.plot(run_train["epoch"], run_train["loss"], marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Training Loss")
        plt.title(f"Training Loss vs Epochs ({model_type} model: {run_name})")
        plt.grid(True)
        out_path = os.path.join(PLOTS_DIR, f"training_loss_{model_type}.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {out_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_metrics()
    summary = summarize_best_per_run(df)

    # Print and save summary table
    print("\n=== Best Test Metrics per Run (one row per configuration) ===")
    print(summary.to_string(index=False))

    summary_path = os.path.join("results", "summary_best_runs.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")

    # Plots required by assignment
    plot_seq_length_effects(summary)
    plot_loss_curves(df, summary)


if __name__ == "__main__":
    main()
