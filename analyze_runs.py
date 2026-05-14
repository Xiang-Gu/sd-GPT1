import csv
import json
from pathlib import Path

import torch


RUNS_ROOT = Path("runs")
OUTPUT_PATH = RUNS_ROOT / "summary.csv"


def read_config(run_dir):
    path = run_dir / "config.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_metrics(run_dir):
    path = run_dir / "metrics.csv"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def best_validation(metrics):
    rows = [row for row in metrics if row["val_loss"]]
    if not rows:
        return None
    return min(rows, key=lambda row: float(row["val_loss"]))


def latest_step(metrics):
    steps = [int(row["step"]) for row in metrics if row["step"]]
    return max(steps) if steps else 0


def latest_nonempty(metrics, field):
    for row in reversed(metrics):
        if row.get(field):
            return row[field]
    return ""


def latest_checkpoint(run_dir):
    path = run_dir / "latest.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def run_status(run_dir, config, metrics):
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        return "no-checkpoint"

    validation_rows = [row for row in metrics if row["val_loss"]]
    if len(validation_rows) >= int(config["epochs"]):
        return "complete"

    patience = config.get("early_stopping_patience")
    epochs_without_improvement = checkpoint.get("epochs_without_improvement")
    if (
        patience is not None
        and epochs_without_improvement is not None
        and int(epochs_without_improvement) >= int(patience)
    ):
        return "early-stopped"

    return "in-progress"


def summarize_run(run_dir):
    config = read_config(run_dir)
    metrics = read_metrics(run_dir)
    best = best_validation(metrics)

    row = {
        "run_dir": str(run_dir),
        "run_name": config["run_name"],
        "status": run_status(run_dir, config, metrics),
        "best_val_loss": "",
        "best_epoch": "",
        "best_step": "",
        "latest_step": latest_step(metrics),
        "latest_tokens_seen": latest_nonempty(metrics, "tokens_seen"),
        "latest_tokens_per_sec": latest_nonempty(metrics, "tokens_per_sec"),
        "latest_elapsed_sec": latest_nonempty(metrics, "elapsed_sec"),
        "latest_epoch_elapsed_sec": latest_nonempty(metrics, "epoch_elapsed_sec"),
        "latest_avg_step_sec": latest_nonempty(metrics, "avg_step_sec"),
        "latest_skipped_updates": latest_nonempty(metrics, "skipped_updates"),
        "latest_grad_norm_before_clip": latest_nonempty(
            metrics, "grad_norm_before_clip"
        ),
        "latest_grad_norm_after_clip": latest_nonempty(metrics, "grad_norm_after_clip"),
        "param_count": config["param_count"],
        "tokenizer_type": config.get("tokenizer_type", "char"),
        "max_vocab_size": config.get("max_vocab_size", ""),
        "vocab_size": config.get("vocab_size", ""),
        "N": config["N"],
        "d_model": config["d_model"],
        "d_ff": config["d_ff"],
        "h": config["h"],
        "B": config["B"],
        "L": config["L"],
        "epochs": config["epochs"],
        "lr": config["lr"],
        "dropout": config["dropout"],
        "grad_clip": config["grad_clip"],
        "weight_tying": config["weight_tying"],
        "best_checkpoint": str(run_dir / "best.pt") if (run_dir / "best.pt").exists() else "",
        "latest_checkpoint": str(run_dir / "latest.pt") if (run_dir / "latest.pt").exists() else "",
    }

    if best is not None:
        row["best_val_loss"] = float(best["val_loss"])
        row["best_epoch"] = int(best["epoch"]) + 1
        row["best_step"] = int(best["step"])

    return row


def collect_summaries(runs_root):
    summaries = []
    for run_dir in sorted(runs_root.iterdir()):
        if run_dir.is_dir() and (run_dir / "metrics.csv").exists():
            summaries.append(summarize_run(run_dir))
    return summaries


def write_summary_csv(rows, path):
    fieldnames = [
        "run_dir",
        "run_name",
        "status",
        "best_val_loss",
        "best_epoch",
        "best_step",
        "latest_step",
        "latest_tokens_seen",
        "latest_tokens_per_sec",
        "latest_elapsed_sec",
        "latest_epoch_elapsed_sec",
        "latest_avg_step_sec",
        "latest_skipped_updates",
        "latest_grad_norm_before_clip",
        "latest_grad_norm_after_clip",
        "param_count",
        "tokenizer_type",
        "max_vocab_size",
        "vocab_size",
        "N",
        "d_model",
        "d_ff",
        "h",
        "B",
        "L",
        "epochs",
        "lr",
        "dropout",
        "grad_clip",
        "weight_tying",
        "best_checkpoint",
        "latest_checkpoint",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    rows = sorted(
        rows,
        key=lambda row: (
            row["best_val_loss"] == "",
            float(row["best_val_loss"]) if row["best_val_loss"] != "" else float("inf"),
        ),
    )
    print(
        "best_val  epoch  step     status       tok/s      step_s  params      tokenizer  vocab  N  d_model  d_ff  h  B    L    run"
    )
    for row in rows:
        best = f"{row['best_val_loss']:.4f}" if row["best_val_loss"] != "" else "n/a"
        tok_s = (
            f"{float(row['latest_tokens_per_sec']):.0f}"
            if row["latest_tokens_per_sec"] != ""
            else "n/a"
        )
        params = (
            f"{int(row['param_count']):,}" if row["param_count"] != "" else "n/a"
        )
        step_s = (
            f"{float(row['latest_avg_step_sec']):.3f}"
            if row["latest_avg_step_sec"] != ""
            else "n/a"
        )
        print(
            f"{best:<8}  {str(row['best_epoch']):<5}  {str(row['best_step']):<7}  "
            f"{row['status']:<11}  {tok_s:<9}  {step_s:<6}  {params:<10}  "
            f"{row['tokenizer_type']:<9}  {row['vocab_size']:<5}  "
            f"{row['N']:<1}  {row['d_model']:<7}  "
            f"{row['d_ff']:<4}  {row['h']:<1}  {row['B']:<4}  {row['L']:<4}  "
            f"{row['run_name']}"
        )


def main():
    rows = collect_summaries(RUNS_ROOT)
    if not rows:
        raise RuntimeError("No runs with metrics.csv found under runs/.")

    write_summary_csv(rows, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")
    print_summary(rows)


if __name__ == "__main__":
    main()
