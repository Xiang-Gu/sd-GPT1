import csv
from pathlib import Path


RUNS_ROOT = Path("runs")
OUTPUT_PATH = RUNS_ROOT / "eval_loss.svg"


def read_validation_points(run_dir):
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return []

    points = []
    with open(metrics_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if not row["val_loss"]:
                continue
            points.append(
                {
                    "epoch": int(row["epoch"]),
                    "step": int(row["step"]),
                    "val_loss": float(row["val_loss"]),
                }
            )
    return points


def collect_runs(runs_root):
    runs = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue

        points = read_validation_points(run_dir)
        if points:
            runs.append((run_dir.name, points))
    return runs


def scale(value, src_min, src_max, dst_min, dst_max):
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def polyline(points, x_min, x_max, y_min, y_max, plot_left, plot_top, plot_w, plot_h):
    coords = []
    for point in points:
        x = scale(point["step"], x_min, x_max, plot_left, plot_left + plot_w)
        y = scale(point["val_loss"], y_min, y_max, plot_top + plot_h, plot_top)
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def make_svg(runs):
    width, height = 1200, 760
    plot_left, plot_top = 90, 70
    plot_w, plot_h = 760, 560
    legend_x = plot_left + plot_w + 45
    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4d7c0f",
    ]

    all_points = [point for _, points in runs for point in points]
    x_min = min(point["step"] for point in all_points)
    x_max = max(point["step"] for point in all_points)
    y_min = min(point["val_loss"] for point in all_points)
    y_max = max(point["val_loss"] for point in all_points)
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="90" y="35" font-family="Arial" font-size="24" font-weight="700">Validation Loss by Training Step</text>',
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]

    for i in range(6):
        y = plot_top + i * plot_h / 5
        val = y_max - i * (y_max - y_min) / 5
        lines.append(
            f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>'
        )
        lines.append(
            f'<text x="{plot_left - 12}" y="{y + 4:.1f}" font-family="Arial" font-size="12" text-anchor="end" fill="#475569">{val:.3f}</text>'
        )

    for i in range(6):
        x = plot_left + i * plot_w / 5
        step = int(x_min + i * (x_max - x_min) / 5)
        lines.append(
            f'<line x1="{x:.1f}" y1="{plot_top}" x2="{x:.1f}" y2="{plot_top + plot_h}" stroke="#e2e8f0"/>'
        )
        lines.append(
            f'<text x="{x:.1f}" y="{plot_top + plot_h + 25}" font-family="Arial" font-size="12" text-anchor="middle" fill="#475569">{step}</text>'
        )

    lines.append(
        f'<text x="{plot_left + plot_w / 2}" y="{plot_top + plot_h + 58}" font-family="Arial" font-size="14" text-anchor="middle">training step</text>'
    )
    lines.append(
        f'<text x="25" y="{plot_top + plot_h / 2}" font-family="Arial" font-size="14" text-anchor="middle" transform="rotate(-90 25 {plot_top + plot_h / 2})">validation loss</text>'
    )

    for i, (name, points) in enumerate(runs):
        color = colors[i % len(colors)]
        path_points = polyline(
            points, x_min, x_max, y_min, y_max, plot_left, plot_top, plot_w, plot_h
        )
        best_point = min(points, key=lambda point: point["val_loss"])
        best_x = scale(best_point["step"], x_min, x_max, plot_left, plot_left + plot_w)
        best_y = scale(best_point["val_loss"], y_min, y_max, plot_top + plot_h, plot_top)
        legend_y = plot_top + i * 58

        lines.append(
            f'<polyline points="{path_points}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        for point in points:
            x = scale(point["step"], x_min, x_max, plot_left, plot_left + plot_w)
            y = scale(point["val_loss"], y_min, y_max, plot_top + plot_h, plot_top)
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')

        lines.append(
            f'<circle cx="{best_x:.1f}" cy="{best_y:.1f}" r="7" fill="white" stroke="{color}" stroke-width="3"/>'
        )
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>'
        )
        lines.append(
            f'<text x="{legend_x + 38}" y="{legend_y + 4}" font-family="Arial" font-size="12" fill="#0f172a">{name}</text>'
        )
        lines.append(
            f'<text x="{legend_x + 38}" y="{legend_y + 22}" font-family="Arial" font-size="12" fill="#64748b">best {best_point["val_loss"]:.4f} @ step {best_point["step"]}</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def print_summary(runs):
    summary = []
    for name, points in runs:
        best = min(points, key=lambda point: point["val_loss"])
        summary.append((best["val_loss"], best["step"], best["epoch"], name))

    for val_loss, step, epoch, name in sorted(summary):
        print(f"{val_loss:.4f} step={step:<7} epoch={epoch + 1:<2} {name}")


def main():
    runs = collect_runs(RUNS_ROOT)
    if not runs:
        raise RuntimeError("No validation metrics found under runs/.")

    OUTPUT_PATH.write_text(make_svg(runs), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print_summary(runs)


if __name__ == "__main__":
    main()
