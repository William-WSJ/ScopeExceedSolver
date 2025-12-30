#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def smooth_ema(scalars: Sequence[float]) -> List[float]:
    """
    EMA smoothing (same idea as LLaMA-Factory `src/llamafactory/extras/ploting.py`).
    """
    if not scalars:
        return []

    last = float(scalars[0])
    smoothed: List[float] = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # sigmoid
    for next_val in scalars:
        next_val_f = float(next_val)
        smoothed_val = last * weight + (1 - weight) * next_val_f
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def smooth_ma(scalars: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return [float(x) for x in scalars]

    smoothed: List[float] = []
    running_sum = 0.0
    buf: List[float] = []
    for x in scalars:
        x_f = float(x)
        buf.append(x_f)
        running_sum += x_f
        if len(buf) > window:
            running_sum -= buf.pop(0)
        smoothed.append(running_sum / len(buf))
    return smoothed


def load_log_history(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("log_history"), list):
        return data["log_history"]

    if isinstance(data, list) and all(isinstance(x, dict) for x in data):
        return data  # also support a raw list of {"epoch": ..., "loss": ...}

    raise ValueError(f"Unsupported JSON schema in: {path}")


def extract_xy(
    log_history: Iterable[Dict[str, Any]],
    metric_key: str,
    x_axis: str,
) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []

    for item in log_history:
        if metric_key not in item:
            continue

        y = item.get(metric_key)
        if y is None:
            continue

        if x_axis == "epoch":
            x = item.get("epoch")
        elif x_axis == "step":
            x = item.get("step")
        elif x_axis == "auto":
            x = item.get("epoch", item.get("step"))
        else:
            raise ValueError(f"Unknown x-axis: {x_axis}")

        if x is None:
            continue

        xs.append(float(x))
        ys.append(float(y))

    return xs, ys


def default_output_path(metric_key: str) -> str:
    if metric_key == "loss":
        return "loss.png"
    return f"training_{metric_key.replace('/', '_')}.png"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loss/metrics from HuggingFace Trainer `trainer_state.json`.",
    )
    parser.add_argument("trainer_state", help="Path to trainer_state.json")
    parser.add_argument(
        "--key",
        default="loss",
        help='Metric key inside log_history (default: "loss"). Example: eval_loss',
    )
    parser.add_argument(
        "--x",
        choices=["auto", "epoch", "step"],
        default="epoch",
        help='X axis key (default: "epoch"; use "step" if your log has no epoch).',
    )
    parser.add_argument("--title", default=None, help="Figure title (optional).")
    parser.add_argument(
        "--smooth",
        choices=["ema", "ma", "none"],
        default="ema",
        help="Smoothing method (default: ema).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Window size for moving average smoothing (default: 5).",
    )
    parser.add_argument("--out", default=None, help="Output image path (png/pdf/svg...).")
    parser.add_argument("--no-grid", action="store_true", help="Disable grid.")
    parser.add_argument(
        "--figsize",
        default="6,4",
        help='Figure size in inches, "W,H" (default: 6,4).',
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show an interactive window (may not work on headless servers).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        fig_w_str, fig_h_str = args.figsize.split(",", 1)
        figsize = (float(fig_w_str), float(fig_h_str))
    except Exception:
        raise SystemExit('--figsize must look like "6,4"')

    log_history = load_log_history(args.trainer_state)
    xs, ys = extract_xy(log_history, metric_key=args.key, x_axis=args.x)
    if not ys:
        raise SystemExit(f'No "{args.key}" found in log_history for x="{args.x}".')

    if args.smooth == "none":
        ys_smooth = None
    elif args.smooth == "ma":
        ys_smooth = smooth_ma(ys, window=args.window)
    else:
        ys_smooth = smooth_ema(ys)

    import matplotlib.pyplot as plt

    if not args.show:
        plt.switch_backend("agg")

    plt.close("all")
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.plot(xs, ys, color="#1f77b4", alpha=0.4, label="original")
    if ys_smooth is not None:
        ax.plot(xs, ys_smooth, color="#1f77b4", label="smoothed")

    if args.title:
        ax.set_title(args.title)

    ax.set_xlabel(args.x)
    ax.set_ylabel(args.key)
    ax.legend()
    if not args.no_grid:
        ax.grid(alpha=0.25, linestyle="--")

    out_path = args.out or default_output_path(args.key)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
