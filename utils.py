"""
utils.py
========
Reproducibility helpers, device selection, and the training/evaluation
plotting suite (spec sections 15 and 19).
"""

import random
from typing import Dict, List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> str:
    """Resolve 'auto'/'cpu'/'cuda' to an actual device string and print it."""
    if preferred == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = preferred
        if device == "cuda" and not torch.cuda.is_available():
            print("[utils] CUDA requested but not available — falling back to CPU.")
            device = "cpu"
    print(f"[utils] Using device: {device}")
    return device


def moving_average(values: Sequence[float], window: int = 20) -> np.ndarray:
    if len(values) < window:
        return np.array(values, dtype=np.float32)
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def plot_training_metrics(history: Dict[str, List[float]], filename: str = "training_results.png") -> None:
    """
    Multi-panel training curves (spec section 19):
      1. Episode reward            4. DQN loss (per training step)
      2. Avg waiting time          5. Epsilon
      3. Avg queue length          6. Phase changes per episode

    `history` keys expected: reward, avg_wait, avg_queue, loss, epsilon, phase_changes
    Uses matplotlib's non-interactive backend so this works headlessly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        ("reward", "Episode Reward", "reward"),
        ("avg_wait", "Avg Waiting Time / episode", "wait (steps)"),
        ("avg_queue", "Avg Queue Length / episode", "vehicles"),
        ("loss", "DQN Loss (per optimizer step)", "Huber loss"),
        ("epsilon", "Epsilon Decay", "epsilon"),
        ("phase_changes", "Phase Changes / episode", "count"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, panels):
        data = history.get(key, [])
        if not data:
            ax.set_visible(False)
            continue
        ax.plot(data, alpha=0.35, color="tab:blue", label="raw")
        if len(data) >= 20:
            ma = moving_average(data, window=20)
            offset = len(data) - len(ma)
            ax.plot(range(offset, len(data)), ma, color="tab:blue", label="moving avg (20)")
        ax.set_title(title)
        ax.set_xlabel("step" if key == "loss" else "episode")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"[utils] Saved training plots -> {filename}")


def plot_controller_comparison(
    results: Dict[str, Dict[str, float]],
    filename: str = "controller_comparison.png",
) -> None:
    """
    Bar-chart comparison across controllers (spec section 19, item 6).
    `results`: {controller_name: {"avg_wait": ..., "avg_queue": ..., "throughput": ...}}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results.keys())
    metrics = ["avg_wait", "avg_queue", "throughput"]
    titles = ["Avg Waiting Time (lower better)", "Avg Queue Length (lower better)", "Throughput (higher better)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, metric, title in zip(axes, metrics, titles):
        values = [results[n][metric] for n in names]
        ax.bar(names, values, color=["tab:gray", "tab:orange", "tab:green", "tab:blue"][: len(names)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"[utils] Saved controller comparison -> {filename}")
