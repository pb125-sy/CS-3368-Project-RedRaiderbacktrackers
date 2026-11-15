"""
Results table + charts: Simulated Annealing vs. Min-Conflicts (Minton et al. 1992)
Author: Akaela Espina

Reads (from simulated_annealing/results/):
    sa_results.csv, sa_history.json   <- simulated_annealing.py
    mc_results.csv, mc_history.json   <- min_conflicts.py
Writes:
    comparison_table.csv              one row per size x model (the slide table)
    fig_final_soft_penalty.png        bar chart: final soft penalty by size
    fig_convergence.png               best cost vs. search effort, one panel per size

Usage (from the repo top folder):
    python simulated_annealing/plot_results.py
Requires: matplotlib  (pip install matplotlib)
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # save files only, no window needed
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS = SCRIPT_DIR / "results"
SCALES = ["small", "medium", "large"]
MODELS = {                                  # label, color, line style
    "sa": ("Simulated annealing", "#00897B", "-"),
    "mc": ("Min-conflicts (Minton 1992)", "#9C4DB0", "--"),
}
INK, MUTED, GRID = "#1F2433", "#5B6475", "#E3E6EB"

plt.rcParams.update({
    "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 16,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False,
})


def read_rows(name):
    path = RESULTS / f"{name}_results.csv"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run the model script first.")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("best_hard", "best_soft", "best_total", "initial_soft"):
            r[k] = int(r[k])
        r["feasible"] = r["feasible"] == "True"
        r["runtime_s"] = float(r["runtime_s"])
        r["effort"] = int(r["iterations"] if "iterations" in r else r["states_evaluated"])
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def build_table(data):
    table = []
    for scale in SCALES:
        for key, (label, _, _) in MODELS.items():
            rs = [r for r in data[key] if r["scale"] == scale]
            if not rs:
                continue
            table.append({
                "size": scale, "model": label, "runs": len(rs),
                "feasible_pct": round(100 * mean([r["feasible"] for r in rs])),
                "mean_start_soft": round(mean([r["initial_soft"] for r in rs]), 1),
                "mean_final_soft": round(mean([r["best_soft"] for r in rs]), 1),
                "best_final_soft": min(r["best_soft"] for r in rs),
                "worst_final_soft": max(r["best_soft"] for r in rs),
                "mean_states_evaluated": round(mean([r["effort"] for r in rs])),
                "mean_runtime_s": round(mean([r["runtime_s"] for r in rs]), 2),
            })
    with open(RESULTS / "comparison_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    return table


def bar_chart(table):
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    width, gap = 0.36, 0.02
    present = [s for s in SCALES if any(t["size"] == s for t in table)]
    for i, (key, (label, color, _)) in enumerate(MODELS.items()):
        xs, ys, lo, hi = [], [], [], []
        for j, scale in enumerate(present):
            t = next((t for t in table if t["size"] == scale and t["model"] == label), None)
            if not t:
                continue
            xs.append(j + (i - 0.5) * (width + gap))
            ys.append(t["mean_final_soft"])
            lo.append(t["mean_final_soft"] - t["best_final_soft"])
            hi.append(t["worst_final_soft"] - t["mean_final_soft"])
        ax.bar(xs, ys, width, color=color, label=label, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1.5,
                    capsize=6, zorder=3)
        for x, y, h in zip(xs, ys, hi):
            ax.text(x, y + h + max(ys) * 0.02, f"{y:.0f}", ha="center", va="bottom",
                    fontsize=15, color=INK)
    top = max(t["worst_final_soft"] for t in table)
    ax.set_ylim(0, top * 1.18)                # headroom for the value labels
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([f"{s.capitalize()}\n({ {'small': 25, 'medium': 50, 'large': 100}[s] } courses)"
                        for s in present])
    ax.set_ylabel("Final soft penalty (lower is better)")
    ax.set_title("Final soft penalty by instance size (bar = mean, whisker = best–worst run)",
                 loc="left", fontsize=17)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_final_soft_penalty.png")
    plt.close(fig)


def convergence_chart(histories, instance_seed="seed42", run="run100"):
    present = [s for s in SCALES if f"{s}_{instance_seed}_{run}" in histories["sa"]]
    fig, axes = plt.subplots(1, len(present), figsize=(15, 5.5), dpi=200, squeeze=False)
    for ax, scale in zip(axes[0], present):
        key = f"{scale}_{instance_seed}_{run}"
        for m, (label, color, style) in MODELS.items():
            h = histories[m].get(key)
            if not h:
                continue
            xs = [max(p[0], 1) for p in h]     # states evaluated (log axis needs > 0)
            ys = [max(p[1], 1) for p in h]     # best total cost
            ax.step(xs, ys, where="post", color=color, linestyle=style, linewidth=2.5,
                    label=label)
            ax.plot(xs[-1], ys[-1], "o", color=color, markersize=9,
                    markeredgecolor="white", markeredgewidth=2)
            ax.annotate(f"{h[-1][3]}", (xs[-1], ys[-1]), xytext=(6, 6),
                        textcoords="offset points", fontsize=14, color=INK)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{scale.capitalize()} ({instance_seed}, {run})", loc="left")
        ax.set_xlabel("States evaluated (log)")
        ax.grid(True, which="major", color=GRID)
    axes[0][0].set_ylabel("Best total cost (log)\nhard × 10,000 + soft")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, -0.02, "Drop of ~4 orders of magnitude = hard violations reach 0. "
             "End labels = final soft penalty.", fontsize=13, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULTS / "fig_convergence.png", bbox_inches="tight")
    plt.close(fig)


def main():
    data = {"sa": read_rows("sa"), "mc": read_rows("mc")}
    histories = {}
    for m in MODELS:
        with open(RESULTS / f"{m}_history.json") as fh:
            histories[m] = json.load(fh)

    table = build_table(data)
    bar_chart(table)
    convergence_chart(histories)

    print(f"{'size':7s} {'model':30s} {'runs':>4s} {'feasible':>8s} {'start soft':>10s} "
          f"{'final soft':>10s} {'best':>5s} {'states':>8s} {'time s':>7s}")
    for t in table:
        print(f"{t['size']:7s} {t['model']:30s} {t['runs']:>4d} {t['feasible_pct']:>7d}% "
              f"{t['mean_start_soft']:>10.1f} {t['mean_final_soft']:>10.1f} "
              f"{t['best_final_soft']:>5d} {t['mean_states_evaluated']:>8d} {t['mean_runtime_s']:>7.2f}")
    print(f"\nWrote comparison_table.csv, fig_final_soft_penalty.png, fig_convergence.png to {RESULTS}")


if __name__ == "__main__":
    main()
