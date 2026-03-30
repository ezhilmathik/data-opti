#!/usr/bin/env python3
"""
plot_p2p.py  —  Bandwidth vs N for versions v1, v2, v4  (1 / 2 / 4 GPUs)
─────────────────────────────────────────────────────────────────────────────
Usage:
    python3 plot_p2p.py

No arguments needed. Automatically finds all results_*/timings.csv files
in the current directory and averages bandwidth across them.
Output is written to plots_final/.
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CSV_GLOB   = "results_*/timings.csv"
OUTPUT_DIR = "plots_final"

VERSIONS_TO_PLOT = ["v1", "v2", "v4"]

# Theoretical peak bandwidths (GB/s); set to None to skip
THEORETICAL = {
    "v1": 100,
    "v2": 200,
    "v4": 300,
}

# Per (framework, version): (label, color, marker)
SERIES = {
    ("cuda",    "v1"): ("CUDA (1 GPU)",    "blue",  "o"),
    ("acc",     "v1"): ("OpenACC (1 GPU)", "blue",  "s"),
    ("omp_off", "v1"): ("OMP Off (1 GPU)", "blue",  "^"),
    ("cuda",    "v2"): ("CUDA (2 GPUs)",   "green", "^"),
    ("acc",     "v2"): ("OpenACC (2 GPUs)","green", "v"),
    ("omp_off", "v2"): ("OMP Off (2 GPUs)","green", "D"),
    ("cuda",    "v4"): ("CUDA (4 GPUs)",   "red",   "D"),
    ("acc",     "v4"): ("OpenACC (4 GPUs)","red",   "x"),
    ("omp_off", "v4"): ("OMP Off (4 GPUs)","red",   "P"),
}

VERSION_COLOR = {"v1": "blue", "v2": "green", "v4": "red"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def scientific_notation_exact(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coefficient = x / (10 ** exponent)
    return f"${coefficient:.2f} \\times 10^{{{exponent}}}$"


def load_csv(path):
    data = defaultdict(list)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("framework", "").strip() == "framework":
                continue
            if row.get("status", "ok").strip() != "ok":
                continue
            version   = row.get("version", "").strip()
            framework = row.get("framework", "").strip()
            bw_raw    = row.get("bandwidth_gbs", "N/A").strip()
            if not version or not framework or bw_raw == "N/A":
                continue
            try:
                N  = int(row["N"].strip())
                bw = float(bw_raw)
            except ValueError:
                continue
            data[(framework, version, N)].append(bw)
    return data


def load_and_average(csv_paths):
    combined = defaultdict(list)
    for path in csv_paths:
        for (fw, ver, N), bws in load_csv(path).items():
            combined[(fw, ver, N)].extend(bws)
    averaged = defaultdict(dict)
    for (fw, ver, N), bws in combined.items():
        averaged[(fw, ver)][N] = float(np.mean(bws))
    return averaged


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    csv_paths = sorted(glob.glob(CSV_GLOB))
    if not csv_paths:
        print(f"No files matched '{CSV_GLOB}' in {os.getcwd()}")
        return

    print(f"Found {len(csv_paths)} file(s):")
    for p in csv_paths:
        print(f"  {p}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_and_average(csv_paths)

    # Collect all N values for v1/v2/v4, sorted numerically
    all_N = sorted({
        n
        for (fw, ver), nd in data.items()
        if ver in VERSIONS_TO_PLOT
        for n in nd
    })

    if not all_N:
        print("No valid bandwidth data found for v1/v2/v4 — nothing to plot.")
        return

    # Equal-spaced integer positions so uneven N gaps don't distort the plot
    x_pos  = list(range(len(all_N)))
    n_to_x = {n: i for i, n in enumerate(all_N)}

    # ── Plot setup (mirrors reference style) ──
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [scientific_notation_exact(n, None) for n in all_N],
        rotation=10, fontsize=18)
    ax1.set_xlabel("Vector N (type:double)", fontsize=18)
    ax1.grid(True)
    ax1.tick_params(axis="y", left=False, labelleft=False)

    ax2 = ax1.twinx()

    # ── Measured data ──
    for (fw, ver), (label, color, marker) in SERIES.items():
        nd = data.get((fw, ver), {})
        if not nd:
            continue
        Ns_sorted = sorted(nd)
        xs  = [n_to_x[n] for n in Ns_sorted]
        bws = [nd[n]      for n in Ns_sorted]
        ax2.plot(xs, bws, marker=marker, color=color, label=label)

    # ── Theoretical lines (no legend entry, annotated with text) ──
    mid = len(x_pos) // 2
    for ver, theor in THEORETICAL.items():
        if theor is None:
            continue
        color = VERSION_COLOR[ver]
        ax2.plot(x_pos, [theor] * len(x_pos),
                 linestyle="--", color=color)
        ax2.text(x_pos[mid], theor + 0.5,
                 f"Theor. BW {theor} GB/s",
                 color=color, ha="center", va="bottom", fontsize=14)

    ax2.set_ylabel("Bandwidth (GB/s)", fontsize=18)
    ax2.tick_params(axis="y", labelsize=18)

    fig.legend(loc="upper left", bbox_to_anchor=(0.04, 0.95), fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "p2p_bandwidth.pdf")
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
