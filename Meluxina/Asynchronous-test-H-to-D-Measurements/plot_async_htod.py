#!/usr/bin/env python3
"""
plot_async_htod.py  —  Bandwidth vs N for versions v1, v2, v4  (Async H-to-D transfer)
─────────────────────────────────────────────────────────────────────────────
Usage:
    python3 plot_async_htod.py

No arguments needed. Automatically finds all results_*/timings.csv files
in the current directory and averages bandwidth across them.
Output is written to plots_final/.

NOTE: handles CSV rows that have an extra empty column after N, e.g.:
    omp_off,v1,940000000,,0.881962,8.526444,ok   (7 fields)
    omp_off,v1,4960000000,3.259117,12.175077,ok  (6 fields)
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
CSV_GLOB         = "results_*/timings.csv"
OUTPUT_DIR       = "plots_final"
VERSIONS_TO_PLOT = ["v1", "v2", "v4"]
THEORETICAL_BW   = 32   # GB/s; set to None to skip

# Legend order: exactly as displayed
SERIES_ORDER = [
    ("cuda",    "v1"),
    ("acc",     "v1"),
    ("omp_off", "v1"),
    ("cuda",    "v2"),
    ("acc",     "v2"),
    ("omp_off", "v2"),
    ("cuda",    "v4"),
    ("acc",     "v4"),
    ("omp_off", "v4"),
]

SERIES = {
    ("cuda",    "v1"): ("CUDA (1 GPU)",    "o"),
    ("acc",     "v1"): ("OpenACC (1 GPU)", "s"),
    ("omp_off", "v1"): ("OMP Off (1 GPU)", "P"),
    ("cuda",    "v2"): ("CUDA (2 GPUs)",   "^"),
    ("acc",     "v2"): ("OpenACC (2 GPUs)","v"),
    ("omp_off", "v2"): ("OMP Off (2 GPUs)","h"),
    ("cuda",    "v4"): ("CUDA (4 GPUs)",   "D"),
    ("acc",     "v4"): ("OpenACC (4 GPUs)","x"),
    ("omp_off", "v4"): ("OMP Off (4 GPUs)","*"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def scientific_notation_exact(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coefficient = x / (10 ** exponent)
    return f"${coefficient:.2f} \\times 10^{{{exponent}}}$"


def parse_row(fields):
    """
    Handle two CSV layouts:
      6-field: framework, version, N, elapsed_sec, bandwidth_gbs, status
      7-field: framework, version, N, (empty), elapsed_sec, bandwidth_gbs, status
    """
    if len(fields) == 6:
        fw, ver, n_str, _elapsed, bw_str, status = fields
    elif len(fields) == 7:
        fw, ver, n_str, _empty, _elapsed, bw_str, status = fields
    else:
        return None
    fw     = fw.strip()
    ver    = ver.strip()
    status = status.strip()
    bw_str = bw_str.strip()
    if fw == "framework":
        return None
    if status != "ok":
        return None
    if not bw_str or bw_str == "N/A":
        return None
    try:
        N  = int(n_str.strip())
        bw = float(bw_str)
    except ValueError:
        return None
    return fw, ver, N, bw


def load_csv(path):
    data = defaultdict(list)
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        for fields in reader:
            result = parse_row(fields)
            if result is None:
                continue
            fw, ver, N, bw = result
            data[(fw, ver, N)].append(bw)
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

    all_N = sorted({
        n
        for (fw, ver), nd in data.items()
        if ver in VERSIONS_TO_PLOT
        for n in nd
    })

    if not all_N:
        print("No valid bandwidth data found — nothing to plot.")
        return

    x_pos  = list(range(len(all_N)))
    n_to_x = {n: i for i, n in enumerate(all_N)}

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [scientific_notation_exact(n, None) for n in all_N],
        rotation=10, fontsize=18)
    ax1.set_xlabel("Vector N (type:double)", fontsize=22)
    ax1.grid(True)
    ax1.tick_params(axis="y", left=False, labelleft=False)

    ax2 = ax1.twinx()

    # ── Measured data — collect handles in legend order ──
    series_handles = []
    for key in SERIES_ORDER:
        label, marker = SERIES[key]
        nd = data.get(key, {})
        if not nd:
            continue
        Ns_sorted = sorted(nd)
        xs  = [n_to_x[n] for n in Ns_sorted]
        bws = [nd[n]      for n in Ns_sorted]
        line, = ax2.plot(xs, bws, marker=marker, label=label)
        series_handles.append(line)

    # ── Theoretical bandwidth: dashed line + text on plot, NOT in legend ──
    if THEORETICAL_BW is not None:
        mid = len(x_pos) // 2
        ax2.plot(x_pos, [THEORETICAL_BW] * len(x_pos),
                 linestyle="--", color="gray")
        ax2.text(x_pos[mid], THEORETICAL_BW + 0.3,
                 f"Theoretical BW ({THEORETICAL_BW} GB/s)",
                 color="gray", ha="center", va="bottom", fontsize=14)

    ax2.set_ylabel("Bandwidth (GB/s)", fontsize=22)
    ax2.tick_params(axis="y", labelsize=18)

    # ── Legend: only the measured series ──
    fig.legend(handles=series_handles,
               loc="upper left", bbox_to_anchor=(0.125, 0.9), fontsize=14)
    plt.tight_layout()

    # ── Save as Async H-to-D ──
    out_path = os.path.join(OUTPUT_DIR, "async_htod_bandwidth.pdf")
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
