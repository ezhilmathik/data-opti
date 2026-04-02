#!/usr/bin/env python3
"""
plot_dtoh.py — Bandwidth vs N for versions v1, v2, v4 (D-to-H transfer)

Usage:
    python3 plot_dtoh.py

Automatically finds all results_*/timings.csv files in the current directory,
averages bandwidth across them, and writes the output to plots_final/.

Handles both row layouts:
    framework,version,N,elapsed_sec,bandwidth_gbs,status
    framework,version,N,,elapsed_sec,bandwidth_gbs,status
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

# Set to None to disable theoretical line
THEORETICAL_BW   = 42  # GB/s

SERIES_ORDER = [
    ("hip",     "v1"),
    ("omp_off", "v1"),
    ("hip",     "v2"),
    ("omp_off", "v2"),
    ("hip",     "v4"),
    ("omp_off", "v4"),
]

SERIES = {
    ("hip",     "v1"): ("HIP (1 GPU)",          "o"),
    ("omp_off", "v1"): ("OMP Offload (1 GPU)",  "s"),
    ("hip",     "v2"): ("HIP (2 GPUs)",         "^"),
    ("omp_off", "v2"): ("OMP Offload (2 GPUs)", "v"),
    ("hip",     "v4"): ("HIP (4 GPUs)",         "D"),
    ("omp_off", "v4"): ("OMP Offload (4 GPUs)", "P"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def scientific_notation_exact(x, pos=None):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coefficient = x / (10 ** exponent)
    return f"${coefficient:.2f} \\times 10^{{{exponent}}}$"


def parse_row(fields):
    """
    Accept:
      6 fields: framework, version, N, elapsed_sec, bandwidth_gbs, status
      7 fields: framework, version, N, '', elapsed_sec, bandwidth_gbs, status
    """
    if len(fields) == 6:
        fw, ver, n_str, _elapsed, bw_str, status = fields
    elif len(fields) == 7:
        fw, ver, n_str, _empty, _elapsed, bw_str, status = fields
    else:
        return None

    fw = fw.strip()
    ver = ver.strip()
    status = status.strip()
    bw_str = bw_str.strip()

    if fw == "framework":
        return None
    if ver not in VERSIONS_TO_PLOT:
        return None
    if status != "ok":
        return None
    if not bw_str or bw_str == "N/A":
        return None

    try:
        N = int(n_str.strip())
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

    x_pos = list(range(len(all_N)))
    n_to_x = {n: i for i, n in enumerate(all_N)}

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [scientific_notation_exact(n) for n in all_N],
        rotation=10,
        fontsize=16
    )
    ax1.set_xlabel("Vector N (type: double)", fontsize=20)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.tick_params(axis="y", left=False, labelleft=False)

    ax2 = ax1.twinx()

    series_handles = []
    for key in SERIES_ORDER:
        label, marker = SERIES[key]
        nd = data.get(key, {})
        if not nd:
            continue

        Ns_sorted = sorted(nd)
        xs = [n_to_x[n] for n in Ns_sorted]
        bws = [nd[n] for n in Ns_sorted]

        line, = ax2.plot(xs, bws, marker=marker, linewidth=2, markersize=8, label=label)
        series_handles.append(line)

    if THEORETICAL_BW is not None:
        mid = len(x_pos) // 2
        ax2.plot(
            x_pos,
            [THEORETICAL_BW] * len(x_pos),
            linestyle="--",
            color="gray",
            linewidth=1.5
        )
        ax2.text(
            x_pos[mid],
            THEORETICAL_BW + 0.4,
            f"Theoretical BW ({THEORETICAL_BW} GB/s)",
            color="gray",
            ha="center",
            va="bottom",
            fontsize=12
        )

    ax2.set_ylabel("Bandwidth (GB/s)", fontsize=20)
    ax2.tick_params(axis="y", labelsize=16)

    max_bw = max(
        bw for (_, _), nd in data.items() for bw in nd.values()
    )
    ymax = max(max_bw * 1.10, (THEORETICAL_BW * 1.08) if THEORETICAL_BW else 0)
    ax2.set_ylim(0, ymax)

    fig.legend(
        handles=series_handles,
        loc="upper left",
        bbox_to_anchor=(0.12, 0.92),
        fontsize=12
    )

    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "dtoh_bandwidth.pdf")
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
