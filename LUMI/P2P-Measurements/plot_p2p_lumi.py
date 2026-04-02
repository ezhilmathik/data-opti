#!/usr/bin/env python3
"""
plot_p2p_lumi.py  —  Bandwidth vs N for versions v1, v2, v4 on LUMI
                     (AMD MI250X, inter-card xGMI, one GCD per physical card)

Automatically finds all results_*/timings.csv files in the current directory
and averages bandwidth across them.
Output is written to plots_final/.

LUMI hardware context (source: docs.lumi-supercomputer.eu/hardware/lumig):
  - Each node: 4 x MI250X modules, each with 2 GCDs (8 GCDs total per node)
  - Run with ROCR_VISIBLE_DEVICES=0,2,4,6 → one GCD per physical card
  - Inter-card single xGMI link : 100 GB/s bidirectional = 50 GB/s per direction
  - Inter-card double xGMI link : 200 GB/s bidirectional = 100 GB/s per direction
  - Intra-card (same MI250X)    : 400 GB/s bidirectional = 200 GB/s per direction (excluded)
  - Topology: non-uniform ring; weight-15 pairs = double link, weight-30 = single link
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

CSV_GLOB   = "results_*/timings.csv"
OUTPUT_DIR = "plots_final"

VERSIONS_TO_PLOT = ["v1", "v2", "v4"]

# Theoretical peak per direction (GB/s) for inter-card xGMI.
# v1 = 1 directed pair  → weight-30 single link → 50 GB/s/direction
# v2 = 2 directed pairs → mix; conservative peak = 100 GB/s
# v4 = 4 directed pairs → all ring edges at single-link rate = 200 GB/s
# These are upper bounds; ring contention lowers achieved BW for v4.
THEORETICAL = {
    "v1": 50,
    "v2": 100,
    "v4": 200,
}

# LUMI only has hip and omp_off — cuda/acc entries kept for safety
FRAMEWORK_LABELS = {
    "hip":     "HIP",
    "omp_off": "OMP Off",
    "cuda":    "CUDA",    # not present on LUMI; ignored if absent
    "acc":     "OpenACC", # not present on LUMI; ignored if absent
}

FRAMEWORK_MARKERS = {
    "hip":     "o",
    "omp_off": "^",
    "cuda":    "s",
    "acc":     "D",
}

FRAMEWORK_LINESTYLE = {
    "hip":     "-",
    "omp_off": "--",
    "cuda":    "-",
    "acc":     ":",
}

VERSION_COLOR = {
    "v1": "steelblue",
    "v2": "seagreen",
    "v4": "tomato",
}


def scientific_notation_exact(x):
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
            version   = row.get("version",   "").strip()
            framework = row.get("framework", "").strip()
            bw_raw    = row.get("bandwidth_gbs", "N/A").strip()
            if not version or not framework or bw_raw == "N/A":
                continue
            try:
                N  = int(row["N"].strip())
                bw = float(bw_raw)
            except (ValueError, KeyError):
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
        print("No valid bandwidth data found for v1/v2/v4 — nothing to plot.")
        return

    available_frameworks = sorted({
        fw for (fw, ver) in data.keys() if ver in VERSIONS_TO_PLOT
    })
    if not available_frameworks:
        print("No frameworks found for requested versions.")
        return

    # Equal-spaced x positions so uneven N gaps don't distort the plot
    x_pos  = list(range(len(all_N)))
    n_to_x = {n: i for i, n in enumerate(all_N)}

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [scientific_notation_exact(n) for n in all_N],
        rotation=10, fontsize=16,
    )
    ax1.set_xlabel("Vector N (type: double)", fontsize=18)
    ax1.grid(True, alpha=0.35)
    ax1.tick_params(axis="y", left=False, labelleft=False)

    ax2 = ax1.twinx()

    for ver in VERSIONS_TO_PLOT:
        color    = VERSION_COLOR.get(ver, "black")
        gpu_lbl  = {"v1": "1 GPU", "v2": "2 GPUs", "v4": "4 GPUs"}.get(ver, ver)

        for fw in available_frameworks:
            nd = data.get((fw, ver), {})
            if not nd:
                continue
            xs  = [n_to_x[n] for n in sorted(nd)]
            bws = [nd[n]      for n in sorted(nd)]
            fw_label = FRAMEWORK_LABELS.get(fw, fw)
            marker   = FRAMEWORK_MARKERS.get(fw, "o")
            ls       = FRAMEWORK_LINESTYLE.get(fw, "-")
            label    = f"{fw_label} ({gpu_lbl})"
            ax2.plot(xs, bws, marker=marker, linestyle=ls, color=color,
                     label=label, linewidth=1.8, markersize=6)

    # Theoretical peak lines
    mid = len(x_pos) // 2
    for ver, theor in THEORETICAL.items():
        if theor is None:
            continue
        color = VERSION_COLOR.get(ver, "black")
        ax2.axhline(theor, linestyle=":", color=color, linewidth=1.0, alpha=0.6)
        ax2.text(
            x_pos[mid], theor + 1.5,
            f"Peak {theor} GB/s",
            color=color, ha="center", va="bottom", fontsize=13,
        )

    ax2.set_ylabel("Bandwidth (GB/s)", fontsize=18)
    ax2.tick_params(axis="y", labelsize=16)

    n_runs = len(csv_paths)
    fig.suptitle(
        f"LUMI — P2P Bandwidth: v1 / v2 / v4\n"
        f"AMD MI250X · inter-card xGMI · ROCR_VISIBLE_DEVICES=0,2,4,6 "
        f"(averaged over {n_runs} run{'s' if n_runs > 1 else ''})",
        fontsize=13,
    )

    # Topology note
    fig.text(
        0.99, 0.01,
        "Theoretical peaks: inter-card single xGMI link (50 GB/s/dir) × N pairs.\n"
        "Ring topology: weight-15 pairs = double link (100 GB/s/dir); "
        "weight-30 = single link (50 GB/s/dir).",
        ha="right", va="bottom", fontsize=9,
        color="dimgray", style="italic",
    )

    fig.legend(loc="upper left", bbox_to_anchor=(0.04, 0.88), fontsize=13)
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = os.path.join(OUTPUT_DIR, "p2p_bandwidth_lumi.pdf")
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
