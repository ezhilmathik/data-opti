#!/usr/bin/env python3
"""
plot_p2p.py  —  Bandwidth vs N for versions v1, v2, v4
Automatically finds all results_*/timings.csv files in the current directory
and averages bandwidth across them.
Output is written to plots_final/.
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

CSV_GLOB = "results_*/timings.csv"
OUTPUT_DIR = "plots_final"

VERSIONS_TO_PLOT = ["v1", "v2", "v4"]

# Theoretical peak bandwidths (GB/s); set to None to skip
THEORETICAL = {
    "v1": 100,
    "v2": 200,
    "v4": 300,
}

# Prefer these display styles when frameworks are present
FRAMEWORK_LABELS = {
    "cuda": "CUDA",
    "hip": "HIP",
    "acc": "OpenACC",
    "omp_off": "OMP Off",
}

FRAMEWORK_MARKERS = {
    "cuda": "o",
    "hip": "o",
    "acc": "s",
    "omp_off": "^",
}

VERSION_COLOR = {
    "v1": "blue",
    "v2": "green",
    "v4": "red",
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

            version = row.get("version", "").strip()
            framework = row.get("framework", "").strip()
            bw_raw = row.get("bandwidth_gbs", "N/A").strip()

            if not version or not framework or bw_raw == "N/A":
                continue

            try:
                N = int(row["N"].strip())
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

    # Equal-spaced integer positions so uneven N gaps don't distort the plot
    x_pos = list(range(len(all_N)))
    n_to_x = {n: i for i, n in enumerate(all_N)}

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(
        [scientific_notation_exact(n) for n in all_N],
        rotation=10,
        fontsize=18,
    )
    ax1.set_xlabel("Vector N (type: double)", fontsize=18)
    ax1.grid(True)
    ax1.tick_params(axis="y", left=False, labelleft=False)

    ax2 = ax1.twinx()

    # Plot only frameworks that actually exist in the CSVs
    for ver in VERSIONS_TO_PLOT:
        color = VERSION_COLOR.get(ver, "black")
        gpu_count = ver.replace("v", "") + " GPU"
        if gpu_count != "1 GPU":
            gpu_count += "s"

        for fw in available_frameworks:
            nd = data.get((fw, ver), {})
            if not nd:
                continue

            xs = [n_to_x[n] for n in sorted(nd)]
            bws = [nd[n] for n in sorted(nd)]

            fw_label = FRAMEWORK_LABELS.get(fw, fw)
            marker = FRAMEWORK_MARKERS.get(fw, "o")
            label = f"{fw_label} ({gpu_count})"

            ax2.plot(xs, bws, marker=marker, color=color, label=label)

    mid = len(x_pos) // 2
    for ver, theor in THEORETICAL.items():
        if theor is None:
            continue
        color = VERSION_COLOR.get(ver, "black")
        ax2.plot(x_pos, [theor] * len(x_pos), linestyle="--", color=color)
        ax2.text(
            x_pos[mid],
            theor + 0.5,
            f"Theor. BW {theor} GB/s",
            color=color,
            ha="center",
            va="bottom",
            fontsize=14,
        )

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
