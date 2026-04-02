#!/usr/bin/env python3
"""
plot_p2p_4to4.py  —  Bandwidth vs N for version v4_4
Automatically finds all results_*/timings.csv files in the current directory
and averages bandwidth across them.
Output is written to plots_final/.
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import defaultdict

CSV_GLOB = "results_*/timings.csv"
OUTPUT_DIR = "plots_final"
VERSION_TO_PLOT = "v4_4"

FRAMEWORK_STYLE = {
    "cuda":    ("CUDA (4 GPUs)",    "blue",   "o", "-"),
    "hip":     ("HIP (4 GPUs)",     "blue",   "o", "-"),
    "acc":     ("OpenACC (4 GPUs)", "green",  "s", "--"),
    "omp_off": ("OMP Off (4 GPUs)", "orange", "^", ":"),
}

THEORETICAL_BW = 1200  # GB/s; set to None to skip


def scientific_notation_exact(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coefficient = x / (10 ** exponent)
    return f"${coefficient:.2f} \\times 10^{{{exponent}}}$"


def load_csv(path, version):
    data = defaultdict(list)
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("framework", "").strip() == "framework":
                continue
            if row.get("version", "").strip() != version:
                continue
            if row.get("status", "ok").strip() != "ok":
                continue

            framework = row.get("framework", "").strip()
            bw_raw = row.get("bandwidth_gbs", "N/A").strip()
            if not framework or bw_raw == "N/A":
                continue

            try:
                N = int(row["N"].strip())
                bw = float(bw_raw)
            except (ValueError, KeyError):
                continue

            data[(framework, N)].append(bw)

    return data


def load_and_average(csv_paths, version):
    combined = defaultdict(list)
    for path in csv_paths:
        for (fw, N), bws in load_csv(path, version).items():
            combined[(fw, N)].extend(bws)

    averaged = defaultdict(dict)
    for (fw, N), bws in combined.items():
        averaged[fw][N] = float(np.mean(bws))

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
    data = load_and_average(csv_paths, VERSION_TO_PLOT)

    all_N = sorted({n for nd in data.values() for n in nd})
    if not all_N:
        print(f"No valid bandwidth data found for {VERSION_TO_PLOT} — nothing to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xticks(all_N)
    ax1.xaxis.set_major_formatter(FuncFormatter(scientific_notation_exact))
    ax1.set_xlabel("Vector N (type: double)", fontsize=18)
    ax1.grid(True, alpha=0.4)
    ax1.tick_params(axis="x", labelsize=14, rotation=10)
    ax1.tick_params(axis="y", left=False, labelleft=False)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Bandwidth (GB/s)", fontsize=18)
    ax2.tick_params(axis="y", labelsize=18)

    for fw in sorted(data.keys()):
        nd = data.get(fw, {})
        if not nd:
            continue

        label, color, mk, ls = FRAMEWORK_STYLE.get(
            fw, (f"{fw} (4 GPUs)", "black", "o", "-")
        )

        Ns_sorted = sorted(nd)
        bws = [nd[n] for n in Ns_sorted]

        ax2.plot(
            Ns_sorted,
            bws,
            marker=mk,
            linestyle=ls,
            color=color,
            label=label,
            linewidth=1.5,
            markersize=6,
        )

        for x, y in zip(Ns_sorted, bws):
            ax2.text(x, y + 1.5, f"{y:.1f}", color=color, fontsize=12, ha="center")

    if THEORETICAL_BW is not None and all_N:
        mid = len(all_N) // 2
        ax2.plot(
            all_N,
            [THEORETICAL_BW] * len(all_N),
            linestyle="--",
            color="black",
            linewidth=1,
            alpha=0.7,
        )
        ax2.text(
            all_N[mid],
            THEORETICAL_BW + 5,
            f"Theor. BW {THEORETICAL_BW} GB/s",
            color="black",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    n = len(csv_paths)
    fig.suptitle(
        f"P2P 4-to-4 Bandwidth (averaged over {n} run{'s' if n > 1 else ''})",
        fontsize=14,
    )
    fig.legend(loc="upper left", bbox_to_anchor=(0.04, 0.95), fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "p2p_4to4_bandwidth.pdf")
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
