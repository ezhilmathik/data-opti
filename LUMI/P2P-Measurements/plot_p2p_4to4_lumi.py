#!/usr/bin/env python3
"""
plot_p2p_4to4_lumi.py  —  Bandwidth vs N for version v4_4 on LUMI
                           (AMD MI250X, inter-card xGMI, one GCD per physical card)

Automatically finds all results_*/timings.csv files in the current directory
and averages bandwidth across them.
Output is written to plots_final/.

LUMI hardware context (source: docs.lumi-supercomputer.eu/hardware/lumig):
  - 4 physical MI250X cards per node (ROCR_VISIBLE_DEVICES=0,2,4,6)
  - xGMI ring topology: weight-15 pairs (double link, 100 GB/s/dir)
                        weight-30 pairs (single link,  50 GB/s/dir)
  - All-to-all theoretical peak is ring-constrained:
      measured hardware ceiling from smoke test: ~150 GB/s aggregate
  - No NVLink; no NVSwitch; all paths through xGMI ring fabric
"""

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from collections import defaultdict

CSV_GLOB       = "results_*/timings.csv"
OUTPUT_DIR     = "plots_final"
VERSION_TO_PLOT = "v4_4"

FRAMEWORK_STYLE = {
    "hip":     ("HIP (4 GPUs)",     "steelblue", "o", "-"),
    "omp_off": ("OMP Off (4 GPUs)", "darkorange", "^", "--"),
    # kept for safety if CSVs from other machines are mixed in
    "cuda":    ("CUDA (4 GPUs)",    "royalblue", "s", "-"),
    "acc":     ("OpenACC (4 GPUs)", "seagreen",  "D", ":"),
}

# Ring-topology-constrained measured ceiling (GB/s).
# Derived from full-node smoke test: 12 directed pairs × 64 MB → ~150 GB/s aggregate.
# Theoretical naive upper bound (12 × 50 GB/s single-link) = 600 GB/s — not achievable
# because shared ring links are the bottleneck.
THEORETICAL_BW         = 150   # measured ring ceiling (GB/s)
NAIVE_THEORETICAL_BW   = 600   # 12 × 50 GB/s — shown as reference only


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
            bw_raw    = row.get("bandwidth_gbs", "N/A").strip()
            if not framework or bw_raw == "N/A":
                continue
            try:
                N  = int(row["N"].strip())
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

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.set_xticks(all_N)
    ax1.xaxis.set_major_formatter(FuncFormatter(scientific_notation_exact))
    ax1.set_xlabel("Vector N (type: double)", fontsize=18)
    ax1.grid(True, alpha=0.35)
    ax1.tick_params(axis="x", labelsize=14, rotation=10)
    ax1.tick_params(axis="y", left=False, labelleft=False)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Bandwidth (GB/s)", fontsize=18)
    ax2.tick_params(axis="y", labelsize=16)

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
            Ns_sorted, bws,
            marker=mk, linestyle=ls, color=color,
            label=label, linewidth=1.8, markersize=6,
        )
        for x, y in zip(Ns_sorted, bws):
            ax2.text(x, y + 1.8, f"{y:.1f}", color=color,
                     fontsize=11, ha="center", va="bottom")

    # Ring-constrained measured ceiling
    if THEORETICAL_BW is not None and all_N:
        mid = len(all_N) // 2
        ax2.axhline(THEORETICAL_BW, linestyle="-.", color="dimgray",
                    linewidth=1.2, alpha=0.8)
        ax2.text(
            all_N[mid], THEORETICAL_BW + 3,
            f"xGMI ring ceiling ~{THEORETICAL_BW} GB/s (measured)",
            color="dimgray", ha="center", va="bottom", fontsize=12,
        )

    # Naive theoretical (shown faintly as upper reference)
    if NAIVE_THEORETICAL_BW is not None and all_N:
        ax2.axhline(NAIVE_THEORETICAL_BW, linestyle=":", color="lightgray",
                    linewidth=1.0, alpha=0.6)
        ax2.text(
            all_N[-1], NAIVE_THEORETICAL_BW + 3,
            f"Naive peak {NAIVE_THEORETICAL_BW} GB/s\n(12 × 50 GB/s, ignores contention)",
            color="lightgray", ha="right", va="bottom", fontsize=10,
        )

    n_runs = len(csv_paths)
    fig.suptitle(
        f"LUMI — P2P All-to-All 4×4 Bandwidth\n"
        f"AMD MI250X · inter-card xGMI ring · ROCR_VISIBLE_DEVICES=0,2,4,6 "
        f"(averaged over {n_runs} run{'s' if n_runs > 1 else ''})",
        fontsize=13,
    )

    # Topology explanation box
    topology_text = (
        "xGMI ring topology (4 physical cards):\n"
        "  weight-15 pairs: double link → 100 GB/s/dir (direct neighbors)\n"
        "  weight-30 pairs: single link →  50 GB/s/dir (diagonal, shared link)\n"
        "Ring contention limits all-to-all aggregate to ~150 GB/s.\n"
        "OMP Off runtime schedules transfers topology-aware; HIP fires all 12 streams naively."
    )
    fig.text(
        0.99, 0.01, topology_text,
        ha="right", va="bottom", fontsize=9,
        color="dimgray", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke",
                  edgecolor="lightgray", alpha=0.8),
    )

    fig.legend(loc="upper left", bbox_to_anchor=(0.04, 0.88), fontsize=13)
    plt.tight_layout(rect=[0, 0.18, 1, 1])

    out_path = os.path.join(OUTPUT_DIR, "p2p_4to4_bandwidth_lumi.pdf")
    plt.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
