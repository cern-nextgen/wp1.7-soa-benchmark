import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

values = {
    "runtime_mean": {
        "label": "Mean Runtime",
        "colormap": "RdYlGn_r",
        "ymax": {"im": 710, "stcl": 250, "nbody": 580},
    },
    "runtime_stddev": {
        "label": "Standard Deviation of Runtime over 5 runs",
        "colormap": "RdYlGn_r",
    },
    "instructions": {
        "label": "Number of Retired Instructions",
        "colormap": "Blues",
        "ymax": {"im": 2.6e10, "stcl": 8e9, "nbody": 1.78e10},
    },
    "fp_arith_inst_retired.scalar": {
        "label": "Number of Scalar Instructions",
        "colormap": "RdYlGn_r",
        "ymax": {"im": 3.7e9, "stcl": 1.67e9, "nbody": 1.38e10},
    },
    "fp_arith_inst_retired.vector": {
        "label": "Number of Vector Instructions",
        "colormap": "RdYlGn",
        "ymax": {"im": 3.4e9, "stcl": 1.25e9, "nbody": 0.9e5},
    },
    "L1-icache-load-misses": {
        "label": "L1 Instruction Cache Load Misses",
        "colormap": "RdYlGn_r",
        "ymax": {"im": 6.8e7, "stcl": 3.8e7, "nbody": 1.3e6},
    },
    "L1-dcache-loads": {
        "label": "L1 Data Cache Loads",
        "colormap": "Blues",
        "ymax": {"im": 6.4e9, "stcl": 2.1e9, "nbody": 4.8e9},
    },
    "L1-dcache-load-misses": {
        "label": "L1 Data Cache Load Misses",
        "colormap": "RdYlGn_r",
        "ymax": {"im": 3.7e8, "stcl": 4.9e8, "nbody": 6.4e8},
    },
    "cache-references": {
        "label": "L2 Cache References",
        "colormap": "Blues",
        "ymax": {"im": 6.1e8, "stcl": 8.2e8, "nbody": 1.6e9},
    },
    "cache-misses": {
        "label": "L2 Cache Misses",
        "colormap": "RdYlGn_r",
        "ymax": {"im": 6.2e8, "stcl": 7.2e8, "nbody": 1.3e6},
    },
    "mem_inst_retired.all_loads": {
        "label": "Memory Loads Retired",
        "colormap": "Blues",
        "ymax": {"im": 6.3e9, "stcl": 2.1e9, "nbody": 4.8e9},
    },
    "alignment-faults": {
        "label": "Alignment Faults",
        "colormap": "RdYlGn_r",
        "ymax": {"im": 0.1, "stcl": 0.1, "nbody": 0.1},
    },
    "branch-misses": {
        "label": "Branch Misses",
        "colormap": "RdYlGn_r",
        "ymax": {"im": 2.8e7, "stcl": 3.3e6, "nbody": 3.4e5},
    },
    "branch-instructions": {
        "label": "Branch Instructions",
        "colormap": "Blues",
        "ymax": {"im": 4.8e9, "stcl": 1.26e9, "nbody": 1.2e9},
    },
    "bus-cycles": {
        "label": "Bus Cycles",
        "colormap": "Blues",
        "ymax": {"im": 1.1e8, "stcl": 5.8e7, "nbody": 3.6e7},
    },
    "cpu-cycles": {
        "label": "CPU Cycles",
        "colormap": "Blues",
        "ymax": {"im": 2.3e10, "stcl": 1.3e10, "nbody": 7.4e9},
    },
    "cycle_activity.stalls_l1d_miss": {
        "label": "Cycles Stalled on L1D Miss",
        "colormap": "RdYlGn_r",
        "ymax": {"im": 2.7e9, "stcl": 5.8e9, "nbody": 1.3e9},
    },
    "major-faults": {
        "label": "Major Faults",
        "colormap": "RdYlGn_r",
        "ymax": {"im": 0.1, "stcl": 0.1, "nbody": 0.1},
    },
    "minor-faults": {
        "label": "Minor Faults",
        "colormap": "RdYlGn_r",
        "ymax": {"im": 2.1e6, "stcl": 8.9e5, "nbody": 1250},
    },
    "frontend_retired.latency_ge_1": {
        "label": "Retired Instructions with Frontend Latency >= 1",
        "colormap": "RdYlGn_r",
        "ymax": {"im": 7.3e8, "stcl": 3.5e8, "nbody": 2.6e6},
    },
    "resource_stalls.any": {
        "label": "Resource Stalls",
        "colormap": "RdYlGn_r",
        "ymax": {"im": 9e9, "stcl": 7.5e9, "nbody": 3.8e9},
    },
}

derived_values = {
    "l1-cache-ratio": {
        "label": "L1D Cache Miss Ratio",
        "colormap": "RdYlGn_r",
        "formula": lambda df: df["L1-dcache-load-misses"] / df["L1-dcache-loads"],
        "ymax": 1,
    },
    "vector-utilization": {
        "label": "Vector Utilization",
        "colormap": "RdYlGn",
        "formula": lambda df: df["fp_arith_inst_retired.vector"]
        / (df["fp_arith_inst_retired.vector"] + df["fp_arith_inst_retired.scalar"]),
        "ymax": 1,
    },
    "branch-miss-ratio": {
        "label": "Branch Miss Ratio",
        "colormap": "RdYlGn_r",
        "formula": lambda df: df["branch-misses"] / df["branch-instructions"],
        "ymax": 1,
    },
    "instructions-per-cycle": {
        "label": "Instructions Per Cycle",
        "colormap": "RdYlGn",
        "formula": lambda df: df.get(
            "instructions",
            (df["fp_arith_inst_retired.vector"] + df["fp_arith_inst_retired.scalar"]), # fallback if instructions not available
        )
        / df["cpu-cycles"],
        "ymax": 4,
    },
}


def plot_nmembers_heatmaps(df, app, scaled=False):
    for layout in ["aos", "soa"]:
        df_layout = df[df["version"] == f"{layout}_manual"].copy()
        plt.figure(figsize=(30, 17))

        n_values = len(df.columns) - 3
        rows = max(1, int(np.floor(np.log2(n_values)) + 1))
        cols = max(len(derived_values), int(np.ceil(np.log2(n_values))))

        for i, k in enumerate(df.columns[3:]):
            v = values[k]

            plt.subplot(rows, cols, i + 1)
            pivot = df_layout.pivot(index="before", columns="after", values=k)
            plt.imshow(pivot, cmap=v["colormap"], aspect="auto", origin="lower")
            plt.colorbar(label=v["label"])
            plt.xlabel(r"Number of double data members $\bf{after}$")
            plt.ylabel(r"Number of double data members $\bf{before}$")
            plt.title(f"{v['label']}")
            plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
            plt.yticks(np.arange(len(pivot.index)), pivot.index)
            if scaled and "ymax" in v and app in v["ymax"]:
                plt.clim([0, v["ymax"][app]])
            else:
                plt.clim(vmin=0)
            plt.tight_layout()

        for i, (k, v) in enumerate(zip(derived_values.keys(), derived_values.values())):
            plt.subplot(rows, cols, i + n_values + 1)
            df_layout[k] = v["formula"](df_layout)
            pivot = df_layout.pivot(index="before", columns="after", values=k)
            plt.imshow(pivot, cmap=v["colormap"], aspect="auto", origin="lower")
            plt.colorbar(label=v["label"])
            plt.xlabel(r"Number of double data members $\bf{after}$")
            plt.ylabel(r"Number of double data members $\bf{before}$")
            plt.title(f"{v['label']}")
            plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
            plt.yticks(np.arange(len(pivot.index)), pivot.index)
            if scaled:
                plt.clim(0, v["ymax"])
            else:
                plt.clim(vmin=0)
            plt.tight_layout()

        plt.savefig(
            f"images/heatmap_{layout}_nmembers_{app}{'_scaled' if scaled else ''}.png",
            bbox_inches="tight",
        )
        plt.close()


def plot_stride_lines(app):
    plt.figure(figsize=(25, 15))
    plt.suptitle(f"Invariant Mass with different Loop Strides", y=1.02, fontsize=16)

    suffix = f"{app}"
    if not os.path.exists(f"perf_output_{suffix}.csv"):
        print(f"File perf_output_{suffix}.csv does not exist. Skipping...")
        return

    with open(f"perf_output_{suffix}.csv", "r") as f:
        df = pd.read_csv(f)

    for layout in ["aos", "soa"]:
        df_layout = df[df["version"] == f"{layout}_manual"].copy()

        n_values = len(df.columns) - 2
        rows = max(1, int(np.floor(np.log2(n_values)) + 1))
        cols = max(len(derived_values), int(np.ceil(np.log2(n_values))))

        for i, k in enumerate(df.columns[2:]):
            v = values[k]

            plt.subplot(rows, cols, i + 1)
            plt.plot(
                df_layout["stride"], df_layout[k], marker="o", label=layout.upper()
            )
            plt.xlabel("Loop stride")
            plt.ylabel(v["label"])
            plt.legend()
            if layout == "soa":
                plt.ylim(bottom=0, top=None)

        # derived values
        for i, (k, v) in enumerate(zip(derived_values.keys(), derived_values.values())):
            df_layout[k] = v["formula"](df_layout)
            plt.subplot(rows, cols, i + n_values + 1)
            plt.plot(
                df_layout["stride"], df_layout[k], marker="o", label=layout.upper()
            )
            plt.xlabel("Loop stride")
            plt.ylabel(v["label"])
            plt.legend()
            plt.ylim(bottom=0, top=v["ymax"])

    plt.tight_layout()
    plt.savefig(f"images/lines_{app}.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for app in ["stcl", "im", "nbody"]:
        for scaled in [False, True]:
            suffix = f"nmembers_{app}"
            if not os.path.exists(f"perf_output_{suffix}.csv"):
                print(f"File perf_output_{suffix}.csv does not exist. Skipping...")
                continue

            with open(f"perf_output_{suffix}.csv", "r") as f:
                df = pd.read_csv(f)

            plot_nmembers_heatmaps(df, app, scaled)

    app = "stride_im"
    plot_stride_lines(app)
