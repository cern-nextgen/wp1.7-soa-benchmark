import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

values = {
    "runtime_mean": {"label": "Mean Runtime", "colormap": "RdYlGn_r", "ymax": 600},
    "runtime_stddev": {
        "label": "Standard Deviation of Runtime",
        "colormap": "RdYlGn_r",
        "ymax": 150,
    },
    "fp_arith_inst_retired.scalar": {
        "label": "Number of Scalar Instructions",
        "colormap": "RdYlGn_r",
        "ymax": 4e9,
    },
    "fp_arith_inst_retired.vector": {
        "label": "Number of Vector Instructions",
        "colormap": "RdYlGn",
        "ymax": 4e9,
    },
    "cache-references": {"label": "Cache References", "colormap": "Blues", "ymax": 8e8},
    "cache-misses": {"label": "Cache Misses", "colormap": "RdYlGn_r", "ymax": 7e8},
    "alignment-faults": {
        "label": "Alignment Faults",
        "colormap": "RdYlGn_r",
        "ymax": 0.2,
    },
    "branch-misses": {"label": "Branch Misses", "colormap": "RdYlGn_r", "ymax": 2.3e7},
    "branch-instructions": {
        "label": "Branch Instructions",
        "colormap": "Blues",
        "ymax": 5e9,
    },
    "bus-cycles": {"label": "Bus Cycles", "colormap": "Blues", "ymax": 1.2e8},
    "cpu-cycles": {"label": "CPU Cycles", "colormap": "Blues", "ymax": 2.5e10},
    "major-faults": {"label": "Major Faults", "colormap": "RdYlGn_r", "ymax": 0.2},
    "minor-faults": {"label": "Minor Faults", "colormap": "RdYlGn_r", "ymax": 2.25e6},
}

derived_values = {
    "cache-ratio": {
        "label": "Cache Miss Ratio",
        "colormap": "RdYlGn_r",
        "formula": lambda df: df["cache-misses"] / df["cache-references"],
        "ymax": 1.0,
    },
    "vector-utilization": {
        "label": "Vector Utilization",
        "colormap": "RdYlGn",
        "formula": lambda df: df["fp_arith_inst_retired.vector"]
        / (df["fp_arith_inst_retired.vector"] + df["fp_arith_inst_retired.scalar"]),
        "ymax": 1.0,
    },
    "branch-miss-ratio": {
        "label": "Branch Miss Ratio",
        "colormap": "RdYlGn_r",
        "formula": lambda df: df["branch-misses"] / df["branch-instructions"],
        "ymax": 1.0,
    },
    "cycles-per-instruction": {
        "label": "Cycles Per Instruction",
        "colormap": "RdYlGn_r",
        "formula": lambda df: df["cpu-cycles"]
        / (df["fp_arith_inst_retired.vector"] + df["fp_arith_inst_retired.scalar"]),
        "ymax": 20,
    },
}


def plot_nmembers_heatmaps(df, layout, app):
    plt.figure(figsize=(30, 17))
    for i, (k, v) in enumerate(zip(values.keys(), values.values())):
        plt.subplot(4, 5, i + 1)
        pivot = df.pivot(index="before", columns="after", values=k)
        plt.imshow(pivot, cmap=v["colormap"], aspect="auto", origin="lower")
        plt.colorbar(label=v["label"])
        plt.xlabel(r"Number of double data members $\bf{after}$")
        plt.ylabel(r"Number of double data members $\bf{before}$")
        plt.title(f"{v['label']}")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        # plt.clim([0, v["ymax"]])
        plt.tight_layout()

    # plt.savefig(f"images/heatmap_{layout}_{app}_{k}.png", bbox_inches="tight")
    # plt.savefig(f"images/heatmap_{layout}_nmembers_{app}.png", bbox_inches="tight")
    # plt.cla()
    # plt.close()

    # plt.figure(figsize=(10, 10))
    for i, (k, v) in enumerate(zip(derived_values.keys(), derived_values.values())):
        plt.subplot(4, 5, i + 16)
        df[k] = v["formula"](df)
        pivot = df.pivot(index="before", columns="after", values=k)
        plt.imshow(pivot, cmap=v["colormap"], aspect="auto", origin="lower")
        plt.colorbar(label=v["label"])
        plt.xlabel(r"Number of double data members $\bf{after}$")
        plt.ylabel(r"Number of double data members $\bf{before}$")
        plt.title(f"{v['label']}")
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        # plt.clim(0, v["ymax"])
        plt.tight_layout()

    plt.savefig(f"images/heatmap_{layout}_nmembers_{app}.png", bbox_inches="tight")
    plt.close()


def plot_stride_lines(app):
    plt.figure(figsize=(25, 15))
    plt.suptitle(f"Invariant Mass with different Loop Strides", y=1.02, fontsize=16)

    for layout in ["aos", "soa"]:
        suffix = f"{app}_{layout}"
        if not os.path.exists(f"perf_output_{suffix}.csv"):
            print(f"File perf_output_{suffix}.csv does not exist. Skipping...")
            continue

        with open(f"perf_output_{suffix}.csv", "r") as f:
            df = pd.read_csv(f)

        for i, (k, v) in enumerate(zip(values.keys(), values.values())):
            plt.subplot(4, 5, i + 1)
            plt.plot(df["stride"], df[k], marker="o", label=layout.upper())
            plt.xlabel("Loop stride")
            plt.ylabel(v["label"])
            plt.legend()

        # derived values
        for i, (k, v) in enumerate(zip(derived_values.keys(), derived_values.values())):
            df[k] = v["formula"](df)
            plt.subplot(4, 5, i + 16)
            plt.plot(df["stride"], df[k], marker="o", label=layout.upper())
            plt.xlabel("Loop stride")
            plt.ylabel(v["label"])
            plt.legend()

    plt.tight_layout()
    plt.savefig(f"images/lines_{app}.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for app in ["stcl", "im"]:
        for layout in ["aos", "soa"]:
            suffix = f"nmembers_{app}_{layout}"
            if not os.path.exists(f"perf_output_{suffix}.csv"):
                print(f"File perf_output_{suffix}.csv does not exist. Skipping...")
                continue

            with open(f"perf_output_{suffix}.csv", "r") as f:
                df = pd.read_csv(f)

            plot_nmembers_heatmaps(df, layout, app)

    app = "stride_im"
    plot_stride_lines(app)
