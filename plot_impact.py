import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

values = {
    "runtime_mean": {
        "label": "Mean Runtime",
        "colormap": "RdYlGn_r",
        "ymax": {
            "im": 850,
            "stcl": 1.5,
            "nbody": 450
        },
    },
    "runtime_stddev": {
        "label": "Standard Deviation of Runtime",
        "colormap": "RdYlGn_r",
    },
    "fp_arith_inst_retired.scalar": {
        "label": "Number of Scalar Instructions",
        "colormap": "RdYlGn_r",
        "ymax": {
            "im": 3.8e9,
            "stcl": 4.1e9,
            "nbody": 9e9
        },
    },
    "fp_arith_inst_retired.vector": {
        "label": "Number of Vector Instructions",
        "colormap": "RdYlGn",
        "ymax": {
            "im": 3e9,
            "stcl": 7e8,
            "nbody": 1.15e5
        },
    },
    "cache-references": {
        "label": "Cache References",
        "colormap": "Blues",
        "ymax": {
            "im": 7.7e8,
            "stcl": 1.68e9,
            "nbody": 1.5e9
        },
    },
    "cache-misses": {
        "label": "Cache Misses",
        "colormap": "RdYlGn_r",
        "ymax": {
            "im": 6.2e8,
            "stcl": 5.2e8,
            "nbody": 3.7e5
        },
    },
    "alignment-faults": {
        "label": "Alignment Faults",
        "colormap": "RdYlGn_r",
        "ymax": {
            "im": 0.1,
            "stcl": 0.1,
            "nbody": 0.1
        },
    },
    "branch-misses": {
        "label": "Branch Misses",
        "colormap": "RdYlGn_r",
        "ymax": {
            "im": 3.1e7,
            "stcl": 1.85e6,
            "nbody": 2.3e5
        },
    },
    "branch-instructions": {
        "label": "Branch Instructions",
        "colormap": "Blues",
        "ymax": {
            "im": 5e9,
            "stcl": 1.3e8,
            "nbody": 8e8
        },
    },
    "bus-cycles": {
        "label": "Bus Cycles",
        "colormap": "Blues",
        "ymax": {
            "im": 1.15e8,
            "stcl": 2.87e7,
            "nbody": 3.1e7
        },
    },
    "cpu-cycles": {
        "label": "CPU Cycles",
        "colormap": "Blues",
        "ymax": {
            "im": 2.5e10,
            "stcl": 6.1e9,
            "nbody": 6.3e9
        },
    },
    "major-faults": {
        "label": "Major Faults",
        "colormap": "RdYlGn_r",
        "ymax": {
            "im": 0.1,
            "stcl": 0.1,
            "nbody": 0.1
        },
    },
    "minor-faults": {
        "label": "Minor Faults",
        "colormap": "RdYlGn_r",
        "ymax": {
            "im": 2.1e6,
            "stcl": 1.1e5,
            "nbody": 820
        },
    },
}

derived_values = {
    "cache-ratio": {
        "label": "Cache Miss Ratio",
        "colormap": "RdYlGn_r",
        "formula": lambda df: df["cache-misses"] / df["cache-references"],
    },
    "vector-utilization": {
        "label": "Vector Utilization",
        "colormap": "RdYlGn",
        "formula": lambda df: df["fp_arith_inst_retired.vector"]
        / (df["fp_arith_inst_retired.vector"] + df["fp_arith_inst_retired.scalar"]),
    },
    "branch-miss-ratio": {
        "label": "Branch Miss Ratio",
        "colormap": "RdYlGn_r",
        "formula": lambda df: df["branch-misses"] / df["branch-instructions"],
    },
    "cycles-per-instruction": {
        "label": "Cycles Per Instruction",
        "colormap": "RdYlGn_r",
        "formula": lambda df: df["cpu-cycles"]
        / (df["fp_arith_inst_retired.vector"] + df["fp_arith_inst_retired.scalar"]),
        "ymax": {
            "im": 20,
            "stcl": 20,
            "nbody": 1,
            "stride_im": 25
        }
    },
}


def plot_nmembers_heatmaps(df, layout, app, scaled=False):
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
        if scaled and "ymax" in v and app in v["ymax"]:
            plt.clim([0, v["ymax"][app]])
        else:
            plt.clim(vmin=0)
        plt.tight_layout()

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
        if scaled:
            if "ymax" in v and app in v["ymax"]:
                plt.clim(0, v["ymax"][app])
            else:
                plt.clim(0, 1)
        else:
            plt.clim(vmin=0)
        plt.tight_layout()

    plt.savefig(f"images/heatmap_{layout}_nmembers_{app}{'_scaled' if scaled else ''}.png", bbox_inches="tight")
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
            if layout == "soa": plt.ylim(bottom=0, top=None)

        # derived values
        for i, (k, v) in enumerate(zip(derived_values.keys(), derived_values.values())):
            df[k] = v["formula"](df)
            plt.subplot(4, 5, i + 16)
            plt.plot(df["stride"], df[k], marker="o", label=layout.upper())
            plt.xlabel("Loop stride")
            plt.ylabel(v["label"])
            plt.legend()
            if layout == "soa": plt.ylim(bottom=0, top=v.get("ymax", {}).get(app, 1))


    plt.tight_layout()
    plt.savefig(f"images/lines_{app}.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    for app in ["stcl", "im", "nbody"]:
        for layout in ["aos", "soa"]:
            for scaled in [False, True]:
                suffix = f"nmembers_{app}_{layout}"
                if not os.path.exists(f"perf_output_{suffix}.csv"):
                    print(f"File perf_output_{suffix}.csv does not exist. Skipping...")
                    continue

                with open(f"perf_output_{suffix}.csv", "r") as f:
                    df = pd.read_csv(f)

                plot_nmembers_heatmaps(df, layout, app, scaled)

    app = "stride_im"
    plot_stride_lines(app)
