import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import glob



values = {
    "runtime_mean": {"label": "Mean Runtime (ms)", "colormap": "RdYlGn_r", "ymax": True},
    "runtime_stddev": {
        "label": "Standard Deviation of Runtime over 5 runs",
        "colormap": "RdYlGn_r",
        "ymax": False,
    },
    "instructions": {
        "label": "Number of Retired Instructions",
        "colormap": "Blues",
        "ymax": True,
    },
    "fp_arith_inst_retired.scalar": {
    "label": "Number of Scalar Instructions",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "fp_ops_retired_by_type.scalar_all": {
        "label": "Number of Scalar Instructions",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "fp_arith_inst_retired.vector": {
        "label": "Number of Vector Instructions",
        "colormap": "RdYlGn",
        "ymax": True,
    },
    "fp_ops_retired_by_type.vector_all": {
        "label": "Number of Vector Instructions",
        "colormap": "RdYlGn",
        "ymax": True,
    },
    "L1-icache-load-misses": {
        "label": "L1 Instruction Cache Load Misses",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "L1-dcache-loads": {
        "label": "L1 Data Cache Loads",
        "colormap": "Blues",
        "ymax": True,
    },
    "L1-dcache-load-misses": {
        "label": "L1 Data Cache Load Misses",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "cache-references": {
        "label": "L2 Cache References",
        "colormap": "Blues",
        "ymax": True,
    },
    "cache-misses": {"label": "L2 Cache Misses", "colormap": "RdYlGn_r", "ymax": True},
    "mem_inst_retired.all_loads": {
        "label": "Memory Loads Retired",
        "colormap": "Blues",
        "ymax": True,
    },
    "mem_inst_retired.any": {
        "label": "Retired Memory Instructions",
        "colormap": "Blues",
        "ymax": True,
    },
    "ls_dispatch.ld_dispatch": {
        "label": "Dispatched Load Operations",
        "colormap": "Blues",
        "ymax": True,
    },
    "ls_dispatch.ld_st_dispatch": {
        "label": "Dispatched Load-Store Operations",
        "colormap": "Blues",
        "ymax": True,
    },
    "alignment-faults": {
        "label": "Alignment Faults",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "branch-misses": {
        "label": "Branch Misses",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "branch-instructions": {
        "label": "Branch Instructions",
        "colormap": "Blues",
        "ymax": True,
    },
    "bus-cycles": {
        "label": "Bus Cycles",
        "colormap": "Blues",
        "ymax": True,
    },
    "cpu-cycles": {
        "label": "CPU Cycles",
        "colormap": "Blues",
        "ymax": True,
    },
    "cycle_activity.stalls_l1d_miss": {
        "label": "Cycles Stalled on L1D Miss",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "cycle_activity.stalls_mem_any": {
        "label": "Cycles Stalled on Memory Access",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "ex_no_retire.load_not_complete": {
        "label": "Cycles Stalled on Load Not Complete",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "major-faults": {
        "label": "Major Faults",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "minor-faults": {
        "label": "Minor Faults",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "frontend_retired.latency_ge_1": {
        "label": "Retired Instructions with Frontend Latency >= 1",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "stalled-cycles-frontend": {
        "label": "Stalled Cycles Frontend",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "resource_stalls.any": {
        "label": "Resource Stalls",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
    "de_no_dispatch_per_slot.backend_stalls": {
        "label": "Backend Stalls",
        "colormap": "RdYlGn_r",
        "ymax": True,
    },
}


def vector_utilization(df):
    if "fp_arith_inst_retired.vector" in df and "fp_arith_inst_retired.scalar" in df:
        return df["fp_arith_inst_retired.vector"] / (
            df["fp_arith_inst_retired.vector"] + df["fp_arith_inst_retired.scalar"]
        )
    elif (
        "fp_ops_retired_by_type.vector_all" in df
        and "fp_ops_retired_by_type.scalar_all" in df
    ):
        return df["fp_ops_retired_by_type.vector_all"] / (
            df["fp_ops_retired_by_type.vector_all"]
            + df["fp_ops_retired_by_type.scalar_all"]
        )
    else:
        raise ValueError("No suitable columns for vector utilization")


def instructions_per_cycle(df):
    if "instructions" in df:
        return df["instructions"] / df["cpu-cycles"]
    elif "fp_arith_inst_retired.vector" in df and "fp_arith_inst_retired.scalar" in df:
        return (
            df["fp_arith_inst_retired.vector"] + df["fp_arith_inst_retired.scalar"]
        ) / df["cpu-cycles"]
    else:
        raise ValueError("No suitable columns for instructions per cycle")

def arithmetic_intensity(df):
    if "fp_arith_inst_retired.vector" in df and "fp_arith_inst_retired.scalar" in df:
        arith = df["fp_arith_inst_retired.vector"] + df["fp_arith_inst_retired.scalar"]
    elif "fp_ops_retired_by_type.vector_all" in df and "fp_ops_retired_by_type.scalar_all" in df:
        arith = df["fp_ops_retired_by_type.vector_all"] + df["fp_ops_retired_by_type.scalar_all"]
    else:
        raise ValueError("No suitable columns for arithmetic intensity", df.columns)

    if "mem_inst_retired.all_loads" in df:
        mem = df["mem_inst_retired.all_loads"]
    elif "mem_inst_retired.any" in df:
        mem = df["mem_inst_retired.any"]
    elif "ls_dispatch.ld_dispatch" in df:
        mem = df["ls_dispatch.ld_dispatch"]
    elif "ls_dispatch.ld_st_dispatch" in df:
        mem = df["ls_dispatch.ld_st_dispatch"]
    else:
        raise ValueError("No suitable columns for memory operations", df.columns)

    return arith / mem

def arithmetic_intensity_label(df):
    if "mem_inst_retired.all_loads" in df or "ls_dispatch.ld_dispatch" in df:
        return "Arithmetic Intensity (FLOP/Mem Loads)"
    elif "mem_inst_retired.any" in df or "ls_dispatch.ld_st_dispatch" in df:
        return "Arithmetic Intensity (FLOP/Mem Instructions)"
    else:
        raise ValueError("No suitable columns for AI label", df.columns)

derived_values = {
    "l1-cache-ratio": {
        "label": "L1D Cache Miss Ratio",
        "colormap": "RdYlGn_r",
        "formula": lambda df: df["L1-dcache-load-misses"] / df["L1-dcache-loads"],
        "ymax": 1,
    },
    # "vector-utilization": {
    #     "label": "Vector Utilization",
    #     "colormap": "RdYlGn",
    #     "formula": lambda df: vector_utilization(df),
    #     "ymax": 1,
    # },
    "arithmetic-intensity": {
        "label": lambda df: arithmetic_intensity_label(df),
        "colormap": "RdYlGn",
        "formula": lambda df: arithmetic_intensity(df),
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
        "formula": lambda df: instructions_per_cycle(df),
        "ymax": 4,
    },
}


def plot_nmembers_heatmaps(df, app, scaled=False):
    n_values = len(df.columns) - 3
    rows = max(1, int(np.ceil(np.log2(n_values)) + 1))
    cols = max(len(derived_values), int(np.floor(np.log2(n_values))))
    print(
        f"Plotting {n_values + len(derived_values)} values in {rows} rows and {cols} columns"
    )

    ymax = np.zeros((2, n_values + len(derived_values)))
    imgs = []
    figs = []
    for il, layout in enumerate(["aos", "soa"]):
        df_layout = df[df["version"] == f"{layout}_manual"].copy()
        figs.append([layout, plt.figure(figsize=(30, 17))])

        ims = []
        for i, k in enumerate(df.columns[3:]):
            v = values[k]

            plt.subplot(rows, cols, i + 1)
            pivot = df_layout.pivot(index="before", columns="after", values=k)
            ims.append(
                plt.imshow(pivot, cmap=v["colormap"], aspect="auto", origin="lower")
            )
            plt.colorbar(label=v["label"])
            plt.xlabel(r"Number of double data members $\bf{after}$")
            plt.ylabel(r"Number of double data members $\bf{before}$")
            plt.title(f"{v['label']}")
            plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
            plt.yticks(np.arange(len(pivot.index)), pivot.index)
            if v["ymax"]:
                ymax[il][i] = np.max(pivot.values)
            plt.tight_layout()

        for i, (k, v) in enumerate(zip(derived_values.keys(), derived_values.values())):
            plt.subplot(rows, cols, i + 17)
            df_layout[k] = v["formula"](df_layout)
            pivot = df_layout.pivot(index="before", columns="after", values=k)
            ims.append(
                plt.imshow(pivot, cmap=v["colormap"], aspect="auto", origin="lower")
            )
            plt.colorbar(label=v["label"])
            plt.xlabel(r"Number of double data members $\bf{after}$")
            plt.ylabel(r"Number of double data members $\bf{before}$")
            plt.title(f"{v['label']}" if not v == "arithmetic-intensity" else v['label'](df_layout))
            plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
            plt.yticks(np.arange(len(pivot.index)), pivot.index)
            if "ymax" in v:
                ymax[il][i + n_values] = v["ymax"]
            plt.tight_layout()

        imgs.append(ims)

    for ims_l in imgs:
        for ia, im in enumerate(ims_l):
            max_val = np.max(ymax[:, ia])
            im.set_clim(0, max_val) if scaled and max_val > 0 else im.set_clim(vmin=0)

    for layout, fig in figs:
        for fmt in formats:
            fig.savefig(
                f"{output_dir}/heatmap_{layout}_nmembers_{app}{'_scaled' if scaled else ''}.{fmt}",
                bbox_inches="tight",
            )
        plt.close(fig)


def plot_stride_lines(app):
    plt.figure(figsize=(25, 15))
    plt.suptitle(f"Invariant Mass with different Loop Strides", y=1.02, fontsize=16)

    suffix = f"{app}"
    if not os.path.exists(f"{input_dir}/perf_output_{suffix}.csv"):
        print(f"File {input_dir}/perf_output_{suffix}.csv does not exist. Skipping...")
        return

    with open(f"{input_dir}/perf_output_{suffix}.csv", "r") as f:
        df = pd.read_csv(f)

    for layout in ["aos", "soa"]:
        df_layout = df[df["version"] == f"{layout}_manual"].copy()

        n_values = len(df.columns) - 2
        rows = max(1, int(np.ceil(np.log2(n_values)) + 1))
        cols = max(len(derived_values), int(np.floor(np.log2(n_values))))

        for i, k in enumerate(df.columns[2:]):
            v = values[k]

            plt.subplot(rows, cols, i + 1)
            plt.plot(
                df_layout["stride"], df_layout[k], marker="o", label=layout.upper()
            )

            if v['label'] == 'Mean Runtime (ms)':
                plt.yscale('symlog')

            plt.xlabel("Loop stride")
            plt.ylabel(v["label"])
            plt.legend()
            if layout == "soa":
                plt.ylim(bottom=0, top=None)

        # derived values
        for i, (k, v) in enumerate(zip(derived_values.keys(), derived_values.values())):
            df_layout[k] = v["formula"](df_layout)
            plt.subplot(rows, cols, i + 17)
            plt.plot(
                df_layout["stride"], df_layout[k], marker="o", label=layout.upper()
            )
            plt.xlabel("Loop stride")
            plt.ylabel(v["label"])
            plt.legend()
            plt.ylim(bottom=0, top=v["ymax"])

    plt.tight_layout()
    for fmt in formats:
        plt.savefig(f"{output_dir}/lines_{app}.{fmt}", bbox_inches="tight")
    plt.close()


# input_dir  = "/home/jollychen/soa-benchmark/wp1.7-soa-benchmark"
input_dir  = "/data/soa-benchmark-results/251001/ngt"
output_dir = "/data/soa-benchmark-results/251001/ngt"
formats = ["png", "pdf"]

if __name__ == "__main__":
    nmembers_files = glob.glob(os.path.join(input_dir, "*nmembers*.csv"))
    for nm_file in nmembers_files:
        print(f"Processing file: {nm_file}")
        app = os.path.splitext(os.path.basename(nm_file))[0].split("perf_output_")[1]
        plot_nmembers_heatmaps(pd.read_csv(nm_file), app, scaled=False)
        plot_nmembers_heatmaps(pd.read_csv(nm_file), app, scaled=True)

    stride_files = glob.glob(os.path.join(input_dir, "*stride*im*.csv"))
    for s_file in stride_files:
        print(f"Processing file: {s_file}")
        app = os.path.splitext(os.path.basename(s_file))[0].split("perf_output_")[1]
        plot_stride_lines(app)
