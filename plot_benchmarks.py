import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
import os
import json
import math
import glob

LABELS = {
    "aos_manual":       "Manual AoS",
    "soa_manual":       "Manual SoA",
    "soa_wrapper":      "Template Metaprogramming SoA",
    "soa_reflections":  "Reflection SoA",
}

def label_for(stem):
    return LABELS.get(stem, stem)

def read_data(filename):
    """
    Reads the Google Benchmark data from a JSON file and returns a DataFrame.
    """
    import pandas as pd
    with open(filename, "r") as read_file:
        data = json.load(read_file)
        df = pd.DataFrame.from_dict(data["benchmarks"]).astype({"real_time": float})
        df = df[df["run_type"] == "aggregate"]
        df["benchmark"] = df["name"].apply(lambda x: x.split('/')[1].split('_')[1])
        return df

def plot_per_benchmark(all_data, out_dir):
    """
    Generates one plot per benchmark, comparing all SoA variants.
    """
    all_benchmarks = set()
    for df, _ in all_data:
        all_benchmarks.update(df["benchmark"].unique())

    for benchmark in all_benchmarks:
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111)

        for df, soa_label in all_data:
            df_mean = df[(df["benchmark"] == benchmark) & (df["aggregate_name"] == "mean")]
            df_std = df[(df["benchmark"] == benchmark) & (df["aggregate_name"] == "stddev")]

            if df_mean.empty:
                continue

            ax.errorbar(df_mean['n_elem'], df_mean['real_time'], yerr=df_std["real_time"],
                        ls="-", marker="o", label=soa_label)

        ax.set_title(f'{benchmark}')
        ax.set_xlabel('Number of Elements')
        ax.set_xscale('symlog')
        ax.set_xticks(sorted(df_mean['n_elem'].unique()),
                      labels=["{:g}".format(x) for x in sorted(df_mean['n_elem'].unique())], minor=False)

        ax.set_ylabel(f'Real Time ({df["time_unit"].iloc[0]})')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))

        plt.legend()

        out_file = f'{out_dir}/{benchmark}_comparison.png'
        plt.savefig(out_file)
        print(f"Saved to {out_file}")


def plot_per_version(df, title, out_dir, min_y=-0.000001, max_y=1000):
    """
    Generates one plot per version, comparing all benchmarks.
    """
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)

    for bm in df["benchmark"].unique():
        df_mean = df[(df["benchmark"] == bm) & (df["aggregate_name"] == "mean")]
        df_std = df[(df["benchmark"] == bm) & (df["aggregate_name"] == "stddev")]
        ax.errorbar(df_mean['n_elem'].astype(int), df_mean['real_time'], yerr=df_std["real_time"],
                    ls="-", marker="o", label=bm)

    ax.set_title(title)
    ax.set_xlabel('Number of Elements')
    ax.set_xscale('symlog')
    ax.set_xticks(df_mean['n_elem'].unique(), labels=["{:g}".format(x) for x in df_mean['n_elem'].unique()], minor=False)

    ax.set_ylabel(f'Real Time ({df["time_unit"].iloc[0]})')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    plt.ylim(min_y, max_y)

    plt.legend()

    out_file = f'{out_dir}/{title.replace(" ", "_")}.png'
    plt.savefig(out_file)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    print("Plotting the benchmark results...")

    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
        json_files = sorted(glob.glob(os.path.join(output_dir, "*.json")))

        all_results = []
        for f in json_files:
            stem = os.path.splitext(os.path.basename(f))[0]
            df = read_data(f)
            all_results.append((df, label_for(stem)))

        if not all_results:
            print("No JSON benchmark result files found in", output_dir)
            sys.exit(1)

        plot_per_benchmark(all_results, output_dir)

        max_y = 10 ** math.ceil(math.log10(max([df[df["aggregate_name"] == "mean"]["real_time"].max() for (df, _) in all_results])))
        min_y = 10 ** math.floor(math.log10(min([df[df["aggregate_name"] == "mean"]["real_time"].min() for (df, _) in all_results])))
        for (df, label) in all_results:
            plot_per_version(df, label, output_dir, min_y, max_y)
    else:
        print("python plot_benchmarks.py <output_dir>")
