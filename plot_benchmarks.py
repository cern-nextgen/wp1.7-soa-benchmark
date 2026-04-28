import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import json
import math
import glob

def read_data(filename):
    """
    Reads the Google Benchmark data from a JSON file.
    Returns (DataFrame, context_name) where context_name comes from context.name
    in the JSON, falling back to the filename stem.
    """
    import pandas as pd
    stem = os.path.splitext(os.path.basename(filename))[0]
    with open(filename, "r") as read_file:
        data = json.load(read_file)
        name = data.get("context", {}).get("name", stem)
        df = pd.DataFrame.from_dict(data["benchmarks"]).astype({"real_time": float})
        df = df[df["run_type"] == "aggregate"]
        df["benchmark"] = df["name"].apply(lambda x: x.split('/')[1].split('_')[1])
        return df, name

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

        for df, label in all_data:
            df_mean = df[(df["benchmark"] == benchmark) & (df["aggregate_name"] == "mean")]
            df_std = df[(df["benchmark"] == benchmark) & (df["aggregate_name"] == "stddev")]

            if df_mean.empty:
                continue

            ax.errorbar(df_mean['n_elem'], df_mean['real_time'], yerr=df_std["real_time"],
                        ls="-", marker="o", label=label)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("--libraries", nargs="+", metavar="FILE",
                        help="JSON files to plot (default: all *.json in output_dir)")
    parser.add_argument("--benchmarks", nargs="+", metavar="NAME",
                        help="Benchmark names to include (e.g. nbody CPUEasyRW)")
    args = parser.parse_args()

    print("Plotting the benchmark results...")

    json_files = sorted(args.libraries) if args.libraries else sorted(glob.glob(os.path.join(args.output_dir, "*.json")))

    all_results = []
    for f in json_files:
        df, name = read_data(f)
        if args.benchmarks:
            df = df[df["benchmark"].isin(args.benchmarks)]
        if not df.empty:
            all_results.append((df, name))

    if not all_results:
        print("No JSON benchmark result files found in", args.output_dir)
        raise SystemExit(1)

    plot_per_benchmark(all_results, args.output_dir)

    max_y = 10 ** math.ceil(math.log10(max([df[df["aggregate_name"] == "mean"]["real_time"].max() for (df, _) in all_results])))
    min_y = 10 ** math.floor(math.log10(min([df[df["aggregate_name"] == "mean"]["real_time"].min() for (df, _) in all_results])))
    for (df, label) in all_results:
        plot_per_version(df, label, args.output_dir, min_y, max_y)
