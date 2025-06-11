import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import subprocess
import sys
import os
import json
import math
import re

def extract_n_elem(name):
    match = re.search(r"/(\d+)(?:\D|$)", name)
    return int(match.group(1)) if match else 0

def read_data(filename):
    """
    Reads the Google Benchmark data from a JSON file and returns a DataFrame.
    """
    with open(filename, "r") as read_file:
        data = json.load(read_file)
        df = pd.DataFrame.from_dict(data["benchmarks"]).astype({"real_time": float})
        df = df[df["run_type"] == "aggregate"]
        df["benchmark"] = df["name"].apply(lambda x: x.split('/')[0])
        df["n_elem"] = df["name"].apply(extract_n_elem)
        return df

def plot_per_benchmark(all_data, title_map, out_dir):
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
        plt.legend()
        plt.savefig(f'{out_dir}/{benchmark}_comparison.png')
        plt.close()

if __name__ == "__main__":
    print("Running the benchmarks...")

    dirname = sys.argv[1]
    soa_versions = {
        'soa_boost': 'Preprocessor Macros SoA',
        'soa_wrapper': 'Template Metaprogramming SoA',
        'soa_manual': 'Manual SoA'
    }

    all_results = []
    for f, label in soa_versions.items():
        filename = f"{dirname}/{f}"
        # Comment the following line to disable benchmarks
        subprocess.run([f"{filename}", "--benchmark_out_format=json", f"--benchmark_out={filename}.json",
                        "--benchmark_counters_tabular=true", "--benchmark_repetitions=3", "--benchmark_min_warmup_time=2"])
        df = read_data(f"{filename}.json")
        all_results.append((df, label))

    plot_per_benchmark(all_results, soa_versions, dirname, min_y, max_y)
