import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import subprocess
import sys
import os
import json
import math

def read_data(filename):
    """
    Reads the Google Benchmark data from a CSV string and returns a DataFrame.
    """
    with open(filename, "r") as read_file:
        data = json.load(read_file)
        df = pd.DataFrame.from_dict(data["benchmarks"]).astype({"real_time": float})
        df["benchmark"] = df["name"].apply(lambda x: x.split('/')[0])
    return df

def plot_results(df, title, out_dir, min_y=-0.000001, max_y=1000):
    """
    Plots the results from the DataFrame.
    """
    # Set the figure size
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)

    # Plot the data
    for bm in df["benchmark"].unique():
        # Filter the DataFrame for the current benchmark
        df_mean = df[(df["benchmark"] == bm) & (df["aggregate_name"] == "mean")]
        df_std = df[(df["benchmark"] == bm) & (df["aggregate_name"] == "stddev")]
        ax.errorbar(df_mean['n_elem'].astype(int), df_mean['real_time'], yerr=df_std["real_time"],
                    ls="-", marker="o", label=bm)

    # Set the title and labels
    ax.set_title(title)
    ax.set_xlabel('Number of Elements')
    ax.set_xscale('log', base=2)
    ax.set_xticks(df_mean['n_elem'].unique(), labels=["{:g}".format(x) for x in df_mean['n_elem'].unique()], minor=False)

    ax.set_ylabel(f'Real Time ({df["time_unit"].iloc[0]})')
    #ax.set_yscale('log', base=2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    plt.ylim(min_y, max_y)
    plt.grid()

    # Add a legend
    plt.legend()

    # Show the plot
    plt.savefig(f'{out_dir}/{"_".join(title.split())}.png')

if __name__ == "__main__":
    print("Running the benchmarks...")

    dirname = sys.argv[1]

    results = {}
    for f, t in zip(['soa_wrapper_gpu'],
                    ['AoS vs SoA for the CUDA kernel x+y+z']):
        filename = f"{dirname}/{f}"

        # Comment this to plot the results from locally saved json files, without running the benchmarks.
        subprocess.run([f"{filename}", "--benchmark_filter=SYNC_GPUAdd_*", "--benchmark_out_format=json", f"--benchmark_out={filename}.json",
                        "--benchmark_counters_tabular=true", "--benchmark_repetitions=3", "--benchmark_min_warmup_time=2"])

        results[f] = (read_data(f"{filename}.json"), t)

    # Round the y-axis up/down to the nearest power of 10
    max_y = max([df[df["aggregate_name"] == "mean"]["real_time"].max() for (df, _) in results.values()])
    min_y = min([df[df["aggregate_name"] == "mean"]["real_time"].min() for (df, _) in results.values()])
    for (results, t) in results.values():
        plot_results(results, t, dirname, min_y, max_y)
