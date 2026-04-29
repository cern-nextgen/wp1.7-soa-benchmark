import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import json
import glob

def read_data(filename):
    import pandas as pd
    stem = os.path.splitext(os.path.basename(filename))[0]
    with open(filename, "r") as f:
        data = json.load(f)
        name = data.get("context", {}).get("name", stem)
        df = pd.DataFrame.from_dict(data["benchmarks"]).astype({"real_time": float})
        df = df[df["run_type"] == "aggregate"]
        df["benchmark"] = df["name"].apply(lambda x: x.split('/')[1].split('_')[1])
        return df, name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, metavar="FILE",
                        help="Output PNG file path")
    parser.add_argument("--libraries", nargs="+", metavar="FILE",
                        help="JSON files to plot")
    parser.add_argument("--benchmarks", nargs="+", metavar="NAME",
                        help="Benchmark names to include")
    args = parser.parse_args()

    json_files = sorted(args.libraries) if args.libraries else []

    all_data = []
    for f in json_files:
        df, name = read_data(f)
        if args.benchmarks:
            df = df[df["benchmark"].isin(args.benchmarks)]
        if not df.empty:
            all_data.append((df, name))

    # Preserve insertion order for deterministic assignment
    all_benchmarks = list(dict.fromkeys(bm for df, _ in all_data for bm in df["benchmark"].unique()))
    all_libraries  = list(dict.fromkeys(name for _, name in all_data))

    colors     = {bm:  plt.cm.tab10(i % 10)                      for i, bm  in enumerate(all_benchmarks)}
    linestyles = {lib: ["-", "--", ":", "-."][i % 4]              for i, lib in enumerate(all_libraries)}

    fig, ax = plt.subplots(figsize=(10, 6))
    time_unit = None

    for df, name in all_data:
        if time_unit is None:
            time_unit = df["time_unit"].iloc[0]
        for bm in df["benchmark"].unique():
            df_mean = df[(df["benchmark"] == bm) & (df["aggregate_name"] == "mean")]
            if df_mean.empty:
                continue
            ax.plot(df_mean['n_elem'], df_mean['real_time'],
                    color=colors[bm], ls=linestyles[name], marker="o",
                    label=f"{name} / {bm}")

    ax.set_xlabel('Number of Elements')
    ax.set_xscale('symlog')
    ax.set_ylabel(f'Real Time ({time_unit})')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:g}'.format(y)))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    plt.savefig(args.output)
    print(f"Saved to {args.output}")
