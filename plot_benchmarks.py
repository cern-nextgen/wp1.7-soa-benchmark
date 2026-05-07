import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import json
import glob

def read_data(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    with open(filename, "r") as f:
        data = json.load(f)
    name = data.get("context", {}).get("name", stem)
    rows = []
    for entry in data["benchmarks"]:
        if entry.get("run_type") != "aggregate":
            continue
        row = dict(entry)
        row["real_time"] = float(row["real_time"])
        row["benchmark"] = row["run_name"].split('/')[1]
        rows.append(row)
    return rows, name

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
        rows, name = read_data(f)
        if args.benchmarks:
            rows = [r for r in rows if r["benchmark"] in args.benchmarks]
        if rows:
            all_data.append((rows, name))

    # Preserve insertion order for deterministic assignment
    all_benchmarks = list(dict.fromkeys(r["benchmark"] for rows, _ in all_data for r in rows))
    all_libraries  = list(dict.fromkeys(name for _, name in all_data))

    colors     = {bm:  plt.cm.tab10(i % 10)                      for i, bm  in enumerate(all_benchmarks)}
    linestyles = {lib: ["-", "--", ":", "-."][i % 4]              for i, lib in enumerate(all_libraries)}

    fig, ax = plt.subplots(figsize=(10, 6))
    time_unit = None

    for rows, name in all_data:
        if time_unit is None:
            time_unit = rows[0]["time_unit"]
        for bm in dict.fromkeys(r["benchmark"] for r in rows):
            mean_rows = [r for r in rows if r["benchmark"] == bm and r.get("aggregate_name") == "mean"]
            if not mean_rows:
                continue
            xs = [r["n_elem"] for r in mean_rows]
            ys = [r["real_time"] for r in mean_rows]
            ax.plot(xs, ys,
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
