import json
import sys

import matplotlib.pyplot as plt

def read_data(json_file, filter_names):
    with open(json_file, "r") as f:
        data = json.load(f)

    results = {name: {"n_elem": [], "real_time": []} for name in filter_names}

    for bench in data.get("benchmarks", []):
        name = bench.get("name", "")
        for filter_name in filter_names:
            if name.startswith(filter_name):
                results[filter_name]["n_elem"].append(bench["n_elem"])
                results[filter_name]["real_time"].append(bench["real_time"])
                break

    return results

def plot(results, title, output_file):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    ticks = results.keys()[0]['n_elem']
    plt.xticks(ticks, labels=["{:g}".format(x) for x in ticks], minor=False)
    plt.grid()
    for key, value in results.items():
        plt.loglog(value['n_elem'], value['real_time'], base=2, marker='o', label=key)
    plt.xlabel('Number of Elements')
    plt.ylabel('Execution Time (ms)')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    json_file = sys.argv[1]
    output_file = sys.argv[2]
    title = sys.argv[3]
    filter_names = sys.argv[4:]
    results = read_data(json_file, filter_names)
    plot(results, title, output_file)

# ./build/soa_wrapper_gpu --benchmark_filter=SYNC_GPUAdd_* --benchmark_counters_tabular=true --benchmark_out_format=json --benchmark_out=build/SYNC_GPUAdd.json --benchmark_min_warmup_time=2