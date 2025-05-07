import sys
import os
import json

import matplotlib.pyplot as plt


def read_json(filename):
    with open(filename, "r") as file_handle:
        dictionary = json.load(file_handle)
    return dictionary

def get_xy(benchmark_dict_list, prefix_selector, suffix_selector):
    n_list = []
    runtime_list = []
    for benchmark_dict in benchmark_dict_list:
        name = benchmark_dict['name']
        if name.startswith(prefix_selector) and name.endswith(suffix_selector):
            n = int(name.split('/')[-1].split('_')[0])
            n_list.append(n)
            real_time = benchmark_dict['real_time']
            runtime_list.append(real_time)
    return n_list, runtime_list

if __name__ == "__main__":
    suffix_selector = '_mean'
    output_dir = sys.argv[1]
    prefix_selector = sys.argv[2]
    jsonfile_list = sys.argv[3:]

    plt.figure()
    plt.title(prefix_selector)
    # plt.legend()
    plt.grid(visible=True)
    for jsonfile in jsonfile_list:
        data = read_json(jsonfile)
        benchmark_dict_list = data['benchmarks']
        x, y = get_xy(benchmark_dict_list, prefix_selector, suffix_selector) # 'BM_CPURealRW' 'BM_CPUEasyCompute'
        plt.loglog(x, y) # , label=os.path.splitext(jsonfile)[0]
        print(jsonfile, x, y)
    output_file = os.path.join(output_dir, f"{prefix_selector}.png")
    print(output_file)
    plt.savefig(output_file)
    plt.close()