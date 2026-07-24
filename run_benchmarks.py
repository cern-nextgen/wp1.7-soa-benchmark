import subprocess
import sys
import pandas as pd

if __name__ == "__main__":

    benchmark_filter = sys.argv[3] if len(sys.argv) > 3 else "BM_stencil"

    if len(sys.argv) > 2:
        output_dir = sys.argv[1]
        csvfile = sys.argv[2]

        all_results = []
        with open(f"{csvfile}", "r") as soa_versions_file:
            reader = pd.read_csv(soa_versions_file)
            soa_versions = reader['version']

            print("Running the benchmarks...")
            for f in soa_versions:
                filename = f"{output_dir}/{f}"
                subprocess.run([f"{filename}", f"--benchmark_filter={benchmark_filter}", "--benchmark_out_format=json", f"--benchmark_out={filename}.json", "--benchmark_min_time=30x",
                                "--benchmark_counters_tabular=true", "--benchmark_repetitions=3", "--benchmark_min_warmup_time=2"])
    else:
        print("python3 run_benchmarks.py <output_dir> <csvfile> [benchmark_filter]")
        print("Provide a CSV file with the SoA versions to benchmark and their labels")
