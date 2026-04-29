import subprocess
import sys
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--executables", nargs="+", required=True, metavar="EXE",
                        help="Paths to benchmark executables")
    parser.add_argument("--benchmarks", nargs="+", required=False, metavar="NAME",
                        help="Run only these benchmarks (default: all)")
    parser.add_argument("--output", required=True, metavar="DIR",
                        help="Directory to store output JSON files")
    args = parser.parse_args()

    benchmark_filter = "|".join(args.benchmarks) if args.benchmarks else ".*"

    print("Running the benchmarks...")
    for exe in args.executables:
        name = os.path.basename(exe)
        out = os.path.join(args.output, name + ".json")
        subprocess.run([exe, "--benchmark_out_format=json", f"--benchmark_out={out}",
                        "--benchmark_counters_tabular=true", "--benchmark_repetitions=3",
                        f"--benchmark_filter={benchmark_filter}", "--benchmark_min_warmup_time=2"])
