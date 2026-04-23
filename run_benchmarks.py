import subprocess
import sys
import os

if __name__ == "__main__":

    if len(sys.argv) > 1:
        output_dir = sys.argv[1]

        executables = [
            f for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
            and os.access(os.path.join(output_dir, f), os.X_OK)
            and not f.endswith((".json", ".py", ".cmake", ".so", ".a"))
        ]

        print("Running the benchmarks...")
        for exe in sorted(executables):
            path = os.path.join(output_dir, exe)
            subprocess.run([path, "--benchmark_out_format=json", f"--benchmark_out={path}.json",
                            "--benchmark_counters_tabular=true", "--benchmark_repetitions=3",
                            "--benchmark_min_warmup_time=2"])
    else:
        print("python run_benchmarks.py <output_dir>")
