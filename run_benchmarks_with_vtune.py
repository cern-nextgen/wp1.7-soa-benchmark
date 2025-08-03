import os
import sys
import shutil
import pandas as pd
import subprocess

def validate_build_dir(build_dir):
    if "vtune" not in os.path.basename(os.path.normpath(build_dir)):
        print(f"Error: build directory '{build_dir}' must contain 'vtune' in its name.")
        sys.exit(1)
    if not os.path.isdir(build_dir):
        print(f"Error: directory '{build_dir}' does not exist.")
        sys.exit(1)

def validate_csv(csvfile):
    if not os.path.isfile(csvfile):
        print(f"Error: CSV file '{csvfile}' not found.")
        sys.exit(1)

def run_vtune_analysis(executable_path, result_dir, analysis_type):
    subprocess.run([
        "vtune",
        "-collect", analysis_type,
        "-result-dir", result_dir,
        executable_path
    ], check=True)

def export_vtune_report(result_dir, report_type, output_csv):
    cmd = [
        "vtune",
        "-report", report_type,
        "-format", "csv",
        "-report-knob", "show-issues=false",
        "-r", result_dir,
        "-report-output", output_csv
    ]

    if report_type == "timeline":
        cmd.extend(["-report-knob", "column-by=CPUTime"])

    subprocess.run(cmd, check=True)

def main(build_dir, csvfile):
    validate_build_dir(build_dir)
    validate_csv(csvfile)

    base_report_dir = "vtune_results"
    os.makedirs(base_report_dir, exist_ok=True)

    df = pd.read_csv(csvfile)
    for version in df['version']:
        executable_path = os.path.join(build_dir, version)
        if not os.path.isfile(executable_path):
            print(f"Warning: Executable '{executable_path}' not found. Skipping.")
            continue

        version_report_dir = os.path.join(base_report_dir, version)
        os.makedirs(version_report_dir, exist_ok=True)

        result_dir = os.path.join(version_report_dir, f"{version}.vtune")

        print(f"\n[VTune] Collecting data for: {version}")
        run_vtune_analysis(executable_path, result_dir, "hotspots")
        export_vtune_report(result_dir, "summary", os.path.join(version_report_dir, f"{version}_summary.csv"))
        export_vtune_report(result_dir, "hotspots", os.path.join(version_report_dir, f"{version}_hotspots.csv"))
        export_vtune_report(result_dir, "timeline", os.path.join(version_report_dir, f"{version}_timeline.csv"))
        export_vtune_report(result_dir, "hw-events", os.path.join(version_report_dir, f"{version}_hwevents.csv"))
        export_vtune_report(result_dir, "callstacks", os.path.join(version_report_dir, f"{version}_callstack.csv"))

    print(f"\nAll VTune reports generated in './{base_report_dir}/<benchmark>/' folders.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 run_benchmarks_with_vtune.py <build_dir> <csvfile>")
        sys.exit(1)

    build_dir = sys.argv[1]
    csvfile = sys.argv[2]

    main(build_dir, csvfile)

