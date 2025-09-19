import multiprocessing
import subprocess
import pandas as pd
from io import StringIO
import os
from itertools import product
import numpy as np

events = [
    "instructions",
    "fp_arith_inst_retired.scalar",
    "fp_arith_inst_retired.vector",
    "cpu-cycles",
    "L1-icache-load-misses",
    "L1-dcache-loads",
    "L1-dcache-load-misses",
    "cycle_activity.stalls_l1d_miss",
    "cache-misses", # l2 misses
    "mem_inst_retired.all_loads",
    "branch-misses",
    "branch-instructions",
    "frontend_retired.latency_ge_1",
    "resource_stalls.any",
]

##
# File modification functions
##

def modify_pxpyzpm_aos_manual(ib, ia):
    """
    ASSUMES that line 40 contains all the data members of Particle
    """
    # Read the file
    with open("aos_manual.cpp", "r") as f:
        lines = f.readlines()

    # Replace line 41 (index 40) with new content
    ib_defs = ia_defs = ""
    for b in range(ib):
        ib_defs += f"b{b}, "
    for a in range(ia):
        ia_defs += f", a{a}"
    lines[40] = f"\tdouble {ib_defs}x, y, z, M{ia_defs};\n"

    # Write back to the file
    with open("aos_manual.cpp", "w") as f:
        f.writelines(lines)

    result = subprocess.run(["make", "aos_manual"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

def modify_pxpyzpm_soa_manual(ib, ia):
    """
    ASSUMES that the struct PxPyPzM is in pxpypzm.h
    """
    ib_defs = "\tdouble" if ib > 0 else ""
    ib_cstrct = ""
    for b in range(ib):
        ib_defs += f"{',' if b > 0 else ''} *__restrict__ b{b}{';' if b == ib-1 else ''}"
        ib_cstrct += f"\t\tb{b} = reinterpret_cast<double *__restrict__>(buf + offset);\n\t\toffset += align_size(n * sizeof(double));\n"
    ia_defs = "\tdouble" if ia > 0 else ""
    ia_cstrct = ""
    for b in range(ia):
        ia_defs += f"{',' if b > 0 else ''} *__restrict__ a{b}{';' if b == ia-1 else ''}"
        ia_cstrct += f"\t\ta{b} = reinterpret_cast<double *__restrict__>(buf + offset);\n\t\toffset += align_size(n * sizeof(double));\n"

    with open("pxpypzm.h", "w") as f:
        f.write(
f"""
struct PxPyPzM {{
{ib_defs}
    double *__restrict__ x, *__restrict__ y, *__restrict__ z, *__restrict__ M;
{ia_defs}

    static size_t size_bytes(size_t n) {{ return align_size(sizeof(double[n])) * {ib+ia+4}; }}

    PxPyPzM(std::byte *buf, size_t n) {{
        size_t offset = 0;
{ib_cstrct}
        x = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        y = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        z = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        M = reinterpret_cast<double *__restrict__>(buf + offset);
{ia_cstrct}
    }}
}};
""")

    result = subprocess.run(["make", "soa_manual"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)


def modify_stride_invariantmass(stride):
    """
    ASSUMES that lines 612-616 in benchmark.h are as follows:
        size_t stride = 1;
        for (auto _ : state) {
            #pragma clang loop vectorize(assume_safety)
            for (size_t start = 0; start < stride; ++start) {
                for (size_t i = start; i < n; i += stride) {
    ASSUMES that the benchmark is instantiated in the exe through resetting the nmembers
    """
    with open("benchmark.h", "r") as f:
        lines = f.readlines()

    # Replace line 94 (index 93) with new content
    lines[614] = f"    size_t stride = {stride};\n"

    # Write back to the file
    with open("benchmark.h", "w") as f:
        f.writelines(lines)

    # Recompile the executable
    result = subprocess.run(["make", "aos_manual", "soa_manual"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

def modify_sstencil_aos_manual(ib, ia):
    """
    ASSUMES that line 36 contains all the data members of Sstencil
    """
    # Read the file
    with open("aos_manual.cpp", "r") as f:
        lines = f.readlines()

    # Replace line 36 (index 35) with new content
    ib_defs = ia_defs = ""
    for b in range(ib):
        ib_defs += f"b{b}, "
    for a in range(ia):
        ia_defs += f", a{a}"
    lines[35] = f"\tdouble {ib_defs}src, dst, rhs{ia_defs};\n"

    # Write back to the file
    with open("aos_manual.cpp", "w") as f:
        f.writelines(lines)

    result = subprocess.run(["make", "aos_manual"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)


def modify_sstencil_soa_manual(ib, ia):
    """
    ASSUMES that the struct Sstencil is in sstencil.h
    """
    ib_defs = "\tdouble" if ib > 0 else ""
    ib_cstrct = ""
    for b in range(ib):
        ib_defs += f"{',' if b > 0 else ''} *__restrict__  b{b}{';' if b == ib-1 else ''}"
        ib_cstrct += f"\t\tb{b} = reinterpret_cast<double *__restrict__>(buf + offset);\n\t\toffset += align_size(n * sizeof(double));\n"
    ia_defs = "\tdouble" if ia > 0 else ""
    ia_cstrct = ""
    for b in range(ia):
        ia_defs += f"{',' if b > 0 else ''} *__restrict__  a{b}{';' if b == ia-1 else ''}"
        ia_cstrct += f"\t\ta{b} = reinterpret_cast<double *__restrict__>(buf + offset);\n\t\toffset += align_size(n * sizeof(double));\n"

    with open("sstencil.h", "w") as f:
        f.write(
f"""
struct Sstencil {{
{ib_defs}
    double *__restrict__ src, *__restrict__ dst, *__restrict__ rhs;
{ia_defs}

    static size_t size_bytes(size_t n) {{ return align_size(sizeof(double[n])) * {ib+ia+3}; }}

    Sstencil(std::byte *buf, size_t n) {{
        size_t offset = 0;
{ib_cstrct}
        src = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        dst = reinterpret_cast<double *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(double));
        rhs = reinterpret_cast<double *__restrict__>(buf + offset);
{ia_cstrct}
    }}
}};
""")

    result = subprocess.run(["make", "soa_manual"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

def modify_nbody_aos_manual(ib, ia):
    """
    ASSUMES that line 32 contains all the data members of Snbody
    """
    # Read the file
    with open("aos_manual.cpp", "r") as f:
        lines = f.readlines()

    # Replace line 32 (index 31) with new content
    ib_defs = ia_defs = ""
    for b in range(ib):
        ib_defs += f"b{b}, "
    for a in range(ia):
        ia_defs += f", a{a}"

    lines[31] = f"\tdouble {ib_defs}x, y, z, vx, vy, vz{ia_defs};\n"

    # Write back to the file
    with open("aos_manual.cpp", "w") as f:
        f.writelines(lines)

    result = subprocess.run(["make", "aos_manual"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

def modify_nbody_soa_manual(ib, ia):
    """
    ASSUMES that the struct Snbody is in snbody.h
    """
    ib_defs = "\tfloat" if ib > 0 else ""
    ib_cstrct = ""
    for b in range(ib):
        ib_defs += f"{',' if b > 0 else ''} *__restrict__  b{b}{';' if b == ib-1 else ''}"
        ib_cstrct += f"\t\tb{b} = reinterpret_cast<float *__restrict__>(buf + offset);\n\t\toffset += align_size(n * sizeof(float));\n"
    ia_defs = "\tfloat" if ia > 0 else ""
    ia_cstrct = ""
    for b in range(ia):
        ia_defs += f"{',' if b > 0 else ''} *__restrict__  a{b}{';' if b == ia-1 else ''}"
        ia_cstrct += f"\t\ta{b} = reinterpret_cast<float *__restrict__>(buf + offset);\n\t\toffset += align_size(n * sizeof(float));\n"

    with open("snbody.h", "w") as f:
        f.write(
f"""
struct Snbody {{
{ib_defs}
    float *__restrict__ x, *__restrict__ y, *__restrict__ z;
    float *__restrict__ vx, *__restrict__ vy, *__restrict__ vz;
{ia_defs}

    static size_t size_bytes(size_t n) {{ return align_size(sizeof(float[n])) * {ib+ia+6}; }}

    Snbody(std::byte *buf, size_t n) {{
        size_t offset = 0;
{ib_cstrct}
        x = reinterpret_cast<float *__restrict__>(buf);
        offset += align_size(n * sizeof(float));
        y = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        z = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        vx = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        vy = reinterpret_cast<float *__restrict__>(buf + offset);
        offset += align_size(n * sizeof(float));
        vz = reinterpret_cast<float *__restrict__>(buf + offset);
{ia_cstrct}
    }}
}};
""")

    result = subprocess.run(["make", "soa_manual"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

###
# Helper functions
##

def get_results(events, filter):
    def run_exe(events, filter):
        p = []
        for c, exe in enumerate(["aos_manual", "soa_manual"]):
            cmd = [
                "likwid-pin",
                "-C",
                str(c),
                "perf",
                "stat",
                "-C",
                str(c),
                "-e",
                ",".join(events),
                "-r",
                "5",
                f"./{exe}",
                "--benchmark_format=csv",
                f"--benchmark_filter={filter}",
            ]
            p.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE))

        aos_result = p[0].communicate()
        soa_result = p[1].communicate()
        # print(aos_result[1].decode())
        # print(soa_result[1].decode())
        # print(aos_result[0].decode())
        # print(soa_result[0].decode())
        return aos_result, soa_result

    def process_df(stdout):
        selected_lines = [
            stdout.splitlines()[i] for i in [0, 1, 3, 5, 7]
        ]
        df = pd.read_csv(StringIO("\n".join(selected_lines)))
        df_mean = df.mean(numeric_only=True)
        df_std = df.std(numeric_only=True)
        return df_mean, df_std

    def process_perfctrs(stderr, n_ctrs):
        lines = stderr.splitlines()
        perf_line_idx = next((i for i, line in enumerate(lines) if "Performance counter stats" in line), -1)
        perf_ctrs = [
            l.replace(",", "").split()
            for l in lines[perf_line_idx + 2 : perf_line_idx + n_ctrs + 2]
        ]
        return perf_ctrs

    # Max number of supported hardware counters is 7
    aos_result, soa_result = run_exe(events[:7], filter)

    df_mean_aos, df_std_aos = process_df(aos_result[0].decode())
    perf_ctrs_aos = process_perfctrs(aos_result[1].decode(), 7)

    df_mean_soa, df_std_soa = process_df(soa_result[0].decode())
    perf_ctrs_soa = process_perfctrs(soa_result[1].decode(), 7)

    # If more than 7 hardware counters, run multiple times
    for ie in range(7, len(events), 7):
        n_ctrs = min(7, len(events) - ie)
        aos_result, soa_result = run_exe(events[ie : ie + n_ctrs], filter)
        new_perf_ctrs_aos = process_perfctrs(aos_result[1].decode(), n_ctrs)
        new_perf_ctrs_soa = process_perfctrs(soa_result[1].decode(), n_ctrs)
        perf_ctrs_aos.extend(new_perf_ctrs_aos)
        perf_ctrs_soa.extend(new_perf_ctrs_soa)

    return ('aos_manual', df_mean_aos, df_std_aos, perf_ctrs_aos), ('soa_manual', df_mean_soa, df_std_soa, perf_ctrs_soa)

###
# Experiment functions
##

def experiment_stride(output_file, app="im"):
    if app == "im":
        modify_pxpyzpm_aos_manual(0, 0)
        modify_pxpyzpm_soa_manual(0, 0)
        filter = "BM_InvariantMass"
    elif app == "stcl":
        modify_sstencil_aos_manual(0, 0)
        modify_sstencil_soa_manual(0, 0)
        filter = "BM_stencil"
    elif app == "nbody":
        modify_nbody_aos_manual(0, 0)
        modify_nbody_soa_manual(0, 0)
        filter = "BM_nbody"
    else:
        raise ValueError(f"Unknown app: {app}")

    header = (
        "version,stride,runtime_mean,runtime_stddev,{}\n".format(
            ",".join(events)
        )
    )
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(header)

    stride_list = range(1, 65)
    for stride in stride_list:
        if app == "im":
            modify_stride_invariantmass(stride)
        else:
            raise ValueError(f"Stride experiment is only implemented for invariant mass.")

        aos_results, soa_results = get_results(events, filter)

        for exe, df_mean, df_std, perf_ctrs in [aos_results, soa_results]:
            with open(output_file, "a") as f:
                f.write(
                    "{},{},{},{},{}\n".format(
                        exe,
                        stride,
                        df_mean["real_time"],
                        df_std["real_time"],
                        ",".join([c[0] for c in perf_ctrs]),
                    )
                )

def experiment_nmembers(output_file, app="im"):
    before_list = range(0, 25)
    after_list = range(0, 25)

    header = (
        "version,before,after,runtime_mean,runtime_stddev,{}\n".format(
            ",".join(events)
        )
    )
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(header)

    for ib, ia in product(before_list, after_list):
        if app == "im":
            modify_stride_invariantmass(1)
            modify_pxpyzpm_aos_manual(ib, ia)
            modify_pxpyzpm_soa_manual(ib, ia)
            filter = "BM_InvariantMass"
        elif app == "stcl":
            modify_sstencil_aos_manual(ib, ia)
            modify_sstencil_soa_manual(ib, ia)
            filter = "BM_stencil"
        elif app == "nbody":
            modify_nbody_aos_manual(ib, ia)
            modify_nbody_soa_manual(ib, ia)
            filter = "BM_nbody"
        else:
            raise ValueError(f"Unknown app: {app}")

        aos_results, soa_results = get_results(events, filter)

        for exe, df_mean, df_std, perf_ctrs in [aos_results, soa_results]:
            with open(output_file, "a") as f:
                f.write(
                    "{},{},{},{},{},{}\n".format(
                        exe,
                        ib,
                        ia,
                        df_mean["real_time"],
                        df_std["real_time"],
                        ",".join([c[0] for c in perf_ctrs]),
                    )
                )


if __name__ == "__main__":
    experiment_nmembers("perf_output_nmembers_im.csv", "im")
    experiment_stride("perf_output_stride_im.csv", "im")
    experiment_nmembers("perf_output_nmembers_stcl.csv", "stcl")
    experiment_nmembers("perf_output_nmembers_nbody.csv", "nbody")
