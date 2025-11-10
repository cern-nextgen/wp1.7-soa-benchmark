import multiprocessing
import subprocess
from matplotlib import lines
import pandas as pd
from io import StringIO
import os
from itertools import product
import numpy as np
from datetime import datetime
import platform

if "amd" in platform.machine().lower():
    events = [
        "instructions",
        "fp_ops_retired_by_type.scalar_all",
        "fp_ops_retired_by_type.vector_all",
        "cpu-cycles",
        "L1-icache-load-misses",
        "L1-dcache-loads",
        "L1-dcache-load-misses",
        "cache-misses",
        "ls_dispatch.ld_st_dispatch",  # Number of memory load-store operations dispatched to the load-store
        "ex_no_retire.load_not_complete",  # Cycles with no retire while the oldest op is waiting for load data
        "branch-misses",
        "branch-instructions",
        "stalled-cycles-frontend",
        "de_no_dispatch_per_slot.backend_stalls",
    ]
else:
    events = [
        "instructions",
        "fp_arith_inst_retired.scalar",
        "fp_arith_inst_retired.vector",
        "cpu-cycles",
        "L1-icache-load-misses",
        "L1-dcache-loads",
        "L1-dcache-load-misses",
        "cache-misses",  # l2 misses
        "mem_inst_retired.any",
        "cycle_activity.stalls_mem_any",
        "branch-misses",
        "branch-instructions",
        "frontend_retired.latency_ge_1",
        "resource_stalls.any",
    ]

N_im = 10000000
N_stencil = 10000000
N_nbody = 10000

run_datetime = datetime.now().strftime("%y%m%d_%H%M")

##
# File modification functions
##


def modify_pxpyzpm_aos_manual(ib, ia):
    """
    ASSUMES that line 14 contains all the data members of Particle
    """
    # Read the file
    with open("aos_manual.cpp", "r") as f:
        lines = f.readlines()

    # Replace line 14 (index 13) with new content
    ib_defs = ia_defs = ""
    for b in range(ib):
        ib_defs += f"b{b}, "
    for a in range(ia):
        ia_defs += f", a{a}"
    lines[13] = f"\tdouble {ib_defs}x, y, z, M{ia_defs};\n"

    # Write back to the file
    with open("aos_manual.cpp", "w") as f:
        f.writelines(lines)


def modify_pxpyzpm_soa_manual(ib, ia):
    """
    ASSUMES that the struct PxPyPzM is in pxpypzm.h
    """
    ib_defs = "\tdouble" if ib > 0 else ""
    ib_cstrct = ""
    for b in range(ib):
        ib_defs += (
            f"{',' if b > 0 else ''} *__restrict__ b{b}{';' if b == ib-1 else ''}"
        )
        ib_cstrct += f"\t\tb{b} = reinterpret_cast<double *__restrict__>(buf + offset);\n\t\toffset += align_size(n * sizeof(double));\n"
    ia_defs = "\tdouble" if ia > 0 else ""
    ia_cstrct = ""
    for b in range(ia):
        ia_defs += (
            f"{',' if b > 0 else ''} *__restrict__ a{b}{';' if b == ia-1 else ''}"
        )
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
"""
        )


def modify_stride_invariantmass(stride, wrap):
    """
    ASSUMES that lines 207-210 in benchmark.h are as follows:
        size_t stride = 1;
        for (auto _ : state) {
            #pragma clang loop vectorize(assume_safety)
            for (size_t start = 0; start < stride; ++start) {
                for (size_t i = start; i < n; i += stride) {
    ASSUMES that the benchmark is instantiated in the exe through resetting the nmembers
    """
    modify_N("im", N_im) if wrap else modify_N("im", N_im * stride)

    with open("benchmark.h", "r") as f:
        lines = f.readlines()

    # Replace line 206 (index 205) with new content
    lines[206] = f"    size_t stride = {stride};\n"

    if wrap:
        lines[209] = (
            "                for (size_t start = 0; start < stride; ++start) { for (size_t i = start; i < n; i += stride) {\n"
        )
        lines[253] = "    }}\n"
    else:
        lines[209] = "        for (size_t i = 0; i < n; i += stride) {\n"
        lines[253] = "    }\n"

    # Write back to the file
    with open("benchmark.h", "w") as f:
        f.writelines(lines)


def modify_sstencil_aos_manual(ib, ia):
    """
    ASSUMES that line 10 contains all the data members of Sstencil
    """
    # Read the file
    with open("aos_manual.cpp", "r") as f:
        lines = f.readlines()

    # Replace line 10 (index 9) with new content
    ib_defs = ia_defs = ""
    for b in range(ib):
        ib_defs += f"b{b}, "
    for a in range(ia):
        ia_defs += f", a{a}"
    lines[9] = f"\tdouble {ib_defs}src, dst, rhs{ia_defs};\n"

    # Write back to the file
    with open("aos_manual.cpp", "w") as f:
        f.writelines(lines)


def modify_sstencil_soa_manual(ib, ia):
    """
    ASSUMES that the struct Sstencil is in sstencil.h
    """
    ib_defs = "\tdouble" if ib > 0 else ""
    ib_cstrct = ""
    for b in range(ib):
        ib_defs += (
            f"{',' if b > 0 else ''} *__restrict__  b{b}{';' if b == ib-1 else ''}"
        )
        ib_cstrct += f"\t\tb{b} = reinterpret_cast<double *__restrict__>(buf + offset);\n\t\toffset += align_size(n * sizeof(double));\n"
    ia_defs = "\tdouble" if ia > 0 else ""
    ia_cstrct = ""
    for b in range(ia):
        ia_defs += (
            f"{',' if b > 0 else ''} *__restrict__  a{b}{';' if b == ia-1 else ''}"
        )
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
"""
        )


def modify_nbody_aos_manual(ib, ia):
    """
    ASSUMES that line 6 contains all the data members of Snbody
    """
    # Read the file
    with open("aos_manual.cpp", "r") as f:
        lines = f.readlines()

    # Replace line 6 (index 5) with new content
    ib_defs = ia_defs = ""
    for b in range(ib):
        ib_defs += f"b{b}, "
    for a in range(ia):
        ia_defs += f", a{a}"

    lines[5] = f"\tdouble {ib_defs}x, y, z, vx, vy, vz{ia_defs};\n"

    # Write back to the file
    with open("aos_manual.cpp", "w") as f:
        f.writelines(lines)


def modify_nbody_soa_manual(ib, ia):
    """
    ASSUMES that the struct Snbody is in snbody.h
    """
    ib_defs = "\tfloat" if ib > 0 else ""
    ib_cstrct = ""
    for b in range(ib):
        ib_defs += (
            f"{',' if b > 0 else ''} *__restrict__  b{b}{';' if b == ib-1 else ''}"
        )
        ib_cstrct += f"\t\tb{b} = reinterpret_cast<float *__restrict__>(buf + offset);\n\t\toffset += align_size(n * sizeof(float));\n"
    ia_defs = "\tfloat" if ia > 0 else ""
    ia_cstrct = ""
    for b in range(ia):
        ia_defs += (
            f"{',' if b > 0 else ''} *__restrict__  a{b}{';' if b == ia-1 else ''}"
        )
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
"""
        )


def modify_N(app, N):
    """
    ASSUMES that lines 22-24 in benchmark.h are as follows:
    constexpr size_t N_im = 10000000;
    constexpr size_t N_stencil = 10000000;
    constexpr size_t N_nbody = 10000;
    """
    with open("benchmark.h", "r") as f:
        lines = f.readlines()

    if app == "im":
        lines[21] = f"constexpr size_t N_im = {N};\n"
    elif app == "stcl":
        lines[22] = f"constexpr size_t N_stencil = {N};\n"
    elif app == "nbody":
        lines[23] = f"constexpr size_t N_nbody = {N};\n"
    else:
        raise ValueError(f"Unknown app: {app}")

    with open("benchmark.h", "w") as f:
        f.writelines(lines)


###
# Helper functions
##


def log(msg, exe=["aos_manual", "soa_manual"]):
    for x in exe:
        with open(f"{run_datetime}_{x}.log", "a") as f:
            f.write(msg)


def compile(exe):
    result = subprocess.run(["make", exe], capture_output=True, text=True)
    log(result.stdout + "\n" + result.stderr, [exe])


def get_results(events, filter, exe=["aos_manual", "soa_manual"]):
    def run_exe(events, filter, exe):
        p = []
        for c, exe in enumerate(exe):
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
            p.append(
                subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            )
            log(" ".join(cmd), ["aos_manual"] if c == 0 else ["soa_manual"])
        aos_result = p[0].communicate()
        soa_result = p[1].communicate()

        with open(f"{run_datetime}_aos_manual.log", "a") as f_aos:
            f_aos.write(aos_result[0].decode())
            f_aos.write(aos_result[1].decode())

        with open(f"{run_datetime}_soa_manual.log", "a") as f_soa:
            f_soa.write(soa_result[0].decode())
            f_soa.write(soa_result[1].decode())

        return aos_result, soa_result

    def process_df(stdout):
        selected_lines = [stdout.splitlines()[i] for i in [0, 1, 3, 5, 7]]
        df = pd.read_csv(StringIO("\n".join(selected_lines)))
        df_mean = df.mean(numeric_only=True)
        df_std = df.std(numeric_only=True)
        return df_mean, df_std

    def process_perfctrs(stderr, n_ctrs):
        lines = stderr.splitlines()
        perf_line_idx = next(
            (i for i, line in enumerate(lines) if "Performance counter stats" in line),
            -1,
        )
        perf_ctrs = [
            l.replace(",", "").split()
            for l in lines[perf_line_idx + 2 : perf_line_idx + n_ctrs + 2]
        ]
        return perf_ctrs

    # Max number of supported hardware counters is 7
    aos_result, soa_result = run_exe(events[:7], filter, exe)

    df_mean_aos, df_std_aos = process_df(aos_result[0].decode())
    perf_ctrs_aos = process_perfctrs(aos_result[1].decode(), 7)

    df_mean_soa, df_std_soa = process_df(soa_result[0].decode())
    perf_ctrs_soa = process_perfctrs(soa_result[1].decode(), 7)

    # If more than 7 hardware counters, run multiple times
    for ie in range(7, len(events), 7):
        n_ctrs = min(7, len(events) - ie)
        aos_result, soa_result = run_exe(events[ie : ie + n_ctrs], filter, exe)
        new_perf_ctrs_aos = process_perfctrs(aos_result[1].decode(), n_ctrs)
        new_perf_ctrs_soa = process_perfctrs(soa_result[1].decode(), n_ctrs)
        perf_ctrs_aos.extend(new_perf_ctrs_aos)
        perf_ctrs_soa.extend(new_perf_ctrs_soa)

    return ("aos_manual", df_mean_aos, df_std_aos, perf_ctrs_aos), (
        "soa_manual",
        df_mean_soa,
        df_std_soa,
        perf_ctrs_soa,
    )


###


def modify_stride(app, stride, compile_now, wrap):
    if app == "im":
        modify_stride_invariantmass(stride, wrap)
    else:
        raise ValueError(f"Stride experiment is only implemented for invariant mass.")

    # Recompile the executables
    if compile_now:
        compile("aos_manual")
        compile("soa_manual")


def modify_nmembers(app, ib, ia, compile_now):
    if app == "im":
        modify_pxpyzpm_aos_manual(ib, ia)
        modify_pxpyzpm_soa_manual(ib, ia)
    elif app == "stcl":
        modify_sstencil_aos_manual(ib, ia)
        modify_sstencil_soa_manual(ib, ia)
    elif app == "nbody":
        modify_nbody_aos_manual(ib, ia)
        modify_nbody_soa_manual(ib, ia)
    else:
        raise ValueError(f"Unknown app: {app}")

    if compile_now:
        compile("aos_manual")
        compile("soa_manual")


def modify_nmembers_stride(app, ib, ia, stride, wrap, compile_now):
    if app == "im":
        modify_stride_invariantmass(stride, wrap)
        modify_pxpyzpm_aos_manual(ib, ia)
        modify_pxpyzpm_soa_manual(ib, ia)
    else:
        raise ValueError(f"App not supported: {app}")

    if compile_now:
        compile("aos_manual")
        compile("soa_manual")


def get_filter(app):
    if app == "im":
        filter = "BM_InvariantMass"
        modify_N("im", N_im)
    elif app == "stcl":
        filter = "BM_stencil"
    elif app == "nbody":
        filter = "BM_nbody"
    else:
        raise ValueError(f"Unknown app: {app}")
    return filter


###
# Experiment functions
##


def experiment_stride(output_file, app, precompiled, wrap):
    header = "version,stride,runtime_mean,runtime_stddev,{}\n".format(",".join(events))
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(header)

    filter = get_filter(app)  # also sets N_im

    modify_pxpyzpm_aos_manual(0, 0)
    modify_pxpyzpm_soa_manual(0, 0)

    stride_list = range(1, 17)
    for stride in stride_list:
        if precompiled:
            log(
                f"Using precompiled binaries for {app} with stride {stride} and wrap {wrap}"
            )
            aos_results, soa_results = get_results(
                events,
                filter,
                exe=[
                    os.path.join(precompiled_dir, f"aos_manual_0_0_{stride}"),
                    os.path.join(precompiled_dir, f"soa_manual_0_0_{stride}"),
                ],
            )
        else:
            log(f"Run {app} with stride {stride} and wrap {wrap}")

            # WARNING: assumes app = im
            modify_stride(app, stride, compile_now=True, wrap=wrap)

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


def experiment_nmembers(output_file, app, precompiled):
    before_list = range(0, 25)
    after_list = range(0, 25)
    filter = get_filter(app)

    if app == "im":
        modify_stride_invariantmass(1, wrap=False)

    header = "version,before,after,runtime_mean,runtime_stddev,{}\n".format(
        ",".join(events)
    )
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(header)

    for ib, ia in product(before_list, after_list):
        if precompiled:
            log(f"Using precompiled binaries for {app} with {ib} before and {ia} after")

            aos_results, soa_results = get_results(
                events,
                filter,
                exe=[
                    os.path.join(precompiled_dir, f"aos_manual_{ib}_{ia}_1"),
                    os.path.join(precompiled_dir, f"soa_manual_{ib}_{ia}_1"),
                ],
            )
        else:
            log(f"Run {app} with {ib} before and {ia} after")

            modify_nmembers(app, ib, ia, compile_now=True)
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


def experiment_nmembers_stride(output_file, app, precompiled, wrap):
    if app != "im":
        raise ValueError(f"App not supported: {app}")

    before_list = range(0, 17)
    after_list = range(0, 17)
    stride_list = range(0, 17)
    filter = get_filter(app)

    header = "version,before,after,stride,runtime_mean,runtime_stddev,{}\n".format(
        ",".join(events)
    )
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(header)

    # Divide the total combinations into 4 roughly equal chunks for parallel processing
    all_combinations = list(product(before_list, after_list, stride_list))
    n_chunks = 1
    chunk_size = (len(all_combinations) + n_chunks - 1) // n_chunks  # ceil division

    # Get the chunk index from environment variable or default to 0
    chunk_idx = 0
    chunk = all_combinations[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]

    for ib, ia, stride in chunk:
        if precompiled:
            log(
                f"Using precompiled binaries for {app} with {ib} before, {ia} after, and stride {stride}"
            )

            aos_results, soa_results = get_results(
                events,
                filter,
                exe=[
                    os.path.join(precompiled_dir, f"aos_manual_{ib}_{ia}_{stride}"),
                    os.path.join(precompiled_dir, f"soa_manual_{ib}_{ia}_{stride}"),
                ],
            )
        else:
            log(f"Run {app} with {ib} before, {ia} after, and stride {stride}")
            modify_nmembers_stride(app, ib, ia, stride, wrap, compile_now=True)
            aos_results, soa_results = get_results(events, filter)

        for exe, df_mean, df_std, perf_ctrs in [aos_results, soa_results]:
            with open(output_file, "a") as f:
                f.write(
                    "{},{},{},{},{},{},{}\n".format(
                        exe,
                        ib,
                        ia,
                        stride,
                        df_mean["real_time"],
                        df_std["real_time"],
                        ",".join([c[0] for c in perf_ctrs]),
                    )
                )


def generate_bin(apps, before_list, after_list, stride_list):
    for ib, ia, stride in product(before_list, after_list, stride_list):
        print(f"Generate bin for {ib} before, {ia} after, and stride {stride}")
        for app in apps:
            if app == "im":
                modify_stride(app, stride, compile_now=False, wrap=False)
            modify_nmembers(app, ib, ia, compile_now=False)

        compile("aos_manual")
        compile("soa_manual")

        subprocess.run(
            [
                "cp",
                "aos_manual",
                os.path.join(precompiled_dir, f"aos_manual_{ib}_{ia}_{stride}"),
            ],
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [
                "cp",
                "soa_manual",
                os.path.join(precompiled_dir, f"soa_manual_{ib}_{ia}_{stride}"),
            ],
            capture_output=True,
            text=True,
        )

precompiled_dir = "/data/soa-benchmark-results/251017/bin"

if __name__ == "__main__":
    experiment_nmembers("perf_output_nmembers_im.csv", "im", precompiled=False)
    experiment_stride("perf_output_stride_im.csv", "im", precompiled=False, wrap=False)

    # experiment_stride("perf_output_stride_im.csv", "im", precompiled=True, wrap=False)
    # experiment_nmembers("perf_output_nmembers_stcl.csv", "stcl", precompiled=True)
    # experiment_nmembers("perf_output_nmembers_nbody.csv", "nbody", precompiled=True)
    # experiment_nmembers_stride("perf_output_nmembers_stride_im.csv", "im", precompiled=True, wrap=False)

    # generate_bin(["nbody", "stcl", "im"], range(0, 25), range(0, 25), [1])
    # generate_bin(["im"], [0], [0], range(2, 38))
    # generate_bin(["im"], range(17), range(17), range(1, 17))
    # generate_bin(["im"], [0], [0], range(1, 38))

