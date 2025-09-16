from attr import ib
import numpy as np
import subprocess
import pandas as pd
from io import StringIO
import os
from itertools import product

events = [
    "fp_arith_inst_retired.scalar",
    "fp_arith_inst_retired.vector",
    "cache-references",
    "cache-misses",
    "alignment-faults",
    "branch-misses",
    "branch-instructions",
    "bus-cycles",
    "cpu-cycles",
    "major-faults",
    "minor-faults",
]

N_im = 10000000
N_stencil = 100000

##
# File modification functions
##

def modify_pxpyzpm_aos_manual(ib, ia):
    """
    ASSUMES that line 41 and 48 in aos_manual.cpp are empty and that members x,y,z,M are on line 44
    ASSUMES that all other benchmark instantiations are commented out
    """
    # Read the file
    with open("aos_manual.cpp", "r") as f:
        lines = f.readlines()

    # Replace line 48 (index 47) with new content
    if ib == 0:
        lines[40] = f"\n"
    else:
        lines[40] = f"\tdouble before[{ib}];\n"

    if ia == 0:
        lines[47] = f"\n"
    else:
        lines[47] = f"\tdouble after[{ia}];\n"

    lines[80] = f"BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM_InvariantMass, Particle, Particle, std::integral_constant<size_t, {N_im}>)->Unit(benchmark::kMillisecond);\n"

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
        ib_defs += f"{',' if b > 0 else ''} *b{b}{';' if b == ib-1 else ''}"
        ib_cstrct += f"\t\tb{b} = reinterpret_cast<double *__restrict__>(buf + offset);\n\t\toffset += align_size(n * sizeof(double));\n"
    ia_defs = "\tdouble" if ia > 0 else ""
    ia_cstrct = ""
    for b in range(ia):
        ia_defs += f"{',' if b > 0 else ''} *a{b}{';' if b == ia-1 else ''}"
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

    with open("soa_manual.cpp", "r") as f:
        lines = f.readlines()
        lines[463] = f"BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM_InvariantMass, PxPyPzM, PxPyPzM, std::integral_constant<size_t, {N_im}>)->Unit(benchmark::kMillisecond);\n"

    with open("soa_manual.cpp", "w") as f:
        f.writelines(lines)

    result = subprocess.run(["make", "soa_manual"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)


def modify_stride_invariantmass(exe, stride):
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
    lines[611] = f"    size_t stride = {stride};\n"

    # Write back to the file
    with open("benchmark.h", "w") as f:
        f.writelines(lines)

    # Recompile the executable
    result = subprocess.run(["make", exe], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

def modify_sstencil_aos_manual(ib, ia):
    """
    ASSUMES that line 36 contains all the data members of Sstencil
    """
    # Read the file
    with open("aos_manual.cpp", "r") as f:
        lines = f.readlines()

    # Replace line 48 (index 47) with new content
    if ib == 0:
        ib_line = f"double "
    else:
        ib_line = f"double before[{ib}], "

    if ia == 0:
        ia_line = f";\n"
    else:
        ia_line = f", after[{ia}];\n"

    lines[35] = f"    {ib_line}src, dst, rhs{ia_line}"
    lines[80] = f"BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM_stencil, Sstencil, std::integral_constant<size_t, {N_stencil}>)->Unit(benchmark::kMillisecond);\n"

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
        ib_defs += f"{',' if b > 0 else ''} *b{b}{';' if b == ib-1 else ''}"
        ib_cstrct += f"\t\tb{b} = reinterpret_cast<double *__restrict__>(buf + offset);\n\t\toffset += align_size(n * sizeof(double));\n"
    ia_defs = "\tdouble" if ia > 0 else ""
    ia_cstrct = ""
    for b in range(ia):
        ia_defs += f"{',' if b > 0 else ''} *a{b}{';' if b == ia-1 else ''}"
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

    with open("soa_manual.cpp", "r") as f:
        lines = f.readlines()
        lines[463] = f"BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM_stencil, Sstencil, std::integral_constant<size_t, {N_stencil}>)->Unit(benchmark::kMillisecond);\n"
    with open("soa_manual.cpp", "w") as f:
        f.writelines(lines)

    result = subprocess.run(["make", "soa_manual"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)

###
# Helper functions
##

def get_results(exe, events):
    def run_exe(exe, events):
        cmd = [
            "likwid-pin",
            "-C",
            "0",
            "perf",
            "stat",
            "-C",
            "0",
            "-e",
            ",".join(events),
            "-r",
            "5",
            f"./{exe}",
            "--benchmark_format=csv",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        # print(result.stdout)
        print(result.stderr)
        return result

    # Max number of supported hardware counters is 7
    result = run_exe(exe, events[:7])
    selected_lines = [
        result.stdout.splitlines()[i] for i in [0, 1, 3, 5, 7]
    ]
    df = pd.read_csv(StringIO("\n".join(selected_lines)))
    df_mean = df.mean(numeric_only=True)
    df_std = df.std(numeric_only=True)

    lines = result.stderr.splitlines()
    perf_line_idx = next((i for i, line in enumerate(lines) if "Performance counter stats" in line), -1,)
    perf_ctrs = [
        l.replace(",", "").split()
        for l in lines[perf_line_idx + 2 : perf_line_idx + 9]
    ]

    # If more than 7 hardware counters, run multiple times
    for ie in range(7, len(events), 7):
        n_ctrs = min(7, len(events) - ie)
        result = run_exe(exe, events[ie : ie + n_ctrs])
        lines = result.stderr.splitlines()
        perf_line_idx = next((i for i, line in enumerate(lines) if "Performance counter stats" in line), -1,)
        perf_ctrs.extend(
            l.replace(",", "").split()
            for l in lines[perf_line_idx + 2 : perf_line_idx + n_ctrs + 2]
        )

    print(perf_ctrs)
    print(df_mean)
    return df_mean, df_std, perf_ctrs

###
# Experiment functions
##

def experiment_nmembers_invariantmass(output_file, layout="aos"):
    exe = "aos_manual" if layout == "aos" else "soa_manual"

    before_list = range(11, 13)
    after_list = range(0, 13)

    header = (
        "version,before,after,{},runtime_mean,runtime_stddev\n".format(
            ",".join(events)
        )
    )
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(header)

    modify_stride_invariantmass(exe, 1)

    for ib, ia in product(before_list, after_list):
        modify_pxpyzpm_aos_manual(ib, ia) if layout == "aos" else modify_pxpyzpm_soa_manual(ib, ia)

        df_mean, df_std, perf_ctrs = get_results(exe, events)

        with open(output_file, "a") as f:
            f.write(
                "{},{},{},{},{},{}\n".format(
                    exe,
                    ib,
                    ia,
                    ",".join([c[0] for c in perf_ctrs]),
                    df_mean["real_time"],
                    df_std["real_time"],
                )
            )

def experiment_stride_invariantmass(output_file, layout="aos"):
    if layout == "aos":
        exe = "aos_manual"
        modify_pxpyzpm_aos_manual(0, 0)
    else:
        exe = "soa_manual"
        modify_pxpyzpm_soa_manual(0, 0)

    header = (
        "version,stride,{},runtime_mean,runtime_stddev\n".format(
            ",".join(events)
        )
    )
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(header)

    stride_list = range(1, 33)
    for stride in stride_list:
        modify_stride_invariantmass(exe, stride)

        df_mean, df_std, perf_ctrs = get_results(exe, events)

        with open(output_file, "a") as f:
            f.write(
                "{},{},{},{},{}\n".format(
                    exe,
                    stride,
                    ",".join([c[0] for c in perf_ctrs]),
                    df_mean["real_time"],
                    df_std["real_time"],
                )
            )

def experiment_nmembers_stencil(output_file, layout="aos"):
    exe = "aos_manual" if layout == "aos" else "soa_manual"

    before_list = range(0, 25)
    after_list = range(0, 25)

    header = (
        "version,before,after,{},runtime_mean,runtime_stddev\n".format(
            ",".join(events)
        )
    )
    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write(header)

    for ib, ia in product(before_list, after_list):
        modify_sstencil_aos_manual(ib, ia) if layout == "aos" else modify_sstencil_soa_manual(ib, ia)

        df_mean, df_std, perf_ctrs = get_results(exe, events)

        with open(output_file, "a") as f:
            f.write(
                "{},{},{},{},{},{}\n".format(
                    exe,
                    ib,
                    ia,
                    ",".join([c[0] for c in perf_ctrs]),
                    df_mean["real_time"],
                    df_std["real_time"],
                )
            )


if __name__ == "__main__":
    # experiment_nmembers_invariantmass("perf_output_nmembers_im_aos.csv", "aos")
    # experiment_nmembers_invariantmass("perf_output_nmembers_im_soa.csv", "soa")
    experiment_stride_invariantmass("perf_output_stride_im_aos.csv", "aos")
    experiment_stride_invariantmass("perf_output_stride_im_soa.csv", "soa")
    experiment_nmembers_stencil("perf_output_nmembers_stcl_aos.csv", "aos")
    experiment_nmembers_stencil("perf_output_nmembers_stcl_soa.csv", "soa")
