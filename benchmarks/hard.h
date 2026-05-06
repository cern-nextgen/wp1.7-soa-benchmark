#ifndef BENCHMARKS_HARD_H
#define BENCHMARKS_HARD_H

#include "benchmarks/common.h"

#include <Eigen/Core>

template <Backend B, class T>
void run_HardRW(benchmark::State &state, std::size_t n, T t)
{
    const Eigen::Matrix3d m0 = Eigen::Matrix3d::Zero();
    const Eigen::Vector3d v0 = Eigen::Vector3d::Zero();
    const Eigen::Matrix3d m2 = Eigen::Matrix3d::Constant(2);
    const Eigen::Vector3d v2 = Eigen::Vector3d::Constant(2);

    // clang-format off
    for (auto _ : state) {
        state.PauseTiming();
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x0,  i) = 0.f;  MEMBER_ACCESS(t, x1,  i) = 0.f;  MEMBER_ACCESS(t, x2,  i) = 0.f;
            MEMBER_ACCESS(t, x3,  i) = 0.f;  MEMBER_ACCESS(t, x4,  i) = 0.f;  MEMBER_ACCESS(t, x5,  i) = 0.f;
            MEMBER_ACCESS(t, x6,  i) = 0.f;  MEMBER_ACCESS(t, x7,  i) = 0.f;  MEMBER_ACCESS(t, x8,  i) = 0.f;
            MEMBER_ACCESS(t, x9,  i) = 0.f;  MEMBER_ACCESS(t, x10, i) = 0.f;  MEMBER_ACCESS(t, x11, i) = 0.f;
            MEMBER_ACCESS(t, x12, i) = 0.f;
            MEMBER_ACCESS(t, x13, i) = 0.;   MEMBER_ACCESS(t, x14, i) = 0.;   MEMBER_ACCESS(t, x15, i) = 0.;
            MEMBER_ACCESS(t, x16, i) = 0.;   MEMBER_ACCESS(t, x17, i) = 0.;   MEMBER_ACCESS(t, x18, i) = 0.;
            MEMBER_ACCESS(t, x19, i) = 0.;   MEMBER_ACCESS(t, x20, i) = 0.;   MEMBER_ACCESS(t, x21, i) = 0.;
            MEMBER_ACCESS(t, x22, i) = 0.;   MEMBER_ACCESS(t, x23, i) = 0.;   MEMBER_ACCESS(t, x24, i) = 0.;
            MEMBER_ACCESS(t, x25, i) = 0.;
            MEMBER_ACCESS(t, x26, i) = 0;    MEMBER_ACCESS(t, x27, i) = 0;    MEMBER_ACCESS(t, x28, i) = 0;
            MEMBER_ACCESS(t, x29, i) = 0;    MEMBER_ACCESS(t, x30, i) = 0;    MEMBER_ACCESS(t, x31, i) = 0;
            MEMBER_ACCESS(t, x32, i) = 0;    MEMBER_ACCESS(t, x33, i) = 0;    MEMBER_ACCESS(t, x34, i) = 0;
            MEMBER_ACCESS(t, x35, i) = 0;    MEMBER_ACCESS(t, x36, i) = 0;    MEMBER_ACCESS(t, x37, i) = 0;
            MEMBER_ACCESS(t, x38, i) = 0;
            MEMBER_ACCESS(t, x39, i) = v0; MEMBER_ACCESS(t, x40, i) = v0;
            MEMBER_ACCESS(t, x41, i) = v0; MEMBER_ACCESS(t, x42, i) = v0;
            MEMBER_ACCESS(t, x43, i) = v0; MEMBER_ACCESS(t, x44, i) = v0;
            MEMBER_ACCESS(t, x45, i) = v0; MEMBER_ACCESS(t, x46, i) = v0;
            MEMBER_ACCESS(t, x47, i) = v0; MEMBER_ACCESS(t, x48, i) = v0;
            MEMBER_ACCESS(t, x49, i) = v0; MEMBER_ACCESS(t, x50, i) = v0;
            MEMBER_ACCESS(t, x51, i) = m0; MEMBER_ACCESS(t, x52, i) = m0;
            MEMBER_ACCESS(t, x53, i) = m0; MEMBER_ACCESS(t, x54, i) = m0;
            MEMBER_ACCESS(t, x55, i) = m0; MEMBER_ACCESS(t, x56, i) = m0;
            MEMBER_ACCESS(t, x57, i) = m0; MEMBER_ACCESS(t, x58, i) = m0;
            MEMBER_ACCESS(t, x59, i) = m0; MEMBER_ACCESS(t, x60, i) = m0;
            MEMBER_ACCESS(t, x61, i) = m0; MEMBER_ACCESS(t, x62, i) = m0;
            MEMBER_ACCESS(t, x63, i) = m0;
        });
        backend_allocator<B>::synchronize();
        state.ResumeTiming();

        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x0,  i) += 2.f; MEMBER_ACCESS(t, x1,  i) += 2.f; MEMBER_ACCESS(t, x2,  i) += 2.f;
            MEMBER_ACCESS(t, x3,  i) += 2.f; MEMBER_ACCESS(t, x4,  i) += 2.f;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x5,  i) += 2.f; MEMBER_ACCESS(t, x6,  i) += 2.f; MEMBER_ACCESS(t, x7,  i) += 2.f;
            MEMBER_ACCESS(t, x8,  i) += 2.f; MEMBER_ACCESS(t, x9,  i) += 2.f;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x10, i) += 2.f; MEMBER_ACCESS(t, x11, i) += 2.f; MEMBER_ACCESS(t, x12, i) += 2.f;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x13, i) += 2.0; MEMBER_ACCESS(t, x14, i) += 2.0; MEMBER_ACCESS(t, x15, i) += 2.0;
            MEMBER_ACCESS(t, x16, i) += 2.0; MEMBER_ACCESS(t, x17, i) += 2.0;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x18, i) += 2.0; MEMBER_ACCESS(t, x19, i) += 2.0; MEMBER_ACCESS(t, x20, i) += 2.0;
            MEMBER_ACCESS(t, x21, i) += 2.0; MEMBER_ACCESS(t, x22, i) += 2.0;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x23, i) += 2.0; MEMBER_ACCESS(t, x24, i) += 2.0; MEMBER_ACCESS(t, x25, i) += 2.0;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x26, i) += 2; MEMBER_ACCESS(t, x27, i) += 2; MEMBER_ACCESS(t, x28, i) += 2;
            MEMBER_ACCESS(t, x29, i) += 2; MEMBER_ACCESS(t, x30, i) += 2;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x31, i) += 2; MEMBER_ACCESS(t, x32, i) += 2; MEMBER_ACCESS(t, x33, i) += 2;
            MEMBER_ACCESS(t, x34, i) += 2; MEMBER_ACCESS(t, x35, i) += 2;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x36, i) += 2; MEMBER_ACCESS(t, x37, i) += 2; MEMBER_ACCESS(t, x38, i) += 2;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x39, i) += v2; MEMBER_ACCESS(t, x40, i) += v2; MEMBER_ACCESS(t, x41, i) += v2;
            MEMBER_ACCESS(t, x42, i) += v2; MEMBER_ACCESS(t, x43, i) += v2;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x44, i) += v2; MEMBER_ACCESS(t, x45, i) += v2; MEMBER_ACCESS(t, x46, i) += v2;
            MEMBER_ACCESS(t, x47, i) += v2; MEMBER_ACCESS(t, x48, i) += v2;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x49, i) += v2; MEMBER_ACCESS(t, x50, i) += v2; MEMBER_ACCESS(t, x51, i) += m0;
            MEMBER_ACCESS(t, x52, i) += m0; MEMBER_ACCESS(t, x53, i) += m0;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x54, i) += m0; MEMBER_ACCESS(t, x55, i) += m0; MEMBER_ACCESS(t, x56, i) += m0;
            MEMBER_ACCESS(t, x57, i) += m0; MEMBER_ACCESS(t, x58, i) += m0;
        });
        parallel_for_n<B>(n, [=] BACKEND_HOST_DEVICE (std::size_t i) mutable {
            MEMBER_ACCESS(t, x59, i) += m0; MEMBER_ACCESS(t, x60, i) += m0; MEMBER_ACCESS(t, x61, i) += m0;
            MEMBER_ACCESS(t, x62, i) += m0; MEMBER_ACCESS(t, x63, i) += m0;
        });
    }

    backend_allocator<B>::synchronize();
    for (std::size_t i = 0; i < n; ++i) {
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x0,  i), "x0");
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x1,  i), "x1");
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x2,  i), "x2");
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x3,  i), "x3");
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x4,  i), "x4");
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x5,  i), "x5");
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x6,  i), "x6");
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x7,  i), "x7");
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x8,  i), "x8");
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x9,  i), "x9");
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x10, i), "x10");
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x11, i), "x11");
        CheckResult(state, 2.f, MEMBER_ACCESS(t, x12, i), "x12");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x13, i), "x13");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x14, i), "x14");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x15, i), "x15");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x16, i), "x16");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x17, i), "x17");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x18, i), "x18");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x19, i), "x19");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x20, i), "x20");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x21, i), "x21");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x22, i), "x22");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x23, i), "x23");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x24, i), "x24");
        CheckResult(state, 2.0, MEMBER_ACCESS(t, x25, i), "x25");
        CheckResult(state, 2, MEMBER_ACCESS(t, x26, i), "x26");
        CheckResult(state, 2, MEMBER_ACCESS(t, x27, i), "x27");
        CheckResult(state, 2, MEMBER_ACCESS(t, x28, i), "x28");
        CheckResult(state, 2, MEMBER_ACCESS(t, x29, i), "x29");
        CheckResult(state, 2, MEMBER_ACCESS(t, x30, i), "x30");
        CheckResult(state, 2, MEMBER_ACCESS(t, x31, i), "x31");
        CheckResult(state, 2, MEMBER_ACCESS(t, x32, i), "x32");
        CheckResult(state, 2, MEMBER_ACCESS(t, x33, i), "x33");
        CheckResult(state, 2, MEMBER_ACCESS(t, x34, i), "x34");
        CheckResult(state, 2, MEMBER_ACCESS(t, x35, i), "x35");
        CheckResult(state, 2, MEMBER_ACCESS(t, x36, i), "x36");
        CheckResult(state, 2, MEMBER_ACCESS(t, x37, i), "x37");
        CheckResult(state, 2, MEMBER_ACCESS(t, x38, i), "x38");
        CheckResult(state, v2, MEMBER_ACCESS(t, x39, i), "x39");
        CheckResult(state, v2, MEMBER_ACCESS(t, x40, i), "x40");
        CheckResult(state, v2, MEMBER_ACCESS(t, x41, i), "x41");
        CheckResult(state, v2, MEMBER_ACCESS(t, x42, i), "x42");
        CheckResult(state, v2, MEMBER_ACCESS(t, x43, i), "x43");
        CheckResult(state, v2, MEMBER_ACCESS(t, x44, i), "x44");
        CheckResult(state, v2, MEMBER_ACCESS(t, x45, i), "x45");
        CheckResult(state, v2, MEMBER_ACCESS(t, x46, i), "x46");
        CheckResult(state, v2, MEMBER_ACCESS(t, x47, i), "x47");
        CheckResult(state, v2, MEMBER_ACCESS(t, x48, i), "x48");
        CheckResult(state, v2, MEMBER_ACCESS(t, x49, i), "x49");
        CheckResult(state, v2, MEMBER_ACCESS(t, x50, i), "x50");
        CheckResult(state, m2, MEMBER_ACCESS(t, x51, i), "x51");
        CheckResult(state, m2, MEMBER_ACCESS(t, x52, i), "x52");
        CheckResult(state, m2, MEMBER_ACCESS(t, x53, i), "x53");
        CheckResult(state, m2, MEMBER_ACCESS(t, x54, i), "x54");
        CheckResult(state, m2, MEMBER_ACCESS(t, x55, i), "x55");
        CheckResult(state, m2, MEMBER_ACCESS(t, x56, i), "x56");
        CheckResult(state, m2, MEMBER_ACCESS(t, x57, i), "x57");
        CheckResult(state, m2, MEMBER_ACCESS(t, x58, i), "x58");
        CheckResult(state, m2, MEMBER_ACCESS(t, x59, i), "x59");
        CheckResult(state, m2, MEMBER_ACCESS(t, x60, i), "x60");
        CheckResult(state, m2, MEMBER_ACCESS(t, x61, i), "x61");
        CheckResult(state, m2, MEMBER_ACCESS(t, x62, i), "x62");
        CheckResult(state, m2, MEMBER_ACCESS(t, x63, i), "x63");
    }
    // clang-format on
    state.counters["n_elem"] = n;
}

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, HardRW)(benchmark::State &state)
{
    constexpr Backend B = std::remove_reference_t<decltype(*this)>::backend;
    run_HardRW<B>(state, this->n, this->t);
}

#endif // BENCHMARKS_HARD_H
