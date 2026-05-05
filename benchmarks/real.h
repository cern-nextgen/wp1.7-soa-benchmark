#ifndef BENCHMARKS_REAL_H
#define BENCHMARKS_REAL_H

#include "benchmarks/common.h"

#include <Eigen/Core>

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, RealRW)(benchmark::State &state)
{
    auto n = this->n;
    auto &t = this->t;

    const Eigen::Matrix3d m2 = Eigen::Matrix3d::Constant(2);
    const Eigen::Vector3d v2 = Eigen::Vector3d::Constant(2);

    for (auto _ : state) {
        state.PauseTiming();
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x0, i) = 0.f;
            MEMBER_ACCESS(t, x1, i) = 0.f;
            MEMBER_ACCESS(t, x2, i) = 0.0;
            MEMBER_ACCESS(t, x3, i) = 0.0;
            MEMBER_ACCESS(t, x4, i) = 0;
            MEMBER_ACCESS(t, x5, i) = 0;
            MEMBER_ACCESS(t, x6, i) = Eigen::Vector3d::Zero();
            MEMBER_ACCESS(t, x7, i) = Eigen::Vector3d::Zero();
            MEMBER_ACCESS(t, x8, i) = Eigen::Matrix3d::Zero();
            MEMBER_ACCESS(t, x9, i) = Eigen::Matrix3d::Zero();
        }
        state.ResumeTiming();

        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x0, i) += 2.f; }
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x1, i) += 2.f; }
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x2, i) += 2.0; }
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x3, i) += 2.0; }
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x4, i) += 2; }
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x5, i) += 2; }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x6, i) += v2; }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x7, i) += v2; }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x8, i) += m2; }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x9, i) += m2; }
    }

    for (int i = 0; i < n; ++i) {
        CheckResult(state, 2.f,  MEMBER_ACCESS(t, x0, i), "x0");
        CheckResult(state, 2.f,  MEMBER_ACCESS(t, x1, i), "x1");
        CheckResult(state, 2.0,  MEMBER_ACCESS(t, x2, i), "x2");
        CheckResult(state, 2.0,  MEMBER_ACCESS(t, x3, i), "x3");
        CheckResult(state, 2,    MEMBER_ACCESS(t, x4, i), "x4");
        CheckResult(state, 2,    MEMBER_ACCESS(t, x5, i), "x5");
        CheckResult(state, v2,   MEMBER_ACCESS(t, x6, i), "x6");
        CheckResult(state, v2,    MEMBER_ACCESS(t, x7, i), "x7");
        CheckResult(state, m2,    MEMBER_ACCESS(t, x8, i), "x8");
        CheckResult(state, m2,    MEMBER_ACCESS(t, x9, i), "x9");
    }
    state.counters["n_elem"] = n;
}

#endif // BENCHMARKS_REAL_H
