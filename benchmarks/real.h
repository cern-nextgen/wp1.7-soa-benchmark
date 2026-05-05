#pragma once
#include "benchmarks/common.h"

BENCHMARK_TEMPLATE_METHOD_F(Fixture1, RealRW)(benchmark::State &state)
{
    auto n = this->n;
    auto &t = this->t;

    const Matrix3D m = Matrix3D::Constant(2);
    const Vector3D v = Vector3D::Constant(2);

    for (auto _ : state) {
        state.PauseTiming();
        for (int i = 0; i < n; ++i) {
            MEMBER_ACCESS(t, x0, i) = 0.f;
            MEMBER_ACCESS(t, x1, i) = 0.f;
            MEMBER_ACCESS(t, x2, i) = 0.0;
            MEMBER_ACCESS(t, x3, i) = 0.0;
            MEMBER_ACCESS(t, x4, i) = 0;
            MEMBER_ACCESS(t, x5, i) = 0;
            MEMBER_ACCESS(t, x6, i) = Vector3D::Zero();
            MEMBER_ACCESS(t, x7, i) = Vector3D::Zero();
            MEMBER_ACCESS(t, x8, i) = Matrix3D::Zero();
            MEMBER_ACCESS(t, x9, i) = Matrix3D::Zero();
        }
        state.ResumeTiming();

        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x0, i) += 2.f; }
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x1, i) += 2.f; }
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x2, i) += 2.0; }
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x3, i) += 2.0; }
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x4, i) += 2; }
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x5, i) += 2; }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x6, i) += v; }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x7, i) += v; }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x8, i) += m; }
        #pragma clang loop vectorize(assume_safety)
        for (int i = 0; i < n; ++i) { MEMBER_ACCESS(t, x9, i) += m; }
    }

    for (int i = 0; i < n; ++i) {
        CheckResult(state, 2.f,  MEMBER_ACCESS(t, x0, i), "x0");
        CheckResult(state, 2.f,  MEMBER_ACCESS(t, x1, i), "x1");
        CheckResult(state, 2.0,  MEMBER_ACCESS(t, x2, i), "x2");
        CheckResult(state, 2.0,  MEMBER_ACCESS(t, x3, i), "x3");
        CheckResult(state, 2,    MEMBER_ACCESS(t, x4, i), "x4");
        CheckResult(state, 2,    MEMBER_ACCESS(t, x5, i), "x5");
        CheckResult(state, v,    MEMBER_ACCESS(t, x6, i), "x6");
        CheckResult(state, v,    MEMBER_ACCESS(t, x7, i), "x7");
        CheckResult(state, m,    MEMBER_ACCESS(t, x8, i), "x8");
        CheckResult(state, m,    MEMBER_ACCESS(t, x9, i), "x9");
    }
    state.counters["n_elem"] = n;
}
