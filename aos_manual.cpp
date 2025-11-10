#define AOS_MANUAL

#include "benchmark.h"

struct Snbody {
	double b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, x, y, z, vx, vy, vz, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23;
};

struct Sstencil {
	double b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23, src, dst, rhs, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23;
};

struct Particle {
	double x, y, z, M;
};

/// Register Benchmarks ///
template <typename S, typename N>
class Fixture1 : public benchmark::Fixture {
 public:
    static constexpr auto n = N::value;
    S t[n];
};


template <typename S1, typename S2, typename N>
class Fixture2 : public benchmark::Fixture {
 public:
    static constexpr auto n = N::value;
    S1 t1[n];
    S2 t2[n];
};

BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM_nbody, Snbody, std::integral_constant<size_t, N_nbody>)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture1, BM_stencil, Sstencil, std::integral_constant<size_t, N_stencil>)->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE_INSTANTIATE_F(Fixture2, BM_InvariantMass, Particle, Particle, std::integral_constant<size_t, N_im>)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
