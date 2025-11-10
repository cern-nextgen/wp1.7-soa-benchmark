#define AOS_MANUAL

#include "benchmark.h"

struct Snbody {
	double x, y, z, vx, vy, vz;
};

struct Sstencil {
	double src, dst, rhs;
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
