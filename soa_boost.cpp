#define SOA_BOOST 

#include "benchmark.h"
#include "SoALayout.h"

GENERATE_SOA_LAYOUT(SoALayout,
    SOA_COLUMN(int, x0),
    SOA_COLUMN(int, x1))

using SoA = SoALayout<>;
using SoAView = SoA::View;
using SoAConstView = SoA::ConstView;

std::unique_ptr<std::byte, decltype(std::free) *> buffer{
    reinterpret_cast<std::byte *>(aligned_alloc(SoA::alignment, SoA::computeDataSize(10))), std::free};
SoA soa(buffer.get(), 10);
SoAView soaView{soa};
SoAConstView soaConstView{soa};

BENCHMARK_CAPTURE(BM_CPUMemoryIntensive, cms_soa, soaView);

BENCHMARK_MAIN();
