#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/mhc/sinkhorn_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &sinkhorn_fwd<2, 16, 1>;
    return 0;
}
