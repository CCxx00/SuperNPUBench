#include <common/pto_tileop.hpp>
#include <cstdint>
#include "deepseek/mhc/norm_fn_pto.hpp"
using namespace supernpu::tile_isa;
int main() {
    [[maybe_unused]] auto fn = &fn_normw_merge_fwd<16, 32, 16, 32>;
    return 0;
}
