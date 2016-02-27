#include "xorshift128plus.h"

namespace jubatus {
namespace util{
namespace math{
namespace random{

static uint64_t _calc_hash(const uint8_t* s, int n) {
    // FNV-1 hash function
    uint64_t hash = 14695981039346656037LLU;
    for (int i = 0; i < n; ++i) {
        hash *= 1099511628211LLU;
        hash ^= s[i];
    }
    return hash;
}

xorshift128plus::xorshift128plus(uint32_t seed) {
    r = 0;
    s[0] = _calc_hash((uint8_t*)&seed, 4);
    s[1] = _calc_hash((uint8_t*)&s[0], 4);
}

xorshift128plus::~xorshift128plus() {}

uint32_t xorshift128plus::next() {
    if (r == 0) {
        uint64_t s1 = s[0];
        const uint64_t s0 = s[1];
        s[0] = s0;
        s1 ^= s1 << 23;
        s[1] = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26);
        uint64_t tmp = s[1] + s0;
        r = tmp | ((uint64_t)1 << 63);
        return tmp >> 32;
    } else {
        uint32_t tmp = (uint32_t)r;
        r = 0;
        return tmp;
    }
}

} // random
} // math
} // util
} // jubatus
