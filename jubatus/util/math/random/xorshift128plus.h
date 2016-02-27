#ifndef JUBATUS_UTIL_MATH_RANDOM_XOR_SHIFT_128_PLUS_H_
#define JUBATUS_UTIL_MATH_RANDOM_XOR_SHIFT_128_PLUS_H_
#include <stdint.h>

namespace jubatus {
namespace util{
namespace math{
namespace random{

class xorshift128plus{
public:
  xorshift128plus(uint32_t seed);
  ~xorshift128plus();

  uint32_t next();

private:
  uint64_t s[2];
  uint64_t r;
};

} // random
} // math
} // util
} // jubatus
#endif // #ifndef JUBATUS_UTIL_MATH_RANDOM_MERSENNE_TWISTER_H_
