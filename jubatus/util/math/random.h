// Copyright (c)2008-2011, Preferred Infrastructure Inc.
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
// 
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
// 
//     * Neither the name of Preferred Infrastructure nor the names of other
//       contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef JUBATUS_UTIL_MATH_RANDOM_H_
#define JUBATUS_UTIL_MATH_RANDOM_H_

#include <cmath>
#include <vector>
#include <map>
#ifdef __AVX2__
#include <immintrin.h>
#endif

#include "../system/time_util.h"
#include "constant.h"
#include "random/mersenne_twister.h"
#include "random/xorshift128plus.h"
#include "avx_mathfun.h"

namespace jubatus {
namespace util{
namespace math{
namespace random{

template <class Gen>
class random{

public:
  random()
    :g(feeder(jubatus::util::system::time::get_clock_time())),
    next_gaussian_stocked(false){
  }
private:
  uint32_t feeder(jubatus::util::system::time::clock_time ct){
    return (uint32_t)(ct.sec)*(uint32_t)1000000+(uint32_t)(ct.usec);
  }
public:
  random(uint32_t seed)
    :g(seed),
    next_gaussian_stocked(false){
  }
  ~random(){
  }

  /// generate [0,0xffffffff] random number
  uint32_t next_int(){
    return g.next();
  }
  
  /// generate [0,\a a) random integer
  uint32_t next_int(uint32_t a){
    return static_cast<uint32_t>(next_double()*a);
  }

  /// generate [\a a,\a b) random integer
  uint32_t next_int(uint32_t a, uint32_t b){
    return a+next_int(b-a);
  }

  /// generates [0,1) random real number with 53bit resolution
  double next_double(){
    uint32_t a=g.next()>>5;
    uint32_t b=g.next()>>6;
    return (a*67108864.0+b)*(1.0/9007199254740992.0);
  }

  /// generate [0,\a a) random real number
  double next_double(double a){
    return a*next_double();
  }

  /// generate [\a a,\a b) random real number
  double next_double(double a, double b){
    return a+next_double(b-a);
  }

  /// generates [0,1) random real number with 24bit resolution
  float next_float(){
    uint32_t a=g.next()>>8;
    return a * (1.0f/16777216.0f);
  }

#ifdef __AVX2__
  __m256 next_m256(){
      static __m256 scale = _mm256_set1_ps(1.0f/16777216.0f);
      __m256i x = _mm256_set_epi32(g.next(), g.next(), g.next(), g.next(),
                                   g.next(), g.next(), g.next(), g.next());
      x = _mm256_srli_epi32(x, 8);
      return _mm256_mul_ps(_mm256_cvtepi32_ps(x), scale);
  }
#endif

  /// generate normalized standard distribution
  double next_gaussian(){
    if(next_gaussian_stocked){
      next_gaussian_stocked=false; 
      return next_gaussian_stock;
    }else{
      double a = 1.0-next_double();
      double b = 1.0-next_double();
      double r1 = std::sqrt(-2.0*std::log(a))*std::sin(2.0*jubatus::util::math::pi*b);
      double r2 = std::sqrt(-2.0*std::log(a))*std::cos(2.0*jubatus::util::math::pi*b);
      next_gaussian_stock=r2;
      next_gaussian_stocked=true;
      return r1;
    }
  }
  
  /// generate standard distribution with specified \a mean and \a deviation
  double next_gaussian(double mean, double deviation){
    return mean + deviation * next_gaussian();
  }

  float next_gaussian32(){
    if(next_gaussian32_stocked){
      next_gaussian32_stocked=false;
      return next_gaussian32_stock;
    }else{
      float f, a, b, r;
      do {
        a = 2.0 * next_float() - 1.0;
        b = 2.0 * next_float() - 1.0;
        r = a * a + b * b;
      } while (r >= 1.0 || r == 0.0);
      f = std::sqrt(-2.0 * std::log(r) / r);
      next_gaussian32_stock = f * a;
      next_gaussian32_stocked = true;
      return f * b;
    }
  }

#ifdef __AVX2__
  void next_gaussian_mm256(__m256& out0, __m256& out1){
#if 1
    // Box-Muller
    static __m256 one = _mm256_set1_ps(1.0);
    static __m256 mtwo = _mm256_set1_ps(-2.0);
    static __m256 twopi = _mm256_set1_ps(2.0 * jubatus::util::math::pi);
    __m256 a = next_m256();
    __m256 b = next_m256();
    a = _mm256_sub_ps(one, a);
    b = _mm256_sub_ps(one, b);
    a = _mm256_sqrt_ps(_mm256_mul_ps(log256_ps(a), mtwo));
    b = _mm256_mul_ps(b, twopi);
    __m256 r_sin, r_cos;
    sincos256_ps(b, &r_sin, &r_cos);
    a = _mm256_mul_ps(a, r_sin);
    b = _mm256_mul_ps(b, r_sin);
    out0 = _mm256_unpacklo_ps(a, b);
    out1 = _mm256_unpackhi_ps(a, b);

    // より厳密な順序
    /*
    out0[0] = a[0]; out0[1] = b[0];
    out0[2] = a[1]; out0[3] = b[1];
    out0[4] = a[2]; out0[5] = b[2];
    out0[6] = a[3]; out0[7] = b[3];
    out1[0] = a[4]; out1[1] = b[4];
    out1[2] = a[5]; out1[3] = b[5];
    out1[4] = a[6]; out1[5] = b[6];
    out1[6] = a[7]; out1[7] = b[7];
    */
#else
    // Polar
    // (r < 1.0 && r != 0)の条件がなかなか達成できないので案の定，めっちゃ遅い
    static __m256 zero = _mm256_set1_ps(0.0);
    static __m256 one = _mm256_set1_ps(1.0);
    static __m256 two = _mm256_set1_ps(2.0);
    static __m256 mtwo = _mm256_set1_ps(-2.0);
    int c;
    __m256 a, b, r, f;
    do {
      a = _mm256_sub_ps(_mm256_mul_ps(next_m256(), two), one);
      b = _mm256_sub_ps(_mm256_mul_ps(next_m256(), two), one);
      r = _mm256_add_ps(_mm256_mul_ps(a, a), _mm256_mul_ps(b, b));
      c = _mm256_movemask_ps(_mm256_cmp_ps(r, one, _CMP_GE_OS)) |
          _mm256_movemask_ps(_mm256_cmp_ps(r, zero, _CMP_EQ_OQ));
    } while (c != 0);
    f = _mm256_sqrt_ps(_mm256_div_ps(_mm256_mul_ps(log256_ps(r), mtwo), r));
    out0 = _mm256_mul_ps(f, b);
    out1 = _mm256_mul_ps(f, a);
#endif
  }
#endif

  ////mul next_int()
  uint32_t operator()(){
    return next_int();
  }


  ////mul next_int(\a a)
  uint32_t operator()(uint32_t a){
    return next_int(a);
  }

  ////mul next_int(\a a,\a b)
  uint32_t operator()(uint32_t a, uint32_t b){
    return next_int(a,b);
  }

private:
  Gen g;
  bool next_gaussian_stocked; double next_gaussian_stock;
  bool next_gaussian32_stocked; float next_gaussian32_stock;
};

typedef random<mersenne_twister> mtrand;
typedef random<xorshift128plus> xorshift128_rand;

/// select k random integer from range [0,n), allowing multiple occurrence. O(k)
template<typename RAND>
bool sample_with_replacement(RAND& r, int n, int k, std::vector<int>& res) {
  if (n<=0||k<0) return false;
  res.resize(k);
  for (int i=0;i<k;++i) res[i]=r(n);
  return true;
}

/// select k random integer from range [0,n), allowing multiple occurrence. O(k log k)
template<typename RAND>
bool sample_without_replacement(RAND& r, int n, int k, std::vector<int>& res) {
  if (n<=0||k<0||k>n) return false;
  std::map<int,int> mm; // pos -> value
  for (int i=0;i<k;++i) mm[i]=i;
  for (int i=0;i<k;++i) {
    int j=i+r(n-i);
    if (!mm.count(j)) mm[j]=j;
    int t=mm[i];
    mm[i]=mm[j];
    mm[j]=t;
  }
  
  res.resize(k);
  for (std::map<int,int>::iterator it=mm.begin();it!=mm.end();++it) 
    if (0<=it->second&&it->second<k)
      res[it->second]=it->first;
  return true;
}

} // random
} // math
} // util
} // jubatus
#endif // #ifndef JUBATUS_UTIL_MATH_RANDOM_H_
