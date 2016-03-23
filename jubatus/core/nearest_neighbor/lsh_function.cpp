// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2011 Preferred Networks and Nippon Telegraph and Telephone Corporation.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License version 2.1 as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

#include <vector>
#include "jubatus/util/math/random.h"
#include "../common/hash.hpp"
#include "../common/type.hpp"
#include "../common/thread_pool.hpp"
#include "../storage/bit_vector.hpp"
#include "lsh_function.hpp"
#include "bit_vector_ranking.hpp"

#if defined(JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

using std::vector;
using jubatus::core::storage::bit_vector;

#if defined(__SSE2__) || defined(JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING)
template <class RND>
inline static void next_gaussian_float8(RND& g, float *out);
#endif

#ifdef JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING
template <class RND> __attribute__((target("avx2")))
inline static void next_gaussian_float16(RND& g, float *out);

static std::vector<float> random_projection_internal(
  const jubatus::core::common::sfv_t& sfv, uint32_t hash_num, size_t start, size_t end)
  __attribute__((target("default")));
static std::vector<float> random_projection_internal(
  const jubatus::core::common::sfv_t& sfv, uint32_t hash_num, size_t start, size_t end)
  __attribute__((target("sse2")));
static std::vector<float> random_projection_internal(
  const jubatus::core::common::sfv_t& sfv, uint32_t hash_num, size_t start, size_t end)
  __attribute__((target("avx2")));
#else
static std::vector<float> random_projection_internal(
  const jubatus::core::common::sfv_t& sfv, uint32_t hash_num, size_t start, size_t end);
#endif
static std::vector<float> random_projection_dispatcher(
  const jubatus::core::common::sfv_t *sfv, uint32_t hash_num, size_t start, size_t end)
{
  return random_projection_internal(*sfv, hash_num, start, end);
}

namespace jubatus {
namespace core {
namespace nearest_neighbor {

bit_vector binarize(const vector<float>& proj) {
  bit_vector bv(proj.size());
  for (size_t i = 0; i < proj.size(); ++i) {
    if (proj[i] > 0) {
      bv.set_bit(i);
    }
  }
  return bv;
}

bit_vector cosine_lsh(const common::sfv_t& sfv, uint32_t hash_num, uint32_t threads) {
  return binarize(random_projection(sfv, hash_num, threads));
}

vector<float> random_projection(const common::sfv_t& sfv, uint32_t hash_num, uint32_t threads) {
  typedef std::vector<jubatus::util::lang::shared_ptr<jubatus::core::common::thread_pool::future<std::vector<float> > > > future_list_t;
  if (threads > 1 && sfv.size() > 0) {
    size_t block_size = static_cast<size_t>(std::ceil(sfv.size() / static_cast<float>(threads)));
    vector<float> proj(hash_num);
    std::vector<jubatus::util::lang::function<std::vector<float>()> > funcs;
    funcs.reserve(sfv.size() / block_size + 1);
    for (size_t t = 0, end = 0; t < threads && end < sfv.size() ; ++t) {
      size_t off = end;
      end += std::min(block_size, sfv.size() - off);
      funcs.push_back(jubatus::util::lang::bind(
        &random_projection_dispatcher, &sfv, hash_num, off, end));
    }
    future_list_t futures = jubatus::core::common::default_thread_pool::async_all(funcs);
    for (typename future_list_t::iterator it = futures.begin(); it != futures.end(); ++it) {
      const std::vector<float>& pj = (*it)->get();
      for (size_t i = 0; i < proj.size(); ++i)
        proj[i] += pj[i];
    }
    return proj;
  } else {
    return random_projection_internal(sfv, hash_num, 0, sfv.size());
  }
}

}  // namespace nearest_neighbor
}  // namespace core
}  // namespace jubatus

#if !defined(__SSE2__) || defined(JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING)
#ifdef JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING
__attribute__((target("default")))
#endif
vector<float> random_projection_internal(
  const jubatus::core::common::sfv_t& sfv, uint32_t hash_num, size_t start, size_t end)
{
  vector<float> proj(hash_num);
  for (size_t i = start; i < end; ++i) {
    const uint32_t seed = jubatus::core::common::hash_util::calc_string_hash(sfv[i].first);
    jubatus::util::math::random::sfmt607rand rnd(seed);
    for (uint32_t j = 0; j < hash_num; ++j) {
      proj[j] += sfv[i].second * rnd.next_gaussian_float();
    }
  }
  return proj;
}
#endif // #if !defined(__SSE2__) || defined(JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING)

#if defined(__SSE2__) || defined(JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING)
#ifdef JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING
__attribute__((target("sse2")))
#endif
vector<float> random_projection_internal(
  const jubatus::core::common::sfv_t& sfv, uint32_t hash_num, size_t start, size_t end)
{
  std::vector<float> proj(hash_num);
  float *p = const_cast<float*>(proj.data());
  const uint32_t hash_num_sse = hash_num & 0xfffffff8;
  float grnd[8] __attribute__((aligned(16)));
  for (size_t i = start; i < end; ++i) {
    const uint32_t seed = jubatus::core::common::hash_util::calc_string_hash(sfv[i].first);
    jubatus::util::math::random::sfmt607rand rnd(seed);
    const float v = sfv[i].second;
    __m128 v4 = _mm_set1_ps(v);
    uint32_t j = 0;
    for (; j < hash_num_sse; j += 8) {
      next_gaussian_float8(rnd, grnd);
      __m128 t0 = _mm_loadu_ps(p + j);
      __m128 t1 = _mm_loadu_ps(p + j + 4);
      __m128 t2 = _mm_mul_ps(v4, _mm_load_ps(grnd));
      __m128 t3 = _mm_mul_ps(v4, _mm_load_ps(grnd + 4));
      _mm_storeu_ps(p + j + 0, _mm_add_ps(t0, t2));
      _mm_storeu_ps(p + j + 4, _mm_add_ps(t1, t3));
    }
    for (; j < hash_num; ++j) {
      proj[j] += v * rnd.next_gaussian_float();
    }
  }
  return proj;
}
#endif // #if defined(__SSE2__) || defined(JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING)

#ifdef JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING
__attribute__((target("avx2")))
vector<float> random_projection_internal(
  const jubatus::core::common::sfv_t& sfv, uint32_t hash_num, size_t start, size_t end)
{
  std::vector<float> proj(hash_num);
  float *p = const_cast<float*>(proj.data());
  uint32_t hash_num_avx = hash_num & 0xfffffff0;
  float grnd[16] __attribute__((aligned(32)));
  for (size_t i = start; i < end; ++i) {
    const uint32_t seed = jubatus::core::common::hash_util::calc_string_hash(sfv[i].first);
    jubatus::util::math::random::sfmt607rand rnd(seed);
    const float v = sfv[i].second;
    __m256 v8 = _mm256_set1_ps(v);
    uint32_t j = 0;
    for (; j < hash_num_avx; j += 16) {
      next_gaussian_float16(rnd, grnd);
      __m256 t0 = _mm256_loadu_ps(p + j);
      __m256 t1 = _mm256_loadu_ps(p + j + 8);
      __m256 t2 = _mm256_mul_ps(v8, _mm256_load_ps(grnd));
      __m256 t3 = _mm256_mul_ps(v8, _mm256_load_ps(grnd + 8));
      _mm256_storeu_ps(p + j + 0, _mm256_add_ps(t0, t2));
      _mm256_storeu_ps(p + j + 8, _mm256_add_ps(t1, t3));
    }
    for (; j < hash_num; ++j) {
      proj[j] += v * rnd.next_gaussian_float();
    }
  }
  return proj;
}
#endif // #ifdef JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING

#if defined(__SSE2__) || defined(JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING)

/*
   log_ps and sincos_ps function is based from below URL.
   http://gruntthepeon.free.fr/ssemath/

   Modifications:
   * remove SSE1+MMX code
   * remove other math functions
   * function multiversioning friendly
*/
/* SIMD (SSE1+MMX or SSE2) implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library

   The default is to use the SSE1 version. If you define USE_SSE2 the
   the SSE2 intrinsics will be used in place of the MMX intrinsics. Do
   not expect any significant performance improvement with SSE2.
*/
/* Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#define _PS_CONST(Name, Val)                                            \
  static const float _ps_##Name[4] __attribute__((aligned(16))) = { Val, Val, Val, Val }
#define _PI32_CONST(Name, Val)                                            \
  static const int _pi32_##Name[4] __attribute__((aligned(16))) = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val)                                 \
  static const Type _ps_##Name[4] __attribute__((aligned(16))) = { Val, Val, Val, Val }

_PS_CONST(1  , 1.0f);
_PS_CONST(0p5, 0.5f);
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);
_PS_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);
_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);
_PS_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS_CONST(cephes_log_p0, 7.0376836292E-2);
_PS_CONST(cephes_log_p1, - 1.1514610310E-1);
_PS_CONST(cephes_log_p2, 1.1676998740E-1);
_PS_CONST(cephes_log_p3, - 1.2420140846E-1);
_PS_CONST(cephes_log_p4, + 1.4249322787E-1);
_PS_CONST(cephes_log_p5, - 1.6668057665E-1);
_PS_CONST(cephes_log_p6, + 2.0000714765E-1);
_PS_CONST(cephes_log_p7, - 2.4999993993E-1);
_PS_CONST(cephes_log_p8, + 3.3333331174E-1);
_PS_CONST(cephes_log_q1, -2.12194440e-4);
_PS_CONST(cephes_log_q2, 0.693359375);
_PS_CONST(minus_cephes_DP1, -0.78515625);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS_CONST(sincof_p0, -1.9515295891E-4);
_PS_CONST(sincof_p1,  8.3321608736E-3);
_PS_CONST(sincof_p2, -1.6666654611E-1);
_PS_CONST(coscof_p0,  2.443315711809948E-005);
_PS_CONST(coscof_p1, -1.388731625493765E-003);
_PS_CONST(coscof_p2,  4.166664568298827E-002);
_PS_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI
_PS_CONST(minus2, -2.0f);
_PS_CONST(scale, static_cast<float>(1.0 / 16777216.0));
_PS_CONST(twopi, static_cast<float>(2.0 * jubatus::util::math::pi));

#ifdef JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING
__attribute__((target("sse2")))
#endif
__m128 log_ps(__m128 x) {
  __m128i emm0;
  __m128 one = *(__m128*)_ps_1;
  __m128 invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());
  x = _mm_max_ps(x, *(__m128*)_ps_min_norm_pos);  /* cut off denormalized stuff */
  emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);
  x = _mm_and_ps(x, *(__m128*)_ps_inv_mant_mask);
  x = _mm_or_ps(x, *(__m128*)_ps_0p5);
  emm0 = _mm_sub_epi32(emm0, *(__m128i*)_pi32_0x7f);
  __m128 e = _mm_cvtepi32_ps(emm0);
  e = _mm_add_ps(e, one);
  __m128 mask = _mm_cmplt_ps(x, *(__m128*)_ps_cephes_SQRTHF);
  __m128 tmp = _mm_and_ps(x, mask);
  x = _mm_sub_ps(x, one);
  e = _mm_sub_ps(e, _mm_and_ps(one, mask));
  x = _mm_add_ps(x, tmp);
  __m128 z = _mm_mul_ps(x,x);
  __m128 y = *(__m128*)_ps_cephes_log_p0;
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p1);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p2);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p3);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p4);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p5);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p6);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p7);
  y = _mm_mul_ps(y, x);
  y = _mm_add_ps(y, *(__m128*)_ps_cephes_log_p8);
  y = _mm_mul_ps(y, x);
  y = _mm_mul_ps(y, z);
  tmp = _mm_mul_ps(e, *(__m128*)_ps_cephes_log_q1);
  y = _mm_add_ps(y, tmp);
  tmp = _mm_mul_ps(z, *(__m128*)_ps_0p5);
  y = _mm_sub_ps(y, tmp);
  tmp = _mm_mul_ps(e, *(__m128*)_ps_cephes_log_q2);
  x = _mm_add_ps(x, y);
  x = _mm_add_ps(x, tmp);
  x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
  return x;
}

#ifdef JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING
__attribute__((target("sse2")))
#endif
void sincos_ps(__m128 x, __m128 *s, __m128 *c) {
  __m128 xmm1, xmm2, xmm3 = _mm_setzero_ps(), sign_bit_sin, y;
  __m128i emm0, emm2, emm4;
  sign_bit_sin = x;
  x = _mm_and_ps(x, *(__m128*)_ps_inv_sign_mask);
  sign_bit_sin = _mm_and_ps(sign_bit_sin, *(__m128*)_ps_sign_mask);
  y = _mm_mul_ps(x, *(__m128*)_ps_cephes_FOPI);
  emm2 = _mm_cvttps_epi32(y);
  emm2 = _mm_add_epi32(emm2, *(__m128i*)_pi32_1);
  emm2 = _mm_and_si128(emm2, *(__m128i*)_pi32_inv1);
  y = _mm_cvtepi32_ps(emm2);
  emm4 = emm2;
  emm0 = _mm_and_si128(emm2, *(__m128i*)_pi32_4);
  emm0 = _mm_slli_epi32(emm0, 29);
  __m128 swap_sign_bit_sin = _mm_castsi128_ps(emm0);
  emm2 = _mm_and_si128(emm2, *(__m128i*)_pi32_2);
  emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
  __m128 poly_mask = _mm_castsi128_ps(emm2);
  xmm1 = *(__m128*)_ps_minus_cephes_DP1;
  xmm2 = *(__m128*)_ps_minus_cephes_DP2;
  xmm3 = *(__m128*)_ps_minus_cephes_DP3;
  xmm1 = _mm_mul_ps(y, xmm1);
  xmm2 = _mm_mul_ps(y, xmm2);
  xmm3 = _mm_mul_ps(y, xmm3);
  x = _mm_add_ps(x, xmm1);
  x = _mm_add_ps(x, xmm2);
  x = _mm_add_ps(x, xmm3);
  emm4 = _mm_sub_epi32(emm4, *(__m128i*)_pi32_2);
  emm4 = _mm_andnot_si128(emm4, *(__m128i*)_pi32_4);
  emm4 = _mm_slli_epi32(emm4, 29);
  __m128 sign_bit_cos = _mm_castsi128_ps(emm4);
  sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);
  __m128 z = _mm_mul_ps(x,x);
  y = *(__m128*)_ps_coscof_p0;
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, *(__m128*)_ps_coscof_p1);
  y = _mm_mul_ps(y, z);
  y = _mm_add_ps(y, *(__m128*)_ps_coscof_p2);
  y = _mm_mul_ps(y, z);
  y = _mm_mul_ps(y, z);
  __m128 tmp = _mm_mul_ps(z, *(__m128*)_ps_0p5);
  y = _mm_sub_ps(y, tmp);
  y = _mm_add_ps(y, *(__m128*)_ps_1);
  __m128 y2 = *(__m128*)_ps_sincof_p0;
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_add_ps(y2, *(__m128*)_ps_sincof_p1);
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_add_ps(y2, *(__m128*)_ps_sincof_p2);
  y2 = _mm_mul_ps(y2, z);
  y2 = _mm_mul_ps(y2, x);
  y2 = _mm_add_ps(y2, x);
  xmm3 = poly_mask;
  __m128 ysin2 = _mm_and_ps(xmm3, y2);
  __m128 ysin1 = _mm_andnot_ps(xmm3, y);
  y2 = _mm_sub_ps(y2,ysin2);
  y = _mm_sub_ps(y, ysin1);
  xmm1 = _mm_add_ps(ysin1,ysin2);
  xmm2 = _mm_add_ps(y,y2);
  *s = _mm_xor_ps(xmm1, sign_bit_sin);
  *c = _mm_xor_ps(xmm2, sign_bit_cos);
}

template <class RND>
#ifdef JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING
__attribute__((target("sse2")))
#endif
inline static void next_gaussian_float8(RND& g, float *out)
{
  __m128 a, b;
  {
    __m128i t[2] __attribute__((aligned(16)));
    g.fill_int_unsafe((uint32_t*)&(t[0]), 8);
    a = _mm_cvtepi32_ps(_mm_srli_epi32(t[0], 8));
    b = _mm_cvtepi32_ps(_mm_srli_epi32(t[1], 8));
    __m128 c = _mm_unpacklo_ps(a, b);
    __m128 d = _mm_unpackhi_ps(a, b);
    a = _mm_unpacklo_ps(c, d);
    b = _mm_unpackhi_ps(c, d);
  }
  a = _mm_mul_ps(a, *(const __m128*)_ps_scale);
  b = _mm_mul_ps(b, *(const __m128*)_ps_scale);
  a = _mm_sub_ps(*(const __m128*)_ps_1, a);
  b = _mm_sub_ps(*(const __m128*)_ps_1, b);
  b = _mm_mul_ps(b, *(const __m128*)_ps_twopi);
  a = log_ps(a);
  __m128 s, c;
  sincos_ps(b, &s, &c);
  a = _mm_mul_ps(a, *(const __m128*)_ps_minus2);
  a = _mm_sqrt_ps(a);
  b = _mm_mul_ps(a, s);
  a = _mm_mul_ps(a, c);
  _mm_storeu_ps(out + 0, _mm_unpacklo_ps(b, a));
  _mm_storeu_ps(out + 4, _mm_unpackhi_ps(b, a));
}

#endif // #if defined(__SSE2__) || defined(JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING)

#ifdef JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING

/*
   log256_ps and sincos256_ps function is based from below URL.
   http://software-lisc.fbk.eu/avx_mathfun/

   Modifications:
   * remove other math functions
   * remove SSE fallback code
   * function multiversioning friendly
   * fixed compiler warning
*/
/*
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#define _PS256_CONST(Name, Val)       \
  static const float _ps256_##Name[8] \
  __attribute__((aligned(32))) =      \
  { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PS256_CONST_TYPE(Name, Type, Val) \
  static const Type _ps256_##Name[8]       \
  __attribute__((aligned(32))) =           \
  { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PI32AVX_CONST(Name, Val)     \
  static const int _pi32avx_##Name[4] \
  __attribute__((aligned(32))) = { Val, Val, Val, Val }
#define _PI32_CONST256(Name, Val)      \
  static const int _pi32_256_##Name[8] \
  __attribute__((aligned(32))) =       \
  { Val, Val, Val, Val, Val, Val, Val, Val }
_PS256_CONST_TYPE(min_norm_pos, int, static_cast<int>(0x00800000));
_PS256_CONST_TYPE(inv_mant_mask, int, static_cast<int>(~0x7f800000));
_PS256_CONST(cephes_log_p0, 7.0376836292E-2);
_PS256_CONST(cephes_log_p1, - 1.1514610310E-1);
_PS256_CONST(cephes_log_p2, 1.1676998740E-1);
_PS256_CONST(cephes_log_p3, - 1.2420140846E-1);
_PS256_CONST(cephes_log_p4, + 1.4249322787E-1);
_PS256_CONST(cephes_log_p5, - 1.6668057665E-1);
_PS256_CONST(cephes_log_p6, + 2.0000714765E-1);
_PS256_CONST(cephes_log_p7, - 2.4999993993E-1);
_PS256_CONST(cephes_log_p8, + 3.3333331174E-1);
_PS256_CONST(cephes_log_q1, -2.12194440e-4);
_PS256_CONST(cephes_log_q2, 0.693359375);
_PS256_CONST(cephes_SQRTHF, 0.707106781186547524);
_PI32_CONST256(0x7f, 0x7f);

_PS256_CONST_TYPE(sign_mask, int, static_cast<int>(0x80000000));
_PS256_CONST_TYPE(inv_sign_mask, int, static_cast<int>(~0x80000000));
_PS256_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI
_PS256_CONST(minus_cephes_DP1, -0.78515625);
_PS256_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS256_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS256_CONST(sincof_p0, -1.9515295891E-4);
_PS256_CONST(sincof_p1,  8.3321608736E-3);
_PS256_CONST(sincof_p2, -1.6666654611E-1);
_PS256_CONST(coscof_p0,  2.443315711809948E-005);
_PS256_CONST(coscof_p1, -1.388731625493765E-003);
_PS256_CONST(coscof_p2,  4.166664568298827E-002);
_PS256_CONST(1  , 1.0f);
_PS256_CONST(0p5, 0.5f);
_PI32AVX_CONST(1, 1);
_PI32AVX_CONST(inv1, ~1);
_PI32AVX_CONST(2, 2);
_PI32AVX_CONST(4, 4);
_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(4, 4);
_PS256_CONST(minus2, -2.0f);
_PS256_CONST(scale , static_cast<float>(1.0 / 16777216.0));
_PS256_CONST(twopi , static_cast<float>(2.0 * jubatus::util::math::pi));

__attribute__((target("avx2")))
static inline __m256 log256_ps(__m256 x) {
  __m256i imm0;
  __m256 one = *(__m256*)_ps256_1;
  __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);
  x = _mm256_max_ps(x, *(__m256*)_ps256_min_norm_pos);
  imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
  x = _mm256_and_ps(x, *(__m256*)_ps256_inv_mant_mask);
  x = _mm256_or_ps(x, *(__m256*)_ps256_0p5);
  imm0 = _mm256_sub_epi32(imm0, *(__m256i*)_pi32_256_0x7f);
  __m256 e = _mm256_cvtepi32_ps(imm0);
  e = _mm256_add_ps(e, one);
  __m256 mask = _mm256_cmp_ps(x, *(__m256*)_ps256_cephes_SQRTHF, _CMP_LT_OS);
  __m256 tmp = _mm256_and_ps(x, mask);
  x = _mm256_sub_ps(x, one);
  e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
  x = _mm256_add_ps(x, tmp);
  __m256 z = _mm256_mul_ps(x,x);
  __m256 y = *(__m256*)_ps256_cephes_log_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p5);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p6);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p7);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, *(__m256*)_ps256_cephes_log_p8);
  y = _mm256_mul_ps(y, x);
  y = _mm256_mul_ps(y, z);
  tmp = _mm256_mul_ps(e, *(__m256*)_ps256_cephes_log_q1);
  y = _mm256_add_ps(y, tmp);
  tmp = _mm256_mul_ps(z, *(__m256*)_ps256_0p5);
  y = _mm256_sub_ps(y, tmp);
  tmp = _mm256_mul_ps(e, *(__m256*)_ps256_cephes_log_q2);
  x = _mm256_add_ps(x, y);
  x = _mm256_add_ps(x, tmp);
  x = _mm256_or_ps(x, invalid_mask);
  return x;
}

__attribute__((target("avx2")))
static inline void sincos256_ps(__m256 x, __m256 *s, __m256 *c) {
  __m256 xmm1, xmm2, xmm3 = _mm256_setzero_ps(), sign_bit_sin, y;
  __m256i imm0, imm2, imm4;
  sign_bit_sin = x;
  x = _mm256_and_ps(x, *(__m256*)_ps256_inv_sign_mask);
  sign_bit_sin = _mm256_and_ps(sign_bit_sin, *(__m256*)_ps256_sign_mask);
  y = _mm256_mul_ps(x, *(__m256*)_ps256_cephes_FOPI);
  imm2 = _mm256_cvttps_epi32(y);
  imm2 = _mm256_add_epi32(imm2, *(__m256i*)_pi32_256_1);
  imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_inv1);
  y = _mm256_cvtepi32_ps(imm2);
  imm4 = imm2;
  imm0 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_4);
  imm0 = _mm256_slli_epi32(imm0, 29);
  imm2 = _mm256_and_si256(imm2, *(__m256i*)_pi32_256_2);
  imm2 = _mm256_cmpeq_epi32(imm2, *(__m256i*)_pi32_256_0);
  __m256 swap_sign_bit_sin = _mm256_castsi256_ps(imm0);
  __m256 poly_mask = _mm256_castsi256_ps(imm2);
  xmm1 = *(__m256*)_ps256_minus_cephes_DP1;
  xmm2 = *(__m256*)_ps256_minus_cephes_DP2;
  xmm3 = *(__m256*)_ps256_minus_cephes_DP3;
  xmm1 = _mm256_mul_ps(y, xmm1);
  xmm2 = _mm256_mul_ps(y, xmm2);
  xmm3 = _mm256_mul_ps(y, xmm3);
  x = _mm256_add_ps(x, xmm1);
  x = _mm256_add_ps(x, xmm2);
  x = _mm256_add_ps(x, xmm3);
  imm4 = _mm256_sub_epi32(imm4, *(__m256i*)_pi32_256_2);
  imm4 = _mm256_andnot_si256(imm4, *(__m256i*)_pi32_256_4);
  imm4 = _mm256_slli_epi32(imm4, 29);
  __m256 sign_bit_cos = _mm256_castsi256_ps(imm4);
  sign_bit_sin = _mm256_xor_ps(sign_bit_sin, swap_sign_bit_sin);
  __m256 z = _mm256_mul_ps(x,x);
  y = *(__m256*)_ps256_coscof_p0;
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(__m256*)_ps256_coscof_p1);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, *(__m256*)_ps256_coscof_p2);
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  __m256 tmp = _mm256_mul_ps(z, *(__m256*)_ps256_0p5);
  y = _mm256_sub_ps(y, tmp);
  y = _mm256_add_ps(y, *(__m256*)_ps256_1);
  __m256 y2 = *(__m256*)_ps256_sincof_p0;
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(__m256*)_ps256_sincof_p1);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_add_ps(y2, *(__m256*)_ps256_sincof_p2);
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);
  xmm3 = poly_mask;
  __m256 ysin2 = _mm256_and_ps(xmm3, y2);
  __m256 ysin1 = _mm256_andnot_ps(xmm3, y);
  y2 = _mm256_sub_ps(y2,ysin2);
  y = _mm256_sub_ps(y, ysin1);
  xmm1 = _mm256_add_ps(ysin1,ysin2);
  xmm2 = _mm256_add_ps(y,y2);
  *s = _mm256_xor_ps(xmm1, sign_bit_sin);
  *c = _mm256_xor_ps(xmm2, sign_bit_cos);
}

template <class RND> __attribute__((target("avx2")))
inline static void next_gaussian_float16(RND& g, float *out)
{
  __m256 a, b;
  {
    __m256i t[2] __attribute__((aligned(32)));
    g.fill_int_unsafe((uint32_t*)&(t[0]), 16);
    a = _mm256_cvtepi32_ps(_mm256_srli_epi32(t[0], 8));
    b = _mm256_cvtepi32_ps(_mm256_srli_epi32(t[1], 8));
    __m256 c = _mm256_unpacklo_ps(a, b);
    __m256 d = _mm256_unpackhi_ps(a, b);
    a = _mm256_unpacklo_ps(c, d);
    b = _mm256_unpackhi_ps(c, d);
  }
  a = _mm256_mul_ps(a, *(const __m256*)_ps256_scale);
  b = _mm256_mul_ps(b, *(const __m256*)_ps256_scale);
  a = _mm256_sub_ps(*(const __m256*)_ps256_1, a);
  b = _mm256_sub_ps(*(const __m256*)_ps256_1, b);
  b = _mm256_mul_ps(b, *(const __m256*)_ps256_twopi);
  a = log256_ps(a);
  __m256 s, c;
  sincos256_ps(b, &s, &c);
  a = _mm256_mul_ps(a, *(const __m256*)_ps256_minus2);
  a = _mm256_sqrt_ps(a);
  b = _mm256_mul_ps(a, s);
  a = _mm256_mul_ps(a, c);
  _mm256_store_ps(out + 0, _mm256_unpacklo_ps(b, a));
  _mm256_store_ps(out + 8, _mm256_unpackhi_ps(b, a));
}

#endif // #ifdef JUBATUS_ENABLED_FUNCTION_MULTIVERSIONING
