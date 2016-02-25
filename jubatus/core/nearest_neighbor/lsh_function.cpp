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

#include <iostream>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
#endif
#include "jubatus/util/math/random.h"
#include "../common/hash.hpp"
#include "../common/type.hpp"
#include "../storage/bit_vector.hpp"
#include "lsh_function.hpp"

#define USE_CACHE
#define USE_POLAR_METHOD
#define USE_AVX2
#if !defined(__AVX2__) && defined(USE_AVX2)
#undef USE_AVX2
#endif

using std::vector;
using jubatus::core::storage::bit_vector;

namespace jubatus {
namespace core {
namespace nearest_neighbor {

#ifdef USE_AVX2

vector<float> random_projection(const common::sfv_t& sfv, uint32_t hash_num, projection_cache_t& cache, rw_mutex& mutex) {
  uint32_t proj_size = hash_num / 8;
  __m256 *proj = (__m256*)_mm_malloc(sizeof(__m256) * proj_size, sizeof(__m256));
  for (uint32_t i = 0; i < proj_size; ++i)
    proj[i] = _mm256_xor_ps(proj[i], proj[i]);
  __m256 r0, r1;

#ifdef USE_CACHE
  bool hold_wlock = false;
  mutex.read_lock();
  for (size_t i = 0; i < sfv.size(); ++i) {
    projection_cache_t::const_iterator it = cache.find(sfv[i].first);
    if (it != cache.end()) {
      const vector<float>& random_vector = it->second;
      __m256 fv = _mm256_set1_ps(sfv[i].second);
      for (uint32_t j = 0; j < proj_size; ++j) {
        r0 = _mm256_loadu_ps(random_vector.data() + j * 8);
        proj[j] = _mm256_add_ps(proj[j], _mm256_mul_ps(fv, r0));
      }
    } else {
      vector<float> random_vector;
      random_vector.reserve(hash_num);
#else
  for (size_t i = 0; i < sfv.size(); ++i) {
#endif
    const uint32_t seed = common::hash_util::calc_string_hash(sfv[i].first);
    jubatus::util::math::random::xorshift128_rand rnd(seed);
    __m256 fv = _mm256_set1_ps(sfv[i].second);
    for (uint32_t j = 0; j < proj_size; j += 2) {
      rnd.next_gaussian_mm256(r0, r1);
      proj[j + 0] = _mm256_add_ps(proj[j + 0], _mm256_mul_ps(fv, r0));
      proj[j + 1] = _mm256_add_ps(proj[j + 1], _mm256_mul_ps(fv, r1));
#ifdef USE_CACHE
      for (int k = 0; k < 8; ++k)
          random_vector.push_back(((float*)&r0)[k]);
      for (int k = 0; k < 8; ++k)
          random_vector.push_back(((float*)&r1)[k]);
    }
    if (!hold_wlock) {
      mutex.unlock();
      mutex.write_lock();
      hold_wlock = true;
      if (cache.find(sfv[i].first) != cache.end())
        continue;
    }
    cache.insert(std::make_pair(sfv[i].first, random_vector));
#else
    }
#endif
  }
#ifdef USE_CACHE
  }
  mutex.unlock();
#endif

  vector<float> ret(hash_num);
  for (uint32_t i = 0; i < proj_size; ++i)
      _mm256_storeu_ps(static_cast<float*>(ret.data()) + i * 8, proj[i]);
  _mm_free(proj);
  return ret;
}

#else

vector<float> random_projection(const common::sfv_t& sfv, uint32_t hash_num, projection_cache_t& cache, rw_mutex& mutex) {
  vector<float> proj(hash_num);
#ifdef USE_CACHE
  bool hold_wlock = false;
  mutex.read_lock();
  for (size_t i = 0; i < sfv.size(); ++i) {
    projection_cache_t::const_iterator it = cache.find(sfv[i].first);
    if (it != cache.end()) {
      const vector<float>& random_vector = it->second;
      for (uint32_t j = 0; j < hash_num; ++j) {
        proj[j] += sfv[i].second * random_vector[j];
      }
    } else {
      vector<float> random_vector;
      random_vector.reserve(hash_num);
#else
  for (size_t i = 0; i < sfv.size(); ++i) {
#endif
      const uint32_t seed = common::hash_util::calc_string_hash(sfv[i].first);
      jubatus::util::math::random::xorshift128_rand rnd(seed);
      for (uint32_t j = 0; j < hash_num; ++j) {
#ifdef USE_POLAR_METHOD
        const float r = rnd.next_gaussian32();
#else
        const float r = rnd.next_gaussian();
#endif
        proj[j] += sfv[i].second * r;
#ifndef USE_CACHE
      }
#else
        random_vector.push_back(r);
      }
      if (!hold_wlock) {
          mutex.unlock();
          mutex.write_lock();
          hold_wlock = true;
          if (cache.find(sfv[i].first) != cache.end())
              continue;
      }
      cache.insert(std::make_pair(sfv[i].first, random_vector));
    }
#endif
  }
#ifdef USE_CACHE
  mutex.unlock();
#endif
  return proj;
}
#endif

bit_vector binarize(const vector<float>& proj) {
  bit_vector bv(proj.size());
  for (size_t i = 0; i < proj.size(); ++i) {
    if (proj[i] > 0) {
      bv.set_bit(i);
    }
  }
  return bv;
}

bit_vector cosine_lsh(const common::sfv_t& sfv, uint32_t hash_num, projection_cache_t& cache, rw_mutex& mutex) {
  return binarize(random_projection(sfv, hash_num, cache, mutex));
}

}  // namespace nearest_neighbor
}  // namespace core
}  // namespace jubatus
