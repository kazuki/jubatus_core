// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2012 Preferred Networks and Nippon Telegraph and Telephone Corporation.
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

#ifndef JUBATUS_CORE_NEAREST_NEIGHBOR_BIT_VECTOR_RANKING_HPP_
#define JUBATUS_CORE_NEAREST_NEIGHBOR_BIT_VECTOR_RANKING_HPP_

#include <stdint.h>
#include <utility>
#include <vector>
#if __cplusplus >= 201103
#include <cmath>
#include "jubatus/core/common/thread_pool.hpp"
#endif

namespace jubatus {
namespace core {
namespace storage {
template <typename bit_base> class bit_vector_base;
typedef bit_vector_base<uint64_t> bit_vector;

template <typename T>
class typed_column;
typedef typed_column<bit_vector> bit_vector_column;
typedef const bit_vector_column const_bit_vector_column;
}
namespace nearest_neighbor {

void ranking_hamming_bit_vectors(
    const storage::bit_vector& query,
    const storage::const_bit_vector_column& bvs,
    std::vector<std::pair<uint64_t, float> >& ret,
    uint64_t ret_num, uint32_t threads);

#if __cplusplus >= 201103

template <typename Function, typename THeap>
void ranking_hamming_bit_vectors_internal(
    Function&& f, size_t size, uint32_t threads, THeap& heap
) {
  size_t block_size = static_cast<size_t>(std::ceil(size / static_cast<float>(threads)));
  if (threads > 1) {
    std::vector<std::future<THeap> > futures;
    for (size_t t = 0, end = 0; t < threads && end < size ; ++t) {
      size_t off = end;
      end += std::min(block_size, size - off);
      futures.push_back(jubatus::core::common::default_thread_pool::async(f, off, end));
    }
    for (auto it = futures.begin(); it != futures.end(); ++it) {
      heap.merge(it->get());
    }
  } else {
    heap.merge(f(0, size));
  }
}

template <typename T>
uint32_t read_threads_config(T& cfg) {
  if (!cfg.bool_test())
    return 0;
  if (*cfg < 0)
    return std::thread::hardware_concurrency();
  return static_cast<uint32_t>(*cfg);
}

#else // #if __cplusplus >= 201103

template <typename T>
uint32_t read_threads_config(T& cfg) {
  return 0;
}

#endif // #if __cplusplus >= 201103

}  // namespace nearest_neighbor
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_NEAREST_NEIGHBOR_BIT_VECTOR_RANKING_HPP_
