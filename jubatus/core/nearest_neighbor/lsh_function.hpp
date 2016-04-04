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

#ifndef JUBATUS_CORE_NEAREST_NEIGHBOR_LSH_FUNCTION_HPP_
#define JUBATUS_CORE_NEAREST_NEIGHBOR_LSH_FUNCTION_HPP_

#include <stdint.h>
#include <vector>
#include "../common/type.hpp"
#include "../storage/bit_vector.hpp"

namespace jubatus {
namespace core {
namespace nearest_neighbor {

std::vector<float> random_projection(const common::sfv_t& sfv, uint32_t hash_num, uint32_t threads);
storage::bit_vector binarize(const std::vector<float>& proj);
storage::bit_vector cosine_lsh(const common::sfv_t& sfv, uint32_t hash_num, uint32_t threads);

}  // namespace nearest_neighbor
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_NEAREST_NEIGHBOR_LSH_FUNCTION_HPP_
