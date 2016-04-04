// Jubatus: Online machine learning framework for distributed environment
// Copyright (C) 2016 Preferred Networks and Nippon Telegraph and Telephone Corporation.
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

#ifndef JUBATUS_CORE_NEAREST_NEIGHBOR_INVERTED_INDEX_EUCLID_HPP_
#define JUBATUS_CORE_NEAREST_NEIGHBOR_INVERTED_INDEX_EUCLID_HPP_

#include <stdint.h>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>
#include "inverted_index.hpp"

namespace jubatus {
namespace core {
namespace nearest_neighbor {

class inverted_index_euclid : public inverted_index {
public:
  explicit inverted_index_euclid(const std::string& id);
  virtual ~inverted_index_euclid();

  std::string type() const {
    return "inverted_index_euclid";
  }
  void neighbor_row(
      const common::sfv_t& query,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const;
  void neighbor_row(
      const std::string& query_id,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const;
};

}  // namespace nearest_neighbor
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_NEAREST_NEIGHBOR_INVERTED_INDEX_EUCLID_HPP_
