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

#include "inverted_index_euclid.hpp"

namespace jubatus {
namespace core {
namespace nearest_neighbor {

inverted_index_euclid::inverted_index_euclid(const std::string& id)
  : inverted_index(id) {
}

inverted_index_euclid::~inverted_index_euclid() {
}

void inverted_index_euclid::neighbor_row(
    const common::sfv_t& query,
    std::vector<std::pair<std::string, float> >& ids,
    uint64_t ret_num) const {
  ids.clear();
  if (ret_num == 0)
    return;
  mixable_storage_->get_model()->calc_euclid_scores(query, ids, ret_num);
  for (size_t i = 0; i < ids.size(); ++i) {
    ids[i].second = -ids[i].second;
  }
}

void inverted_index_euclid::neighbor_row(
    const std::string& query_id,
    std::vector<std::pair<std::string, float> >& ids,
    uint64_t ret_num) const {
  throw std::runtime_error("NOT IMPLEMENTED inverted_index_euclid::neighbor_row");
}

}  // namespace nearest_neighbor
}  // namespace core
}  // namespace jubatus
