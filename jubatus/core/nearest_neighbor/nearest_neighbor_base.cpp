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

#include <string>
#include <utility>
#include <vector>

#include "exception.hpp"
#include "nearest_neighbor_base.hpp"

using std::pair;
using std::string;
using std::vector;

namespace jubatus {
namespace core {
namespace nearest_neighbor {

nearest_neighbor_base::nearest_neighbor_base(const std::string& id)
    : my_id_(id) {
}

void nearest_neighbor_base::similar_row(
    const common::sfv_t& query,
    vector<pair<string, float> >& ids,
    uint64_t ret_num) const {
  neighbor_row(query, ids, ret_num);  // lock acquired inside
  for (size_t i = 0; i < ids.size(); ++i) {
    ids[i].second = calc_similarity(ids[i].second);
  }
}

void nearest_neighbor_base::similar_row(
    const string& query_id,
    vector<pair<string, float> >& ids,
    uint64_t ret_num) const {
  neighbor_row(query_id, ids, ret_num);  // lock acquired inside
  for (size_t i = 0; i < ids.size(); ++i) {
    ids[i].second = calc_similarity(ids[i].second);
  }
}

}  // namespace nearest_neighbor
}  // namespcae core
}  // namespace jubatus
