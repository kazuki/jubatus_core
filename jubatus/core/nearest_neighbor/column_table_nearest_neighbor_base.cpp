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
#include "column_table_nearest_neighbor_base.hpp"
#include "../storage/column_table.hpp"
#include "jubatus/util/concurrent/rwmutex.h"

using std::pair;
using std::string;
using std::vector;
using jubatus::util::lang::shared_ptr;

namespace jubatus {
namespace core {
namespace nearest_neighbor {

column_table_nearest_neighbor_base::column_table_nearest_neighbor_base(
    shared_ptr<storage::column_table> table,
    const std::string& id)
    : nearest_neighbor_base(id),
      mixable_table_(new framework::mixable_versioned_table) {
  mixable_table_->set_model(table);
}

void column_table_nearest_neighbor_base::get_all_row_ids(vector<string>& ids) const {
  vector<string> ret;
  shared_ptr<const storage::column_table> table = get_const_table();
  util::concurrent::scoped_rlock lk(table->get_mutex());

  /* table lock acquired; all subsequent table operations must be nolock */

  uint64_t table_size = table->size_nolock();
  ret.reserve(table_size);
  for (size_t i = 0; i < table_size; ++i) {
    ret.push_back(table->get_key_nolock(i));
  }
  ret.swap(ids);
}

void column_table_nearest_neighbor_base::delete_row(const std::string& id) {
  get_table()->delete_row(id);
}

void column_table_nearest_neighbor_base::clear() {
  mixable_table_->get_model()->clear();  // lock acquired inside
}

void column_table_nearest_neighbor_base::pack(framework::packer& packer) const {
  get_const_table()->pack(packer);  // lock acquired inside
}

void column_table_nearest_neighbor_base::unpack(msgpack::object o) {
  get_table()->unpack(o);  // lock acquired inside
}

framework::mixable* column_table_nearest_neighbor_base::get_mixable() const {
  return mixable_table_.get();
}

}  // namespace nearest_neighbor
}  // namespcae core
}  // namespace jubatus
