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

#ifndef JUBATUS_CORE_NEAREST_NEIGHBOR_INVERTED_INDEX_HPP_
#define JUBATUS_CORE_NEAREST_NEIGHBOR_INVERTED_INDEX_HPP_

#include <stdint.h>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>
#include "nearest_neighbor_base.hpp"
#include "../storage/inverted_index_storage.hpp"
#include "../unlearner/unlearner_base.hpp"

namespace jubatus {
namespace core {
namespace nearest_neighbor {

class inverted_index : public nearest_neighbor_base {
public:
  explicit inverted_index(const std::string& id);
  inverted_index(const std::string& id,
                 const jubatus::util::lang::shared_ptr<unlearner::unlearner_base> &unl);
  virtual ~inverted_index();

  virtual void get_all_row_ids(std::vector<std::string>& ids) const;
  virtual std::string type() const {
    return "inverted_index";
  }
  virtual void clear();
  virtual void set_row(const std::string& id, const common::sfv_t& sfv);
  virtual void delete_row(const std::string& id);
  virtual void neighbor_row(
      const common::sfv_t& query,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const;
  virtual void neighbor_row(
      const std::string& query_id,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const;

  virtual void pack(framework::packer& packer) const;
  virtual void unpack(msgpack::object o);

  virtual framework::mixable* get_mixable() const;

  virtual void setup_original_storage(const jubatus::util::lang::shared_ptr<storage::sparse_matrix_storage> &orig);

protected:
  jubatus::util::lang::shared_ptr<storage::mixable_inverted_index_storage>
    mixable_storage_;
  jubatus::util::lang::shared_ptr<storage::sparse_matrix_storage> orig_;
};

}  // namespace nearest_neighbor
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_NEAREST_NEIGHBOR_INVERTED_INDEX_HPP_
