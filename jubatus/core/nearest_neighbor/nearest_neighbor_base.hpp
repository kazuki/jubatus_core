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

#ifndef JUBATUS_CORE_NEAREST_NEIGHBOR_NEAREST_NEIGHBOR_BASE_HPP_
#define JUBATUS_CORE_NEAREST_NEIGHBOR_NEAREST_NEIGHBOR_BASE_HPP_

#include <stdint.h>
#include <iosfwd>
#include <string>
#include <utility>
#include <vector>
#include "jubatus/util/lang/shared_ptr.h"
#include "../common/type.hpp"
#include "../framework/mixable.hpp"
#include "../framework/packer.hpp"

namespace jubatus {
namespace core {
namespace nearest_neighbor {

class nearest_neighbor_base {
 public:
  explicit nearest_neighbor_base(const std::string& id);
  virtual ~nearest_neighbor_base() {}

  virtual void get_all_row_ids(std::vector<std::string>& ids) const = 0;
  virtual std::string type() const = 0;
  virtual void clear() = 0;
  virtual void set_row(const std::string& id, const common::sfv_t& sfv) = 0;
  virtual void delete_row(const std::string& id) = 0;
  virtual void neighbor_row(
      const common::sfv_t& query,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const = 0;
  virtual void neighbor_row(
      const std::string& query_id,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const = 0;
  virtual float calc_similarity(float distance) const {
    return 1 - distance;
  }

  virtual void similar_row(
      const common::sfv_t& query,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const;
  virtual void similar_row(
      const std::string& query_id,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const;

  virtual void pack(framework::packer& packer) const = 0;
  virtual void unpack(msgpack::object o) = 0;

  virtual framework::mixable* get_mixable() const = 0;

 protected:
  std::string my_id_;

};

}  // namespace nearest_neighbor
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_NEAREST_NEIGHBOR_NEAREST_NEIGHBOR_BASE_HPP_
