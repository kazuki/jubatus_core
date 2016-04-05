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

#ifndef JUBATUS_CORE_NEAREST_NEIGHBOR_EUCLID_LSH_2_HPP_
#define JUBATUS_CORE_NEAREST_NEIGHBOR_EUCLID_LSH_2_HPP_

#include "jubatus/util/data/serialization.h"
#include "jubatus/util/data/optional.h"
#include "../storage/lsh_index_storage.hpp"
#include "nearest_neighbor_base.hpp"
#include "lsh_function.hpp"

namespace jubatus {
namespace core {
namespace nearest_neighbor {

class euclid_lsh2 : public nearest_neighbor_base {
public:
  struct config {
    config() :
      hash_num(64), table_num(4), bin_width(100),
      probe_num(64), seed(), threads(), cache_size() {}

    int64_t hash_num;
    int64_t table_num;
    float bin_width;
    int32_t probe_num;
    jubatus::util::data::optional<int32_t> seed;
    jubatus::util::data::optional<int32_t> threads;
    jubatus::util::data::optional<int32_t> cache_size;

    template<typename Ar>
    void serialize(Ar& ar) {
      ar & JUBA_MEMBER(hash_num)
        & JUBA_MEMBER(table_num)
        & JUBA_MEMBER(bin_width)
        & JUBA_MEMBER(probe_num)
        & JUBA_MEMBER(seed)
        & JUBA_MEMBER(threads)
        & JUBA_MEMBER(cache_size);
    }
  };

  explicit euclid_lsh2(const config& conf, const std::string& id);
  virtual ~euclid_lsh2();

  virtual std::string type() const {
    return "euclid_lsh2";
  }

  virtual void get_all_row_ids(std::vector<std::string>& ids) const;
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
  virtual void similar_row(
      const common::sfv_t& query,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const;
  virtual void similar_row(
      const std::string& query_id,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const;

  virtual void pack(framework::packer& packer) const;
  virtual void unpack(msgpack::object o);
  virtual framework::mixable* get_mixable() const;

private:
  std::vector<float> calculate_lsh(const common::sfv_t& query) const;

  jubatus::util::lang::shared_ptr<storage::mixable_lsh_index_storage>
    mixable_storage_;
  float bin_width_;
  uint32_t num_probe_;
  uint32_t threads_;
  mutable cache_t cache_;
};

}  // namespace nearest_neighbor
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_NEAREST_NEIGHBOR_EUCLID_LSH_2_HPP_
