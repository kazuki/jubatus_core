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

#ifndef JUBATUS_CORE_NEAREST_NEIGHBOR_EUCLID_LSH_HPP_
#define JUBATUS_CORE_NEAREST_NEIGHBOR_EUCLID_LSH_HPP_

#include <map>
#include <string>
#include <utility>
#include <vector>
#include "jubatus/util/data/serialization.h"
#include "jubatus/util/lang/shared_ptr.h"
#include "nearest_neighbor_base.hpp"
#include "lsh_function.hpp"

namespace jubatus {
namespace core {
namespace storage {
class column_table;
}
namespace nearest_neighbor {

class euclid_lsh : public nearest_neighbor_base {
 public:
  struct config {
    config()
        : hash_num(64u) {
    }

    // TODO(beam2d): make it uint32_t (by modifying pficommon)
    int32_t hash_num;

    template <typename Ar>
    void serialize(Ar& ar) {
      ar & JUBA_MEMBER(hash_num);
    }
  };

  euclid_lsh(
      const config& conf,
      jubatus::util::lang::shared_ptr<storage::column_table> table,
      const std::string& id);
  euclid_lsh(
      const config& conf,
      jubatus::util::lang::shared_ptr<storage::column_table> table,
      std::vector<storage::column_type>& schema,
      const std::string& id);
#ifdef USE_OPENCL
  virtual ~euclid_lsh();
#endif

  virtual std::string type() const {
    return "euclid_lsh";
  }

  virtual void set_row(const std::string& id, const common::sfv_t& sfv);
  virtual void neighbor_row(
      const common::sfv_t& query,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const;
  virtual void neighbor_row(
      const std::string& query,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const;

  virtual float calc_similarity(float distance) const {
    return -distance;
  }

 private:
  void set_config(const config& conf);
  void fill_schema(std::vector<storage::column_type>& schema);
  storage::const_bit_vector_column& lsh_column() const;
  storage::const_float_column& norm_column() const;

  void neighbor_row_from_hash(
      const storage::bit_vector& bv,
      float norm,
      std::vector<std::pair<std::string, float> >& ids,
      uint64_t ret_num) const;

  uint64_t first_column_id_;
  uint32_t hash_num_;
  mutable projection_cache_t cache_;
  mutable rw_mutex cache_mutex_;
#ifdef USE_OPENCL
  jubatus::util::cl::Program cl_prog_;
  cl_kernel cl_kernel_;
  std::function<void(const uint64_t*,float, float*)> cl_func_;
  uint64_t *cl_bv_;
  mutable float *cl_scores_;
  mutable size_t cl_scores_capacity_;
  void init_cl();
#endif
};

}  // namespace nearest_neighbor_base
}  // namespace core
}  // namespace jubatus

#endif  // JUBATUS_CORE_NEAREST_NEIGHBOR_EUCLID_LSH_HPP_
