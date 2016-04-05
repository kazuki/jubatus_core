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

#include "euclid_lsh2.hpp"
#include "bit_vector_ranking.hpp"

namespace jubatus {
namespace core {
namespace nearest_neighbor {

namespace {
static float calc_norm(const common::sfv_t& sfv) {
  float sqnorm = 0;
  for (size_t i = 0; i < sfv.size(); ++i) {
    sqnorm += sfv[i].second * sfv[i].second;
  }
  return std::sqrt(sqnorm);
}
}

euclid_lsh2::euclid_lsh2(const config& config, const std::string& id)
  : nearest_neighbor_base(id),
    mixable_storage_(),
    bin_width_(config.bin_width),
    num_probe_(config.probe_num),
    threads_(jubatus::core::nearest_neighbor::read_threads_config(config.threads)),
    cache_() {

  if (!(1 <= config.hash_num)) {
    throw JUBATUS_EXCEPTION(
        common::invalid_parameter("1 <= hash_num"));
  }

  if (!(1 <= config.table_num)) {
    throw JUBATUS_EXCEPTION(
        common::invalid_parameter("1 <= table_num"));
  }

  if (!(0.f < config.bin_width)) {
    throw JUBATUS_EXCEPTION(
        common::invalid_parameter("0.0 < bin_width"));
  }

  if (!(0 <= config.probe_num)) {
    throw JUBATUS_EXCEPTION(
        common::invalid_parameter("0 <= probe_num"));
  }

  int32_t seed = 1091;
  if (config.seed.bool_test()) {
    if (!(0 <= *config.seed)) {
      throw JUBATUS_EXCEPTION(common::invalid_parameter("0 <= seed"));
    }
    seed = *config.seed;
  }

  jubatus::core::nearest_neighbor::init_cache_from_config(cache_, config.cache_size);

  typedef storage::mixable_lsh_index_storage mli_storage;
  typedef mli_storage::model_ptr model_ptr;
  typedef storage::lsh_index_storage li_storage;

  model_ptr p(new li_storage(config.hash_num, config.table_num, seed));
  mixable_storage_.reset(new mli_storage(p));
}

euclid_lsh2::~euclid_lsh2() {
}

void euclid_lsh2::get_all_row_ids(std::vector<std::string>& ids) const {
  mixable_storage_->get_model()->get_all_row_ids(ids);
}

void euclid_lsh2::clear() {
  mixable_storage_->get_model()->clear();
}

void euclid_lsh2::set_row(const std::string& id, const common::sfv_t& sfv) {
  storage::lsh_index_storage& lsh_index = *mixable_storage_->get_model();
  const std::vector<float> hash = calculate_lsh(sfv);
  const float norm = calc_norm(sfv);
  lsh_index.set_row(id, hash, norm);
}

void euclid_lsh2::delete_row(const std::string& id) {
  mixable_storage_->get_model()->remove_row(id);
}

void euclid_lsh2::similar_row(
    const common::sfv_t& query,
    std::vector<std::pair<std::string, float> >& ids,
    uint64_t ret_num) const {
  storage::lsh_index_storage& lsh_index = *mixable_storage_->get_model();
  ids.clear();

  const std::vector<float> hash = calculate_lsh(query);
  const float norm = calc_norm(query);
  lsh_index.similar_row(hash, norm, num_probe_, ret_num, ids);
}

void euclid_lsh2::similar_row(
    const std::string& query_id,
    std::vector<std::pair<std::string, float> >& ids,
    uint64_t ret_num) const {
  ids.clear();
  mixable_storage_->get_model()->similar_row(query_id, ret_num, ids);
}

void euclid_lsh2::neighbor_row(
    const common::sfv_t& query,
    std::vector<std::pair<std::string, float> >& ids,
    uint64_t ret_num) const {
  similar_row(query, ids, ret_num);
  for (size_t i = 0; i < ids.size(); ++i) {
    ids[i].second = -ids[i].second;
  }
}

void euclid_lsh2::neighbor_row(
    const std::string& query_id,
    std::vector<std::pair<std::string, float> >& ids,
    uint64_t ret_num) const {
  similar_row(query_id, ids, ret_num);
  for (size_t i = 0; i < ids.size(); ++i) {
    ids[i].second = -ids[i].second;
  }
}

std::vector<float> euclid_lsh2::calculate_lsh(const common::sfv_t& query) const {
  std::vector<float> hash = jubatus::core::nearest_neighbor::random_projection(
    query, mixable_storage_->get_model()->all_lsh_num(), threads_, cache_);
  for (size_t j = 0; j < hash.size(); ++j) {
    hash[j] /= bin_width_;
  }
  return hash;
}

void euclid_lsh2::pack(framework::packer& packer) const {
  mixable_storage_->get_model()->pack(packer);
}

void euclid_lsh2::unpack(msgpack::object o) {
  mixable_storage_->get_model()->unpack(o);
}

framework::mixable* euclid_lsh2::get_mixable() const {
  return mixable_storage_.get();
}

}  // namespace nearest_neighbor
}  // namespace core
}  // namespace jubatus
