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

#include "inverted_index.hpp"

namespace jubatus {
namespace core {
namespace nearest_neighbor {

inverted_index::inverted_index(const std::string& id)
  : nearest_neighbor_base(id), mixable_storage_(), orig_() {
  typedef storage::inverted_index_storage ii_storage;
  typedef storage::mixable_inverted_index_storage mii_storage;
  jubatus::util::lang::shared_ptr<ii_storage> p(new ii_storage);
  mixable_storage_.reset(new mii_storage(p));
}

inverted_index::inverted_index(
    const std::string& id,
    const jubatus::util::lang::shared_ptr<unlearner::unlearner_base> &unl)
  : nearest_neighbor_base(id), mixable_storage_(), orig_() {
  typedef storage::inverted_index_storage ii_storage;
  typedef storage::mixable_inverted_index_storage mii_storage;
  jubatus::util::lang::shared_ptr<ii_storage> p(new ii_storage);
  p->set_unlearner(unl);
  mixable_storage_.reset(new mii_storage(p));
}

inverted_index::~inverted_index() {
}

void inverted_index::get_all_row_ids(std::vector<std::string>& ids) const {
  mixable_storage_->get_model()->get_all_column_ids(ids);
}

void inverted_index::clear() {
  mixable_storage_->get_model()->clear();
}

void inverted_index::set_row(const std::string& id, const common::sfv_t& sfv){
  storage::inverted_index_storage& inv = *mixable_storage_->get_model();
  if (orig_) {
    orig_->set_row(id, sfv);
  }
  for (size_t i = 0; i < sfv.size(); ++i) {
    inv.set(sfv[i].first, id, sfv[i].second);
  }
}

void inverted_index::delete_row(const std::string& id) {
  if (!orig_) {
    throw std::runtime_error("needs to call setup_original_storage method");
  }
  std::vector<std::pair<std::string, float> > columns;
  orig_->get_row(id, columns);
  storage::inverted_index_storage& inv = *mixable_storage_->get_model();
  for (size_t i = 0; i < columns.size(); ++i) {
    inv.remove(columns[i].first, id);
  }
  orig_->remove_row(id);
}

void inverted_index::neighbor_row(
    const common::sfv_t& query,
    std::vector<std::pair<std::string, float> >& ids,
    uint64_t ret_num) const {
  ids.clear();
  if (ret_num == 0)
    return;
  mixable_storage_->get_model()->calc_scores(query, ids, ret_num);
  for (size_t i = 0; i < ids.size(); ++i) {
    ids[i].second = 1 - ids[i].second;
  }
}

void inverted_index::neighbor_row(
    const std::string& query_id,
    std::vector<std::pair<std::string, float> >& ids,
    uint64_t ret_num) const {
  if (!orig_) {
    throw std::runtime_error("needs to call setup_original_storage method");
  }
  common::sfv_t sfv;
  orig_->get_row(query_id, sfv);
  neighbor_row(sfv, ids, ret_num);
}

void inverted_index::pack(framework::packer& packer) const {
  mixable_storage_->get_model()->pack(packer);
}

void inverted_index::unpack(msgpack::object o) {
  mixable_storage_->get_model()->unpack(o);
}

framework::mixable* inverted_index::get_mixable() const {
  return mixable_storage_.get();
}

void inverted_index::setup_original_storage(
    const jubatus::util::lang::shared_ptr<storage::sparse_matrix_storage> &orig) {
  orig_ = orig;
}

}  // namespace nearest_neighbor
}  // namespace core
}  // namespace jubatus
