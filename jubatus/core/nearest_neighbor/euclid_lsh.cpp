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

#include "euclid_lsh.hpp"

#include <map>
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include "jubatus/util/lang/cast.h"
#include "../storage/fixed_size_heap.hpp"
#include "../storage/column_table.hpp"
#include "lsh_function.hpp"

#ifdef USE_OPENCL
#include "jubatus/util/opencl.hpp"
#endif

using std::map;
using std::pair;
using std::make_pair;
using std::string;
using std::vector;
using jubatus::util::lang::lexical_cast;
using jubatus::core::storage::column_table;
using jubatus::core::storage::column_type;
using jubatus::core::storage::owner;
using jubatus::core::storage::bit_vector;
using jubatus::core::storage::const_bit_vector_column;
using jubatus::core::storage::const_float_column;

namespace jubatus {
namespace core {
namespace nearest_neighbor {
namespace {

float squared_l2norm(const common::sfv_t& sfv) {
  float sqnorm = 0;
  for (size_t i = 0; i < sfv.size(); ++i) {
    sqnorm += sfv[i].second * sfv[i].second;
  }
  return sqnorm;
}

float l2norm(const common::sfv_t& sfv) {
  return std::sqrt(squared_l2norm(sfv));
}

}  // namespace

euclid_lsh::euclid_lsh(
    const config& conf,
    jubatus::util::lang::shared_ptr<column_table> table,
    const std::string& id)
    : nearest_neighbor_base(table, id),
      first_column_id_(), hash_num_(),
      cache_(), cache_mutex_()
#ifdef USE_OPENCL
    , cl_prog_(jubatus::util::cl::GPU), cl_kernel_(NULL), cl_scores_(NULL), cl_scores_capacity_(0)
#endif
{
  set_config(conf);

  vector<column_type> schema;
  fill_schema(schema);
  get_table()->init(schema);
#ifdef USE_OPENCL
  init_cl();
#endif
}

euclid_lsh::euclid_lsh(
    const config& conf,
    jubatus::util::lang::shared_ptr<column_table> table,
    vector<column_type>& schema,
    const std::string& id)
    : nearest_neighbor_base(table, id),
      first_column_id_(), hash_num_(),
      cache_(), cache_mutex_()
#ifdef USE_OPENCL
    , cl_prog_(jubatus::util::cl::GPU), cl_kernel_(NULL), cl_scores_(NULL), cl_scores_capacity_(0)
#endif
{
  set_config(conf);
  fill_schema(schema);
#ifdef USE_OPENCL
  init_cl();
#endif
}

#ifdef USE_OPENCL
euclid_lsh:: ~euclid_lsh()
{
  if (cl_bv_)
    clSVMFree(jubatus::util::cl::GPU.context(), cl_bv_);
  if (cl_scores_)
    clSVMFree(jubatus::util::cl::GPU.context(), cl_scores_);
  if (cl_kernel_)
    clReleaseKernel(cl_kernel_);
}
#endif

void euclid_lsh::set_row(const string& id, const common::sfv_t& sfv) {
  // TODO(beam2d): support nested algorithm, e.g. when used by lof and then we
  // cannot suppose that the first two columns are assigned to euclid_lsh.
  get_table()->add(id, owner(my_id_), cosine_lsh(sfv, hash_num_, cache_, cache_mutex_), l2norm(sfv));
}

void euclid_lsh::neighbor_row(
    const common::sfv_t& query,
    vector<pair<string, float> >& ids,
    uint64_t ret_num) const {
  neighbor_row_from_hash(
      cosine_lsh(query, hash_num_, cache_, cache_mutex_),
      l2norm(query),
      ids,
      ret_num);
}

void euclid_lsh::neighbor_row(
    const std::string& query_id,
    vector<pair<string, float> >& ids,
    uint64_t ret_num) const {
  const pair<bool, uint64_t> maybe_index =
      get_const_table()->exact_match(query_id);
  if (!maybe_index.first) {
    ids.clear();
    return;
  }

  const bit_vector bv = lsh_column()[maybe_index.second];
  const float norm = norm_column()[maybe_index.second];
  neighbor_row_from_hash(bv, norm, ids, ret_num);
}

void euclid_lsh::set_config(const config& conf) {
  if (!(1 <= conf.hash_num)) {
    throw JUBATUS_EXCEPTION(
        common::invalid_parameter("1 <= hash_num"));
  }

  hash_num_ = conf.hash_num;
}

void euclid_lsh::fill_schema(vector<column_type>& schema) {
  first_column_id_ = schema.size();
  schema.push_back(column_type(column_type::bit_vector_type, hash_num_));
  schema.push_back(column_type(column_type::float_type));
}

const_bit_vector_column& euclid_lsh::lsh_column() const {
  return get_const_table()->get_bit_vector_column(first_column_id_);
}

const_float_column& euclid_lsh::norm_column() const {
  return get_const_table()->get_float_column(first_column_id_ + 1);
}

void euclid_lsh::neighbor_row_from_hash(
    const bit_vector& bv,
    float norm,
    vector<pair<string, float> >& ids,
    uint64_t ret_num) const {
  jubatus::util::lang::shared_ptr<const column_table> table = get_const_table();

  jubatus::core::storage::fixed_size_heap<pair<float, size_t> > heap(ret_num);
  {
#ifdef USE_OPENCL
    if (cl_scores_capacity_ < table->size()) {
      if (cl_scores_)
        clSVMFree(jubatus::util::cl::GPU.context(), cl_scores_);
      cl_scores_ = (float*)clSVMAlloc(jubatus::util::cl::GPU.context(),
                                       CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                                       table->size() * sizeof(float), 0);
      cl_scores_capacity_ = table->size();
    }
    std::memcpy(cl_bv_, bv.raw_data_unsafe(), hash_num_ / 8);
    cl_func_(cl_bv_, norm, cl_scores_);
    for (size_t i = 0; i < table->size(); ++i) {
      heap.push(make_pair(cl_scores_[i], i));
    }
#else
    const_bit_vector_column& bv_col = lsh_column();
    const_float_column& norm_col = norm_column();
    const float denom = bv.bit_num();
    for (size_t i = 0; i < table->size(); ++i) {
      const size_t hamm_dist = bv.calc_hamming_distance_raw(bv_col.get_pointer_at(i),
                                                            bv_col.type().bit_vector_length());
      const float theta = hamm_dist * M_PI / denom;
      const float score =
          norm_col[i] * (norm_col[i] - 2 * norm * std::cos(theta));
      heap.push(make_pair(score, i));
    }
#endif
  }

  vector<pair<float, size_t> > sorted;
  heap.get_sorted(sorted);

  ids.clear();
  const float squared_norm = norm * norm;
  for (size_t i = 0; i < sorted.size(); ++i) {
    ids.push_back(make_pair(table->get_key(sorted[i].second),
                            std::sqrt(squared_norm + sorted[i].first)));
  }
}

#ifdef USE_OPENCL
const char *OPENCL_CODE =
"// #define BVSIZE = bitvectorのサイズ(ビット数 / 64)\n"
"// #define DENOM = bitvectorのビット数\n"
"// #define PI = M_PI\n"
"__kernel void calc_euclid_lsh_score(__global const ulong *target, __global const ulong *bitvectors, __global const float *norms, float NORM, __global float* restrict scores) {\n"
"  size_t i = get_global_id(0);\n"
"  __global const ulong *bv = bitvectors + (i * BVSIZE);\n"
"  uint hamm_dist = 0;\n"
"  for (int j = 0; j < BVSIZE; ++j) {\n"
"    hamm_dist += popcount(target[j] ^ bv[j]);\n"
"  }\n"
"  const float theta = hamm_dist * PI / DENOM;\n"
"  const float norm = norms[i];\n"
"  scores[i] = norm * (norm - 2 * NORM * native_cos(theta));\n"
"}";

void euclid_lsh::init_cl() {
  std::string opt("-cl-std=CL2.0 -DPI=3.14159265358979323846 -DBVSIZE=");
  opt += std::to_string(hash_num_ / 64);
  opt += " -DDENOM=";
  opt += std::to_string(hash_num_);
  cl_prog_.build(OPENCL_CODE, opt);
  cl_kernel_ = cl_prog_.get("calc_euclid_lsh_score");
  cl_bv_ = (uint64_t*)clSVMAlloc(jubatus::util::cl::GPU.context(),
                                 CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER,
                                 hash_num_ / 8, 0);
  cl_func_ = [this](const uint64_t *bv, float norm, float *scores) {
    const_bit_vector_column& bv_col = this->lsh_column();
    const_float_column& norm_col = this->norm_column();
    const uint64_t *bv_ptr = bv_col.data();
    const float *norm_ptr = norm_col.data();
    size_t global_size = this->get_const_table()->size();
    if (clSetKernelArgSVMPointer(this->cl_kernel_, 0, bv) != CL_SUCCESS)
      throw std::runtime_error("failed clSetKernelArgSVMPointer");
    if (clSetKernelArgSVMPointer(this->cl_kernel_, 1, bv_ptr) != CL_SUCCESS)
      throw std::runtime_error("failed clSetKernelArgSVMPointer");
    if (clSetKernelArgSVMPointer(this->cl_kernel_, 2, norm_ptr) != CL_SUCCESS)
      throw std::runtime_error("failed clSetKernelArgSVMPointer");
    if (clSetKernelArg(this->cl_kernel_, 3, sizeof(float), &norm) != CL_SUCCESS)
      throw std::runtime_error("failed clSetKernelArg");
    if (clSetKernelArgSVMPointer(this->cl_kernel_, 4, scores) != CL_SUCCESS)
      throw std::runtime_error("failed clSetKernelArgSVMPointer");
    cl_event evt;
    if (clEnqueueNDRangeKernel(this->cl_prog_.queue(), this->cl_kernel_,
                               1, NULL, &global_size,
                               NULL, 0, NULL, &evt) != CL_SUCCESS)
      throw std::runtime_error("failed clEnqueueNDRangeKernel");
    if (clFlush(this->cl_prog_.queue()) != CL_SUCCESS)
      throw std::runtime_error("failed clFlush");
    clWaitForEvents(1, &evt);
  };
}
#endif

}  // namespace nearest_neighbor
}  // namespace core
}  // namespace jubatus
