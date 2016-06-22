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

#include "thread_pool.hpp"
#include <vector>

#ifdef JUBATUS_CORE_NUMA
#include "../storage/bit_vector.hpp"
#include <numa.h>
#endif

#include <iostream>

using jubatus::util::concurrent::scoped_lock;
using jubatus::util::lang::bind;
using jubatus::util::lang::function;
using jubatus::util::lang::shared_ptr;

namespace jubatus {
namespace core {
namespace common {

thread_pool::thread_pool(int max_threads)
  : pool_(), queue_(), mutex_(), cond_(), shutdown_(false) {
  if (max_threads < 0)
    max_threads = thread::hardware_concurrency();
  pool_.reserve(max_threads);
}

thread_pool::~thread_pool() {
  {
    scoped_lock lk(mutex_);
    this->shutdown_ = true;
    cond_.notify_all();
  }
  for (std::vector<shared_ptr<thread> >::iterator it = pool_.begin();
       it != pool_.end(); ++it) {
    (*it)->join();
  }
}

void thread_pool::worker(thread_pool *tp) {
  while (true) {
    function<void()> task;
    {
      scoped_lock lk(tp->mutex_);
      while (tp->queue_.empty()) {
        if (tp->shutdown_)
          return;
        tp->cond_.wait(tp->mutex_);
      }
      task = tp->queue_.front();
      tp->queue_.pop();
    }
    task();
  }
}

namespace default_thread_pool {
  thread_pool instance(-1);
}

#ifdef JUBATUS_CORE_NUMA
numa_thread_pool default_numa_thread_pool;

numa_thread_pool::numa_thread_pool() : nodes_(), shutdown_(false) {
  if (numa_available() < 0) {
    return;
  }
  const int num_nodes = numa_num_configured_nodes();
  const int num_task_nodes = numa_num_task_nodes();
  if (num_nodes == 1 || num_task_nodes == 1) {
    return;
  }
  pagesize_ = static_cast<size_t>(numa_pagesize());

  bitmask* bm = numa_bitmask_alloc(numa_num_configured_cpus());
  const unsigned long bm_cnt = bm->size / sizeof(unsigned long) / 8;
  for (int i = 0; i < num_nodes; ++i) {
    int num = 0;
    numa_node_to_cpus(i, bm);
    for (unsigned long j = 0; j < bm_cnt; ++j) {
      num += jubatus::core::storage::detail::bitcount(bm->maskp[j]);
    }

    shared_ptr<node_info2> ni(new node_info2);
    ni->id = i;
    ni->num_of_cpus = num;
    long long tmp;
    ni->size = numa_node_size64(i, &tmp);
    for (int j = 0; j < num; ++j) {
      shared_ptr<thread> th(new thread(bind(&worker, this, ni.get())));
      th->start();
      ni->threads.push_back(th);
    }
    nodes_.push_back(ni);
  }
}

numa_thread_pool::~numa_thread_pool() {
  shutdown_ = true;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    node_info2& ni = *nodes_[i];
    {
      scoped_lock lk(ni.lock);
      ni.cond.notify_all();
    }
    for (size_t j = 0; j < ni.threads.size(); ++j) {
      ni.threads[j]->join();
    }
  }
}

void numa_thread_pool::worker(numa_thread_pool* ntp, node_info2 *ni) {
  numa_run_on_node(ni->id);
  while (true) {
    function<void()> task;
    {
      scoped_lock lk(ni->lock);
      while (ni->queue.empty()) {
        if (ntp->shutdown_)
          return;
        ni->cond.wait(ni->lock);
      }
      task = ni->queue.front();
      ni->queue.pop();
    }
    task();
  }
}
#endif  // JUBATUS_CORE_NUMA
  

}  // namespace common
}  // namespace core
}  // namespace jubatus
