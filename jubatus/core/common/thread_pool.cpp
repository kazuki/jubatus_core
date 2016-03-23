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

#if __cplusplus >= 201103
#include "thread_pool.hpp"

namespace jubatus {
namespace core {
namespace common {

thread_pool::thread_pool(int max_threads)
  : pool_(), queue_(), mutex_(), cond_(), shutdown_(false)
{
  if (max_threads < 0)
    max_threads = std::thread::hardware_concurrency();
  pool_.reserve(max_threads);
}

thread_pool::~thread_pool() {
  this->shutdown_ = true;
  {
    std::lock_guard<std::mutex> lk(mutex_);
    cond_.notify_all();
  }
  for (auto it = pool_.begin(); it != pool_.end(); ++it)
    it->join();
}

void thread_pool::worker(thread_pool *tp) {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lk(tp->mutex_);
      tp->cond_.wait(lk, [tp]{ return tp->shutdown_ || !tp->queue_.empty(); });
      if (tp->shutdown_ && tp->queue_.empty())
        return;
      task = std::move(tp->queue_.front());
      tp->queue_.pop();
    }
    task();
  }
}

namespace default_thread_pool {
  thread_pool instance(-1);
}


}  // namespace common
}  // namespace core
}  // namespace jubatus

#endif // #if __cplusplus >= 201103
