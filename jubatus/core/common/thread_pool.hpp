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

#if !defined(JUBATUS_CORE_COMMON_THREAD_POOL_HPP_) && __cplusplus >= 201103
#define JUBATUS_CORE_COMMON_THREAD_POOL_HPP_

#include <atomic>
#include <future>
#include <mutex>
#include <queue>
#include <vector>

#include <iostream>

namespace jubatus {
namespace core {
namespace common {

class thread_pool {
public:
  thread_pool() = delete;
  thread_pool(const thread_pool&) = delete;
  explicit thread_pool(int max_threads);
  virtual ~thread_pool();

  template< class Function, class... Args>
  std::future<typename std::result_of<Function(Args...)>::type>
  async(Function&& f, Args&&... args) {
    using return_type = typename std::result_of<Function(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<return_type()> >(
      std::bind(std::forward<Function>(f), std::forward<Args>(args)...));
    std::future<return_type> res = task->get_future();
    if (pool_.capacity() == 0) {
      (*task)();
    } else {
      std::lock_guard<std::mutex> lk(mutex_);
      const bool empty_flag = queue_.empty();
      queue_.emplace([task](){ (*task)(); });
      if (pool_.size() < pool_.capacity()) {
        pool_.emplace_back(worker, this);
      } else if (empty_flag) {
        cond_.notify_all();
      }
    }
    return res;
  }
private:
  static void worker(thread_pool *tp);

  std::vector<std::thread> pool_;
  std::queue<std::function<void()> > queue_;
  std::mutex mutex_;
  std::condition_variable cond_;
  bool shutdown_;
};

namespace default_thread_pool {
  extern thread_pool instance;

  template< class Function, class... Args>
  std::future<typename std::result_of<Function(Args...)>::type>
  async(Function&& f, Args&&... args) {
    return instance.async(f, args ...);
  }
}

}  // namespace common
}  // namespace core
}  // namespace jubatus

#endif  // #if defined(JUBATUS_CORE_COMMON_THREAD_POOL_HPP_) && __cplusplus >= 201103
