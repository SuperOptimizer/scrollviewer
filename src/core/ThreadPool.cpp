#include "core/ThreadPool.h"

namespace sv {

ThreadPool::ThreadPool(std::string name, unsigned threadCount)
    : name_(std::move(name)) {
  if (threadCount == 0) threadCount = 1;
  threads_.reserve(threadCount);
  for (unsigned i = 0; i < threadCount; ++i) {
    threads_.emplace_back([this](std::stop_token st) { workerLoop(st); });
  }
}

ThreadPool::~ThreadPool() {
  for (auto& t : threads_) t.request_stop();
  cv_.notify_all();
}

void ThreadPool::post(Job job) {
  {
    std::lock_guard lock(mutex_);
    queue_.push_back(std::move(job));
  }
  cv_.notify_one();
}

std::size_t ThreadPool::queuedJobs() const {
  std::lock_guard lock(mutex_);
  return queue_.size();
}

void ThreadPool::workerLoop(std::stop_token st) {
  for (;;) {
    Job job;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, st, [this] { return !queue_.empty(); });
      if (st.stop_requested()) return;
      job = std::move(queue_.front());
      queue_.pop_front();
    }
    job(st);
  }
}

}  // namespace sv
