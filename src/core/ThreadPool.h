#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace sv {

// Fixed-size worker pool. Jobs receive the pool's stop_token so long-running
// work can bail out during shutdown.
class ThreadPool {
 public:
  using Job = std::move_only_function<void(std::stop_token)>;

  ThreadPool(std::string name, unsigned threadCount);
  ~ThreadPool();  // requests stop, drains nothing: queued jobs are discarded

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void post(Job job);

  std::size_t queuedJobs() const;
  unsigned threadCount() const { return static_cast<unsigned>(threads_.size()); }

 private:
  void workerLoop(std::stop_token st);

  std::string name_;
  mutable std::mutex mutex_;
  std::condition_variable_any cv_;
  std::deque<Job> queue_;
  std::vector<std::jthread> threads_;  // last member: joins before rest dies
};

}  // namespace sv
