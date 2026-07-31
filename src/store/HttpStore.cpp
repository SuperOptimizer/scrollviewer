#include "store/HttpStore.h"

#include <curl/curl.h>

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/Log.h"

namespace sv::store {

namespace {

// Growable receive buffer; converted to ByteBuffer on completion.
struct RecvBuffer {
  std::vector<std::byte> data;
};

std::size_t writeCallback(char* ptr, std::size_t size, std::size_t nmemb,
                          void* userdata) {
  auto* buf = static_cast<RecvBuffer*>(userdata);
  const auto* bytes = reinterpret_cast<const std::byte*>(ptr);
  buf->data.insert(buf->data.end(), bytes, bytes + size * nmemb);
  return size * nmemb;
}

struct CurlGlobal {
  CurlGlobal() { curl_global_init(CURL_GLOBAL_DEFAULT); }
  ~CurlGlobal() { curl_global_cleanup(); }
};

}  // namespace

class HttpStore::Impl {
 public:
  Impl(std::string baseUrl, HttpConfig config)
      : baseUrl_(std::move(baseUrl)), config_(std::move(config)) {
    static CurlGlobal g_curlGlobal;
    if (!baseUrl_.empty() && baseUrl_.back() != '/') baseUrl_ += '/';
    multi_ = curl_multi_init();
    curl_multi_setopt(multi_, CURLMOPT_PIPELINING, CURLPIPE_MULTIPLEX);
    // S3 (and most object stores) speak HTTP/1.1 only: one request per
    // connection, no multiplexing. 8 connections capped chunk streaming at
    // a fraction of the line; size the pool to the pipeline's inflight cap
    // so HTTP/1.1 concurrency matches what the scheduler issues.
    curl_multi_setopt(multi_, CURLMOPT_MAX_TOTAL_CONNECTIONS, 64L);
    curl_multi_setopt(multi_, CURLMOPT_MAX_HOST_CONNECTIONS, 64L);
    thread_ = std::jthread([this](std::stop_token st) { loop(st); });
  }

  ~Impl() {
    thread_.request_stop();
    curl_multi_wakeup(multi_);
    thread_.join();
    curl_multi_cleanup(multi_);
  }

  void submit(std::string key, std::string range, std::stop_token st,
              StoreCallback cb) {
    {
      std::lock_guard lock(mutex_);
      pending_.push_back(Request{std::move(key), std::move(range),
                                 std::move(st), std::move(cb), 0});
    }
    curl_multi_wakeup(multi_);
  }

 private:
  struct Request {
    std::string key;
    std::string range;  // empty or "start-end" (inclusive)
    std::stop_token stop;
    StoreCallback cb;
    int attempt;
  };

  struct Transfer {
    Request req;
    RecvBuffer recv;
    char errbuf[CURL_ERROR_SIZE] = {};
  };

  void loop(std::stop_token st) {
    std::unordered_map<CURL*, std::unique_ptr<Transfer>> active;
    std::deque<Request> retryQueue;

    while (!st.stop_requested()) {
      // Admit pending/retry requests up to the stream cap.
      {
        std::lock_guard lock(mutex_);
        while (!pending_.empty() &&
               active.size() <
                   static_cast<std::size_t>(config_.maxConcurrentStreams)) {
          startTransfer(active, std::move(pending_.front()));
          pending_.pop_front();
        }
      }
      while (!retryQueue.empty() &&
             active.size() <
                 static_cast<std::size_t>(config_.maxConcurrentStreams)) {
        startTransfer(active, std::move(retryQueue.front()));
        retryQueue.pop_front();
      }

      // Drop cancelled transfers.
      for (auto it = active.begin(); it != active.end();) {
        if (it->second->req.stop.stop_requested()) {
          curl_multi_remove_handle(multi_, it->first);
          curl_easy_cleanup(it->first);
          it->second->req.cb(std::unexpected(
              StoreError{StoreError::Kind::Cancelled, 0, {}}));
          it = active.erase(it);
        } else {
          ++it;
        }
      }

      int running = 0;
      curl_multi_perform(multi_, &running);

      // Reap completions.
      int msgsLeft = 0;
      while (CURLMsg* msg = curl_multi_info_read(multi_, &msgsLeft)) {
        if (msg->msg != CURLMSG_DONE) continue;
        CURL* easy = msg->easy_handle;
        const CURLcode rc = msg->data.result;
        auto it = active.find(easy);
        if (it == active.end()) continue;
        std::unique_ptr<Transfer> t = std::move(it->second);
        active.erase(it);

        long status = 0;
        curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &status);
        curl_multi_remove_handle(multi_, easy);
        curl_easy_cleanup(easy);

        finishTransfer(std::move(t), rc, status, retryQueue);
      }

      int numfds = 0;
      curl_multi_poll(multi_, nullptr, 0, 100, &numfds);
    }

    // Shutdown: fail everything still in flight or queued.
    for (auto& [easy, t] : active) {
      curl_multi_remove_handle(multi_, easy);
      curl_easy_cleanup(easy);
      t->req.cb(std::unexpected(StoreError{StoreError::Kind::Cancelled, 0, {}}));
    }
    std::lock_guard lock(mutex_);
    for (auto& r : pending_)
      r.cb(std::unexpected(StoreError{StoreError::Kind::Cancelled, 0, {}}));
    pending_.clear();
  }

  void startTransfer(
      std::unordered_map<CURL*, std::unique_ptr<Transfer>>& active,
      Request req) {
    if (req.stop.stop_requested()) {
      req.cb(std::unexpected(StoreError{StoreError::Kind::Cancelled, 0, {}}));
      return;
    }
    auto t = std::make_unique<Transfer>();
    t->req = std::move(req);

    CURL* easy = curl_easy_init();
    curl_easy_setopt(easy, CURLOPT_URL, (baseUrl_ + t->req.key).c_str());
    curl_easy_setopt(easy, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(easy, CURLOPT_WRITEDATA, &t->recv);
    curl_easy_setopt(easy, CURLOPT_ERRORBUFFER, t->errbuf);
    curl_easy_setopt(easy, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(easy, CURLOPT_MAXREDIRS, 4L);
    curl_easy_setopt(easy, CURLOPT_ACCEPT_ENCODING, "");  // allow gzip metadata
    curl_easy_setopt(easy, CURLOPT_HTTP_VERSION, CURL_HTTP_VERSION_2TLS);
    // No PIPEWAIT: it holds transfers back hoping to multiplex over an
    // existing connection, which never pays off on HTTP/1.1 object stores
    // and serializes the ramp-up to the connection cap.
    curl_easy_setopt(easy, CURLOPT_CONNECTTIMEOUT_MS, config_.connectTimeoutMs);
    curl_easy_setopt(easy, CURLOPT_TIMEOUT_MS, config_.requestTimeoutMs);
    curl_easy_setopt(easy, CURLOPT_FAILONERROR, 0L);  // inspect status ourselves
    if (!config_.username.empty()) {
      curl_easy_setopt(easy, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
      curl_easy_setopt(easy, CURLOPT_USERNAME, config_.username.c_str());
      curl_easy_setopt(easy, CURLOPT_PASSWORD, config_.password.c_str());
    }
    if (!t->req.range.empty())
      curl_easy_setopt(easy, CURLOPT_RANGE, t->req.range.c_str());

    active.emplace(easy, std::move(t));
    curl_multi_add_handle(multi_, easy);
  }

  void finishTransfer(std::unique_ptr<Transfer> t, CURLcode rc, long status,
                      std::deque<Request>& retryQueue) {
    const bool transportError = (rc != CURLE_OK);
    const bool retryableStatus = (status >= 500 || status == 429);

    if ((transportError || retryableStatus) &&
        t->req.attempt + 1 < config_.maxRetries &&
        !t->req.stop.stop_requested()) {
      Request retry = std::move(t->req);
      ++retry.attempt;
      retryQueue.push_back(std::move(retry));
      return;
    }

    if (transportError) {
      t->req.cb(std::unexpected(StoreError{
          StoreError::Kind::Http, static_cast<int>(rc),
          t->errbuf[0] ? t->errbuf : curl_easy_strerror(rc)}));
      return;
    }
    if (status == 404 || status == 403) {
      // S3 signals missing keys as 403 for anonymous callers without list
      // permission; treat both as NotFound (missing chunk => fill_value).
      t->req.cb(std::unexpected(
          StoreError{StoreError::Kind::NotFound, static_cast<int>(status), {}}));
      return;
    }
    if (status == 401) {
      t->req.cb(std::unexpected(StoreError{StoreError::Kind::Auth,
                                           static_cast<int>(status),
                                           "authentication required"}));
      return;
    }
    if (status < 200 || status >= 300) {
      t->req.cb(std::unexpected(StoreError{
          StoreError::Kind::Http, static_cast<int>(status), "HTTP error"}));
      return;
    }

    ByteBuffer out = ByteBuffer::copyOf(t->recv.data);
    t->req.cb(std::move(out));
  }

  std::string baseUrl_;
  HttpConfig config_;
  CURLM* multi_ = nullptr;
  std::mutex mutex_;
  std::deque<Request> pending_;
  std::jthread thread_;  // last member
};

HttpStore::HttpStore(std::string baseUrl, HttpConfig config)
    : impl_(std::make_unique<Impl>(std::move(baseUrl), std::move(config))) {}

HttpStore::~HttpStore() = default;

void HttpStore::read(std::string key, std::stop_token st, StoreCallback cb) {
  impl_->submit(std::move(key), {}, std::move(st), std::move(cb));
}

void HttpStore::readRange(std::string key, std::uint64_t offset,
                          std::uint64_t length, std::stop_token st,
                          StoreCallback cb) {
  impl_->submit(std::move(key),
                std::to_string(offset) + "-" +
                    std::to_string(offset + length - 1),
                std::move(st), std::move(cb));
}

}  // namespace sv::store
