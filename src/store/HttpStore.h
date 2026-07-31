#pragma once

#include <memory>
#include <string>

#include "store/ChunkStore.h"

namespace sv::store {

struct HttpConfig {
  std::string username;      // basic auth; empty = anonymous
  std::string password;
  int maxConcurrentStreams = 128;
  int maxRetries = 3;
  long connectTimeoutMs = 10000;
  long requestTimeoutMs = 60000;
};

// HTTP(S) backend over libcurl. One dedicated curl-multi thread multiplexes
// all transfers (HTTP/2 when the server offers it). Cancellation via the
// request's stop_token is exact: the transfer is removed from the multi
// handle on the next loop iteration.
class HttpStore final : public ChunkStore {
 public:
  HttpStore(std::string baseUrl, HttpConfig config = {});
  ~HttpStore() override;

  void read(std::string key, std::stop_token st, StoreCallback cb) override;
  void readRange(std::string key, std::uint64_t offset, std::uint64_t length,
                 std::stop_token st, StoreCallback cb) override;
  bool isRemote() const override { return true; }

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sv::store
