#include "store/ChunkStore.h"

namespace sv::store {

void ChunkStore::readRange(std::string key, std::uint64_t offset,
                           std::uint64_t length, std::stop_token st,
                           StoreCallback cb) {
  read(std::move(key), std::move(st),
       [offset, length, cb = std::move(cb)](StoreResult r) mutable {
         if (!r) {
           cb(std::unexpected(r.error()));
           return;
         }
         if (offset + length > r->size()) {
           cb(std::unexpected(StoreError{StoreError::Kind::Io, 0,
                                         "range beyond object size"}));
           return;
         }
         cb(ByteBuffer::copyOf(r->span().subspan(offset, length)));
       });
}

}  // namespace sv::store
