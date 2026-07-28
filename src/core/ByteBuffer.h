#pragma once

#include <cstddef>
#include <memory>
#include <span>

namespace sv {

// Move-only owned byte buffer. Cheaper and more explicit than std::vector for
// large IO buffers (no zero-fill on allocation via forUninitialized).
class ByteBuffer {
 public:
  ByteBuffer() = default;

  static ByteBuffer uninitialized(std::size_t size) {
    ByteBuffer b;
    b.data_ = std::make_unique_for_overwrite<std::byte[]>(size);
    b.size_ = size;
    return b;
  }

  static ByteBuffer copyOf(std::span<const std::byte> src) {
    ByteBuffer b = uninitialized(src.size());
    std::copy(src.begin(), src.end(), b.data_.get());
    return b;
  }

  ByteBuffer(ByteBuffer&&) noexcept = default;
  ByteBuffer& operator=(ByteBuffer&&) noexcept = default;
  ByteBuffer(const ByteBuffer&) = delete;
  ByteBuffer& operator=(const ByteBuffer&) = delete;

  std::byte* data() { return data_.get(); }
  const std::byte* data() const { return data_.get(); }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  std::span<std::byte> span() { return {data_.get(), size_}; }
  std::span<const std::byte> span() const { return {data_.get(), size_}; }

  // Shrink the logical size (e.g. after a short read). Never grows.
  void truncate(std::size_t newSize) {
    if (newSize < size_) size_ = newSize;
  }

 private:
  std::unique_ptr<std::byte[]> data_;
  std::size_t size_ = 0;
};

}  // namespace sv
