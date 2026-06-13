#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen3tts {

template <typename T>
class Tensor {
 public:
  Tensor() = default;
  Tensor(std::vector<int64_t> shape, std::vector<T> values)
      : shape_(std::move(shape)), values_(std::move(values)) {
    if (NumElements(shape_) != values_.size()) {
      throw std::invalid_argument("tensor shape/value count mismatch: shape=" + ShapeToString(shape_) +
                                  " expected=" + std::to_string(NumElements(shape_)) +
                                  " values=" + std::to_string(values_.size()));
    }
  }
  explicit Tensor(std::vector<int64_t> shape) : shape_(std::move(shape)), values_(NumElements(shape_)) {}

  const std::vector<int64_t>& shape() const { return shape_; }
  std::vector<int64_t>& shape() { return shape_; }
  const std::vector<T>& values() const { return values_; }
  std::vector<T>& values() { return values_; }
  const T* data() const { return values_.data(); }
  T* data() { return values_.data(); }
  size_t size() const { return values_.size(); }
  bool empty() const { return values_.empty(); }

  static size_t NumElements(const std::vector<int64_t>& shape) {
    size_t n = 1;
    for (int64_t d : shape) {
      if (d < 0) throw std::invalid_argument("negative tensor dimension");
      n *= static_cast<size_t>(d);
    }
    return n;
  }

  static std::string ShapeToString(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i) out += ",";
      out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
  }

 private:
  std::vector<int64_t> shape_;
  std::vector<T> values_;
};

using FloatTensor = Tensor<float>;
using Int64Tensor = Tensor<int64_t>;

FloatTensor SliceAxis1(const FloatTensor& tensor, int64_t begin, int64_t end);
Int64Tensor SliceAxis1(const Int64Tensor& tensor, int64_t begin, int64_t end);
FloatTensor ConcatAxis1(const std::vector<FloatTensor>& tensors);
Int64Tensor ConcatAxis1(const std::vector<Int64Tensor>& tensors);
FloatTensor Add(const FloatTensor& left, const FloatTensor& right);
FloatTensor RepeatAxis1(const FloatTensor& tensor, int64_t repeats);
Int64Tensor SliceCodes(const Int64Tensor& codes, int64_t begin, int64_t end);
Int64Tensor ConcatCodesBatch1(const Int64Tensor& left, const Int64Tensor& right);

}  // namespace qwen3tts
