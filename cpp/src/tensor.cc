#include "qwen3tts/tensor.h"

#include <algorithm>

namespace qwen3tts {

FloatTensor SliceAxis1(const FloatTensor& tensor, int64_t begin, int64_t end) {
  const auto& s = tensor.shape();
  if (s.size() != 3) throw std::invalid_argument("SliceAxis1 expects rank-3 tensor");
  const int64_t b = s[0], t = s[1], d = s[2];
  begin = std::max<int64_t>(0, begin);
  end = std::min<int64_t>(t, end);
  if (end < begin) end = begin;
  FloatTensor out({b, end - begin, d});
  for (int64_t bi = 0; bi < b; ++bi) {
    auto src = tensor.values().begin() + (bi * t + begin) * d;
    auto dst = out.values().begin() + bi * (end - begin) * d;
    std::copy(src, src + (end - begin) * d, dst);
  }
  return out;
}

Int64Tensor SliceAxis1(const Int64Tensor& tensor, int64_t begin, int64_t end) {
  const auto& s = tensor.shape();
  if (s.size() != 2 && s.size() != 3) throw std::invalid_argument("SliceAxis1 expects rank-2 or rank-3 tensor");
  const int64_t b = s[0], t = s[1], d = s.size() == 3 ? s[2] : 1;
  begin = std::max<int64_t>(0, begin);
  end = std::min<int64_t>(t, end);
  if (end < begin) end = begin;
  std::vector<int64_t> out_shape = s.size() == 3 ? std::vector<int64_t>{b, end - begin, d}
                                                 : std::vector<int64_t>{b, end - begin};
  Int64Tensor out(out_shape);
  for (int64_t bi = 0; bi < b; ++bi) {
    auto src = tensor.values().begin() + (bi * t + begin) * d;
    auto dst = out.values().begin() + bi * (end - begin) * d;
    std::copy(src, src + (end - begin) * d, dst);
  }
  return out;
}

FloatTensor ConcatAxis1(const std::vector<FloatTensor>& tensors) {
  if (tensors.empty()) throw std::invalid_argument("ConcatAxis1 expects at least one tensor");
  const auto& first = tensors.front().shape();
  if (first.size() != 3) throw std::invalid_argument("ConcatAxis1 expects rank-3 tensors");
  const int64_t b = first[0], d = first[2];
  int64_t total_t = 0;
  for (const auto& tensor : tensors) {
    const auto& s = tensor.shape();
    if (s.size() != 3 || s[0] != b || s[2] != d) {
      throw std::invalid_argument("ConcatAxis1 rank-3 shape mismatch");
    }
    total_t += s[1];
  }
  FloatTensor out({b, total_t, d});
  int64_t offset = 0;
  for (const auto& tensor : tensors) {
    const int64_t t = tensor.shape()[1];
    for (int64_t bi = 0; bi < b; ++bi) {
      auto dst = out.values().begin() + (bi * total_t + offset) * d;
      auto src = tensor.values().begin() + bi * t * d;
      std::copy(src, src + t * d, dst);
    }
    offset += t;
  }
  return out;
}

Int64Tensor ConcatAxis1(const std::vector<Int64Tensor>& tensors) {
  if (tensors.empty()) throw std::invalid_argument("ConcatAxis1 expects at least one tensor");
  const auto& first = tensors.front().shape();
  if (first.size() != 2 && first.size() != 3) throw std::invalid_argument("ConcatAxis1 expects rank-2 or rank-3");
  const int64_t b = first[0], d = first.size() == 3 ? first[2] : 1;
  int64_t total_t = 0;
  for (const auto& tensor : tensors) {
    const auto& s = tensor.shape();
    if (s.size() != first.size() || s[0] != b || (s.size() == 3 ? s[2] : 1) != d) {
      throw std::invalid_argument("ConcatAxis1 int64 shape mismatch");
    }
    total_t += s[1];
  }
  std::vector<int64_t> out_shape = first.size() == 3 ? std::vector<int64_t>{b, total_t, d}
                                                     : std::vector<int64_t>{b, total_t};
  Int64Tensor out(out_shape);
  int64_t offset = 0;
  for (const auto& tensor : tensors) {
    const int64_t t = tensor.shape()[1];
    for (int64_t bi = 0; bi < b; ++bi) {
      auto dst = out.values().begin() + (bi * total_t + offset) * d;
      auto src = tensor.values().begin() + bi * t * d;
      std::copy(src, src + t * d, dst);
    }
    offset += t;
  }
  return out;
}

FloatTensor Add(const FloatTensor& left, const FloatTensor& right) {
  if (left.shape() != right.shape()) throw std::invalid_argument("Add expects equal shapes");
  FloatTensor out(left.shape());
  for (size_t i = 0; i < out.size(); ++i) out.values()[i] = left.values()[i] + right.values()[i];
  return out;
}

FloatTensor RepeatAxis1(const FloatTensor& tensor, int64_t repeats) {
  const auto& s = tensor.shape();
  if (s.size() != 3 || s[1] != 1) throw std::invalid_argument("RepeatAxis1 expects [B,1,D]");
  if (repeats < 0) throw std::invalid_argument("RepeatAxis1 repeats must be non-negative");
  const int64_t b = s[0], d = s[2];
  FloatTensor out({b, repeats, d});
  for (int64_t bi = 0; bi < b; ++bi) {
    for (int64_t ti = 0; ti < repeats; ++ti) {
      auto src = tensor.values().begin() + bi * d;
      auto dst = out.values().begin() + (bi * repeats + ti) * d;
      std::copy(src, src + d, dst);
    }
  }
  return out;
}

Int64Tensor SliceCodes(const Int64Tensor& codes, int64_t begin, int64_t end) {
  const auto& s = codes.shape();
  if (s.size() != 3 || s[0] != 1) throw std::invalid_argument("SliceCodes expects [1,T,G]");
  const int64_t t = s[1], g = s[2];
  begin = std::max<int64_t>(0, begin);
  end = std::min<int64_t>(t, end);
  if (end < begin) end = begin;
  Int64Tensor out({1, end - begin, g});
  std::copy(codes.values().begin() + begin * g, codes.values().begin() + end * g, out.values().begin());
  return out;
}

Int64Tensor ConcatCodesBatch1(const Int64Tensor& left, const Int64Tensor& right) {
  if (left.shape().size() != 3 || right.shape().size() != 3 || left.shape()[0] != 1 || right.shape()[0] != 1 ||
      left.shape()[2] != right.shape()[2]) {
    throw std::invalid_argument("ConcatCodesBatch1 expects [1,T,G] tensors with same G");
  }
  const int64_t lt = left.shape()[1], rt = right.shape()[1], g = left.shape()[2];
  Int64Tensor out({1, lt + rt, g});
  std::copy(left.values().begin(), left.values().end(), out.values().begin());
  std::copy(right.values().begin(), right.values().end(), out.values().begin() + lt * g);
  return out;
}

}  // namespace qwen3tts
