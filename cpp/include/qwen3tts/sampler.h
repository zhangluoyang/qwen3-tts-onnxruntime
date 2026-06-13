#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace qwen3tts {

struct SamplingOptions {
  bool do_sample = false;
  int top_k = 50;
  float top_p = 1.0f;
  float temperature = 0.9f;
  float repetition_penalty = 1.0f;
  int64_t eos_token_id = 2150;
  int64_t vocab_size = 3072;
  int64_t first_codebook_mask_tail = 1024;
  int64_t min_new_tokens = 0;
  uint64_t seed = 1234;
};

class MainTokenSampler {
 public:
  explicit MainTokenSampler(SamplingOptions options);

  int64_t Sample(const std::vector<float>& logits, const std::vector<int64_t>& generated_first_tokens);

 private:
  void ApplyMasks(std::vector<float>& scores, const std::vector<int64_t>& generated_first_tokens) const;
  int64_t Greedy(const std::vector<float>& scores) const;
  int64_t SampleTopKTopP(const std::vector<float>& scores);

  SamplingOptions options_;
  std::mt19937_64 rng_;
};

}  // namespace qwen3tts
