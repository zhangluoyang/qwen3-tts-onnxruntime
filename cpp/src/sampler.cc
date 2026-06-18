#include "qwen3tts/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace qwen3tts {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

std::vector<float> SoftmaxSelected(const std::vector<float>& scores, const std::vector<int64_t>& indices) {
  float max_score = kNegInf;
  for (int64_t idx : indices) max_score = std::max(max_score, scores[static_cast<size_t>(idx)]);
  std::vector<float> probs(indices.size(), 0.0f);
  double total = 0.0;
  for (size_t i = 0; i < indices.size(); ++i) {
    float score = scores[static_cast<size_t>(indices[i])];
    double v = std::isfinite(score) ? std::exp(static_cast<double>(score - max_score)) : 0.0;
    probs[i] = static_cast<float>(v);
    total += v;
  }
  if (!std::isfinite(total) || total <= 0.0) {
    std::fill(probs.begin(), probs.end(), 1.0f / static_cast<float>(std::max<size_t>(1, probs.size())));
    return probs;
  }
  for (float& p : probs) p = static_cast<float>(static_cast<double>(p) / total);
  return probs;
}

}  // namespace

MainTokenSampler::MainTokenSampler(SamplingOptions options)
    : options_(options), owned_rng_(options.seed), rng_(&owned_rng_) {}

MainTokenSampler::MainTokenSampler(SamplingOptions options, std::mt19937_64* rng)
    : options_(options), owned_rng_(options.seed), rng_(rng == nullptr ? &owned_rng_ : rng) {}

int64_t MainTokenSampler::Sample(const std::vector<float>& logits,
                                 const std::vector<int64_t>& generated_first_tokens) {
  if (logits.empty()) throw std::invalid_argument("empty logits");
  std::vector<float> scores = logits;
  ApplyMasks(scores, generated_first_tokens);
  if (!options_.do_sample) return Greedy(scores);

  if (options_.temperature > 0.0f && options_.temperature != 1.0f) {
    for (float& score : scores) {
      if (std::isfinite(score)) score /= options_.temperature;
    }
  }
  return SampleTopKTopP(scores);
}

void MainTokenSampler::ApplyMasks(std::vector<float>& scores,
                                  const std::vector<int64_t>& generated_first_tokens) const {
  if (options_.repetition_penalty != 1.0f && !generated_first_tokens.empty()) {
    std::vector<int64_t> tokens = generated_first_tokens;
    std::sort(tokens.begin(), tokens.end());
    tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());
    for (int64_t token : tokens) {
      if (token < 0 || token >= static_cast<int64_t>(scores.size())) continue;
      float& score = scores[static_cast<size_t>(token)];
      score = score < 0.0f ? score * options_.repetition_penalty : score / options_.repetition_penalty;
    }
  }

  int64_t start = std::max<int64_t>(0, options_.vocab_size - options_.first_codebook_mask_tail);
  int64_t end = std::min<int64_t>(options_.vocab_size, static_cast<int64_t>(scores.size()));
  for (int64_t token = start; token < end; ++token) {
    if (token != options_.eos_token_id) scores[static_cast<size_t>(token)] = kNegInf;
  }
  if (static_cast<int64_t>(generated_first_tokens.size()) < options_.min_new_tokens &&
      options_.eos_token_id >= 0 && options_.eos_token_id < static_cast<int64_t>(scores.size())) {
    scores[static_cast<size_t>(options_.eos_token_id)] = kNegInf;
  }
}

int64_t MainTokenSampler::Greedy(const std::vector<float>& scores) const {
  return static_cast<int64_t>(std::distance(scores.begin(), std::max_element(scores.begin(), scores.end())));
}

int64_t MainTokenSampler::SampleTopKTopP(const std::vector<float>& scores) {
  std::vector<int64_t> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  indices.erase(std::remove_if(indices.begin(), indices.end(), [&](int64_t idx) {
                  return !std::isfinite(scores[static_cast<size_t>(idx)]);
                }),
                indices.end());
  if (indices.empty()) return Greedy(scores);

  if (options_.top_k > 0 && options_.top_k < static_cast<int>(indices.size())) {
    size_t k = static_cast<size_t>(options_.top_k);
    std::nth_element(indices.begin(), indices.end() - static_cast<std::ptrdiff_t>(k), indices.end(),
                     [&](int64_t a, int64_t b) { return scores[static_cast<size_t>(a)] < scores[static_cast<size_t>(b)]; });
    indices.erase(indices.begin(), indices.end() - static_cast<std::ptrdiff_t>(k));
  }

  if (options_.top_p < 1.0f && indices.size() > 1) {
    std::sort(indices.begin(), indices.end(),
              [&](int64_t a, int64_t b) { return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)]; });
    auto probs = SoftmaxSelected(scores, indices);
    std::vector<int64_t> kept;
    double cum = 0.0;
    for (size_t i = 0; i < indices.size(); ++i) {
      if (!kept.empty() && cum > options_.top_p) break;
      kept.push_back(indices[i]);
      cum += probs[i];
    }
    indices = std::move(kept);
  }

  auto probs = SoftmaxSelected(scores, indices);
  std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
  return indices[dist(*rng_)];
}

}  // namespace qwen3tts
