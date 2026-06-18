#pragma once

#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "qwen3tts/ort_runner.h"
#include "qwen3tts/sampler.h"
#include "qwen3tts/tensor.h"

namespace qwen3tts {

struct CloneRuntimeConfig {
  std::filesystem::path onnx_dir = "onnx_fp16";
  bool use_cuda = true;
  int cuda_device_id = 0;

  int64_t max_new_tokens = 512;
  int64_t min_new_tokens = 0;
  int64_t eos_token_id = 2150;
  int64_t vocab_size = 3072;
  int64_t first_codebook_mask_tail = 1024;
  int64_t num_hidden_layers = 28;
  int64_t num_key_value_heads = 8;
  int64_t head_dim = 128;
  int64_t num_code_groups = 16;
  int64_t decode_upsample_rate = 1920;
  int64_t audio_sample_rate = 24000;
  int64_t tokenizer_decode_chunk_frames = 300;
  int64_t tokenizer_decode_context_frames = 25;

  bool do_sample = false;
  int top_k = 50;
  float top_p = 1.0f;
  float temperature = 0.9f;
  float repetition_penalty = 1.0f;
  uint64_t seed = 1234;
};

struct CloneInputs {
  FloatTensor inputs_embeds;
  Int64Tensor attention_mask;
  FloatTensor trailing_text_hidden;
  FloatTensor tts_pad_embed;
  Int64Tensor ref_code;
};

struct CloneResult {
  Int64Tensor generated_codes;
  FloatTensor audio;
  int64_t sample_rate = 24000;
  bool stopped = false;
  std::string stop_reason = "max_new_tokens";
  std::vector<std::pair<std::string, double>> timings;
};

class ClonePipeline {
 public:
  explicit ClonePipeline(CloneRuntimeConfig config);

  CloneResult Run(const CloneInputs& inputs);
  CloneResult Run(const CloneInputs& inputs, std::mt19937_64* rng);
  CloneResult Run(const CloneInputs& inputs, const CloneRuntimeConfig& runtime_config);
  CloneResult Run(const CloneInputs& inputs, const CloneRuntimeConfig& runtime_config, std::mt19937_64* rng);
  std::vector<std::pair<std::string, double>> SessionLoadTimings() const;

 private:
  struct TalkerState {
    std::vector<float> logits;
    Ort::Value last_hidden{nullptr};
    std::vector<Ort::Value> past_keys;
    std::vector<Ort::Value> past_values;
    int64_t past_len = 0;
  };
  struct DecodeStepOutput {
    TalkerState state;
    Int64Tensor codebook_tokens;
    double sub_talker_sample_seconds = 0.0;
    double talker_core_decode_seconds = 0.0;
  };

  TalkerState RunPrefill(const CloneInputs& inputs);
  DecodeStepOutput RunDecodeStep(TalkerState state, int64_t first_token, const FloatTensor& text_embed);
  FloatTensor TextEmbedForStep(const CloneInputs& inputs, int64_t step) const;
  FloatTensor DecodeAudio(const Int64Tensor& full_codes, int64_t context_frames) const;
  std::vector<std::string> TalkerOutputNames() const;
  std::unordered_set<std::string> TalkerDeviceOutputs() const;
  Int64Tensor MakeGeneratedCodes(const std::vector<int64_t>& frames) const;

  CloneRuntimeConfig config_;
  Ort::Env env_;
  OrtRunner talker_core_;
  OrtRunner sub_talker_sample_;
  OrtRunner tokenizer_decode_;
};

}  // namespace qwen3tts
