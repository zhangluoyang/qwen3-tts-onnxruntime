#pragma once

#include <chrono>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "qwen3tts/bpe_tokenizer.h"
#include "qwen3tts/clone_pipeline.h"
#include "qwen3tts/ort_runner.h"
#include "qwen3tts/tensor.h"

namespace qwen3tts {

struct ProductCloneConfig {
  std::filesystem::path model_dir =
      "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base";
  std::filesystem::path onnx_dir = "onnx_fp16";
  bool use_cuda = true;
  int cuda_device_id = 0;
  int64_t max_new_tokens = 512;
  int64_t min_new_tokens = 0;
  bool do_sample = false;
  int top_k = 50;
  float top_p = 1.0f;
  float temperature = 0.9f;
  float repetition_penalty = 1.0f;
  uint64_t seed = 1234;
};

struct ProductCloneRequest {
  std::string text;
  std::filesystem::path reference_audio;
  std::string reference_text;
  std::string language = "auto";
};

struct ProductCloneResult {
  CloneResult generation;
  Int64Tensor reference_codes;
  FloatTensor speaker_embedding;
  std::vector<std::pair<std::string, double>> timings;
};

class ProductCloneRuntime {
 public:
  explicit ProductCloneRuntime(ProductCloneConfig config);

  ProductCloneResult Generate(const ProductCloneRequest& request);
  std::vector<std::pair<std::string, double>> SessionLoadTimings() const;

 private:
  using Clock = std::chrono::steady_clock;

  struct ModelIds {
    int64_t tts_bos_token_id = 151672;
    int64_t tts_eos_token_id = 151673;
    int64_t tts_pad_token_id = 151671;
    int64_t hidden_size = 2048;
    int64_t num_hidden_layers = 28;
    int64_t num_key_value_heads = 8;
    int64_t head_dim = 128;
    int64_t vocab_size = 3072;
    int64_t first_codebook_mask_tail = 1024;
    int64_t num_code_groups = 16;
    int64_t codec_bos_id = 2149;
    int64_t codec_eos_token_id = 2150;
    int64_t codec_pad_id = 2148;
    int64_t codec_think_id = 2154;
    int64_t codec_nothink_id = 2155;
    int64_t codec_think_bos_id = 2156;
    int64_t codec_think_eos_id = 2157;
    int64_t audio_sample_rate = 24000;
    int64_t decode_upsample_rate = 1920;
    std::unordered_map<std::string, int64_t> codec_language_id;
  };

  static ModelIds LoadModelIds(const std::filesystem::path& model_dir);
  static std::string BuildAssistantText(const std::string& text);
  static std::string BuildReferenceText(const std::string& text);
  static double SecondsSince(Clock::time_point start, Clock::time_point end);

  CloneRuntimeConfig MakeCloneRuntimeConfig() const;
  std::vector<int64_t> CodecPrefillIds(const std::string& language) const;
  int64_t LanguageId(const std::string& language) const;

  Int64Tensor TokenizeToTensor(const std::string& text) const;
  FloatTensor TextProject(const Int64Tensor& input_ids) const;
  FloatTensor CodecEmbed(const Int64Tensor& token_ids) const;
  FloatTensor RefCodeEmbed(const Int64Tensor& ref_code) const;
  Int64Tensor EncodeReferenceAudio(const std::vector<float>& audio) const;
  FloatTensor EncodeSpeakerEmbedding(const std::vector<float>& audio) const;
  CloneInputs BuildPrompt(const ProductCloneRequest& request,
                          const Int64Tensor& ref_code,
                          const FloatTensor& speaker_embedding) const;

  ProductCloneConfig config_;
  ModelIds ids_;
  Ort::Env env_;
  Qwen2BpeTokenizer tokenizer_;
  OrtRunner text_project_;
  OrtRunner codec_embed_;
  OrtRunner tokenizer_encode_;
  OrtRunner speaker_encoder_;
  ClonePipeline pipeline_;
};

}  // namespace qwen3tts
