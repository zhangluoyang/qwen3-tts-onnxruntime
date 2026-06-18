#pragma once

#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "qwen3tts/bpe_tokenizer.h"
#include "qwen3tts/clone_pipeline.h"
#include "qwen3tts/ort_runner.h"
#include "qwen3tts/tensor.h"

namespace qwen3tts {

enum class ComputeDType {
  kAuto,
  kFloat32,
  kFloat16,
};

struct Qwen3TTSModelConfig {
  std::filesystem::path model_dir;
  std::filesystem::path onnx_dir;
  bool use_cuda = true;
  int cuda_device_id = 0;
  ComputeDType dtype = ComputeDType::kAuto;
};

struct GenerationOptions {
  int64_t max_new_tokens = 2048;
  int64_t min_new_tokens = 2;
  bool do_sample = true;
  int top_k = 50;
  float top_p = 1.0f;
  float temperature = 0.9f;
  float repetition_penalty = 1.05f;
  int64_t eos_token_id = -1;
  uint64_t seed = 1234;
};

struct SegmentStreamOptions {
  GenerationOptions generation;
  int min_text_chunk_chars = 20;
  int max_text_chunk_chars = 80;
  int kv_anchor_segment_count = 3;
  int pinned_anchor_segment_count = 0;
};

struct VoiceCloneReference {
  std::filesystem::path audio_path;
  std::string text;
};

class Qwen3TTSOnnxModelBase {
 public:
  explicit Qwen3TTSOnnxModelBase(Qwen3TTSModelConfig config);
  virtual ~Qwen3TTSOnnxModelBase();

  CloneResult GenerateAudioFromPrompt(const CloneInputs& prompt, const GenerationOptions& options);
  CloneResult GenerateAudioFromPrompt(const CloneInputs& prompt,
                                      const GenerationOptions& options,
                                      std::mt19937_64* rng);
  std::vector<std::pair<std::string, double>> SessionLoadTimings() const;
  int64_t AudioSampleRate() const;
  int64_t NumCodeGroups() const;

 protected:
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
    std::string tts_model_size;
    std::unordered_map<std::string, int64_t> codec_language_id;
    std::unordered_map<std::string, int64_t> spk_id;
    std::unordered_map<std::string, std::string> spk_is_dialect;
  };

  static ModelIds LoadModelIds(const std::filesystem::path& model_dir);
  static std::string BuildAssistantText(const std::string& text);
  static std::string BuildReferenceText(const std::string& text);
  static std::string BuildInstructText(const std::string& text);

  CloneRuntimeConfig MakeRuntimeConfig(const GenerationOptions& options) const;
  int64_t LanguageId(const std::string& language) const;
  std::vector<int64_t> CodecPrefillIds(const std::string& language) const;
  Int64Tensor EmptyCodes() const;
  Int64Tensor RowTensor(std::vector<int64_t> values) const;
  Int64Tensor ShapeCodes(const Int64Tensor& codes) const;

  Int64Tensor TokenizeToTensor(const std::string& text) const;
  FloatTensor TextProject(const Int64Tensor& input_ids) const;
  FloatTensor CodecEmbed(const Int64Tensor& token_ids) const;
  FloatTensor RefCodeEmbed(const Int64Tensor& ref_code) const;
  OrtRunner& TokenizerEncodeRunner() const;
  OrtRunner& SpeakerEncoderRunner() const;
  Int64Tensor EncodeReferenceAudioFile(const std::filesystem::path& path) const;
  FloatTensor EncodeSpeakerEmbeddingFile(const std::filesystem::path& path) const;

  FloatTensor TargetTextEmbeds(const std::string& text) const;
  Int64Tensor TargetTextIds(const std::string& text) const;
  FloatTensor InstructEmbed(const std::string& instruct) const;
  FloatTensor CodecTailEmbed() const;
  std::tuple<FloatTensor, FloatTensor, FloatTensor> TtsSpecialEmbeds() const;
  std::tuple<FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor> BuildTalkerPrefix(
      const Int64Tensor& input_id,
      const std::string& language,
      const std::optional<FloatTensor>& speaker_embed) const;

  std::pair<Int64Tensor, int64_t> NormalizeAnchorCode(const Int64Tensor& anchor_code) const;
  FloatTensor RoleAlignedCodecPart(const FloatTensor& codec_input,
                                   const FloatTensor& tts_bos_embed,
                                   const FloatTensor& tts_pad_embed) const;
  std::pair<FloatTensor, FloatTensor> AppendNonStreamingText(const FloatTensor& talker_input_embed,
                                                             const Int64Tensor& input_id,
                                                             const FloatTensor& tts_eos_embed,
                                                             const FloatTensor& tts_pad_embed) const;
  std::pair<FloatTensor, FloatTensor> GenerateIclPrompt(const Int64Tensor& text_id,
                                                        const Int64Tensor& ref_id,
                                                        const Int64Tensor& ref_code,
                                                        const FloatTensor& tts_pad_embed,
                                                        const FloatTensor& tts_eos_embed,
                                                        bool non_streaming_mode,
                                                        bool include_text_eos = true) const;
  CloneInputs BuildConditionedPrompt(const std::string& text,
                                     const FloatTensor& codec_input,
                                     const std::string& instruct,
                                     bool non_streaming_mode,
                                     const std::string& anchor_text,
                                     const Int64Tensor& anchor_code) const;

  Qwen3TTSModelConfig config_;
  ModelIds ids_;
  Ort::Env env_;
  Qwen2BpeTokenizer tokenizer_;
  OrtRunner text_project_;
  OrtRunner codec_embed_;
  mutable std::unique_ptr<OrtRunner> tokenizer_encode_;
  mutable std::unique_ptr<OrtRunner> speaker_encoder_;
  ClonePipeline pipeline_;
};

class BaseQwen3TTSOnnxModel : public Qwen3TTSOnnxModelBase {
 public:
  explicit BaseQwen3TTSOnnxModel(Qwen3TTSModelConfig config);

  CloneInputs BuildClonePrompt(const std::string& text,
                               const std::string& language,
                               const std::string& ref_text,
                               const Int64Tensor& ref_code,
                               const FloatTensor& ref_spk_embedding,
                               bool x_vector_only_mode,
                               bool non_streaming_mode) const;
  CloneInputs BuildClonePromptFromReference(const std::string& text,
                                            const VoiceCloneReference& reference,
                                            const std::string& language = "auto",
                                            bool x_vector_only_mode = false,
                                            bool non_streaming_mode = false) const;
  CloneResult GenerateCloneAudioFromReference(const std::string& text,
                                              const VoiceCloneReference& reference,
                                              const std::string& language = "auto",
                                              bool x_vector_only_mode = false,
                                              bool non_streaming_mode = false,
                                              const GenerationOptions& options = GenerationOptions{});
  std::vector<CloneResult> StreamCloneAudioFromReference(const std::vector<std::string>& text_deltas,
                                                         const VoiceCloneReference& reference,
                                                         const std::string& language = "auto",
                                                         const SegmentStreamOptions& options = SegmentStreamOptions{});
};

class CustomQwen3TTSOnnxModel : public Qwen3TTSOnnxModelBase {
 public:
  explicit CustomQwen3TTSOnnxModel(Qwen3TTSModelConfig config);

  std::vector<std::string> SupportedSpeakers() const;
  CloneInputs BuildCustomVoicePrompt(const std::string& text,
                                     const std::string& speaker,
                                     const std::string& language = "auto",
                                     const std::string& instruct = "",
                                     bool non_streaming_mode = true,
                                     const std::string& anchor_text = "",
                                     const Int64Tensor& anchor_code = Int64Tensor{}) const;
  CloneResult GenerateCustomVoice(const std::string& text,
                                  const std::string& speaker,
                                  const std::string& language = "auto",
                                  const std::string& instruct = "",
                                  const GenerationOptions& options = GenerationOptions{});
  std::vector<CloneResult> StreamCustomVoice(const std::vector<std::string>& text_deltas,
                                             const std::string& speaker,
                                             const std::string& language = "auto",
                                             const std::string& instruct = "",
                                             const SegmentStreamOptions& options = SegmentStreamOptions{});

 private:
  std::string NormalizeSpeaker(const std::string& speaker) const;
  std::string EffectiveLanguage(const std::string& language, const std::string& speaker) const;
  FloatTensor SpeakerEmbed(const std::string& speaker) const;
  FloatTensor CodecConditioning(const std::string& language, const std::string& speaker) const;
};

class DesignQwen3TTSOnnxModel : public Qwen3TTSOnnxModelBase {
 public:
  explicit DesignQwen3TTSOnnxModel(Qwen3TTSModelConfig config);

  CloneInputs BuildVoiceDesignPrompt(const std::string& text,
                                     const std::string& instruct,
                                     const std::string& language = "auto",
                                     bool non_streaming_mode = true,
                                     const std::string& anchor_text = "",
                                     const Int64Tensor& anchor_code = Int64Tensor{}) const;
  CloneResult GenerateVoiceDesign(const std::string& text,
                                  const std::string& instruct,
                                  const std::string& language = "auto",
                                  const GenerationOptions& options = GenerationOptions{});
  std::vector<CloneResult> StreamVoiceDesign(const std::vector<std::string>& text_deltas,
                                             const std::string& instruct,
                                             const std::string& language = "auto",
                                             const SegmentStreamOptions& options = SegmentStreamOptions{});

 private:
  FloatTensor CodecConditioning(const std::string& language) const;
};

}  // namespace qwen3tts
