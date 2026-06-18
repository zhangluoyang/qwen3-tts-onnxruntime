#include "qwen3tts/qwen_model.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <functional>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

#include "qwen3tts/audio_frontend.h"

namespace qwen3tts {
namespace {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool ExtractInt(const std::string& text, const std::string& key, int64_t* value) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*(-?[0-9]+)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) return false;
  *value = std::stoll(match[1].str());
  return true;
}

bool ExtractString(const std::string& text, const std::string& key, std::string* value) {
  const std::regex pattern("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) return false;
  *value = match[1].str();
  return true;
}

std::string ExtractObject(const std::string& text, const std::string& key) {
  const auto pos = text.find("\"" + key + "\"");
  if (pos == std::string::npos) return {};
  const auto begin = text.find('{', pos);
  if (begin == std::string::npos) return {};
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (size_t i = begin; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      if (escaped) escaped = false;
      else if (c == '\\') escaped = true;
      else if (c == '"') in_string = false;
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == '{') ++depth;
    if (c == '}') {
      --depth;
      if (depth == 0) return text.substr(begin, i - begin + 1);
    }
  }
  return {};
}

std::string KeepTopLevelObjectFields(std::string object_text) {
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (char& c : object_text) {
    const char original = c;
    if (in_string) {
      if (escaped) escaped = false;
      else if (original == '\\') escaped = true;
      else if (original == '"') in_string = false;
      if (depth > 1) c = ' ';
      continue;
    }
    if (original == '"') {
      in_string = true;
      if (depth > 1) c = ' ';
      continue;
    }
    if (original == '{' || original == '[') {
      ++depth;
      continue;
    }
    if (original == '}' || original == ']') {
      --depth;
      continue;
    }
    if (depth > 1) c = ' ';
  }
  return object_text;
}

std::unordered_map<std::string, int64_t> ExtractIntMap(const std::string& text, const std::string& key) {
  std::unordered_map<std::string, int64_t> out;
  const auto object = ExtractObject(text, key);
  if (object.empty()) return out;
  const std::regex item("\"([^\"]+)\"\\s*:\\s*(-?[0-9]+)");
  for (auto it = std::sregex_iterator(object.begin(), object.end(), item); it != std::sregex_iterator(); ++it) {
    out[(*it)[1].str()] = std::stoll((*it)[2].str());
  }
  return out;
}

std::unordered_map<std::string, std::string> ExtractStringMap(const std::string& text, const std::string& key) {
  std::unordered_map<std::string, std::string> out;
  const auto object = ExtractObject(text, key);
  if (object.empty()) return out;
  const std::regex item("\"([^\"]+)\"\\s*:\\s*\"([^\"]*)\"");
  for (auto it = std::sregex_iterator(object.begin(), object.end(), item); it != std::sregex_iterator(); ++it) {
    out[(*it)[1].str()] = (*it)[2].str();
  }
  return out;
}

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string Trim(std::string value) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

FloatTensor ReshapeFloat(const FloatTensor& tensor, std::vector<int64_t> shape) {
  if (FloatTensor::NumElements(shape) != tensor.size()) {
    throw std::invalid_argument("float reshape count mismatch from " + FloatTensor::ShapeToString(tensor.shape()) +
                                " to " + FloatTensor::ShapeToString(shape));
  }
  return FloatTensor(std::move(shape), tensor.values());
}

Int64Tensor ReshapeInt64(const Int64Tensor& tensor, std::vector<int64_t> shape) {
  if (Int64Tensor::NumElements(shape) != tensor.size()) {
    throw std::invalid_argument("int64 reshape count mismatch from " + Int64Tensor::ShapeToString(tensor.shape()) +
                                " to " + Int64Tensor::ShapeToString(shape));
  }
  return Int64Tensor(std::move(shape), tensor.values());
}

Int64Tensor FullInt64(std::vector<int64_t> shape, int64_t value) {
  return Int64Tensor(shape, std::vector<int64_t>(Int64Tensor::NumElements(shape), value));
}

Int64Tensor OnesMask(int64_t length) {
  return FullInt64({1, length}, 1);
}

bool HasFrames(const Int64Tensor& codes) {
  return codes.shape().size() == 3 && codes.shape()[0] == 1 && codes.shape()[1] > 0;
}

Int64Tensor ConcatCodeTensors(const std::vector<Int64Tensor>& codes, int64_t num_groups) {
  int64_t total = 0;
  for (const auto& item : codes) {
    if (item.shape().size() != 3 || item.shape()[0] != 1 || item.shape()[2] != num_groups) {
      throw std::invalid_argument("ConcatCodeTensors expects [1,T,G] tensors");
    }
    total += item.shape()[1];
  }
  Int64Tensor out({1, total, num_groups});
  size_t offset = 0;
  for (const auto& item : codes) {
    std::copy(item.values().begin(), item.values().end(), out.values().begin() + static_cast<std::ptrdiff_t>(offset));
    offset += item.values().size();
  }
  return out;
}

size_t Utf8CharCount(const std::string& text) {
  size_t count = 0;
  for (unsigned char c : text) {
    if ((c & 0xC0) != 0x80) ++count;
  }
  return count;
}

size_t ByteOffsetForUtf8Chars(const std::string& text, size_t chars) {
  size_t count = 0;
  for (size_t i = 0; i < text.size(); ++i) {
    if ((static_cast<unsigned char>(text[i]) & 0xC0) != 0x80) {
      if (count == chars) return i;
      ++count;
    }
  }
  return text.size();
}

bool StartsWithAt(const std::string& text, size_t pos, const std::string& token) {
  return pos + token.size() <= text.size() && text.compare(pos, token.size(), token) == 0;
}

class StreamingTextBuffer {
 public:
  StreamingTextBuffer(int min_chars, int max_chars)
      : min_chars_(std::max(0, min_chars)), max_chars_(std::max(1, max_chars)) {}

  std::vector<std::string> Push(const std::string& text) {
    cache_ += text;
    return Extract(false);
  }

  std::vector<std::string> Finish() {
    return Extract(true);
  }

 private:
  std::vector<std::string> Extract(bool force) {
    if (force) {
      std::string tail = cache_;
      cache_.clear();
      return tail.empty() ? std::vector<std::string>{} : std::vector<std::string>{tail};
    }

    std::vector<std::string> segments;
    while (!cache_.empty()) {
      std::optional<size_t> cut;
      if (Utf8CharCount(cache_) >= static_cast<size_t>(min_chars_)) {
        cut = FindDelimiterCut();
      }
      if (!cut && Utf8CharCount(cache_) >= static_cast<size_t>(max_chars_)) {
        const auto space = cache_.rfind(' ');
        cut = space != std::string::npos && space > 0 ? space + 1 : ByteOffsetForUtf8Chars(cache_, max_chars_);
      }
      if (!cut) break;
      segments.push_back(cache_.substr(0, *cut));
      cache_.erase(0, *cut);
    }
    return segments;
  }

  std::optional<size_t> FindDelimiterCut() const {
    static const std::vector<std::string> delimiters = {
        "。", "！", "？", "…", "；", "，", "!", "?", ".", ";", ",", "\n"};
    size_t chars_seen = 0;
    for (size_t i = 0; i < cache_.size();) {
      if ((static_cast<unsigned char>(cache_[i]) & 0xC0) != 0x80) ++chars_seen;
      for (const auto& delimiter : delimiters) {
        if (StartsWithAt(cache_, i, delimiter) && chars_seen >= static_cast<size_t>(min_chars_)) {
          size_t end = i + delimiter.size();
          while (end < cache_.size() && std::isspace(static_cast<unsigned char>(cache_[end]))) ++end;
          return end;
        }
      }
      ++i;
      while (i < cache_.size() && (static_cast<unsigned char>(cache_[i]) & 0xC0) == 0x80) ++i;
    }
    return std::nullopt;
  }

  int min_chars_;
  int max_chars_;
  std::string cache_;
};

using SegmentPromptBuilder =
    std::function<CloneInputs(const std::string& segment, const std::string& anchor_text, const Int64Tensor& anchor_code)>;

std::vector<CloneResult> RunSegmentStream(Qwen3TTSOnnxModelBase& model,
                                          const std::vector<std::string>& text_deltas,
                                          const SegmentStreamOptions& options,
                                          const SegmentPromptBuilder& prompt_builder) {
  StreamingTextBuffer text_buffer(options.min_text_chunk_chars, options.max_text_chunk_chars);
  std::vector<std::pair<std::string, Int64Tensor>> pinned_segments;
  std::deque<std::pair<std::string, Int64Tensor>> rolling_segments;
  bool rolling_evicted = false;
  std::vector<CloneResult> outputs;
  std::mt19937_64 stream_rng(options.generation.seed);

  auto all_anchor_segments = [&]() {
    std::vector<std::pair<std::string, Int64Tensor>> out;
    if (!rolling_evicted) {
      out.insert(out.end(), pinned_segments.begin(), pinned_segments.end());
    }
    out.insert(out.end(), rolling_segments.begin(), rolling_segments.end());
    return out;
  };

  auto combined_anchor = [&]() -> std::pair<std::string, Int64Tensor> {
    const auto anchors = all_anchor_segments();
    if (anchors.empty()) return {"", Int64Tensor({1, 0, model.NumCodeGroups()})};
    std::string text;
    std::vector<Int64Tensor> code_parts;
    for (const auto& item : anchors) {
      text += item.first;
      code_parts.push_back(item.second);
    }
    return {text, ConcatCodeTensors(code_parts, model.NumCodeGroups())};
  };

  auto save_anchor = [&](const std::string& segment, const Int64Tensor& generated_codes) {
    if (!HasFrames(generated_codes)) return;
    if (static_cast<int>(pinned_segments.size()) < std::max(0, options.pinned_anchor_segment_count)) {
      pinned_segments.push_back({segment, generated_codes});
      return;
    }
    const int max_rolling = std::max(0, options.kv_anchor_segment_count);
    if (max_rolling <= 0) return;
    if (static_cast<int>(rolling_segments.size()) == max_rolling) {
      rolling_segments.pop_front();
      rolling_evicted = true;
    }
    rolling_segments.push_back({segment, generated_codes});
  };

  auto generate_segment = [&](const std::string& segment) {
    if (segment.empty()) return;
    auto [anchor_text, anchor_code] = combined_anchor();
    CloneInputs prompt = prompt_builder(segment, anchor_text, anchor_code);
    // Python 的 ConditionedSegmentStreamingSession 会把同一个 rng 传给每个 segment，
    // seed=None；这里也让随机状态跨 segment 继续推进，避免每段都从同一 seed 重新采样。
    CloneResult result = model.GenerateAudioFromPrompt(prompt, options.generation, &stream_rng);
    save_anchor(segment, result.generated_codes);
    outputs.push_back(std::move(result));
  };

  for (const auto& delta : text_deltas) {
    for (const auto& segment : text_buffer.Push(delta)) {
      generate_segment(segment);
    }
  }
  for (const auto& segment : text_buffer.Finish()) {
    generate_segment(segment);
  }
  return outputs;
}

}  // namespace

Qwen3TTSOnnxModelBase::Qwen3TTSOnnxModelBase(Qwen3TTSModelConfig config)
    : config_(std::move(config)),
      ids_(LoadModelIds(config_.model_dir)),
      env_(ORT_LOGGING_LEVEL_WARNING, "qwen3tts_model_cpp"),
      tokenizer_(config_.model_dir),
      text_project_(env_, OrtRunnerOptions{config_.onnx_dir / "text_project" / "text_project.onnx",
                                           config_.use_cuda, config_.cuda_device_id}),
      codec_embed_(env_, OrtRunnerOptions{config_.onnx_dir / "codec_embed" / "codec_embed.onnx",
                                          config_.use_cuda, config_.cuda_device_id}),
      pipeline_(MakeRuntimeConfig(GenerationOptions{})) {}

Qwen3TTSOnnxModelBase::~Qwen3TTSOnnxModelBase() = default;

Qwen3TTSOnnxModelBase::ModelIds Qwen3TTSOnnxModelBase::LoadModelIds(const std::filesystem::path& model_dir) {
  ModelIds ids;
  const auto config = ReadTextFile(model_dir / "config.json");
  const auto talker_config = ExtractObject(config, "talker_config");
  const auto talker_top_level = KeepTopLevelObjectFields(talker_config.empty() ? config : talker_config);
  (void)ExtractInt(config, "tts_bos_token_id", &ids.tts_bos_token_id);
  (void)ExtractInt(config, "tts_eos_token_id", &ids.tts_eos_token_id);
  (void)ExtractInt(config, "tts_pad_token_id", &ids.tts_pad_token_id);
  (void)ExtractString(config, "tts_model_size", &ids.tts_model_size);
  (void)ExtractInt(talker_top_level, "hidden_size", &ids.hidden_size);
  (void)ExtractInt(talker_top_level, "num_hidden_layers", &ids.num_hidden_layers);
  (void)ExtractInt(talker_top_level, "num_key_value_heads", &ids.num_key_value_heads);
  (void)ExtractInt(talker_top_level, "head_dim", &ids.head_dim);
  (void)ExtractInt(talker_top_level, "vocab_size", &ids.vocab_size);
  (void)ExtractInt(talker_top_level, "first_codebook_mask_tail", &ids.first_codebook_mask_tail);
  (void)ExtractInt(talker_top_level, "num_code_groups", &ids.num_code_groups);
  (void)ExtractInt(talker_top_level, "codec_bos_id", &ids.codec_bos_id);
  (void)ExtractInt(talker_top_level, "codec_eos_token_id", &ids.codec_eos_token_id);
  (void)ExtractInt(talker_top_level, "codec_pad_id", &ids.codec_pad_id);
  (void)ExtractInt(talker_top_level, "codec_think_id", &ids.codec_think_id);
  (void)ExtractInt(talker_top_level, "codec_nothink_id", &ids.codec_nothink_id);
  (void)ExtractInt(talker_top_level, "codec_think_bos_id", &ids.codec_think_bos_id);
  (void)ExtractInt(talker_top_level, "codec_think_eos_id", &ids.codec_think_eos_id);
  ids.codec_language_id = ExtractIntMap(talker_config.empty() ? config : talker_config, "codec_language_id");
  ids.spk_id = ExtractIntMap(talker_config.empty() ? config : talker_config, "spk_id");
  ids.spk_is_dialect = ExtractStringMap(talker_config.empty() ? config : talker_config, "spk_is_dialect");

  const auto audio_config_path = model_dir / "speech_tokenizer" / "config.json";
  if (std::filesystem::exists(audio_config_path)) {
    const auto audio_config = ReadTextFile(audio_config_path);
    (void)ExtractInt(audio_config, "output_sample_rate", &ids.audio_sample_rate);
    (void)ExtractInt(audio_config, "decode_upsample_rate", &ids.decode_upsample_rate);
  }
  return ids;
}

std::string Qwen3TTSOnnxModelBase::BuildAssistantText(const std::string& text) {
  return "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string Qwen3TTSOnnxModelBase::BuildReferenceText(const std::string& text) {
  return "<|im_start|>assistant\n" + text + "<|im_end|>\n";
}

std::string Qwen3TTSOnnxModelBase::BuildInstructText(const std::string& text) {
  return "<|im_start|>user\n" + text + "<|im_end|>\n";
}

CloneRuntimeConfig Qwen3TTSOnnxModelBase::MakeRuntimeConfig(const GenerationOptions& options) const {
  CloneRuntimeConfig runtime;
  runtime.onnx_dir = config_.onnx_dir;
  runtime.use_cuda = config_.use_cuda;
  runtime.cuda_device_id = config_.cuda_device_id;
  runtime.max_new_tokens = options.max_new_tokens;
  runtime.min_new_tokens = options.min_new_tokens;
  runtime.eos_token_id = options.eos_token_id >= 0 ? options.eos_token_id : ids_.codec_eos_token_id;
  runtime.vocab_size = ids_.vocab_size;
  runtime.first_codebook_mask_tail = ids_.first_codebook_mask_tail;
  runtime.num_hidden_layers = ids_.num_hidden_layers;
  runtime.num_key_value_heads = ids_.num_key_value_heads;
  runtime.head_dim = ids_.head_dim;
  runtime.num_code_groups = ids_.num_code_groups;
  runtime.decode_upsample_rate = ids_.decode_upsample_rate;
  runtime.audio_sample_rate = ids_.audio_sample_rate;
  runtime.do_sample = options.do_sample;
  runtime.top_k = options.top_k;
  runtime.top_p = options.top_p;
  runtime.temperature = options.temperature;
  runtime.repetition_penalty = options.repetition_penalty;
  runtime.seed = options.seed;
  return runtime;
}

CloneResult Qwen3TTSOnnxModelBase::GenerateAudioFromPrompt(const CloneInputs& prompt,
                                                           const GenerationOptions& options) {
  return pipeline_.Run(prompt, MakeRuntimeConfig(options));
}

CloneResult Qwen3TTSOnnxModelBase::GenerateAudioFromPrompt(const CloneInputs& prompt,
                                                           const GenerationOptions& options,
                                                           std::mt19937_64* rng) {
  return pipeline_.Run(prompt, MakeRuntimeConfig(options), rng);
}

std::vector<std::pair<std::string, double>> Qwen3TTSOnnxModelBase::SessionLoadTimings() const {
  auto timings = std::vector<std::pair<std::string, double>>{
      {"text_project", text_project_.LoadSeconds()},
      {"codec_embed", codec_embed_.LoadSeconds()},
  };
  if (tokenizer_encode_) timings.push_back({"tokenizer_encode", tokenizer_encode_->LoadSeconds()});
  if (speaker_encoder_) timings.push_back({"speaker_encoder", speaker_encoder_->LoadSeconds()});
  for (const auto& item : pipeline_.SessionLoadTimings()) timings.push_back(item);
  return timings;
}

int64_t Qwen3TTSOnnxModelBase::AudioSampleRate() const {
  return ids_.audio_sample_rate;
}

int64_t Qwen3TTSOnnxModelBase::NumCodeGroups() const {
  return ids_.num_code_groups;
}

int64_t Qwen3TTSOnnxModelBase::LanguageId(const std::string& language) const {
  const auto key = Lower(language.empty() ? "auto" : language);
  if (key == "auto") return -1;
  auto it = ids_.codec_language_id.find(key);
  if (it == ids_.codec_language_id.end()) throw std::runtime_error("unsupported language: " + language);
  return it->second;
}

std::vector<int64_t> Qwen3TTSOnnxModelBase::CodecPrefillIds(const std::string& language) const {
  const int64_t language_id = LanguageId(language);
  if (language_id < 0) return {ids_.codec_nothink_id, ids_.codec_think_bos_id, ids_.codec_think_eos_id};
  return {ids_.codec_think_id, ids_.codec_think_bos_id, language_id, ids_.codec_think_eos_id};
}

Int64Tensor Qwen3TTSOnnxModelBase::EmptyCodes() const {
  return Int64Tensor({1, 0, ids_.num_code_groups});
}

Int64Tensor Qwen3TTSOnnxModelBase::RowTensor(std::vector<int64_t> values) const {
  const int64_t len = static_cast<int64_t>(values.size());
  return Int64Tensor({1, len}, std::move(values));
}

Int64Tensor Qwen3TTSOnnxModelBase::ShapeCodes(const Int64Tensor& codes) const {
  if (codes.empty()) return EmptyCodes();
  if (codes.shape().size() == 3) {
    if (codes.shape()[0] != 1 || codes.shape()[2] != ids_.num_code_groups) {
      throw std::invalid_argument("codes must have shape [1,T,G]");
    }
    return codes;
  }
  if (codes.shape().size() == 2) {
    if (codes.shape()[1] != ids_.num_code_groups) {
      throw std::invalid_argument("codes codebook count mismatch");
    }
    return ReshapeInt64(codes, {1, codes.shape()[0], codes.shape()[1]});
  }
  throw std::invalid_argument("codes must have shape [T,G] or [1,T,G]");
}

Int64Tensor Qwen3TTSOnnxModelBase::TokenizeToTensor(const std::string& text) const {
  return RowTensor(tokenizer_.Encode(text));
}

FloatTensor Qwen3TTSOnnxModelBase::TextProject(const Int64Tensor& input_ids) const {
  if (input_ids.shape().size() != 2) throw std::invalid_argument("TextProject expects [1,T] input ids");
  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("input_ids", text_project_.MakeInt64Input(input_ids));
  auto outputs = text_project_.RunIo(feed, {"text_embed"}, {});
  auto output = text_project_.CopyFloatTensor(outputs[0]);
  return ReshapeFloat(output, {1, input_ids.shape()[1], ids_.hidden_size});
}

FloatTensor Qwen3TTSOnnxModelBase::CodecEmbed(const Int64Tensor& token_ids) const {
  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("token_ids", codec_embed_.MakeInt64Input(token_ids));
  Int64Tensor dummy_ref({1, 1, ids_.num_code_groups});
  feed.emplace("ref_code", codec_embed_.MakeInt64Input(dummy_ref));
  auto outputs = codec_embed_.RunIo(feed, {"embed"}, {});
  auto output = codec_embed_.CopyFloatTensor(outputs[0]);
  return ReshapeFloat(output, {1, token_ids.shape()[1], ids_.hidden_size});
}

FloatTensor Qwen3TTSOnnxModelBase::RefCodeEmbed(const Int64Tensor& ref_code) const {
  const auto codes = ShapeCodes(ref_code);
  if (codes.shape()[1] <= 0) throw std::invalid_argument("ref_code_embed requires at least one frame");
  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("token_ids", codec_embed_.MakeInt64Input(RowTensor({ids_.codec_bos_id})));
  feed.emplace("ref_code", codec_embed_.MakeInt64Input(codes));
  auto outputs = codec_embed_.RunIo(feed, {"ref_code_embed"}, {});
  auto output = codec_embed_.CopyFloatTensor(outputs[0]);
  return ReshapeFloat(output, {1, codes.shape()[1], ids_.hidden_size});
}

OrtRunner& Qwen3TTSOnnxModelBase::TokenizerEncodeRunner() const {
  if (!tokenizer_encode_) {
    tokenizer_encode_ = std::make_unique<OrtRunner>(
        const_cast<Ort::Env&>(env_),
        OrtRunnerOptions{config_.onnx_dir / "tokenizer" / "tokenizer12hz_encode.onnx",
                         config_.use_cuda, config_.cuda_device_id});
  }
  return *tokenizer_encode_;
}

OrtRunner& Qwen3TTSOnnxModelBase::SpeakerEncoderRunner() const {
  if (!speaker_encoder_) {
    speaker_encoder_ = std::make_unique<OrtRunner>(
        const_cast<Ort::Env&>(env_),
        OrtRunnerOptions{config_.onnx_dir / "speaker_encoder" / "speaker_encoder.onnx",
                         config_.use_cuda, config_.cuda_device_id});
  }
  return *speaker_encoder_;
}

Int64Tensor Qwen3TTSOnnxModelBase::EncodeReferenceAudioFile(const std::filesystem::path& path) const {
  auto audio = LoadAudioMono(path, static_cast<int>(ids_.audio_sample_rate));
  FloatTensor input({1, static_cast<int64_t>(audio.samples.size())}, audio.samples);
  auto& runner = TokenizerEncodeRunner();
  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("audio", runner.MakeFloatInput(input, "audio"));
  auto outputs = runner.RunIo(feed, {"codes"}, {});
  return ShapeCodes(runner.CopyInt64Tensor(outputs[0]));
}

FloatTensor Qwen3TTSOnnxModelBase::EncodeSpeakerEmbeddingFile(const std::filesystem::path& path) const {
  auto audio = LoadAudioMono(path, static_cast<int>(ids_.audio_sample_rate));
  auto mel = MelSpectrogram(audio.samples, static_cast<int>(ids_.audio_sample_rate));
  auto& runner = SpeakerEncoderRunner();
  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("mel", runner.MakeFloatInput(mel, "mel"));
  auto outputs = runner.RunIo(feed, {"speaker_embedding"}, {});
  auto speaker = runner.CopyFloatTensor(outputs[0]);
  if (speaker.shape().size() == 2 && speaker.shape()[0] == 1) return ReshapeFloat(speaker, {1, 1, speaker.shape()[1]});
  if (speaker.shape().size() == 3) return speaker;
  return ReshapeFloat(speaker, {1, 1, static_cast<int64_t>(speaker.size())});
}

Int64Tensor Qwen3TTSOnnxModelBase::TargetTextIds(const std::string& text) const {
  if (text.empty()) return Int64Tensor({1, 0});
  const auto ids = TokenizeToTensor(BuildAssistantText(text));
  return SliceAxis1(ids, 3, ids.shape()[1] - 5);
}

FloatTensor Qwen3TTSOnnxModelBase::TargetTextEmbeds(const std::string& text) const {
  const auto ids = TargetTextIds(text);
  if (ids.shape()[1] == 0) return FloatTensor({1, 0, ids_.hidden_size});
  return TextProject(ids);
}

FloatTensor Qwen3TTSOnnxModelBase::InstructEmbed(const std::string& instruct) const {
  if (instruct.empty()) return FloatTensor({1, 0, ids_.hidden_size});
  return TextProject(TokenizeToTensor(BuildInstructText(instruct)));
}

FloatTensor Qwen3TTSOnnxModelBase::CodecTailEmbed() const {
  return CodecEmbed(RowTensor({ids_.codec_pad_id, ids_.codec_bos_id}));
}

std::tuple<FloatTensor, FloatTensor, FloatTensor> Qwen3TTSOnnxModelBase::TtsSpecialEmbeds() const {
  const auto special = TextProject(RowTensor({ids_.tts_bos_token_id, ids_.tts_eos_token_id, ids_.tts_pad_token_id}));
  return {SliceAxis1(special, 0, 1), SliceAxis1(special, 1, 2), SliceAxis1(special, 2, 3)};
}

std::tuple<FloatTensor, FloatTensor, FloatTensor, FloatTensor, FloatTensor>
Qwen3TTSOnnxModelBase::BuildTalkerPrefix(const Int64Tensor& input_id,
                                         const std::string& language,
                                         const std::optional<FloatTensor>& speaker_embed) const {
  auto [tts_bos_embed, tts_eos_embed, tts_pad_embed] = TtsSpecialEmbeds();
  const auto codec_input_0 = CodecEmbed(RowTensor(CodecPrefillIds(language)));
  const auto codec_input_1 = CodecTailEmbed();
  FloatTensor codec_input = speaker_embed
                                ? ConcatAxis1({codec_input_0,
                                               ReshapeFloat(*speaker_embed, {1, 1, static_cast<int64_t>(speaker_embed->size())}),
                                               codec_input_1})
                                : ConcatAxis1({codec_input_0, codec_input_1});
  const auto role_embed = TextProject(SliceAxis1(input_id, 0, 3));
  const auto pad_prefix = RepeatAxis1(tts_pad_embed, codec_input.shape()[1] - 2);
  const auto text_side = ConcatAxis1({pad_prefix, tts_bos_embed});
  const auto codec_side = SliceAxis1(codec_input, 0, codec_input.shape()[1] - 1);
  const auto talker_input = ConcatAxis1({role_embed, Add(text_side, codec_side)});
  return {talker_input, codec_input, tts_bos_embed, tts_eos_embed, tts_pad_embed};
}

std::pair<Int64Tensor, int64_t> Qwen3TTSOnnxModelBase::NormalizeAnchorCode(const Int64Tensor& anchor_code) const {
  const auto shaped = ShapeCodes(anchor_code);
  return {shaped, shaped.shape()[1]};
}

FloatTensor Qwen3TTSOnnxModelBase::RoleAlignedCodecPart(const FloatTensor& codec_input,
                                                        const FloatTensor& tts_bos_embed,
                                                        const FloatTensor& tts_pad_embed) const {
  const auto pad_prefix = RepeatAxis1(tts_pad_embed, codec_input.shape()[1] - 2);
  const auto text_side = ConcatAxis1({pad_prefix, tts_bos_embed});
  return Add(text_side, SliceAxis1(codec_input, 0, codec_input.shape()[1] - 1));
}

std::pair<FloatTensor, FloatTensor> Qwen3TTSOnnxModelBase::AppendNonStreamingText(
    const FloatTensor& talker_input_embed,
    const Int64Tensor& input_id,
    const FloatTensor& tts_eos_embed,
    const FloatTensor& tts_pad_embed) const {
  const auto target_ids = SliceAxis1(input_id, 3, input_id.shape()[1] - 5);
  const auto target_text = target_ids.shape()[1] ? TextProject(target_ids) : FloatTensor({1, 0, ids_.hidden_size});
  const auto text_with_eos = ConcatAxis1({target_text, tts_eos_embed});
  const auto codec_pad = CodecEmbed(FullInt64({1, text_with_eos.shape()[1]}, ids_.codec_pad_id));
  const auto codec_bos = CodecEmbed(RowTensor({ids_.codec_bos_id}));
  const auto input_without_tail = SliceAxis1(talker_input_embed, 0, talker_input_embed.shape()[1] - 1);
  const auto updated = ConcatAxis1({input_without_tail, Add(text_with_eos, codec_pad), Add(tts_pad_embed, codec_bos)});
  return {updated, tts_pad_embed};
}

std::pair<FloatTensor, FloatTensor> Qwen3TTSOnnxModelBase::GenerateIclPrompt(
    const Int64Tensor& text_id,
    const Int64Tensor& ref_id,
    const Int64Tensor& ref_code,
    const FloatTensor& tts_pad_embed,
    const FloatTensor& tts_eos_embed,
    bool non_streaming_mode,
    bool include_text_eos) const {
  const auto ref_text = ref_id.shape()[1] ? TextProject(ref_id) : FloatTensor({1, 0, ids_.hidden_size});
  const auto target_text = text_id.shape()[1] ? TextProject(text_id) : FloatTensor({1, 0, ids_.hidden_size});
  FloatTensor text_embed = include_text_eos ? ConcatAxis1({ref_text, target_text, tts_eos_embed})
                                            : ConcatAxis1({ref_text, target_text});
  const auto code_embed = RefCodeEmbed(ref_code);
  const auto codec_bos = CodecEmbed(RowTensor({ids_.codec_bos_id}));
  const auto codec_embed = ConcatAxis1({codec_bos, code_embed});
  const int64_t text_len = text_embed.shape()[1];
  const int64_t codec_len = codec_embed.shape()[1];

  if (non_streaming_mode) {
    const auto codec_pad = CodecEmbed(FullInt64({1, text_len}, ids_.codec_pad_id));
    const auto icl_text = Add(text_embed, codec_pad);
    const auto icl_codec = Add(codec_embed, RepeatAxis1(tts_pad_embed, codec_len));
    return {ConcatAxis1({icl_text, icl_codec}), tts_pad_embed};
  }

  if (text_len > codec_len) {
    return {Add(SliceAxis1(text_embed, 0, codec_len), codec_embed), SliceAxis1(text_embed, codec_len, text_len)};
  }
  if (text_len < codec_len) {
    text_embed = ConcatAxis1({text_embed, RepeatAxis1(tts_pad_embed, codec_len - text_len)});
  }
  return {Add(text_embed, codec_embed), FloatTensor({1, 0, ids_.hidden_size})};
}

CloneInputs Qwen3TTSOnnxModelBase::BuildConditionedPrompt(const std::string& text,
                                                          const FloatTensor& codec_input,
                                                          const std::string& instruct,
                                                          bool non_streaming_mode,
                                                          const std::string& anchor_text,
                                                          const Int64Tensor& anchor_code) const {
  const auto input_id = TokenizeToTensor(BuildAssistantText(text));
  const auto instruct_embed = InstructEmbed(instruct);
  auto [tts_bos_embed, tts_eos_embed, tts_pad_embed] = TtsSpecialEmbeds();
  const auto role_embed = TextProject(SliceAxis1(input_id, 0, 3));
  const auto codec_part = RoleAlignedCodecPart(codec_input, tts_bos_embed, tts_pad_embed);
  FloatTensor talker_input = ConcatAxis1({role_embed, codec_part});

  auto [normalized_anchor_code, anchor_code_len] = NormalizeAnchorCode(anchor_code);
  FloatTensor trailing_text_hidden;
  Int64Tensor ref_code_for_audio = EmptyCodes();
  if (non_streaming_mode && anchor_code_len > 0) {
    const auto anchor_text_ids = TargetTextIds(anchor_text);
    const auto anchor_text_embed =
        anchor_text_ids.shape()[1] ? TextProject(anchor_text_ids) : FloatTensor({1, 0, ids_.hidden_size});
    const auto target_ids = SliceAxis1(input_id, 3, input_id.shape()[1] - 5);
    const auto target_text_embed =
        target_ids.shape()[1] ? TextProject(target_ids) : FloatTensor({1, 0, ids_.hidden_size});
    const auto text_with_eos = ConcatAxis1({anchor_text_embed, target_text_embed, tts_eos_embed});
    const auto codec_pad = CodecEmbed(FullInt64({1, text_with_eos.shape()[1]}, ids_.codec_pad_id));
    const auto text_side = Add(text_with_eos, codec_pad);
    const auto codec_bos = CodecEmbed(RowTensor({ids_.codec_bos_id}));
    const auto codec_side = Add(ConcatAxis1({codec_bos, RefCodeEmbed(normalized_anchor_code)}),
                                RepeatAxis1(tts_pad_embed, anchor_code_len + 1));
    talker_input = ConcatAxis1({talker_input, text_side, codec_side});
    trailing_text_hidden = tts_pad_embed;
    ref_code_for_audio = normalized_anchor_code;
  } else {
    const auto first_text = Add(TextProject(SliceAxis1(input_id, 3, 4)),
                                SliceAxis1(codec_input, codec_input.shape()[1] - 1, codec_input.shape()[1]));
    talker_input = ConcatAxis1({talker_input, first_text});
    if (non_streaming_mode) {
      auto appended = AppendNonStreamingText(talker_input, input_id, tts_eos_embed, tts_pad_embed);
      talker_input = std::move(appended.first);
      trailing_text_hidden = std::move(appended.second);
    } else {
      const auto target_tail_ids = SliceAxis1(input_id, 4, input_id.shape()[1] - 5);
      const auto target_tail = target_tail_ids.shape()[1] ? TextProject(target_tail_ids)
                                                          : FloatTensor({1, 0, ids_.hidden_size});
      trailing_text_hidden = ConcatAxis1({target_tail, tts_eos_embed});
    }
  }

  if (instruct_embed.shape()[1] > 0) {
    talker_input = ConcatAxis1({instruct_embed, talker_input});
  }

  CloneInputs prompt;
  prompt.inputs_embeds = std::move(talker_input);
  prompt.attention_mask = OnesMask(prompt.inputs_embeds.shape()[1]);
  prompt.trailing_text_hidden = std::move(trailing_text_hidden);
  prompt.tts_pad_embed = std::move(tts_pad_embed);
  prompt.ref_code = std::move(ref_code_for_audio);
  return prompt;
}

BaseQwen3TTSOnnxModel::BaseQwen3TTSOnnxModel(Qwen3TTSModelConfig config)
    : Qwen3TTSOnnxModelBase(std::move(config)) {}

CloneInputs BaseQwen3TTSOnnxModel::BuildClonePrompt(const std::string& text,
                                                    const std::string& language,
                                                    const std::string& ref_text,
                                                    const Int64Tensor& ref_code,
                                                    const FloatTensor& ref_spk_embedding,
                                                    bool x_vector_only_mode,
                                                    bool non_streaming_mode) const {
  if (!x_vector_only_mode && ref_text.empty()) {
    throw std::invalid_argument("ref_text is required when x_vector_only_mode=false");
  }
  if (!x_vector_only_mode && !HasFrames(ShapeCodes(ref_code))) {
    throw std::invalid_argument("ref_code is required when x_vector_only_mode=false");
  }
  if (ref_spk_embedding.empty()) throw std::invalid_argument("ref_spk_embedding is required");

  const auto input_id = TokenizeToTensor(BuildAssistantText(text));
  auto [talker_input, codec_input, tts_bos_embed, tts_eos_embed, tts_pad_embed] =
      BuildTalkerPrefix(input_id, language, ref_spk_embedding);

  FloatTensor trailing_text_hidden;
  Int64Tensor ref_code_for_audio = EmptyCodes();
  if (!x_vector_only_mode) {
    const auto ref_id = TokenizeToTensor(BuildReferenceText(ref_text));
    ref_code_for_audio = ShapeCodes(ref_code);
    const auto icl = GenerateIclPrompt(
        SliceAxis1(input_id, 3, input_id.shape()[1] - 5),
        SliceAxis1(ref_id, 3, ref_id.shape()[1] - 2),
        ref_code_for_audio,
        tts_pad_embed,
        tts_eos_embed,
        non_streaming_mode);
    talker_input = ConcatAxis1({talker_input, icl.first});
    trailing_text_hidden = icl.second;
  } else {
    const auto first_text = Add(TextProject(SliceAxis1(input_id, 3, 4)),
                                SliceAxis1(codec_input, codec_input.shape()[1] - 1, codec_input.shape()[1]));
    talker_input = ConcatAxis1({talker_input, first_text});
    if (non_streaming_mode) {
      talker_input = SliceAxis1(talker_input, 0, talker_input.shape()[1] - 1);
      const auto target_ids = SliceAxis1(input_id, 3, input_id.shape()[1] - 5);
      const auto target_text = target_ids.shape()[1] ? TextProject(target_ids) : FloatTensor({1, 0, ids_.hidden_size});
      const auto text_with_eos = ConcatAxis1({target_text, tts_eos_embed});
      const auto codec_pad = CodecEmbed(FullInt64({1, target_ids.shape()[1] + 1}, ids_.codec_pad_id));
      const auto codec_bos = CodecEmbed(RowTensor({ids_.codec_bos_id}));
      talker_input = ConcatAxis1({talker_input, Add(text_with_eos, codec_pad), Add(tts_pad_embed, codec_bos)});
      trailing_text_hidden = tts_pad_embed;
    } else {
      const auto target_tail_ids = SliceAxis1(input_id, 4, input_id.shape()[1] - 5);
      const auto target_tail = target_tail_ids.shape()[1] ? TextProject(target_tail_ids)
                                                          : FloatTensor({1, 0, ids_.hidden_size});
      trailing_text_hidden = ConcatAxis1({target_tail, tts_eos_embed});
    }
  }

  CloneInputs prompt;
  prompt.inputs_embeds = std::move(talker_input);
  prompt.attention_mask = OnesMask(prompt.inputs_embeds.shape()[1]);
  prompt.trailing_text_hidden = std::move(trailing_text_hidden);
  prompt.tts_pad_embed = std::move(tts_pad_embed);
  prompt.ref_code = std::move(ref_code_for_audio);
  return prompt;
}

CloneInputs BaseQwen3TTSOnnxModel::BuildClonePromptFromReference(const std::string& text,
                                                                 const VoiceCloneReference& reference,
                                                                 const std::string& language,
                                                                 bool x_vector_only_mode,
                                                                 bool non_streaming_mode) const {
  const auto ref_code = x_vector_only_mode ? EmptyCodes() : EncodeReferenceAudioFile(reference.audio_path);
  const auto speaker = EncodeSpeakerEmbeddingFile(reference.audio_path);
  return BuildClonePrompt(text, language, reference.text, ref_code, speaker, x_vector_only_mode, non_streaming_mode);
}

CloneResult BaseQwen3TTSOnnxModel::GenerateCloneAudioFromReference(const std::string& text,
                                                                  const VoiceCloneReference& reference,
                                                                  const std::string& language,
                                                                  bool x_vector_only_mode,
                                                                  bool non_streaming_mode,
                                                                  const GenerationOptions& options) {
  auto prompt = BuildClonePromptFromReference(text, reference, language, x_vector_only_mode, non_streaming_mode);
  return GenerateAudioFromPrompt(prompt, options);
}

std::vector<CloneResult> BaseQwen3TTSOnnxModel::StreamCloneAudioFromReference(
    const std::vector<std::string>& text_deltas,
    const VoiceCloneReference& reference,
    const std::string& language,
    const SegmentStreamOptions& options) {
  const auto base_ref_code = EncodeReferenceAudioFile(reference.audio_path);
  const auto speaker = EncodeSpeakerEmbeddingFile(reference.audio_path);

  return RunSegmentStream(
      *this, text_deltas, options,
      [&](const std::string& segment, const std::string& anchor_text, const Int64Tensor& anchor_code) {
        Int64Tensor ref_code = base_ref_code;
        if (HasFrames(anchor_code)) {
          ref_code = ConcatCodeTensors({base_ref_code, anchor_code}, ids_.num_code_groups);
        }
        return BuildClonePrompt(segment,
                                language,
                                reference.text + anchor_text,
                                ref_code,
                                speaker,
                                false,
                                true);
      });
}

CustomQwen3TTSOnnxModel::CustomQwen3TTSOnnxModel(Qwen3TTSModelConfig config)
    : Qwen3TTSOnnxModelBase(std::move(config)) {}

std::vector<std::string> CustomQwen3TTSOnnxModel::SupportedSpeakers() const {
  std::vector<std::string> speakers;
  speakers.reserve(ids_.spk_id.size());
  for (const auto& item : ids_.spk_id) speakers.push_back(item.first);
  std::sort(speakers.begin(), speakers.end());
  return speakers;
}

std::string CustomQwen3TTSOnnxModel::NormalizeSpeaker(const std::string& speaker) const {
  auto key = Lower(Trim(speaker));
  if (key.empty()) throw std::invalid_argument("speaker must not be empty");
  return key;
}

std::string CustomQwen3TTSOnnxModel::EffectiveLanguage(const std::string& language, const std::string& speaker) const {
  auto language_key = Lower(language.empty() ? "auto" : language);
  auto speaker_key = NormalizeSpeaker(speaker);
  auto dialect = ids_.spk_is_dialect.find(speaker_key);
  if ((language_key == "chinese" || language_key == "auto") && dialect != ids_.spk_is_dialect.end() &&
      !dialect->second.empty()) {
    return Lower(dialect->second);
  }
  return language_key;
}

FloatTensor CustomQwen3TTSOnnxModel::SpeakerEmbed(const std::string& speaker) const {
  const auto key = NormalizeSpeaker(speaker);
  auto it = ids_.spk_id.find(key);
  if (it == ids_.spk_id.end()) throw std::runtime_error("unsupported speaker: " + speaker);
  return CodecEmbed(RowTensor({it->second}));
}

FloatTensor CustomQwen3TTSOnnxModel::CodecConditioning(const std::string& language, const std::string& speaker) const {
  const auto effective_language = EffectiveLanguage(language, speaker);
  return ConcatAxis1({CodecEmbed(RowTensor(CodecPrefillIds(effective_language))),
                      SpeakerEmbed(speaker),
                      CodecTailEmbed()});
}

CloneInputs CustomQwen3TTSOnnxModel::BuildCustomVoicePrompt(const std::string& text,
                                                            const std::string& speaker,
                                                            const std::string& language,
                                                            const std::string& instruct,
                                                            bool non_streaming_mode,
                                                            const std::string& anchor_text,
                                                            const Int64Tensor& anchor_code) const {
  if (!instruct.empty() && Lower(ids_.tts_model_size) == "0b6") {
    throw std::runtime_error("0.6B CustomVoice does not support instruct");
  }
  return BuildConditionedPrompt(text,
                                CodecConditioning(language, speaker),
                                instruct,
                                non_streaming_mode,
                                anchor_text,
                                anchor_code);
}

CloneResult CustomQwen3TTSOnnxModel::GenerateCustomVoice(const std::string& text,
                                                        const std::string& speaker,
                                                        const std::string& language,
                                                        const std::string& instruct,
                                                        const GenerationOptions& options) {
  auto prompt = BuildCustomVoicePrompt(text, speaker, language, instruct, true);
  return GenerateAudioFromPrompt(prompt, options);
}

std::vector<CloneResult> CustomQwen3TTSOnnxModel::StreamCustomVoice(const std::vector<std::string>& text_deltas,
                                                                    const std::string& speaker,
                                                                    const std::string& language,
                                                                    const std::string& instruct,
                                                                    const SegmentStreamOptions& options) {
  return RunSegmentStream(
      *this, text_deltas, options,
      [&](const std::string& segment, const std::string& anchor_text, const Int64Tensor& anchor_code) {
        return BuildCustomVoicePrompt(segment, speaker, language, instruct, true, anchor_text, anchor_code);
      });
}

DesignQwen3TTSOnnxModel::DesignQwen3TTSOnnxModel(Qwen3TTSModelConfig config)
    : Qwen3TTSOnnxModelBase(std::move(config)) {}

FloatTensor DesignQwen3TTSOnnxModel::CodecConditioning(const std::string& language) const {
  const auto language_key = Lower(language.empty() ? "auto" : language);
  return ConcatAxis1({CodecEmbed(RowTensor(CodecPrefillIds(language_key))), CodecTailEmbed()});
}

CloneInputs DesignQwen3TTSOnnxModel::BuildVoiceDesignPrompt(const std::string& text,
                                                            const std::string& instruct,
                                                            const std::string& language,
                                                            bool non_streaming_mode,
                                                            const std::string& anchor_text,
                                                            const Int64Tensor& anchor_code) const {
  return BuildConditionedPrompt(text,
                                CodecConditioning(language),
                                instruct,
                                non_streaming_mode,
                                anchor_text,
                                anchor_code);
}

CloneResult DesignQwen3TTSOnnxModel::GenerateVoiceDesign(const std::string& text,
                                                        const std::string& instruct,
                                                        const std::string& language,
                                                        const GenerationOptions& options) {
  auto prompt = BuildVoiceDesignPrompt(text, instruct, language, true);
  return GenerateAudioFromPrompt(prompt, options);
}

std::vector<CloneResult> DesignQwen3TTSOnnxModel::StreamVoiceDesign(const std::vector<std::string>& text_deltas,
                                                                    const std::string& instruct,
                                                                    const std::string& language,
                                                                    const SegmentStreamOptions& options) {
  return RunSegmentStream(
      *this, text_deltas, options,
      [&](const std::string& segment, const std::string& anchor_text, const Int64Tensor& anchor_code) {
        return BuildVoiceDesignPrompt(segment, instruct, language, true, anchor_text, anchor_code);
      });
}

}  // namespace qwen3tts
