#include "qwen3tts/product_clone_runtime.h"

#include <algorithm>
#include <cctype>
#include <fstream>
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

std::unordered_map<std::string, int64_t> ExtractIntMap(const std::string& text, const std::string& key) {
  std::unordered_map<std::string, int64_t> out;
  const auto pos = text.find("\"" + key + "\"");
  if (pos == std::string::npos) return out;
  const auto begin = text.find('{', pos);
  if (begin == std::string::npos) return out;
  int depth = 0;
  size_t end = begin;
  for (; end < text.size(); ++end) {
    if (text[end] == '{') ++depth;
    if (text[end] == '}') {
      --depth;
      if (depth == 0) break;
    }
  }
  if (end >= text.size()) return out;
  const std::string body = text.substr(begin + 1, end - begin - 1);
  const std::regex item("\"([^\"]+)\"\\s*:\\s*(-?[0-9]+)");
  for (auto it = std::sregex_iterator(body.begin(), body.end(), item); it != std::sregex_iterator(); ++it) {
    out[(*it)[1].str()] = std::stoll((*it)[2].str());
  }
  return out;
}

std::string ExtractObject(const std::string& text, const std::string& key) {
  const auto pos = text.find("\"" + key + "\"");
  if (pos == std::string::npos) return {};
  const auto begin = text.find('{', pos);
  if (begin == std::string::npos) return {};
  int depth = 0;
  size_t end = begin;
  for (; end < text.size(); ++end) {
    if (text[end] == '{') ++depth;
    if (text[end] == '}') {
      --depth;
      if (depth == 0) break;
    }
  }
  if (end >= text.size()) return {};
  return text.substr(begin, end - begin + 1);
}

std::string KeepTopLevelObjectFields(std::string object_text) {
  int depth = 0;
  bool in_string = false;
  bool escaped = false;
  for (char& c : object_text) {
    char original = c;
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (original == '\\') {
        escaped = true;
      } else if (original == '"') {
        in_string = false;
      }
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

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

Int64Tensor RowTensor(std::vector<int64_t> values) {
  const int64_t len = static_cast<int64_t>(values.size());
  return Int64Tensor({1, len}, std::move(values));
}

FloatTensor Reshape(const FloatTensor& tensor, std::vector<int64_t> shape) {
  if (FloatTensor::NumElements(shape) != tensor.size()) {
    throw std::invalid_argument("float reshape count mismatch: target=" + FloatTensor::ShapeToString(shape) +
                                " target_count=" + std::to_string(FloatTensor::NumElements(shape)) +
                                " value_count=" + std::to_string(tensor.size()) +
                                " source_shape=" + FloatTensor::ShapeToString(tensor.shape()));
  }
  return FloatTensor(std::move(shape), tensor.values());
}

Int64Tensor Reshape(const Int64Tensor& tensor, std::vector<int64_t> shape) {
  if (Int64Tensor::NumElements(shape) != tensor.size()) throw std::invalid_argument("int64 reshape count mismatch");
  return Int64Tensor(std::move(shape), tensor.values());
}

}  // namespace

ProductCloneRuntime::ProductCloneRuntime(ProductCloneConfig config)
    : config_(std::move(config)),
      ids_(LoadModelIds(config_.model_dir)),
      env_(ORT_LOGGING_LEVEL_WARNING, "qwen3tts_product_clone_cpp"),
      tokenizer_(config_.model_dir),
      text_project_(env_, OrtRunnerOptions{config_.onnx_dir / "text_project" / "text_project.onnx",
                                           config_.use_cuda, config_.cuda_device_id}),
      codec_embed_(env_, OrtRunnerOptions{config_.onnx_dir / "codec_embed" / "codec_embed.onnx",
                                          config_.use_cuda, config_.cuda_device_id}),
      tokenizer_encode_(env_, OrtRunnerOptions{config_.onnx_dir / "tokenizer" / "tokenizer12hz_encode.onnx",
                                               config_.use_cuda, config_.cuda_device_id}),
      speaker_encoder_(env_, OrtRunnerOptions{config_.onnx_dir / "speaker_encoder" / "speaker_encoder.onnx",
                                              config_.use_cuda, config_.cuda_device_id}),
      pipeline_(MakeCloneRuntimeConfig()) {}

ProductCloneRuntime::ModelIds ProductCloneRuntime::LoadModelIds(const std::filesystem::path& model_dir) {
  ModelIds ids;
  const auto config = ReadTextFile(model_dir / "config.json");
  const auto talker_config = ExtractObject(config, "talker_config");
  const auto talker_top_level = KeepTopLevelObjectFields(talker_config.empty() ? config : talker_config);
  (void)ExtractInt(config, "tts_bos_token_id", &ids.tts_bos_token_id);
  (void)ExtractInt(config, "tts_eos_token_id", &ids.tts_eos_token_id);
  (void)ExtractInt(config, "tts_pad_token_id", &ids.tts_pad_token_id);
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

  const auto audio_config_path = model_dir / "speech_tokenizer" / "config.json";
  if (std::filesystem::exists(audio_config_path)) {
    const auto audio_config = ReadTextFile(audio_config_path);
    (void)ExtractInt(audio_config, "output_sample_rate", &ids.audio_sample_rate);
    (void)ExtractInt(audio_config, "decode_upsample_rate", &ids.decode_upsample_rate);
  }
  return ids;
}

std::string ProductCloneRuntime::BuildAssistantText(const std::string& text) {
  return "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
}

std::string ProductCloneRuntime::BuildReferenceText(const std::string& text) {
  return "<|im_start|>assistant\n" + text + "<|im_end|>\n";
}

double ProductCloneRuntime::SecondsSince(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

CloneRuntimeConfig ProductCloneRuntime::MakeCloneRuntimeConfig() const {
  CloneRuntimeConfig runtime;
  runtime.onnx_dir = config_.onnx_dir;
  runtime.use_cuda = config_.use_cuda;
  runtime.cuda_device_id = config_.cuda_device_id;
  runtime.max_new_tokens = config_.max_new_tokens;
  runtime.min_new_tokens = config_.min_new_tokens;
  runtime.eos_token_id = ids_.codec_eos_token_id;
  runtime.vocab_size = ids_.vocab_size;
  runtime.first_codebook_mask_tail = ids_.first_codebook_mask_tail;
  runtime.num_hidden_layers = ids_.num_hidden_layers;
  runtime.num_key_value_heads = ids_.num_key_value_heads;
  runtime.head_dim = ids_.head_dim;
  runtime.num_code_groups = ids_.num_code_groups;
  runtime.decode_upsample_rate = ids_.decode_upsample_rate;
  runtime.audio_sample_rate = ids_.audio_sample_rate;
  runtime.do_sample = config_.do_sample;
  runtime.top_k = config_.top_k;
  runtime.top_p = config_.top_p;
  runtime.temperature = config_.temperature;
  runtime.repetition_penalty = config_.repetition_penalty;
  runtime.seed = config_.seed;
  return runtime;
}

int64_t ProductCloneRuntime::LanguageId(const std::string& language) const {
  const auto key = Lower(language);
  if (key == "auto") return -1;
  auto it = ids_.codec_language_id.find(key);
  if (it == ids_.codec_language_id.end()) throw std::runtime_error("unsupported language: " + language);
  return it->second;
}

std::vector<int64_t> ProductCloneRuntime::CodecPrefillIds(const std::string& language) const {
  const int64_t language_id = LanguageId(language);
  if (language_id < 0) return {ids_.codec_nothink_id, ids_.codec_think_bos_id, ids_.codec_think_eos_id};
  return {ids_.codec_think_id, ids_.codec_think_bos_id, language_id, ids_.codec_think_eos_id};
}

Int64Tensor ProductCloneRuntime::TokenizeToTensor(const std::string& text) const {
  return RowTensor(tokenizer_.Encode(text));
}

FloatTensor ProductCloneRuntime::TextProject(const Int64Tensor& input_ids) const {
  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("input_ids", text_project_.MakeInt64Input(input_ids));
  auto outputs = text_project_.RunIo(feed, {"text_embed"}, {});
  auto output = text_project_.CopyFloatTensor(outputs[0]);
  return Reshape(output, {1, input_ids.shape()[1], ids_.hidden_size});
}

FloatTensor ProductCloneRuntime::CodecEmbed(const Int64Tensor& token_ids) const {
  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("token_ids", codec_embed_.MakeInt64Input(token_ids));
  Int64Tensor dummy_ref({1, 1, ids_.num_code_groups});
  feed.emplace("ref_code", codec_embed_.MakeInt64Input(dummy_ref));
  auto outputs = codec_embed_.RunIo(feed, {"embed"}, {});
  auto output = codec_embed_.CopyFloatTensor(outputs[0]);
  return Reshape(output, {1, token_ids.shape()[1], ids_.hidden_size});
}

FloatTensor ProductCloneRuntime::RefCodeEmbed(const Int64Tensor& ref_code) const {
  std::unordered_map<std::string, Ort::Value> feed;
  auto dummy_tokens = RowTensor({ids_.codec_bos_id});
  feed.emplace("token_ids", codec_embed_.MakeInt64Input(dummy_tokens));
  feed.emplace("ref_code", codec_embed_.MakeInt64Input(ref_code));
  auto outputs = codec_embed_.RunIo(feed, {"ref_code_embed"}, {});
  auto output = codec_embed_.CopyFloatTensor(outputs[0]);
  return Reshape(output, {1, ref_code.shape()[1], ids_.hidden_size});
}

Int64Tensor ProductCloneRuntime::EncodeReferenceAudio(const std::vector<float>& audio) const {
  FloatTensor input({1, static_cast<int64_t>(audio.size())}, audio);
  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("audio", tokenizer_encode_.MakeFloatInput(input, "audio"));
  auto outputs = tokenizer_encode_.RunIo(feed, {"codes"}, {});
  auto codes = tokenizer_encode_.CopyInt64Tensor(outputs[0]);
  if (codes.shape().size() == 3 && codes.shape()[0] == 1) return codes;
  if (codes.shape().size() == 2) return Reshape(codes, {1, codes.shape()[0], codes.shape()[1]});
  throw std::runtime_error("tokenizer encoder returned unexpected codes shape");
}

FloatTensor ProductCloneRuntime::EncodeSpeakerEmbedding(const std::vector<float>& audio) const {
  auto mel = MelSpectrogram(audio, static_cast<int>(ids_.audio_sample_rate));
  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("mel", speaker_encoder_.MakeFloatInput(mel, "mel"));
  auto outputs = speaker_encoder_.RunIo(feed, {"speaker_embedding"}, {});
  auto speaker = speaker_encoder_.CopyFloatTensor(outputs[0]);
  if (speaker.shape().size() == 2 && speaker.shape()[0] == 1) {
    return Reshape(speaker, {1, 1, speaker.shape()[1]});
  }
  if (speaker.shape().size() == 3) return speaker;
  return Reshape(speaker, {1, 1, static_cast<int64_t>(speaker.size())});
}

CloneInputs ProductCloneRuntime::BuildPrompt(const ProductCloneRequest& request,
                                             const Int64Tensor& ref_code,
                                             const FloatTensor& speaker_embedding) const {
  const auto input_id = TokenizeToTensor(BuildAssistantText(request.text));
  const auto ref_id = TokenizeToTensor(BuildReferenceText(request.reference_text));

  const auto special_ids = RowTensor({ids_.tts_bos_token_id, ids_.tts_eos_token_id, ids_.tts_pad_token_id});
  const auto special_embeds = TextProject(special_ids);
  const auto tts_bos_embed = SliceAxis1(special_embeds, 0, 1);
  const auto tts_eos_embed = SliceAxis1(special_embeds, 1, 2);
  const auto tts_pad_embed = SliceAxis1(special_embeds, 2, 3);

  const auto codec_prefill = CodecEmbed(RowTensor(CodecPrefillIds(request.language)));
  const auto codec_tail = CodecEmbed(RowTensor({ids_.codec_pad_id, ids_.codec_bos_id}));
  const auto codec_input = ConcatAxis1({codec_prefill, speaker_embedding, codec_tail});

  const auto role_embed = TextProject(SliceAxis1(input_id, 0, 3));
  const auto pad_prefix = RepeatAxis1(tts_pad_embed, codec_input.shape()[1] - 2);
  const auto text_side = ConcatAxis1({pad_prefix, tts_bos_embed});
  const auto codec_side = SliceAxis1(codec_input, 0, codec_input.shape()[1] - 1);
  auto talker_input = ConcatAxis1({role_embed, Add(text_side, codec_side)});

  const auto ref_text_ids = SliceAxis1(ref_id, 3, ref_id.shape()[1] - 2);
  const auto target_text_ids = SliceAxis1(input_id, 3, input_id.shape()[1] - 5);
  const auto ref_text_embed =
      ref_text_ids.shape()[1] > 0 ? TextProject(ref_text_ids) : FloatTensor({1, 0, ids_.hidden_size});
  const auto target_text_embed =
      target_text_ids.shape()[1] > 0 ? TextProject(target_text_ids) : FloatTensor({1, 0, ids_.hidden_size});
  auto text_embed = ConcatAxis1({ref_text_embed, target_text_embed, tts_eos_embed});

  const auto codec_bos_embed = CodecEmbed(RowTensor({ids_.codec_bos_id}));
  const auto ref_codec_embed = RefCodeEmbed(ref_code);
  const auto codec_embed = ConcatAxis1({codec_bos_embed, ref_codec_embed});
  const auto codec_pad = CodecEmbed(Int64Tensor({1, text_embed.shape()[1]},
                                                std::vector<int64_t>(static_cast<size_t>(text_embed.shape()[1]),
                                                                     ids_.codec_pad_id)));
  const auto icl_text_side = Add(text_embed, codec_pad);
  const auto icl_codec_side = Add(codec_embed, RepeatAxis1(tts_pad_embed, codec_embed.shape()[1]));
  const auto icl_input = ConcatAxis1({icl_text_side, icl_codec_side});
  talker_input = ConcatAxis1({talker_input, icl_input});

  CloneInputs inputs;
  inputs.inputs_embeds = std::move(talker_input);
  inputs.attention_mask =
      Int64Tensor({1, inputs.inputs_embeds.shape()[1]},
                  std::vector<int64_t>(static_cast<size_t>(inputs.inputs_embeds.shape()[1]), 1));
  inputs.trailing_text_hidden = tts_pad_embed;
  inputs.tts_pad_embed = tts_pad_embed;
  inputs.ref_code = ref_code;
  return inputs;
}

ProductCloneResult ProductCloneRuntime::Generate(const ProductCloneRequest& request) {
  ProductCloneResult result;

  auto audio_start = Clock::now();
  auto audio = LoadAudioMono(request.reference_audio, static_cast<int>(ids_.audio_sample_rate));
  result.timings.push_back({"load_reference_audio", SecondsSince(audio_start, Clock::now())});

  auto ref_code_start = Clock::now();
  result.reference_codes = EncodeReferenceAudio(audio.samples);
  result.timings.push_back({"encode_reference_audio", SecondsSince(ref_code_start, Clock::now())});

  auto speaker_start = Clock::now();
  result.speaker_embedding = EncodeSpeakerEmbedding(audio.samples);
  result.timings.push_back({"encode_speaker_embedding", SecondsSince(speaker_start, Clock::now())});

  auto prompt_start = Clock::now();
  auto inputs = BuildPrompt(request, result.reference_codes, result.speaker_embedding);
  result.timings.push_back({"build_prompt", SecondsSince(prompt_start, Clock::now())});

  auto generate_start = Clock::now();
  result.generation = pipeline_.Run(inputs);
  result.timings.push_back({"generation", SecondsSince(generate_start, Clock::now())});
  return result;
}

std::vector<std::pair<std::string, double>> ProductCloneRuntime::SessionLoadTimings() const {
  auto timings = std::vector<std::pair<std::string, double>>{
      {"text_project", text_project_.LoadSeconds()},
      {"codec_embed", codec_embed_.LoadSeconds()},
      {"tokenizer_encode", tokenizer_encode_.LoadSeconds()},
      {"speaker_encoder", speaker_encoder_.LoadSeconds()},
  };
  for (const auto& item : pipeline_.SessionLoadTimings()) timings.push_back(item);
  return timings;
}

}  // namespace qwen3tts
