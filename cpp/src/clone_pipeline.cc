#include "qwen3tts/clone_pipeline.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "qwen3tts/npy.h"

namespace qwen3tts {
namespace {

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

std::vector<int64_t> Range(int64_t begin, int64_t end) {
  std::vector<int64_t> out(static_cast<size_t>(std::max<int64_t>(0, end - begin)));
  for (size_t i = 0; i < out.size(); ++i) out[i] = begin + static_cast<int64_t>(i);
  return out;
}

std::string Trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

bool ParseBool(const std::string& value) {
  return value == "1" || value == "true" || value == "True" || value == "yes";
}

void ValidateRank3(const FloatTensor& tensor, const std::string& name) {
  if (tensor.shape().size() != 3 || tensor.shape()[0] != 1) {
    throw std::invalid_argument(name + " must have shape [1,T,H]");
  }
}

}  // namespace

ClonePipeline::ClonePipeline(CloneRuntimeConfig config)
    : config_(std::move(config)),
      env_(ORT_LOGGING_LEVEL_WARNING, "qwen3tts_clone_cpp"),
      talker_core_(env_, OrtRunnerOptions{config_.onnx_dir / "talker" / "talker_core.onnx",
                                          config_.use_cuda, config_.cuda_device_id}),
      sub_talker_sample_(env_, OrtRunnerOptions{config_.onnx_dir / "decode" / "sub_talker_sample.onnx",
                                                config_.use_cuda, config_.cuda_device_id}),
      tokenizer_decode_(env_, OrtRunnerOptions{config_.onnx_dir / "tokenizer" / "tokenizer12hz_decode_chunk.onnx",
                                               config_.use_cuda, config_.cuda_device_id}) {}

std::vector<std::pair<std::string, double>> ClonePipeline::SessionLoadTimings() const {
  return {
      {"talker_core", talker_core_.LoadSeconds()},
      {"sub_talker_sample", sub_talker_sample_.LoadSeconds()},
      {"tokenizer_decode", tokenizer_decode_.LoadSeconds()},
  };
}

CloneResult ClonePipeline::Run(const CloneInputs& inputs) {
  ValidateRank3(inputs.inputs_embeds, "inputs_embeds");
  ValidateRank3(inputs.trailing_text_hidden, "trailing_text_hidden");
  ValidateRank3(inputs.tts_pad_embed, "tts_pad_embed");

  SamplingOptions sampling;
  sampling.do_sample = config_.do_sample;
  sampling.top_k = config_.top_k;
  sampling.top_p = config_.top_p;
  sampling.temperature = config_.temperature;
  sampling.repetition_penalty = config_.repetition_penalty;
  sampling.eos_token_id = config_.eos_token_id;
  sampling.vocab_size = config_.vocab_size;
  sampling.first_codebook_mask_tail = config_.first_codebook_mask_tail;
  sampling.min_new_tokens = config_.min_new_tokens;
  sampling.seed = config_.seed;
  MainTokenSampler sampler(sampling);

  auto prefill_start = Clock::now();
  auto state = RunPrefill(inputs);
  auto prefill_end = Clock::now();
  const int64_t max_codec_frames = std::max<int64_t>(config_.max_new_tokens - 1, 1);
  std::vector<int64_t> generated_first_tokens;
  std::vector<int64_t> frames;
  frames.reserve(static_cast<size_t>(max_codec_frames * config_.num_code_groups));

  CloneResult result;
  double main_token_sample_seconds = 0.0;
  double frame_decode_seconds = 0.0;
  double sub_talker_sample_seconds = 0.0;
  double talker_core_decode_seconds = 0.0;
  auto decode_loop_start = Clock::now();
  for (int64_t step = 0; step < max_codec_frames; ++step) {
    auto sample_start = Clock::now();
    int64_t first_token = sampler.Sample(state.logits, generated_first_tokens);
    main_token_sample_seconds += SecondsSince(sample_start, Clock::now());
    if (first_token == config_.eos_token_id) {
      result.stopped = true;
      result.stop_reason = "eos";
      break;
    }

    FloatTensor text_embed = TextEmbedForStep(inputs, step);
    auto frame_start = Clock::now();
    auto decoded = RunDecodeStep(std::move(state), first_token, text_embed);
    frame_decode_seconds += SecondsSince(frame_start, Clock::now());
    sub_talker_sample_seconds += decoded.sub_talker_sample_seconds;
    talker_core_decode_seconds += decoded.talker_core_decode_seconds;
    state = std::move(decoded.state);
    generated_first_tokens.push_back(first_token);
    const auto& row = decoded.codebook_tokens.values();
    if (static_cast<int64_t>(row.size()) != config_.num_code_groups) {
      throw std::runtime_error("sub_talker_sample returned unexpected codebook token count");
    }
    frames.insert(frames.end(), row.begin(), row.end());
  }
  auto decode_loop_end = Clock::now();

  result.generated_codes = MakeGeneratedCodes(frames);
  Int64Tensor full_codes = ConcatCodesBatch1(inputs.ref_code, result.generated_codes);
  auto audio_start = Clock::now();
  result.audio = DecodeAudio(full_codes, inputs.ref_code.shape()[1]);
  auto audio_end = Clock::now();
  result.sample_rate = config_.audio_sample_rate;
  result.timings = {
      {"prefill", SecondsSince(prefill_start, prefill_end)},
      {"decode_loop", SecondsSince(decode_loop_start, decode_loop_end)},
      {"main_token_sample", main_token_sample_seconds},
      {"frame_decode", frame_decode_seconds},
      {"sub_talker_sample", sub_talker_sample_seconds},
      {"talker_core_decode", talker_core_decode_seconds},
      {"audio_decode", SecondsSince(audio_start, audio_end)},
  };
  return result;
}

CloneResult ClonePipeline::Run(const CloneInputs& inputs, const CloneRuntimeConfig& runtime_config) {
  CloneRuntimeConfig previous = config_;
  config_ = runtime_config;
  try {
    CloneResult result = Run(inputs);
    config_ = previous;
    return result;
  } catch (...) {
    config_ = previous;
    throw;
  }
}

ClonePipeline::TalkerState ClonePipeline::RunPrefill(const CloneInputs& inputs) {
  const int64_t seq_len = inputs.inputs_embeds.shape()[1];
  std::unordered_map<std::string, Ort::Value> feed;
  feed.emplace("inputs_embeds", talker_core_.MakeFloatInput(inputs.inputs_embeds, "inputs_embeds"));
  feed.emplace("attention_mask", talker_core_.MakeInt64Input(inputs.attention_mask));
  auto cache_position = Range(0, seq_len);
  feed.emplace("cache_position", talker_core_.MakeInt64Input({seq_len}, cache_position));

  for (int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    FloatTensor empty_kv({1, config_.num_key_value_heads, 0, config_.head_dim});
    feed.emplace("past_key_" + std::to_string(layer), talker_core_.MakeFloatInput(empty_kv, "past_key_" + std::to_string(layer)));
    feed.emplace("past_value_" + std::to_string(layer), talker_core_.MakeFloatInput(empty_kv, "past_value_" + std::to_string(layer)));
  }

  auto outputs = talker_core_.RunIo(feed, TalkerOutputNames(), TalkerDeviceOutputs());
  TalkerState state;
  state.logits = talker_core_.CopyLastLogits(outputs[0]);
  state.last_hidden = std::move(outputs[1]);
  state.past_keys.reserve(static_cast<size_t>(config_.num_hidden_layers));
  state.past_values.reserve(static_cast<size_t>(config_.num_hidden_layers));
  for (int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    state.past_keys.push_back(std::move(outputs[2 + static_cast<size_t>(layer) * 2]));
    state.past_values.push_back(std::move(outputs[3 + static_cast<size_t>(layer) * 2]));
  }
  state.past_len = seq_len;
  return state;
}

ClonePipeline::DecodeStepOutput ClonePipeline::RunDecodeStep(TalkerState state, int64_t first_token,
                                                             const FloatTensor& text_embed) {
  std::unordered_map<std::string, Ort::Value> sample_feed;
  std::vector<int64_t> first_token_values{first_token};
  sample_feed.emplace("first_token", sub_talker_sample_.MakeInt64Input({1}, first_token_values));
  sample_feed.emplace("last_hidden", std::move(state.last_hidden));
  sample_feed.emplace("text_embed", sub_talker_sample_.MakeFloatInput(text_embed, "text_embed"));
  auto sample_start = Clock::now();
  auto sample_outputs = sub_talker_sample_.RunIo(sample_feed, {"codebook_tokens", "decode_embed"}, {"decode_embed"});
  auto sample_end = Clock::now();
  Int64Tensor codebook_tokens = sub_talker_sample_.CopyInt64Tensor(sample_outputs[0]);

  std::unordered_map<std::string, Ort::Value> core_feed;
  core_feed.emplace("inputs_embeds", std::move(sample_outputs[1]));
  std::vector<int64_t> attention_values(static_cast<size_t>(state.past_len + 1), 1);
  core_feed.emplace("attention_mask", talker_core_.MakeInt64Input({1, state.past_len + 1}, attention_values));
  std::vector<int64_t> cache_pos{state.past_len};
  core_feed.emplace("cache_position", talker_core_.MakeInt64Input({1}, cache_pos));
  for (int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    core_feed.emplace("past_key_" + std::to_string(layer), std::move(state.past_keys[static_cast<size_t>(layer)]));
    core_feed.emplace("past_value_" + std::to_string(layer), std::move(state.past_values[static_cast<size_t>(layer)]));
  }

  auto core_start = Clock::now();
  auto core_outputs = talker_core_.RunIo(core_feed, TalkerOutputNames(), TalkerDeviceOutputs());
  auto core_end = Clock::now();
  TalkerState next;
  next.logits = talker_core_.CopyLastLogits(core_outputs[0]);
  next.last_hidden = std::move(core_outputs[1]);
  next.past_keys.reserve(static_cast<size_t>(config_.num_hidden_layers));
  next.past_values.reserve(static_cast<size_t>(config_.num_hidden_layers));
  for (int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    next.past_keys.push_back(std::move(core_outputs[2 + static_cast<size_t>(layer) * 2]));
    next.past_values.push_back(std::move(core_outputs[3 + static_cast<size_t>(layer) * 2]));
  }
  next.past_len = state.past_len + 1;
  return DecodeStepOutput{std::move(next), std::move(codebook_tokens),
                          SecondsSince(sample_start, sample_end), SecondsSince(core_start, core_end)};
}

FloatTensor ClonePipeline::TextEmbedForStep(const CloneInputs& inputs, int64_t step) const {
  if (step < inputs.trailing_text_hidden.shape()[1]) return SliceAxis1(inputs.trailing_text_hidden, step, step + 1);
  return inputs.tts_pad_embed;
}

FloatTensor ClonePipeline::DecodeAudio(const Int64Tensor& full_codes, int64_t context_frames) const {
  std::vector<float> chunks;
  const int64_t total_frames = full_codes.shape()[1];
  int64_t start_frame = context_frames;
  while (start_frame < total_frames) {
    int64_t end_frame = std::min(start_frame + config_.tokenizer_decode_chunk_frames, total_frames);
    int64_t left_context = std::min(config_.tokenizer_decode_context_frames, start_frame);
    int64_t input_start = start_frame - left_context;
    Int64Tensor code_chunk = SliceCodes(full_codes, input_start, end_frame);

    std::unordered_map<std::string, Ort::Value> feed;
    feed.emplace("audio_codes", tokenizer_decode_.MakeInt64Input(code_chunk));
    std::vector<int64_t> context_value{left_context};
    feed.emplace("context_frames", tokenizer_decode_.MakeInt64Input({}, context_value));
    auto outputs = tokenizer_decode_.RunIo(feed, {"audio_values", "lengths"}, {});
    FloatTensor audio_values = tokenizer_decode_.CopyFloatTensor(outputs[0]);
    Int64Tensor lengths = tokenizer_decode_.CopyInt64Tensor(outputs[1]);

    int64_t expected = (end_frame - start_frame) * config_.decode_upsample_rate;
    int64_t reported = lengths.empty() ? static_cast<int64_t>(audio_values.size()) : lengths.values()[0];
    if (reported > static_cast<int64_t>(audio_values.size())) {
      throw std::runtime_error("tokenizer decode output is shorter than its reported length");
    }
    if (reported < expected || static_cast<int64_t>(audio_values.size()) < expected) {
      throw std::runtime_error("tokenizer decode returned fewer samples than expected");
    }
    const int64_t valid_end = std::min<int64_t>(reported, static_cast<int64_t>(audio_values.size()));
    chunks.insert(chunks.end(),
                  audio_values.values().begin() + static_cast<std::ptrdiff_t>(valid_end - expected),
                  audio_values.values().begin() + static_cast<std::ptrdiff_t>(valid_end));
    start_frame = end_frame;
  }
  const int64_t sample_count = static_cast<int64_t>(chunks.size());
  return FloatTensor({sample_count}, std::move(chunks));
}

std::vector<std::string> ClonePipeline::TalkerOutputNames() const {
  std::vector<std::string> names{"logits", "last_hidden"};
  for (int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    names.push_back("new_past_key_" + std::to_string(layer));
    names.push_back("new_past_value_" + std::to_string(layer));
  }
  return names;
}

std::unordered_set<std::string> ClonePipeline::TalkerDeviceOutputs() const {
  std::unordered_set<std::string> names{"last_hidden"};
  for (int64_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    names.insert("new_past_key_" + std::to_string(layer));
    names.insert("new_past_value_" + std::to_string(layer));
  }
  return names;
}

Int64Tensor ClonePipeline::MakeGeneratedCodes(const std::vector<int64_t>& frames) const {
  if (frames.empty()) return Int64Tensor({1, 0, config_.num_code_groups});
  if (frames.size() % static_cast<size_t>(config_.num_code_groups) != 0) {
    throw std::invalid_argument("generated frame/codebook count mismatch");
  }
  return Int64Tensor({1, static_cast<int64_t>(frames.size() / static_cast<size_t>(config_.num_code_groups)),
                      config_.num_code_groups},
                     frames);
}

CloneRuntimeConfig LoadConfigFile(const std::filesystem::path& path, CloneRuntimeConfig defaults) {
  std::ifstream in(path);
  if (!in) return defaults;
  std::string line;
  while (std::getline(in, line)) {
    line = Trim(line);
    if (line.empty() || line[0] == '#') continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = Trim(line.substr(0, eq));
    std::string value = Trim(line.substr(eq + 1));
    if (key == "max_new_tokens") defaults.max_new_tokens = std::stoll(value);
    else if (key == "min_new_tokens") defaults.min_new_tokens = std::stoll(value);
    else if (key == "eos_token_id") defaults.eos_token_id = std::stoll(value);
    else if (key == "vocab_size") defaults.vocab_size = std::stoll(value);
    else if (key == "first_codebook_mask_tail") defaults.first_codebook_mask_tail = std::stoll(value);
    else if (key == "num_hidden_layers") defaults.num_hidden_layers = std::stoll(value);
    else if (key == "num_key_value_heads") defaults.num_key_value_heads = std::stoll(value);
    else if (key == "head_dim") defaults.head_dim = std::stoll(value);
    else if (key == "num_code_groups") defaults.num_code_groups = std::stoll(value);
    else if (key == "decode_upsample_rate") defaults.decode_upsample_rate = std::stoll(value);
    else if (key == "audio_sample_rate") defaults.audio_sample_rate = std::stoll(value);
    else if (key == "tokenizer_decode_chunk_frames") defaults.tokenizer_decode_chunk_frames = std::stoll(value);
    else if (key == "tokenizer_decode_context_frames") defaults.tokenizer_decode_context_frames = std::stoll(value);
    else if (key == "do_sample") defaults.do_sample = ParseBool(value);
    else if (key == "top_k") defaults.top_k = std::stoi(value);
    else if (key == "top_p") defaults.top_p = std::stof(value);
    else if (key == "temperature") defaults.temperature = std::stof(value);
    else if (key == "repetition_penalty") defaults.repetition_penalty = std::stof(value);
    else if (key == "seed") defaults.seed = static_cast<uint64_t>(std::stoull(value));
  }
  return defaults;
}

CloneInputs LoadCloneInputs(const std::filesystem::path& fixture_dir) {
  CloneInputs inputs;
  inputs.inputs_embeds = ReadFloatNpy(fixture_dir / "inputs_embeds.npy");
  inputs.attention_mask = ReadInt64Npy(fixture_dir / "attention_mask.npy");
  inputs.trailing_text_hidden = ReadFloatNpy(fixture_dir / "trailing_text_hidden.npy");
  inputs.tts_pad_embed = ReadFloatNpy(fixture_dir / "tts_pad_embed.npy");
  inputs.ref_code = ReadInt64Npy(fixture_dir / "ref_code.npy");
  if (inputs.ref_code.shape().size() == 2) {
    inputs.ref_code.shape().insert(inputs.ref_code.shape().begin(), 1);
  }
  return inputs;
}

}  // namespace qwen3tts
