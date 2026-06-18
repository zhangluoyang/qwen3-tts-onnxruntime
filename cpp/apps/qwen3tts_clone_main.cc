#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "qwen3tts/product_clone_runtime.h"
#include "qwen3tts/wav_writer.h"

namespace {

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " [--model-dir PATH] [--onnx-dir PATH] [--ref-audio PATH] [--ref-text TEXT]\n"
            << "       [--text TEXT] [--output PATH] [--max-new-tokens N] [--sample] [--cpu]\n";
}

bool ParseBoolFlag(const std::string& value) {
  return value == "1" || value == "true" || value == "True" || value == "yes";
}

void PrintAudioStats(const qwen3tts::FloatTensor& audio) {
  float peak = 0.0f;
  double sum_abs = 0.0;
  for (float v : audio.values()) {
    peak = std::max(peak, std::abs(v));
    sum_abs += std::abs(v);
  }
  const double mean_abs = audio.empty() ? 0.0 : sum_abs / static_cast<double>(audio.size());
  std::cout << "audio_samples=" << audio.size() << "\n";
  std::cout << "audio_peak_abs=" << peak << "\n";
  std::cout << "audio_mean_abs=" << mean_abs << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    qwen3tts::ProductCloneConfig config;
    qwen3tts::ProductCloneRequest request;
    std::filesystem::path output_path = "outputs/cpp_product_clone.wav";

    request.reference_audio = "data/ref_from_mp3_24k_mono.wav";
    request.reference_text = "告诉自己，不要怕";
    request.text = "小医仙低声说道：我本来以为，这一路会很难走，可现在，好像也没有那么怕了。";

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--model-dir" && i + 1 < argc) {
        config.model_dir = argv[++i];
      } else if (arg == "--onnx-dir" && i + 1 < argc) {
        config.onnx_dir = argv[++i];
      } else if (arg == "--ref-audio" && i + 1 < argc) {
        request.reference_audio = argv[++i];
      } else if (arg == "--ref-text" && i + 1 < argc) {
        request.reference_text = argv[++i];
      } else if (arg == "--text" && i + 1 < argc) {
        request.text = argv[++i];
      } else if (arg == "--language" && i + 1 < argc) {
        request.language = argv[++i];
      } else if (arg == "--output" && i + 1 < argc) {
        output_path = argv[++i];
      } else if (arg == "--max-new-tokens" && i + 1 < argc) {
        config.max_new_tokens = std::stoll(argv[++i]);
      } else if (arg == "--min-new-tokens" && i + 1 < argc) {
        config.min_new_tokens = std::stoll(argv[++i]);
      } else if (arg == "--top-k" && i + 1 < argc) {
        config.top_k = std::stoi(argv[++i]);
      } else if (arg == "--top-p" && i + 1 < argc) {
        config.top_p = std::stof(argv[++i]);
      } else if (arg == "--temperature" && i + 1 < argc) {
        config.temperature = std::stof(argv[++i]);
      } else if (arg == "--repetition-penalty" && i + 1 < argc) {
        config.repetition_penalty = std::stof(argv[++i]);
      } else if (arg == "--seed" && i + 1 < argc) {
        config.seed = static_cast<uint64_t>(std::stoull(argv[++i]));
      } else if (arg == "--sample") {
        config.do_sample = true;
      } else if (arg == "--do-sample" && i + 1 < argc) {
        config.do_sample = ParseBoolFlag(argv[++i]);
      } else if (arg == "--cpu") {
        config.use_cuda = false;
      } else if (arg == "--help" || arg == "-h") {
        PrintUsage(argv[0]);
        return 0;
      } else {
        PrintUsage(argv[0]);
        return 2;
      }
    }

    std::filesystem::create_directories(output_path.parent_path().empty() ? "." : output_path.parent_path());

    auto total_start = Clock::now();
    auto init_start = Clock::now();
    qwen3tts::ProductCloneRuntime runtime(config);
    auto init_end = Clock::now();
    auto generate_start = Clock::now();
    auto result = runtime.Generate(request);
    auto generate_end = Clock::now();
    auto save_start = Clock::now();
    qwen3tts::WriteWavMono16(output_path, result.generation.audio.values(), result.generation.sample_rate);
    auto save_end = Clock::now();

    std::cout << "output_wav=" << output_path.string() << "\n";
    std::cout << "sample_rate=" << result.generation.sample_rate << "\n";
    std::cout << "generated_frames=" << result.generation.generated_codes.shape()[1] << "\n";
    std::cout << "reference_frames=" << result.reference_codes.shape()[1] << "\n";
    std::cout << "stopped=" << (result.generation.stopped ? "true" : "false") << "\n";
    std::cout << "stop_reason=" << result.generation.stop_reason << "\n";
    PrintAudioStats(result.generation.audio);
    std::cout << "timing.init_seconds=" << SecondsSince(init_start, init_end) << "\n";
    std::cout << "timing.generate_seconds=" << SecondsSince(generate_start, generate_end) << "\n";
    std::cout << "timing.save_seconds=" << SecondsSince(save_start, save_end) << "\n";
    std::cout << "timing.total_seconds=" << SecondsSince(total_start, Clock::now()) << "\n";
    for (const auto& item : runtime.SessionLoadTimings()) {
      std::cout << "timing.session_load." << item.first << "_seconds=" << item.second << "\n";
    }
    for (const auto& item : result.timings) {
      std::cout << "timing.product." << item.first << "_seconds=" << item.second << "\n";
    }
    for (const auto& item : result.generation.timings) {
      std::cout << "timing.pipeline." << item.first << "_seconds=" << item.second << "\n";
    }
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
