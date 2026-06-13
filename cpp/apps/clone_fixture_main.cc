#include <filesystem>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

#include "qwen3tts/clone_pipeline.h"
#include "qwen3tts/npy.h"
#include "qwen3tts/wav_writer.h"

namespace {

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

void PrintUsage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --onnx-dir onnx_fp16 --fixture-dir outputs/cpp_fixture --out-dir outputs/cpp_fixture [--cpu]\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path onnx_dir = "onnx_fp16";
    std::filesystem::path fixture_dir = "outputs/cpp_fixture";
    std::filesystem::path out_dir = fixture_dir;
    bool use_cuda = true;

    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--onnx-dir" && i + 1 < argc) {
        onnx_dir = argv[++i];
      } else if (arg == "--fixture-dir" && i + 1 < argc) {
        fixture_dir = argv[++i];
      } else if (arg == "--out-dir" && i + 1 < argc) {
        out_dir = argv[++i];
      } else if (arg == "--cpu") {
        use_cuda = false;
      } else if (arg == "--help" || arg == "-h") {
        PrintUsage(argv[0]);
        return 0;
      } else {
        PrintUsage(argv[0]);
        return 2;
      }
    }

    qwen3tts::CloneRuntimeConfig config;
    config.onnx_dir = onnx_dir;
    config.use_cuda = use_cuda;
    config = qwen3tts::LoadConfigFile(fixture_dir / "meta.txt", config);
    config.onnx_dir = onnx_dir;
    config.use_cuda = use_cuda;

    std::filesystem::create_directories(out_dir);
    auto total_start = Clock::now();
    auto load_start = Clock::now();
    auto inputs = qwen3tts::LoadCloneInputs(fixture_dir);
    auto load_end = Clock::now();
    auto init_start = Clock::now();
    qwen3tts::ClonePipeline pipeline(config);
    auto init_end = Clock::now();
    auto inference_start = Clock::now();
    auto result = pipeline.Run(inputs);
    auto inference_end = Clock::now();

    auto save_start = Clock::now();
    qwen3tts::WriteInt64Npy(out_dir / "cpp_generated_codes.npy", result.generated_codes);
    qwen3tts::WriteFloatNpy(out_dir / "cpp_audio.npy", result.audio);
    qwen3tts::WriteWavMono16(out_dir / "cpp_clone.wav", result.audio.values(), result.sample_rate);
    auto save_end = Clock::now();

    std::cout << "generated_codes_shape=[";
    for (size_t i = 0; i < result.generated_codes.shape().size(); ++i) {
      if (i) std::cout << ",";
      std::cout << result.generated_codes.shape()[i];
    }
    std::cout << "]\n";
    std::cout << "audio_samples=" << result.audio.size() << "\n";
    std::cout << "sample_rate=" << result.sample_rate << "\n";
    std::cout << "stopped=" << (result.stopped ? "true" : "false") << "\n";
    std::cout << "stop_reason=" << result.stop_reason << "\n";
    std::cout << "saved_wav=" << (out_dir / "cpp_clone.wav").string() << "\n";
    std::cout << "timing.load_inputs_seconds=" << SecondsSince(load_start, load_end) << "\n";
    std::cout << "timing.init_sessions_seconds=" << SecondsSince(init_start, init_end) << "\n";
    std::cout << "timing.inference_seconds=" << SecondsSince(inference_start, inference_end) << "\n";
    std::cout << "timing.save_outputs_seconds=" << SecondsSince(save_start, save_end) << "\n";
    std::cout << "timing.total_seconds=" << SecondsSince(total_start, Clock::now()) << "\n";
    for (const auto& item : pipeline.SessionLoadTimings()) {
      std::cout << "timing.session_load." << item.first << "_seconds=" << item.second << "\n";
    }
    for (const auto& item : result.timings) {
      std::cout << "timing.pipeline." << item.first << "_seconds=" << item.second << "\n";
    }
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
