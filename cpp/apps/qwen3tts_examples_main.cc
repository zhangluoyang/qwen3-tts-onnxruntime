#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "qwen3tts/qwen_model.h"
#include "qwen3tts/wav_writer.h"

namespace {

const char* kBaseModelDir = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base";
const char* kCustomModelDir = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice";

std::vector<std::filesystem::path> DesignModelCandidates() {
  return {
      "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
      "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1___7B-VoiceDesign",
  };
}

std::filesystem::path ResolveDesignModelDir() {
  for (const auto& path : DesignModelCandidates()) {
    if (std::filesystem::exists(path)) return path;
  }
  throw std::runtime_error("VoiceDesign model dir not found in default local model directories");
}

std::string NonstreamText() {
  return "小医仙身着淡紫色长裙，纤细腰肢束着一条银丝软带，一头乌黑长发如瀑般垂至腰际，几缕碎发轻掩着那张清丽中略带苍白的俏脸；她眉如远山含黛，眸若秋水映月，唇边总挂着一丝若有若无的浅笑，仿佛能化解世间所有伤痛——然而那笑意深处却藏着一抹令人心疼的孤寂，那是厄难毒体与生俱来的诅咒，是她以毕生之力抗争的宿命；玉手纤纤，指尖时常萦绕着淡淡的七彩毒雾，却偏偏能用这些夺人性命的毒物炼就救死扶伤的灵药，一如她矛盾而动人的存在：既是令人闻风丧胆的毒女，又是那个在青山镇小医馆里温柔为穷苦百姓诊治的善良姑娘，待到后来与萧炎并肩而行，那双素来沉静的眸子终于多了几分生机与暖意，宛如被春风拂过的寒潭，泛起粼粼波光。";
}

std::vector<std::string> StreamDeltas() {
  return {
      "小医仙身着淡紫色长裙,",
      "纤细腰肢束着一条银丝软带,",
      "一头乌黑长发如瀑般垂至腰际,",
      "几缕碎发轻掩着那张清丽中略带苍白的俏脸；",
      "她眉如远山含黛，",
      "眸若秋水映月，",
      "唇边总挂着一丝若有若无的浅笑，",
      "仿佛能化解世间所有伤痛——然而那笑意深处却藏着一抹令人心疼的孤寂，",
      "那是厄难毒体与生俱来的诅咒，",
      "是她以毕生之力抗争的宿命；",
      "玉手纤纤，",
      "指尖时常萦绕着淡淡的七彩毒雾，",
      "却偏偏能用这些夺人性命的毒物炼就救死扶伤的灵药，",
      "一如她矛盾而动人的存在：",
      "既是令人闻风丧胆的毒女，",
      "又是那个在青山镇小医馆里温柔为穷苦百姓诊治的善良姑娘，",
      "待到后来与萧炎并肩而行，",
      "那双素来沉静的眸子终于多了几分生机与暖意，",
      "宛如被春风拂过的寒潭，",
      "泛起粼粼波光。",
      "小医仙身着淡紫色长裙,",
      "纤细腰肢束着一条银丝软带,",
      "一头乌黑长发如瀑般垂至腰际,",
      "几缕碎发轻掩着那张清丽中略带苍白的俏脸；",
      "她眉如远山含黛，",
      "眸若秋水映月，",
      "唇边总挂着一丝若有若无的浅笑，",
      "仿佛能化解世间所有伤痛——然而那笑意深处却藏着一抹令人心疼的孤寂，",
      "那是厄难毒体与生俱来的诅咒，",
      "是她以毕生之力抗争的宿命；",
      "玉手纤纤，",
      "指尖时常萦绕着淡淡的七彩毒雾，",
      "却偏偏能用这些夺人性命的毒物炼就救死扶伤的灵药，",
      "一如她矛盾而动人的存在：",
      "既是令人闻风丧胆的毒女，",
      "又是那个在青山镇小医馆里温柔为穷苦百姓诊治的善良姑娘，",
      "待到后来与萧炎并肩而行，",
      "那双素来沉静的眸子终于多了几分生机与暖意，",
      "宛如被春风拂过的寒潭，",
      "泛起粼粼波光。",
  };
}

std::string JoinText(const std::vector<std::string>& parts) {
  std::string out;
  for (const auto& item : parts) out += item;
  return out;
}

std::vector<float> ConcatStreamAudio(const std::vector<qwen3tts::CloneResult>& chunks, int64_t* sample_rate) {
  std::vector<float> audio;
  *sample_rate = 24000;
  for (const auto& chunk : chunks) {
    audio.insert(audio.end(), chunk.audio.values().begin(), chunk.audio.values().end());
    *sample_rate = chunk.sample_rate;
  }
  return audio;
}

void PrintAudioStats(const std::string& label, const std::vector<float>& audio) {
  float peak = 0.0f;
  double sum_abs = 0.0;
  for (float v : audio) {
    peak = std::max(peak, std::abs(v));
    sum_abs += std::abs(v);
  }
  const double mean_abs = audio.empty() ? 0.0 : sum_abs / static_cast<double>(audio.size());
  std::cout << label << ".audio_samples=" << audio.size() << "\n";
  std::cout << label << ".audio_peak_abs=" << peak << "\n";
  std::cout << label << ".audio_mean_abs=" << mean_abs << "\n";
}

void SaveResult(const std::filesystem::path& path, const qwen3tts::CloneResult& result) {
  std::filesystem::create_directories(path.parent_path());
  qwen3tts::WriteWavMono16(path, result.audio.values(), result.sample_rate);
  std::cout << "saved=" << path.string() << "\n";
  std::cout << "generated_frames=" << result.generated_codes.shape()[1] << "\n";
  std::cout << "stopped=" << (result.stopped ? "true" : "false") << "\n";
  std::cout << "stop_reason=" << result.stop_reason << "\n";
  PrintAudioStats(path.stem().string(), result.audio.values());
}

void SaveStreamResult(const std::filesystem::path& path, const std::vector<qwen3tts::CloneResult>& chunks) {
  std::filesystem::create_directories(path.parent_path());
  int64_t sample_rate = 24000;
  auto audio = ConcatStreamAudio(chunks, &sample_rate);
  qwen3tts::WriteWavMono16(path, audio, sample_rate);
  std::cout << "saved=" << path.string() << "\n";
  std::cout << "stream_chunks=" << chunks.size() << "\n";
  PrintAudioStats(path.stem().string(), audio);
}

}  // namespace

int main() {
  try {
    // 这个示例程序不需要命令行参数，所有路径和文本都在这里固定好。
    // 默认从仓库根目录运行，直接执行 ./cpp/build/qwen3tts_examples 即可。
    const std::filesystem::path base_model_dir = kBaseModelDir;
    const std::filesystem::path custom_model_dir = kCustomModelDir;
    const std::filesystem::path design_model_dir = ResolveDesignModelDir();
    const std::filesystem::path base_onnx_dir = "onnx_fp16";
    const std::filesystem::path custom_onnx_dir = "onnx_custom_fp16";
    const std::filesystem::path design_onnx_dir = "onnx_design_fp16";
    const std::filesystem::path output_dir = "outputs/cpp_examples";

    // Base Clone 参考音频和参考文本，与 Python 样例保持一致。
    const qwen3tts::VoiceCloneReference reference{"data/ref_from_mp3_24k_mono.wav", "告诉自己，不要怕"};

    // CustomVoice / VoiceDesign 条件，与 Python 样例保持一致。
    const std::string speaker = "serena";
    const std::string language = "chinese";
    const std::string custom_instruct = "语气温柔自然，情绪平稳，典型的御姐甜美风";
    const std::string design_instruct = "年轻女性，声音温柔清澈，语速适中，情绪坚定。";

    // 默认使用 CUDA 和 Python 样例一致的采样参数。
    const bool use_cuda = true;
    const qwen3tts::GenerationOptions generation;

    const auto deltas = StreamDeltas();
    const auto nonstream_text = NonstreamText();
    const auto design_nonstream_text = JoinText(deltas);

    // Base 模型单独放在一个作用域里：跑完两个 Base 样例后立即析构，
    // 释放 ONNX Runtime session 和 CUDA 显存，再加载下一个模型。
    {
      qwen3tts::BaseQwen3TTSOnnxModel base({base_model_dir, base_onnx_dir, use_cuda});

      // 1. Base Clone 非流式。
      {
        auto result = base.GenerateCloneAudioFromReference(nonstream_text, reference, language, false, false, generation);
        SaveResult(output_dir / "base_clone_nonstream.wav", result);
      }

      // 2. Base Clone 流式，采用 ref-code ICL 分段重组 prompt。
      {
        qwen3tts::SegmentStreamOptions stream_options;
        stream_options.generation = generation;
        stream_options.generation.repetition_penalty = 1.0f;
        stream_options.kv_anchor_segment_count = 4;
        auto chunks = base.StreamCloneAudioFromReference(deltas, reference, language, stream_options);
        SaveStreamResult(output_dir / "base_clone_stream.wav", chunks);
      }
    }

    // CustomVoice 模型单独作用域，避免和 Base / VoiceDesign 同时占用显存。
    {
      qwen3tts::CustomQwen3TTSOnnxModel custom({custom_model_dir, custom_onnx_dir, use_cuda});

      // 3. CustomVoice 非流式。
      {
        auto result = custom.GenerateCustomVoice(nonstream_text, speaker, language, custom_instruct, generation);
        SaveResult(output_dir / "custom_voice_nonstream.wav", result);
      }

      // 4. CustomVoice 流式，分段生成并把最近几段 code 作为下一段 anchor。
      {
        qwen3tts::SegmentStreamOptions stream_options;
        stream_options.generation = generation;
        stream_options.kv_anchor_segment_count = 4;
        auto chunks = custom.StreamCustomVoice(deltas, speaker, language, custom_instruct, stream_options);
        SaveStreamResult(output_dir / "custom_voice_stream.wav", chunks);
      }
    }

    // VoiceDesign 模型最后加载，前两类模型已经离开作用域并释放资源。
    {
      qwen3tts::DesignQwen3TTSOnnxModel design({design_model_dir, design_onnx_dir, use_cuda});

      // 5. VoiceDesign 非流式。
      {
        auto result = design.GenerateVoiceDesign(design_nonstream_text, design_instruct, language, generation);
        SaveResult(output_dir / "voice_design_nonstream.wav", result);
      }

      // 6. VoiceDesign 流式，保留首段 pinned anchor，并滚动保留最近几段 anchor。
      {
        qwen3tts::SegmentStreamOptions stream_options;
        stream_options.generation = generation;
        stream_options.kv_anchor_segment_count = 3;
        stream_options.pinned_anchor_segment_count = 1;
        stream_options.min_text_chunk_chars = 16;
        stream_options.max_text_chunk_chars = 64;
        auto chunks = design.StreamVoiceDesign(deltas, design_instruct, language, stream_options);
        SaveStreamResult(output_dir / "voice_design_stream.wav", chunks);
      }
    }

    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "error: " << exc.what() << "\n";
    return 1;
  }
}
