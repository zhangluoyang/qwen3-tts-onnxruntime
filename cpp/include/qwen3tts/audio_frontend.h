#pragma once

// 参考音频前端。
//
// 声音克隆时，参考音频既要被 tokenizer encoder 编成 codec codes，
// 又要被 speaker encoder 抽成说话人 embedding。这个文件负责把任意输入音频
// 统一成 24k 单声道 waveform，并计算和 Python mel_spectrogram 对齐的 mel。

#include <filesystem>
#include <vector>

#include "qwen3tts/tensor.h"

namespace qwen3tts {

// 参考音频前端使用的最小音频容器。
// Qwen3-TTS Base 声音克隆里，同一段参考音频会走两条分支：
// 1. waveform -> tokenizer12hz_encode.onnx -> 参考 codec codes；
// 2. waveform -> mel 频谱 -> speaker_encoder.onnx -> 说话人 embedding。
struct AudioBuffer {
  std::vector<float> samples;
  int sample_rate = 24000;
};

// 加载任意常见音频文件并转换成单声道目标采样率。
// WAV 16-bit PCM / 32-bit float 会在 C++ 内部直接解析；其它格式交给 ffmpeg。
AudioBuffer LoadAudioMono(const std::filesystem::path& path, int target_sample_rate = 24000);

// 计算和 Python Qwen3-TTS 前端对齐的 Slaney mel 频谱。
// 输出形状为 [1, frames, num_mels]，正好对应 speaker_encoder.onnx 的 mel 输入。
FloatTensor MelSpectrogram(const std::vector<float>& audio,
                           int sample_rate = 24000,
                           int n_fft = 1024,
                           int num_mels = 128,
                           int hop_size = 256,
                           int win_size = 1024,
                           int fmin = 0,
                           int fmax = 12000);

}  // namespace qwen3tts
