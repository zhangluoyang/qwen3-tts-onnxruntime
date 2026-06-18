#include "qwen3tts/audio_frontend.h"

// C++ 参考音频前端，目标是尽量贴近 Python librosa + qwen_tts
// mel_spectrogram 的行为。这里的数值对齐直接影响 speaker_embedding，
// 因此注释里会标出哪些地方是为了和 Python 保持一致。

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <fftw3.h>

namespace qwen3tts {
namespace {

constexpr double kPi = 3.14159265358979323846;

double HzToMel(double hz) {
  constexpr double f_min = 0.0;
  constexpr double f_sp = 200.0 / 3.0;
  double mel = (hz - f_min) / f_sp;
  constexpr double min_log_hz = 1000.0;
  constexpr double min_log_mel = (min_log_hz - f_min) / f_sp;
  constexpr double logstep = 1.8562979903656263 / 27.0;  // log(6.4) / 27，和 Slaney mel 标度保持一致
  if (hz >= min_log_hz) mel = min_log_mel + std::log(hz / min_log_hz) / logstep;
  return mel;
}

double MelToHz(double mel) {
  constexpr double f_min = 0.0;
  constexpr double f_sp = 200.0 / 3.0;
  double hz = f_min + f_sp * mel;
  constexpr double min_log_hz = 1000.0;
  constexpr double min_log_mel = (min_log_hz - f_min) / f_sp;
  constexpr double logstep = 1.8562979903656263 / 27.0;  // log(6.4) / 27，和 Slaney mel 标度保持一致
  if (mel >= min_log_mel) hz = min_log_hz * std::exp(logstep * (mel - min_log_mel));
  return hz;
}

std::vector<float> ReflectPad(const std::vector<float>& x, int pad) {
  // PyTorch/Librosa 前端常用 reflect padding。它比 zero padding 更不容易
  // 在首尾引入突兀能量，speaker encoder 对参考音频边界也更稳定。
  if (x.empty()) return {};
  std::vector<float> out(static_cast<size_t>(x.size() + 2 * pad));
  const int n = static_cast<int>(x.size());
  for (int i = 0; i < pad; ++i) {
    int idx = pad - i;
    if (idx >= n) idx = n - 1;
    out[static_cast<size_t>(i)] = x[static_cast<size_t>(idx)];
  }
  std::copy(x.begin(), x.end(), out.begin() + pad);
  for (int i = 0; i < pad; ++i) {
    int idx = n - 2 - i;
    if (idx < 0) idx = 0;
    out[static_cast<size_t>(pad + n + i)] = x[static_cast<size_t>(idx)];
  }
  return out;
}

std::vector<float> MakeMelBasis(int sample_rate, int n_fft, int num_mels, int fmin, int fmax) {
  // 构造 Slaney 风格 mel 滤波器组，输出布局为 [num_mels, n_fft/2+1]。
  // 后面每一帧的幅度谱会左乘这个矩阵得到 mel bins。
  const int bins = n_fft / 2 + 1;
  std::vector<double> fftfreqs(static_cast<size_t>(bins));
  for (int i = 0; i < bins; ++i) fftfreqs[static_cast<size_t>(i)] = static_cast<double>(i) * sample_rate / n_fft;

  std::vector<double> mel_f(static_cast<size_t>(num_mels + 2));
  const double min_mel = HzToMel(fmin);
  const double max_mel = HzToMel(fmax);
  for (int i = 0; i < num_mels + 2; ++i) {
    mel_f[static_cast<size_t>(i)] = MelToHz(min_mel + (max_mel - min_mel) * i / (num_mels + 1));
  }

  std::vector<float> weights(static_cast<size_t>(num_mels * bins), 0.0f);
  for (int m = 0; m < num_mels; ++m) {
    const double fdiff0 = mel_f[static_cast<size_t>(m + 1)] - mel_f[static_cast<size_t>(m)];
    const double fdiff1 = mel_f[static_cast<size_t>(m + 2)] - mel_f[static_cast<size_t>(m + 1)];
    const double enorm = 2.0 / (mel_f[static_cast<size_t>(m + 2)] - mel_f[static_cast<size_t>(m)]);
    for (int b = 0; b < bins; ++b) {
      const double lower = (fftfreqs[static_cast<size_t>(b)] - mel_f[static_cast<size_t>(m)]) / fdiff0;
      const double upper = (mel_f[static_cast<size_t>(m + 2)] - fftfreqs[static_cast<size_t>(b)]) / fdiff1;
      weights[static_cast<size_t>(m * bins + b)] = static_cast<float>(std::max(0.0, std::min(lower, upper)) * enorm);
    }
  }
  return weights;
}

uint32_t ReadU32(std::istream& in) {
  // WAV RIFF 头使用 little-endian。
  unsigned char b[4];
  in.read(reinterpret_cast<char*>(b), 4);
  return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
         (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
}

uint16_t ReadU16(std::istream& in) {
  unsigned char b[2];
  in.read(reinterpret_cast<char*>(b), 2);
  return static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
}

std::string ShellQuote(const std::string& s) {
  // ffmpeg/ffprobe 通过 shell 管道调用，路径必须安全转义。
  std::string out = "'";
  for (char c : s) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  out += "'";
  return out;
}

AudioBuffer LoadAudioWithFfmpeg(const std::filesystem::path& path, int target_sample_rate) {
  // WAV 以外的格式，例如 mp3/m4a/flac，交给 ffmpeg 解码成 f32le。
  // 这样 C++ 示例可以直接使用 data/林志玲.mp3，不要求用户预先转 wav。
  int channels = 1;
  {
    const std::string probe_command =
        "ffprobe -v error -select_streams a:0 -show_entries stream=channels "
        "-of default=nw=1:nk=1 " + ShellQuote(path.string());
    FILE* probe = popen(probe_command.c_str(), "r");
    if (probe) {
      char text[64] = {0};
      if (fgets(text, sizeof(text), probe)) {
        std::istringstream in(text);
        int parsed = 0;
        if (in >> parsed && parsed > 0) channels = parsed;
      }
      pclose(probe);
    }
  }

  // Do not ask ffmpeg to downmix with "-ac 1": its default stereo matrix uses
  // 0.707 coefficients, while librosa mono=True averages channels. Decode
  // interleaved float samples and average here so C++ matches Python better.
  const std::string command = "ffmpeg -v error -i " + ShellQuote(path.string()) +
                              " -f f32le -ar " + std::to_string(target_sample_rate) + " pipe:1";
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe) {
    throw std::runtime_error("Failed to start ffmpeg for audio decode: " + path.string());
  }

  std::vector<float> interleaved;
  std::vector<unsigned char> buffer(64 * 1024);
  while (true) {
    const size_t n = fread(buffer.data(), 1, buffer.size(), pipe);
    if (n > 0) {
      const size_t old_bytes = interleaved.size() * sizeof(float);
      const size_t new_bytes = old_bytes + n;
      interleaved.resize((new_bytes + sizeof(float) - 1) / sizeof(float), 0.0f);
      std::memcpy(reinterpret_cast<unsigned char*>(interleaved.data()) + old_bytes, buffer.data(), n);
    }
    if (n < buffer.size()) {
      if (feof(pipe)) break;
      if (ferror(pipe)) {
        pclose(pipe);
        throw std::runtime_error("Failed while reading decoded audio from ffmpeg: " + path.string());
      }
    }
  }

  const int status = pclose(pipe);
  if (status != 0 || interleaved.empty()) {
    throw std::runtime_error("ffmpeg failed to decode audio; install ffmpeg or convert to 24 kHz mono WAV: " +
                             path.string());
  }

  std::vector<float> samples;
  if (channels <= 1) {
    samples = std::move(interleaved);
  } else {
    const size_t frames = interleaved.size() / static_cast<size_t>(channels);
    samples.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
      double sum = 0.0;
      for (int ch = 0; ch < channels; ++ch) {
        sum += interleaved[i * static_cast<size_t>(channels) + static_cast<size_t>(ch)];
      }
      samples[i] = static_cast<float>(sum / channels);
    }
  }
  return {std::move(samples), target_sample_rate};
}

}  // 匿名命名空间

AudioBuffer LoadAudioMono(const std::filesystem::path& path, int target_sample_rate) {
  // 快路径：直接解析常见 WAV，避免每次都启动 ffmpeg 进程。
  // 如果不是 RIFF/WAVE，则自动退回 ffmpeg。
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Failed to open audio: " + path.string());
  char riff[4], wave[4];
  in.read(riff, 4);
  (void)ReadU32(in);
  in.read(wave, 4);
  if (std::strncmp(riff, "RIFF", 4) != 0 || std::strncmp(wave, "WAVE", 4) != 0) {
    return LoadAudioWithFfmpeg(path, target_sample_rate);
  }

  uint16_t format = 0;
  uint16_t channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits = 0;
  std::vector<unsigned char> data;
  while (in && (!format || data.empty())) {
    char id[4];
    in.read(id, 4);
    if (!in) break;
    const uint32_t size = ReadU32(in);
    if (std::strncmp(id, "fmt ", 4) == 0) {
      format = ReadU16(in);
      channels = ReadU16(in);
      sample_rate = ReadU32(in);
      (void)ReadU32(in);
      (void)ReadU16(in);
      bits = ReadU16(in);
      if (size > 16) in.seekg(size - 16, std::ios::cur);
    } else if (std::strncmp(id, "data", 4) == 0) {
      data.resize(size);
      in.read(reinterpret_cast<char*>(data.data()), size);
    } else {
      in.seekg(size, std::ios::cur);
    }
    if (size % 2 == 1) in.seekg(1, std::ios::cur);
  }
  if (!format || channels == 0 || sample_rate == 0 || data.empty()) {
    throw std::runtime_error("Invalid or unsupported WAV file: " + path.string());
  }

  const int bytes_per_sample = bits / 8;
  const size_t frames = data.size() / bytes_per_sample / channels;
  std::vector<float> mono(frames);
  for (size_t i = 0; i < frames; ++i) {
    // 多声道按算术平均混成 mono，与 librosa.load(..., mono=True) 更接近。
    double s = 0.0;
    for (uint16_t ch = 0; ch < channels; ++ch) {
      const unsigned char* p = data.data() + (i * channels + ch) * bytes_per_sample;
      if (format == 1 && bits == 16) {
        int16_t v;
        std::memcpy(&v, p, sizeof(v));
        s += static_cast<double>(v) / 32768.0;
      } else if (format == 3 && bits == 32) {
        float v;
        std::memcpy(&v, p, sizeof(v));
        s += v;
      } else {
        throw std::runtime_error("Unsupported WAV encoding; use 16-bit PCM or 32-bit float WAV");
      }
    }
    mono[i] = static_cast<float>(s / channels);
  }

  if (static_cast<int>(sample_rate) == target_sample_rate) return {mono, target_sample_rate};

  // 简单线性重采样。用于 WAV 快路径；复杂格式已经由 ffmpeg 负责重采样。
  const double ratio = static_cast<double>(target_sample_rate) / sample_rate;
  std::vector<float> resampled(static_cast<size_t>(std::ceil(mono.size() * ratio)));
  for (size_t i = 0; i < resampled.size(); ++i) {
    const double src = static_cast<double>(i) / ratio;
    const size_t j = static_cast<size_t>(std::floor(src));
    const double frac = src - j;
    const float a = mono[std::min(j, mono.size() - 1)];
    const float b = mono[std::min(j + 1, mono.size() - 1)];
    resampled[i] = static_cast<float>(a + (b - a) * frac);
  }
  return {resampled, target_sample_rate};
}

FloatTensor MelSpectrogram(const std::vector<float>& audio,
                           int sample_rate,
                           int n_fft,
                           int num_mels,
                           int hop_size,
                           int win_size,
                           int fmin,
                           int fmax) {
  if (audio.empty()) throw std::invalid_argument("Cannot compute mel for empty audio");
  if (n_fft != win_size) throw std::invalid_argument("MelSpectrogram currently expects n_fft == win_size");

  const int padding = (n_fft - hop_size) / 2;
  const auto padded = ReflectPad(audio, padding);
  const int frames = std::max<int>(0, (static_cast<int>(padded.size()) - n_fft) / hop_size + 1);
  const int bins = n_fft / 2 + 1;
  const auto mel_basis = MakeMelBasis(sample_rate, n_fft, num_mels, fmin, fmax);

  std::vector<float> window(static_cast<size_t>(win_size));
  for (int i = 0; i < win_size; ++i) {
    // Hann window，对齐常见 STFT 前端。
    window[static_cast<size_t>(i)] = static_cast<float>(0.5 - 0.5 * std::cos(2.0 * kPi * i / win_size));
  }

  FloatTensor mel({1, frames, num_mels});
  std::vector<float> frame(static_cast<size_t>(n_fft));
  std::vector<float> spec(static_cast<size_t>(bins));
  std::vector<fftwf_complex> fft_out(static_cast<size_t>(bins));
  std::unique_ptr<std::remove_pointer_t<fftwf_plan>, decltype(&fftwf_destroy_plan)> fft_plan(
      fftwf_plan_dft_r2c_1d(n_fft, frame.data(), fft_out.data(), FFTW_ESTIMATE),
      fftwf_destroy_plan);
  if (!fft_plan) throw std::runtime_error("Failed to create FFTW mel spectrogram plan");
  for (int t = 0; t < frames; ++t) {
    // 每一帧：加窗 -> rFFT -> 幅度谱 -> mel 滤波 -> log 压缩。
    const int start = t * hop_size;
    for (int i = 0; i < n_fft; ++i) {
      frame[static_cast<size_t>(i)] = padded[static_cast<size_t>(start + i)] * window[static_cast<size_t>(i)];
    }
    fftwf_execute(fft_plan.get());
    for (int b = 0; b < bins; ++b) {
      const auto real = fft_out[static_cast<size_t>(b)][0];
      const auto imag = fft_out[static_cast<size_t>(b)][1];
      spec[static_cast<size_t>(b)] = std::sqrt(real * real + imag * imag + 1e-9f);
    }
    for (int m = 0; m < num_mels; ++m) {
      double sum = 0.0;
      for (int b = 0; b < bins; ++b) {
        sum += mel_basis[static_cast<size_t>(m * bins + b)] * spec[static_cast<size_t>(b)];
      }
      mel.values()[static_cast<size_t>(t * num_mels + m)] = static_cast<float>(std::log(std::max(sum, 1e-5)));
    }
  }
  return mel;
}

}  // namespace qwen3tts
