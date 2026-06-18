#include "qwen3tts/wav_writer.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace qwen3tts {
namespace {
void WriteU16(std::ofstream& out, uint16_t v) {
  out.put(static_cast<char>(v & 0xff));
  out.put(static_cast<char>((v >> 8) & 0xff));
}
void WriteU32(std::ofstream& out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out.put(static_cast<char>((v >> (8 * i)) & 0xff));
}
}  // namespace

void WriteWavMono16(const std::filesystem::path& path, const std::vector<float>& audio, int sample_rate) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("failed to write wav: " + path.string());
  const uint16_t channels = 1;
  const uint16_t bits = 16;
  const uint32_t data_bytes = static_cast<uint32_t>(audio.size() * sizeof(int16_t));
  out.write("RIFF", 4);
  WriteU32(out, 36 + data_bytes);
  out.write("WAVEfmt ", 8);
  WriteU32(out, 16);
  WriteU16(out, 1);
  WriteU16(out, channels);
  WriteU32(out, static_cast<uint32_t>(sample_rate));
  WriteU32(out, static_cast<uint32_t>(sample_rate * channels * bits / 8));
  WriteU16(out, channels * bits / 8);
  WriteU16(out, bits);
  out.write("data", 4);
  WriteU32(out, data_bytes);
  for (float x : audio) {
    x = std::max(-1.0f, std::min(1.0f, x));
    int16_t s = static_cast<int16_t>(x * 32767.0f);
    WriteU16(out, static_cast<uint16_t>(s));
  }
}

}  // namespace qwen3tts
