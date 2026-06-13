#pragma once

#include <filesystem>
#include <vector>

namespace qwen3tts {

void WriteWavMono16(const std::filesystem::path& path, const std::vector<float>& audio, int sample_rate);

}
