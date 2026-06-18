#pragma once

#include <filesystem>
#include <string>

#include "qwen3tts/tensor.h"

namespace qwen3tts {

FloatTensor ReadFloatNpy(const std::filesystem::path& path);
Int64Tensor ReadInt64Npy(const std::filesystem::path& path);
void WriteFloatNpy(const std::filesystem::path& path, const FloatTensor& tensor);
void WriteInt64Npy(const std::filesystem::path& path, const Int64Tensor& tensor);

}  // namespace qwen3tts
