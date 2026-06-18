#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "qwen3tts/tensor.h"

namespace qwen3tts {

struct OrtRunnerOptions {
  std::filesystem::path model_path;
  bool use_cuda = true;
  int cuda_device_id = 0;
  int log_severity_level = 3;
};

class OrtRunner {
 public:
  OrtRunner() = default;
  OrtRunner(Ort::Env& env, OrtRunnerOptions options);

  const std::vector<std::string>& InputNames() const { return input_names_; }
  const std::vector<std::string>& OutputNames() const { return output_names_; }
  const std::filesystem::path& ModelPath() const { return options_.model_path; }
  double LoadSeconds() const { return load_seconds_; }
  ONNXTensorElementDataType InputType(const std::string& name) const;
  ONNXTensorElementDataType OutputType(const std::string& name) const;

  Ort::Value MakeFloatInput(const FloatTensor& tensor, const std::string& input_name) const;
  Ort::Value MakeInt64Input(const Int64Tensor& tensor) const;
  Ort::Value MakeInt64Input(const std::vector<int64_t>& shape, const std::vector<int64_t>& values) const;

  std::vector<Ort::Value> RunIo(std::unordered_map<std::string, Ort::Value>& inputs,
                                const std::vector<std::string>& outputs,
                                const std::unordered_set<std::string>& device_outputs) const;

  FloatTensor CopyFloatTensor(const Ort::Value& value) const;
  std::vector<float> CopyLastLogits(const Ort::Value& value) const;
  Int64Tensor CopyInt64Tensor(const Ort::Value& value) const;
  int64_t CopyInt64Scalar(const Ort::Value& value) const;

 private:
  void InitNames();
  static Ort::SessionOptions BuildOptions(const OrtRunnerOptions& options);
  bool UsesCuda() const { return options_.use_cuda; }

  OrtRunnerOptions options_;
  std::unique_ptr<Ort::Session> session_;
  Ort::MemoryInfo cpu_memory_{nullptr};
  Ort::MemoryInfo cuda_memory_{nullptr};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::unordered_map<std::string, ONNXTensorElementDataType> input_types_;
  std::unordered_map<std::string, ONNXTensorElementDataType> output_types_;
  mutable std::vector<std::vector<Ort::Float16_t>> fp16_storage_;
  mutable size_t fp16_storage_index_ = 0;
  double load_seconds_ = 0.0;
};

bool IsCudaTensor(const Ort::Value& value);

}  // namespace qwen3tts
