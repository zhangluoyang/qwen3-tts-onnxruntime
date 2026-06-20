#include "qwen3tts/ort_runner.h"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "qwen3tts/build_config.h"

namespace qwen3tts {
namespace {

class CudaRuntime {
 public:
  using CudaMemcpyFn = int (*)(void*, const void*, size_t, int);

  static CudaRuntime& Instance() {
    static CudaRuntime runtime;
    return runtime;
  }

  void CopyDeviceToHost(void* dst, const void* src, size_t bytes) const {
    if (bytes == 0) return;
    if (!cuda_memcpy_) throw std::runtime_error("libcudart cudaMemcpy not available");
    constexpr int kCudaMemcpyDeviceToHost = 2;
    int status = cuda_memcpy_(dst, src, bytes, kCudaMemcpyDeviceToHost);
    if (status != 0) throw std::runtime_error("cudaMemcpyDeviceToHost failed: " + std::to_string(status));
  }

 private:
  CudaRuntime() {
    const char* names[] = {"libcudart.so", "libcudart.so.13", "libcudart.so.12"};
    for (const char* name : names) {
      handle_ = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
      if (handle_) break;
    }
    if (handle_) cuda_memcpy_ = reinterpret_cast<CudaMemcpyFn>(dlsym(handle_, "cudaMemcpy"));
  }
  void* handle_ = nullptr;
  CudaMemcpyFn cuda_memcpy_ = nullptr;
};

bool HasCudaProvider() {
  auto providers = Ort::GetAvailableProviders();
  return std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end();
}

void PreloadSharedLibraries(const std::vector<std::filesystem::path>& paths) {
  for (const auto& path : paths) {
    if (!std::filesystem::exists(path)) continue;
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      throw std::runtime_error("failed to preload " + path.string() + ": " + dlerror());
    }
  }
}

void PreloadCudaProviderDependencies() {
  static std::once_flag once;
  std::call_once(once, [] {
    const std::vector<std::string> library_names = {
        "libcudart.so.12",
        "libcublas.so.12",
        "libcublasLt.so.12",
        "libcudnn.so.9",
        "libcudnn_adv.so.9",
        "libcudnn_cnn.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_graph.so.9",
        "libcudnn_heuristic.so.9",
        "libcudnn_engines_runtime_compiled.so.9",
        "libcudnn_engines_precompiled.so.9",
        "libcufft.so.11",
        "libcurand.so.10",
    };
    std::vector<std::filesystem::path> paths;
    for (const auto& name : library_names) {
      for (const auto& dir : kCudaProviderLibDirs) {
        auto path = std::filesystem::path(std::string(dir)) / name;
        if (std::filesystem::exists(path)) {
          paths.push_back(path);
          break;
        }
      }
    }
    PreloadSharedLibraries(paths);
  });
}

bool ShapeMatchesCount(const std::vector<int64_t>& shape, size_t count) {
  size_t n = 1;
  for (int64_t d : shape) {
    if (d < 0) return false;
    n *= static_cast<size_t>(d);
  }
  return n == count;
}

}  // namespace

bool IsCudaTensor(const Ort::Value& value) {
  return value.GetTensorMemoryInfo().GetDeviceType() == OrtMemoryInfoDeviceType_GPU;
}

OrtRunner::OrtRunner(Ort::Env& env, OrtRunnerOptions options)
    : options_(std::move(options)),
      cpu_memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      cuda_memory_("Cuda", OrtDeviceAllocator, options_.cuda_device_id, OrtMemTypeDefault) {
  auto start = std::chrono::steady_clock::now();
  auto session_options = BuildOptions(options_);
  session_ = std::make_unique<Ort::Session>(env, options_.model_path.c_str(), session_options);
  InitNames();
  load_seconds_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

Ort::SessionOptions OrtRunner::BuildOptions(const OrtRunnerOptions& options) {
  Ort::SessionOptions session_options;
  session_options.SetLogSeverityLevel(options.log_severity_level);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (options.use_cuda) {
    if (!HasCudaProvider()) throw std::runtime_error("CUDAExecutionProvider is not available");
    PreloadCudaProviderDependencies();
    Ort::CUDAProviderOptions cuda_options;
    cuda_options.Update({
        {"device_id", std::to_string(options.cuda_device_id)},
        {"cudnn_conv_algo_search", "EXHAUSTIVE"},
        {"cudnn_conv_use_max_workspace", "1"},
        {"do_copy_in_default_stream", "1"},
        {"use_tf32", "1"},
    });
    session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
  }
  return session_options;
}

void OrtRunner::InitNames() {
  Ort::AllocatorWithDefaultOptions allocator;
  for (size_t i = 0; i < session_->GetInputCount(); ++i) {
    auto name_alloc = session_->GetInputNameAllocated(i, allocator);
    std::string name = name_alloc.get();
    input_names_.push_back(name);
    input_types_[name] = session_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
  }
  for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
    auto name_alloc = session_->GetOutputNameAllocated(i, allocator);
    std::string name = name_alloc.get();
    output_names_.push_back(name);
    output_types_[name] = session_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();
  }
}

ONNXTensorElementDataType OrtRunner::InputType(const std::string& name) const {
  auto it = input_types_.find(name);
  if (it == input_types_.end()) throw std::runtime_error("unknown input: " + name);
  return it->second;
}

ONNXTensorElementDataType OrtRunner::OutputType(const std::string& name) const {
  auto it = output_types_.find(name);
  if (it == output_types_.end()) throw std::runtime_error("unknown output: " + name);
  return it->second;
}

Ort::Value OrtRunner::MakeFloatInput(const FloatTensor& tensor, const std::string& input_name) const {
  auto type = InputType(input_name);
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return Ort::Value::CreateTensor<float>(cpu_memory_, const_cast<float*>(tensor.data()), tensor.size(),
                                           tensor.shape().data(), tensor.shape().size());
  }
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    if (fp16_storage_index_ >= fp16_storage_.size()) fp16_storage_.emplace_back();
    auto& storage = fp16_storage_[fp16_storage_index_++];
    storage.resize(tensor.size());
    for (size_t i = 0; i < tensor.size(); ++i) storage[i] = Ort::Float16_t(tensor.values()[i]);
    return Ort::Value::CreateTensor<Ort::Float16_t>(cpu_memory_, storage.data(), storage.size(),
                                                    tensor.shape().data(), tensor.shape().size());
  }
  throw std::runtime_error("float input type mismatch for " + input_name);
}

Ort::Value OrtRunner::MakeInt64Input(const Int64Tensor& tensor) const {
  return Ort::Value::CreateTensor<int64_t>(cpu_memory_, const_cast<int64_t*>(tensor.data()), tensor.size(),
                                           tensor.shape().data(), tensor.shape().size());
}

Ort::Value OrtRunner::MakeInt64Input(const std::vector<int64_t>& shape, const std::vector<int64_t>& values) const {
  return Ort::Value::CreateTensor<int64_t>(cpu_memory_, const_cast<int64_t*>(values.data()), values.size(),
                                           shape.data(), shape.size());
}

std::vector<Ort::Value> OrtRunner::RunIo(std::unordered_map<std::string, Ort::Value>& inputs,
                                         const std::vector<std::string>& outputs,
                                         const std::unordered_set<std::string>& device_outputs) const {
  (void)device_outputs;
  Ort::IoBinding binding(*session_);
  for (const auto& name : input_names_) {
    auto it = inputs.find(name);
    if (it != inputs.end()) binding.BindInput(name.c_str(), it->second);
  }
  for (const auto& name : outputs) {
    if (options_.use_cuda) {
      binding.BindOutput(name.c_str(), cuda_memory_);
    } else {
      binding.BindOutput(name.c_str(), cpu_memory_);
    }
  }
  session_->Run(Ort::RunOptions{nullptr}, binding);
  binding.SynchronizeOutputs();
  auto out = binding.GetOutputValues();
  fp16_storage_index_ = 0;
  return out;
}

FloatTensor OrtRunner::CopyFloatTensor(const Ort::Value& value) const {
  auto info = value.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  const size_t count = info.GetElementCount();
  if (!ShapeMatchesCount(shape, count)) shape = {static_cast<int64_t>(count)};
  std::vector<float> values(count);
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const auto* src = value.GetTensorData<Ort::Float16_t>();
    std::vector<Ort::Float16_t> tmp;
    if (IsCudaTensor(value)) {
      tmp.resize(count);
      CudaRuntime::Instance().CopyDeviceToHost(tmp.data(), src, count * sizeof(Ort::Float16_t));
      src = tmp.data();
    }
    for (size_t i = 0; i < count; ++i) values[i] = src[i].ToFloat();
  } else if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    const float* src = value.GetTensorData<float>();
    if (IsCudaTensor(value)) {
      CudaRuntime::Instance().CopyDeviceToHost(values.data(), src, count * sizeof(float));
    } else {
      std::copy(src, src + count, values.begin());
    }
  } else {
    throw std::runtime_error("expected float tensor");
  }
  const int64_t value_count = static_cast<int64_t>(values.size());
  return FloatTensor({value_count}, std::move(values));
}

std::vector<float> OrtRunner::CopyLastLogits(const Ort::Value& value) const {
  auto info = value.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  if (shape.size() != 3) throw std::runtime_error("logits must be rank-3");
  const int64_t t = shape[1], vocab = shape[2];
  const size_t begin = static_cast<size_t>((t - 1) * vocab);
  std::vector<float> out(static_cast<size_t>(vocab));
  if (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
    const auto* src = value.GetTensorData<Ort::Float16_t>();
    std::vector<Ort::Float16_t> tmp;
    if (IsCudaTensor(value)) {
      tmp.resize(static_cast<size_t>(vocab));
      CudaRuntime::Instance().CopyDeviceToHost(tmp.data(), src + begin, tmp.size() * sizeof(Ort::Float16_t));
      src = tmp.data();
      for (int64_t i = 0; i < vocab; ++i) out[static_cast<size_t>(i)] = src[i].ToFloat();
    } else {
      for (int64_t i = 0; i < vocab; ++i) out[static_cast<size_t>(i)] = src[begin + i].ToFloat();
    }
  } else {
    const float* src = value.GetTensorData<float>();
    if (IsCudaTensor(value)) {
      CudaRuntime::Instance().CopyDeviceToHost(out.data(), src + begin, out.size() * sizeof(float));
    } else {
      std::copy(src + begin, src + begin + vocab, out.begin());
    }
  }
  return out;
}

Int64Tensor OrtRunner::CopyInt64Tensor(const Ort::Value& value) const {
  auto info = value.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) throw std::runtime_error("expected int64 tensor");
  auto shape = info.GetShape();
  const size_t count = info.GetElementCount();
  if (!ShapeMatchesCount(shape, count)) shape = {static_cast<int64_t>(count)};
  std::vector<int64_t> values(count);
  const int64_t* src = value.GetTensorData<int64_t>();
  if (IsCudaTensor(value)) {
    CudaRuntime::Instance().CopyDeviceToHost(values.data(), src, count * sizeof(int64_t));
  } else {
    std::copy(src, src + count, values.begin());
  }
  return Int64Tensor(std::move(shape), std::move(values));
}

int64_t OrtRunner::CopyInt64Scalar(const Ort::Value& value) const {
  auto tensor = CopyInt64Tensor(value);
  if (tensor.empty()) throw std::runtime_error("empty int64 scalar");
  return tensor.values()[0];
}

}  // namespace qwen3tts
