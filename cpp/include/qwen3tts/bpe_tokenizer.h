#pragma once

// C++ 版 Qwen2 byte-level BPE tokenizer。
//
// 它的目标是和 Python AutoProcessor/Qwen tokenizer 的 encode 结果对齐，
// 以便 C++ 可以脱离 Python 直接构造 assistant/ref_text token ids。

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace qwen3tts {

// Qwen3-TTS Base 的文本侧沿用 Qwen2 byte-level BPE tokenizer。
// C++ 运行时只需要 encode：把带有 <|im_start|>/<|im_end|> 的对话模板文本
// 转成 token id，再交给 text_project.onnx 得到 talker 可用的文本 embedding。
class Qwen2BpeTokenizer {
 public:
  explicit Qwen2BpeTokenizer(const std::filesystem::path& model_dir);

  // 返回 tokenizer id 序列，包含匹配到的 added/special token。
  std::vector<int64_t> Encode(const std::string& text) const;

 private:
  // 三个加载函数分别对应 HuggingFace tokenizer 目录下的词表、BPE merge 表
  // 和 added_tokens。这里保持轻量解析，避免 C++ runtime 依赖完整 JSON 库。
  void LoadVocab(const std::filesystem::path& path);
  void LoadMerges(const std::filesystem::path& path);
  void LoadAddedTokens(const std::filesystem::path& path);

  // byte-level BPE 的三个核心步骤：UTF-8 字节映射、按 merge rank 合并、查 id。
  std::vector<std::string> ByteEncode(const std::string& bytes) const;
  std::vector<std::string> Bpe(const std::string& token) const;
  int64_t TokenId(const std::string& token) const;

  std::unordered_map<std::string, int64_t> vocab_;
  std::unordered_map<std::string, int64_t> added_tokens_;
  std::unordered_map<std::string, int> merge_ranks_;
  std::vector<std::string> byte_encoder_;
  std::vector<std::string> special_tokens_sorted_;
};

}  // namespace qwen3tts
