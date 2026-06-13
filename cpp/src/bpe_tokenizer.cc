#include "qwen3tts/bpe_tokenizer.h"

// 轻量 BPE tokenizer 实现。
//
// 注意：这里不是完整复刻 HuggingFace tokenizers 的所有边界情况，
// 而是覆盖 Qwen3-TTS 推理文本里最关键的路径：special token、中文、
// ASCII 单词、标点和 byte-level BPE merge。用于学习时可以重点看
// Encode() -> PreTokenize() -> Bpe() -> TokenId() 这条链。

#include <algorithm>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace qwen3tts {
namespace {

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string JsonUnescape(const std::string& s) {
  // tokenizer JSON 里的 token 可能包含 \"、\n 等转义。
  std::string out;
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] != '\\' || i + 1 >= s.size()) {
      out.push_back(s[i]);
      continue;
    }
    const char c = s[++i];
    switch (c) {
      case '"': out.push_back('"'); break;
      case '\\': out.push_back('\\'); break;
      case '/': out.push_back('/'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'n': out.push_back('\n'); break;
      case 'r': out.push_back('\r'); break;
      case 't': out.push_back('\t'); break;
      default: out.push_back(c); break;
    }
  }
  return out;
}

std::string PairKey(const std::string& a, const std::string& b) {
  return a + '\x1f' + b;
}

bool IsContinuation(unsigned char c) {
  return (c & 0xC0) == 0x80;
}

std::vector<std::string> Utf8Chars(const std::string& text) {
  // 按 UTF-8 codepoint 切分，避免把中文字符的多个字节拆散。
  std::vector<std::string> out;
  for (size_t i = 0; i < text.size();) {
    size_t j = i + 1;
    while (j < text.size() && IsContinuation(static_cast<unsigned char>(text[j]))) ++j;
    out.push_back(text.substr(i, j - i));
    i = j;
  }
  return out;
}

bool IsAsciiAlpha(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool IsAsciiDigit(char c) {
  return c >= '0' && c <= '9';
}

bool IsSpacePiece(const std::string& s) {
  return s == " " || s == "\n" || s == "\r" || s == "\t";
}

bool IsNewlinePiece(const std::string& s) {
  return s == "\n" || s == "\r";
}

bool IsKnownUnicodePunct(const std::string& s) {
  static const std::vector<std::string> punct = {
      "，", "。", "、", "；", "：", "？", "！", "（", "）", "《", "》", "“", "”", "‘", "’", "—", "…"};
  return std::find(punct.begin(), punct.end(), s) != punct.end();
}

bool IsNonAsciiWordPiece(const std::string& s) {
  return s.size() > 1 && !IsKnownUnicodePunct(s);
}

bool IsLetterPiece(const std::string& s) {
  return (s.size() == 1 && IsAsciiAlpha(s[0])) || IsNonAsciiWordPiece(s);
}

bool IsDigitPiece(const std::string& s) {
  return s.size() == 1 && IsAsciiDigit(s[0]);
}

bool IsPunctuationPiece(const std::string& s) {
  return !IsSpacePiece(s) && !IsLetterPiece(s) && !IsDigitPiece(s);
}

std::vector<std::string> PreTokenize(const std::string& text) {
  // 近似 Qwen2/tiktoken 的 pre-tokenizer：
  // - 字母/中文连续成段，允许一个前置非字母数字符号（如 +b、_bar）；
  // - 数字逐位成段；
  // - 连续标点整体进入 BPE（如 ++、==、--），避免把可 merge 的 pair 先拆碎；
  // - 单个前导空格并入后续字母或标点段，符合 byte-level BPE 的空格语义。
  std::vector<std::string> out;
  const auto chars = Utf8Chars(text);
  for (size_t i = 0; i < chars.size();) {
    std::string prefix;
    if (chars[i] == " ") {
      size_t j = i;
      while (j < chars.size() && chars[j] == " ") ++j;
      if (j < chars.size() && !IsSpacePiece(chars[j])) {
        if (j > i + 1) out.push_back(std::string(j - i - 1, ' '));
        prefix = " ";
        i = j;
      } else {
        out.push_back(std::string(j - i, ' '));
        i = j;
        continue;
      }
    }
    if (i >= chars.size()) {
      out.push_back(prefix);
      break;
    }

    const std::string& ch = chars[i];
    if (IsLetterPiece(ch)) {
      std::string s = prefix;
      while (i < chars.size() && IsLetterPiece(chars[i])) s += chars[i++];
      out.push_back(s);
    } else if (IsDigitPiece(ch)) {
      if (!prefix.empty()) out.push_back(prefix);
      out.push_back(ch);
      ++i;
    } else if (IsPunctuationPiece(ch)) {
      std::string s = prefix;
      while (i < chars.size() && IsPunctuationPiece(chars[i])) s += chars[i++];
      while (i < chars.size() && IsNewlinePiece(chars[i])) s += chars[i++];
      out.push_back(s);
    } else {
      if (!prefix.empty()) out.push_back(prefix);
      std::string s;
      while (i < chars.size() && IsSpacePiece(chars[i])) s += chars[i++];
      out.push_back(s);
      continue;
    }
  }
  return out;
}

bool IsContractionTail(const std::vector<std::string>& chars, size_t i, size_t* end) {
  if (i >= chars.size() || chars[i] != "'") return false;
  static const std::vector<std::vector<std::string>> tails = {
      {"s"}, {"t"}, {"r", "e"}, {"v", "e"}, {"m"}, {"l", "l"}, {"d"},
      {"S"}, {"T"}, {"R", "E"}, {"V", "E"}, {"M"}, {"L", "L"}, {"D"}};
  for (const auto& tail : tails) {
    if (i + tail.size() >= chars.size()) continue;
    bool ok = true;
    for (size_t j = 0; j < tail.size(); ++j) {
      if (chars[i + 1 + j] != tail[j]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      *end = i + 1 + tail.size();
      return true;
    }
  }
  return false;
}

std::vector<std::string> PreTokenizeWithContractions(const std::string& text) {
  std::vector<std::string> out;
  const auto chars = Utf8Chars(text);
  size_t start = 0;
  for (size_t i = 0; i < chars.size();) {
    size_t end = i;
    if (IsContractionTail(chars, i, &end)) {
      std::string prefix;
      for (size_t j = start; j < i; ++j) prefix += chars[j];
      for (const auto& piece : PreTokenize(prefix)) {
        if (!piece.empty()) out.push_back(piece);
      }
      std::string contraction;
      for (size_t j = i; j < end; ++j) contraction += chars[j];
      out.push_back(contraction);
      i = end;
      start = i;
    } else {
      ++i;
    }
  }
  std::string suffix;
  for (size_t j = start; j < chars.size(); ++j) suffix += chars[j];
  for (const auto& piece : PreTokenize(suffix)) {
    if (!piece.empty()) out.push_back(piece);
  }
  return out;
}

std::vector<std::pair<int, int>> ByteRanges() {
  // GPT/Qwen byte-level BPE 会把 0..255 映射到可打印 unicode 字符，
  // 避免原始控制字符直接出现在 vocab token 中。
  std::vector<std::pair<int, int>> ranges{{'!', '~'}, {0xA1, 0xAC}, {0xAE, 0xFF}};
  return ranges;
}

}  // 匿名命名空间

Qwen2BpeTokenizer::Qwen2BpeTokenizer(const std::filesystem::path& model_dir) {
  // 构建 bytes_to_unicode 映射，再加载 vocab/merges/added tokens。
  byte_encoder_.resize(256);
  std::vector<int> bs;
  for (auto [a, b] : ByteRanges()) {
    for (int i = a; i <= b; ++i) bs.push_back(i);
  }
  std::vector<int> cs = bs;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n++);
    }
  }
  for (size_t i = 0; i < bs.size(); ++i) {
    int cp = cs[i];
    std::string s;
    if (cp < 0x80) {
      s.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
      s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    byte_encoder_[static_cast<size_t>(bs[i])] = s;
  }
  LoadVocab(model_dir / "vocab.json");
  LoadMerges(model_dir / "merges.txt");
  LoadAddedTokens(model_dir / "tokenizer_config.json");
}

void Qwen2BpeTokenizer::LoadVocab(const std::filesystem::path& path) {
  // vocab.json: token string -> token id。
  const auto text = ReadText(path);
  std::regex item("\\\"((?:[^\\\\\\\"]|\\\\.)*)\\\"\\s*:\\s*([0-9]+)");
  for (auto it = std::sregex_iterator(text.begin(), text.end(), item); it != std::sregex_iterator(); ++it) {
    vocab_[JsonUnescape((*it)[1].str())] = std::stoll((*it)[2].str());
  }
}

void Qwen2BpeTokenizer::LoadMerges(const std::filesystem::path& path) {
  // merges.txt 的行号就是 merge rank，rank 越小越优先合并。
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open " + path.string());
  std::string a, b;
  int rank = 0;
  while (in >> a >> b) {
    merge_ranks_[PairKey(a, b)] = rank++;
  }
}

void Qwen2BpeTokenizer::LoadAddedTokens(const std::filesystem::path& path) {
  // special/added tokens 例如 <|im_start|> 必须优先整体匹配，不能被普通 BPE 切碎。
  const auto text = ReadText(path);
  std::regex item("\\\"([0-9]+)\\\"\\s*:\\s*\\{[^\\}]*\\\"content\\\"\\s*:\\s*\\\"((?:[^\\\\\\\"]|\\\\.)*)\\\"");
  for (auto it = std::sregex_iterator(text.begin(), text.end(), item); it != std::sregex_iterator(); ++it) {
    const auto token = JsonUnescape((*it)[2].str());
    const auto id = std::stoll((*it)[1].str());
    added_tokens_[token] = id;
    special_tokens_sorted_.push_back(token);
  }
  std::sort(special_tokens_sorted_.begin(), special_tokens_sorted_.end(),
            [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
}

std::vector<std::string> Qwen2BpeTokenizer::ByteEncode(const std::string& bytes) const {
  std::vector<std::string> out;
  out.reserve(bytes.size());
  for (unsigned char c : bytes) out.push_back(byte_encoder_[c]);
  return out;
}

std::vector<std::string> Qwen2BpeTokenizer::Bpe(const std::string& token) const {
  // 贪心地反复合并 rank 最小的相邻 pair，直到没有可合并 pair。
  auto word = ByteEncode(token);
  if (word.size() <= 1) return word;
  while (true) {
    int best_rank = std::numeric_limits<int>::max();
    size_t best = 0;
    for (size_t i = 0; i + 1 < word.size(); ++i) {
      auto it = merge_ranks_.find(PairKey(word[i], word[i + 1]));
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best = i;
      }
    }
    if (best_rank == std::numeric_limits<int>::max()) break;
    word[best] += word[best + 1];
    word.erase(word.begin() + static_cast<long>(best + 1));
    if (word.size() == 1) break;
  }
  return word;
}

int64_t Qwen2BpeTokenizer::TokenId(const std::string& token) const {
  auto it = vocab_.find(token);
  if (it == vocab_.end()) throw std::runtime_error("Token not in vocab: " + token);
  return it->second;
}

std::vector<int64_t> Qwen2BpeTokenizer::Encode(const std::string& text) const {
  std::vector<int64_t> ids;
  for (size_t i = 0; i < text.size();) {
    // 先从当前位置尝试 special token 最长匹配。对话模板里的
    // <|im_start|>/<|im_end|> 都依赖这个逻辑。
    bool matched = false;
    for (const auto& special : special_tokens_sorted_) {
      if (!special.empty() && text.compare(i, special.size(), special) == 0) {
        ids.push_back(added_tokens_.at(special));
        i += special.size();
        matched = true;
        break;
      }
    }
    if (matched) continue;

    size_t next = text.size();
    // 普通文本只处理到下一个 special token 之前，防止跨 special 边界做 BPE。
    for (const auto& special : special_tokens_sorted_) {
      const auto pos = text.find(special, i);
      if (pos != std::string::npos) next = std::min(next, pos);
    }
    for (const auto& piece : PreTokenizeWithContractions(text.substr(i, next - i))) {
      for (const auto& bpe : Bpe(piece)) ids.push_back(TokenId(bpe));
    }
    i = next;
  }
  return ids;
}

}  // namespace qwen3tts
