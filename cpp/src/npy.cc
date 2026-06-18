#include "qwen3tts/npy.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

namespace qwen3tts {
namespace {

struct Header {
  std::string descr;
  std::vector<int64_t> shape;
};

std::string ReadHeader(std::ifstream& in) {
  char magic[6];
  in.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") throw std::runtime_error("not an npy file");
  unsigned char ver[2];
  in.read(reinterpret_cast<char*>(ver), 2);
  uint32_t header_len = 0;
  if (ver[0] == 1) {
    unsigned char len[2];
    in.read(reinterpret_cast<char*>(len), 2);
    header_len = static_cast<uint32_t>(len[0]) | (static_cast<uint32_t>(len[1]) << 8);
  } else {
    unsigned char len[4];
    in.read(reinterpret_cast<char*>(len), 4);
    header_len = static_cast<uint32_t>(len[0]) | (static_cast<uint32_t>(len[1]) << 8) |
                 (static_cast<uint32_t>(len[2]) << 16) | (static_cast<uint32_t>(len[3]) << 24);
  }
  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));
  return header;
}

Header ParseHeader(const std::string& text) {
  Header h;
  auto descr_pos = text.find("'descr':");
  if (descr_pos == std::string::npos) descr_pos = text.find("\"descr\":");
  auto q1 = text.find_first_of("'\"", descr_pos + 8);
  auto q2 = text.find_first_of("'\"", q1 + 1);
  h.descr = text.substr(q1 + 1, q2 - q1 - 1);

  auto shape_pos = text.find("'shape':");
  if (shape_pos == std::string::npos) shape_pos = text.find("\"shape\":");
  auto l = text.find('(', shape_pos);
  auto r = text.find(')', l);
  std::string body = text.substr(l + 1, r - l - 1);
  std::stringstream ss(body);
  while (ss.good()) {
    std::string item;
    std::getline(ss, item, ',');
    item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c) { return std::isspace(c); }), item.end());
    if (!item.empty()) h.shape.push_back(std::stoll(item));
  }
  return h;
}

template <typename T>
void WriteNpyTyped(const std::filesystem::path& path, const Tensor<T>& tensor, const char* descr) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  if (!out) throw std::runtime_error("failed to write npy: " + path.string());
  std::string shape = "(";
  for (size_t i = 0; i < tensor.shape().size(); ++i) {
    if (i) shape += ", ";
    shape += std::to_string(tensor.shape()[i]);
  }
  if (tensor.shape().size() == 1) shape += ",";
  shape += ")";
  std::string header = "{'descr': '" + std::string(descr) + "', 'fortran_order': False, 'shape': " + shape + ", }";
  size_t padding = 16 - ((10 + header.size() + 1) % 16);
  header.append(padding, ' ');
  header.push_back('\n');
  out.write("\x93NUMPY", 6);
  char ver[2] = {1, 0};
  out.write(ver, 2);
  uint16_t len = static_cast<uint16_t>(header.size());
  char len_buf[2] = {static_cast<char>(len & 0xff), static_cast<char>((len >> 8) & 0xff)};
  out.write(len_buf, 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(tensor.data()), static_cast<std::streamsize>(tensor.size() * sizeof(T)));
}

}  // namespace

FloatTensor ReadFloatNpy(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to read npy: " + path.string());
  Header h = ParseHeader(ReadHeader(in));
  const size_t count = FloatTensor::NumElements(h.shape);
  std::vector<float> values(count);
  if (h.descr == "<f4" || h.descr == "|f4") {
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
  } else if (h.descr == "<f2" || h.descr == "|f2") {
    std::vector<Ort::Float16_t> tmp(count);
    in.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(count * sizeof(Ort::Float16_t)));
    for (size_t i = 0; i < count; ++i) values[i] = tmp[i].ToFloat();
  } else {
    throw std::runtime_error("unsupported float npy dtype: " + h.descr);
  }
  return FloatTensor(std::move(h.shape), std::move(values));
}

Int64Tensor ReadInt64Npy(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("failed to read npy: " + path.string());
  Header h = ParseHeader(ReadHeader(in));
  if (h.descr != "<i8" && h.descr != "|i8") throw std::runtime_error("unsupported int64 npy dtype: " + h.descr);
  const size_t count = Int64Tensor::NumElements(h.shape);
  std::vector<int64_t> values(count);
  in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(int64_t)));
  return Int64Tensor(std::move(h.shape), std::move(values));
}

void WriteFloatNpy(const std::filesystem::path& path, const FloatTensor& tensor) {
  WriteNpyTyped(path, tensor, "<f4");
}

void WriteInt64Npy(const std::filesystem::path& path, const Int64Tensor& tensor) {
  WriteNpyTyped(path, tensor, "<i8");
}

}  // namespace qwen3tts
