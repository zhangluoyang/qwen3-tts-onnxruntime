# qwen3-tts-onnxruntime

A lightweight ONNX Runtime implementation for Qwen3-TTS, providing model export tools and Python inference examples for Base, CustomVoice, and VoiceDesign models.

项目介绍：[Qwen3-TTS ONNX Runtime 实践](https://zhuanlan.zhihu.com/p/2049974407729243443)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 导出示例

### Base

```bash
python export_onnx.py \
  --model-path /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output-dir ./onnx_fp16 \
  --dtype float16 \
  --device cuda:0 \
  --components all
```

### CustomVoice

```bash
python export_onnx.py \
  --model-path /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --output-dir ./onnx_custom_fp16 \
  --dtype float16 \
  --device cuda:0 \
  --components all
```

### VoiceDesign

```bash
python export_onnx.py \
  --model-path /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --output-dir ./onnx_design_fp16 \
  --dtype float16 \
  --device cuda:0 \
  --components all
```

## 测试示例

运行测试前，请先按上面的示例导出对应 ONNX 目录。

### Base Clone

```bash
python test_base_clone_stream_and_nonstream.py
```

输出文件：

- `outputs/base_clone_nonstream.wav`
- `outputs/base_clone_stream.wav`

### CustomVoice

```bash
python test_custom_voice_stream_and_nonstream.py
```

输出文件：

- `outputs/custom_voice_nonstream.wav`
- `outputs/custom_voice_stream.wav`

### VoiceDesign

```bash
python test_voice_design_stream_and_nonstream.py
```

输出文件：

- `outputs/voice_design_nonstream.wav`
- `outputs/voice_design_stream.wav`

## C++ 推理运行

C++ 版本支持 Base Clone、CustomVoice、VoiceDesign 三种模型，并提供非流式和流式示例。下面命令默认从仓库根目录运行。

### 下载 C++ 依赖

下载脚本放在 `cpp/scripts/download_deps.sh`，第三方依赖会放到 `cpp/third_party/`，不会混到源码目录里。

```bash
bash cpp/scripts/download_deps.sh
```

脚本会准备：

- ONNX Runtime GPU C++ 包：`cpp/third_party/onnxruntime-local/`
- FFTW3f 单精度静态库：`cpp/third_party/fftw3-local/`

CUDA、cuDNN、cuBLAS、cuFFT、cuRAND 这些运行库仍然使用当前 Python 环境或系统里已有的安装。

### 编译

```bash
python build_cpp.py
```

`build_cpp.py` 会自动使用 `cpp/third_party/` 下面下载好的 ONNX Runtime 和 FFTW3f，不需要手动传 CMake 参数。

编译完成后会生成：

- `cpp/build/qwen3tts_examples`: 三种模型的流式/非流式统一示例。
- `cpp/build/qwen3tts_clone`: 旧版 Base Clone 示例，保留兼容。

### 直接运行

运行前请先导出对应 ONNX 目录：

- Base: `onnx_fp16`
- CustomVoice: `onnx_custom_fp16`
- VoiceDesign: `onnx_design_fp16`

然后直接执行：

```bash
./cpp/build/qwen3tts_examples
```

默认会一次跑完六个 C++ 示例：Base Clone、CustomVoice、VoiceDesign 三种模型，每种模型都会跑非流式和流式。

会输出：

- `outputs/cpp_examples/base_clone_nonstream.wav`
- `outputs/cpp_examples/base_clone_stream.wav`
- `outputs/cpp_examples/custom_voice_nonstream.wav`
- `outputs/cpp_examples/custom_voice_stream.wav`
- `outputs/cpp_examples/voice_design_nonstream.wav`
- `outputs/cpp_examples/voice_design_stream.wav`

默认读取的路径：

- Base 模型目录：`/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- CustomVoice 模型目录：`/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- VoiceDesign 模型目录：`/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- 参考音频：`data/ref_from_mp3_24k_mono.wav`
- 输出目录：`outputs/cpp_examples`

旧版 Base Clone 示例也可以直接运行：

```bash
./cpp/build/qwen3tts_clone
```
