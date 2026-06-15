# qwen3-tts-onnxruntime

A lightweight ONNX Runtime implementation for Qwen3-TTS, providing model export tools and Python inference examples for Base, CustomVoice, and VoiceDesign models.

## 安装依赖

```bash
pip install -r requirements.txt
```

## 导出示例

### Base

```bash
python export_onnx.py \
  --model-path /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --output-dir ./onnx_fp16_small \
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

## 参数说明

- `--model-path`: 本地模型目录。
- `--output-dir`: ONNX 导出目录。
- `--dtype`: 导出精度，例如 `float16`。
- `--device`: 导出设备，例如 `cuda:0` 或 `cpu`。
- `--components`: 导出组件，通常使用 `all`。

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
