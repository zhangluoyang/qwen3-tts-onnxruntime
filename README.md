python export_onnx.py \
  --model-path /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --output-dir ./onnx_fp16_small \
  --dtype float16 \
  --device cuda:0 \
  --components all

python export_onnx.py \
  --model-path  /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --output-dir ./onnx_custom_fp16 \
  --dtype float16 \
  --device cuda:0 \
  --components all

python export_onnx.py \
  --model-path /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --output-dir ./onnx_design_fp16 \
  --dtype float16 \
  --device cuda:0 \
  --components all

# /home/zhang/.cache/pip/wheels/50/cc/85/34451a5b4827594563d9d4ed713e4e93e5f1b59929dd51811c
# 

## C++ 推理运行

先编译 C++ 版本：

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j 8
```

直接运行 clone 版本：

```bash
./cpp/build/qwen3tts_clone
```

默认会读取：

```text
model_dir=/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base
onnx_dir=onnx_fp16
ref_audio=data/ref_from_mp3_24k_mono.wav
ref_text=告诉自己，不要怕
output=outputs/cpp_product_clone.wav
max_new_tokens=512
do_sample=false
```

也可以手动指定参数：

```bash
./cpp/build/qwen3tts_clone \
  --model-dir /home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --onnx-dir onnx_fp16 \
  --ref-audio data/ref_from_mp3_24k_mono.wav \
  --ref-text "告诉自己，不要怕" \
  --text "小医仙身着淡紫色长裙，纤细腰肢束着一条银丝软带，一头乌黑长发如瀑般垂至腰际，几缕碎发轻掩着那张清丽中略带苍白的俏脸；她眉如远山含黛，眸若秋水映月，唇边总挂着一丝若有若无的浅笑，仿佛能化解世间所有伤痛——然而那笑意深处却藏着一抹令人心疼的孤寂，那是厄难毒体与生俱来的诅咒，是她以毕生之力抗争的宿命；玉手纤纤，指尖时常萦绕着淡淡的七彩毒雾，却偏偏能用这些夺人性命的毒物炼就救死扶伤的灵药，一如她矛盾而动人的存在：既是令人闻风丧胆的毒女，又是那个在青山镇小医馆里温柔为穷苦百姓诊治的善良姑娘，待到后来与萧炎并肩而行，那双素来沉静的眸子终于多了几分生机与暖意，宛如被春风拂过的寒潭，泛起粼粼波光。" \
  --max-new-tokens 2048 \
  --output outputs/cpp_product_clone.wav \
  --sample \
  --top-k 50 \
  --top-p 1.0 \
  --temperature 0.9
```

如果要开启采样：

```bash
./cpp/build/qwen3tts_clone --sample --top-k 50 --top-p 1.0 --temperature 0.9
```

CPU 运行：

```bash
./cpp/build/qwen3tts_clone --cpu
```
