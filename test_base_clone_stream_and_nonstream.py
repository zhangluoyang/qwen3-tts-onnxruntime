from __future__ import annotations

import os
from pathlib import Path
import time
import numpy as np
import soundfile as sf

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

from src.models import BaseQwen3TTSOnnxModel

providers = ['CUDAExecutionProvider']

MODEL_DIR = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ONNX_DIR = "onnx_fp16"
REF_AUDIO = "data/ref_from_mp3_24k_mono.wav"
REF_TEXT = "告诉自己，不要怕"
OUTPUT_DIR = Path("outputs")

NONSTREAM_TEXT = "小医仙身着淡紫色长裙，纤细腰肢束着一条银丝软带，一头乌黑长发如瀑般垂至腰际，几缕碎发轻掩着那张清丽中略带苍白的俏脸；她眉如远山含黛，眸若秋水映月，唇边总挂着一丝若有若无的浅笑，仿佛能化解世间所有伤痛——然而那笑意深处却藏着一抹令人心疼的孤寂，那是厄难毒体与生俱来的诅咒，是她以毕生之力抗争的宿命；玉手纤纤，指尖时常萦绕着淡淡的七彩毒雾，却偏偏能用这些夺人性命的毒物炼就救死扶伤的灵药，一如她矛盾而动人的存在：既是令人闻风丧胆的毒女，又是那个在青山镇小医馆里温柔为穷苦百姓诊治的善良姑娘，待到后来与萧炎并肩而行，那双素来沉静的眸子终于多了几分生机与暖意，宛如被春风拂过的寒潭，泛起粼粼波光。"
STREAM_TEXT_DELTAS =[
    "小医仙身着淡紫色长裙,",
    "纤细腰肢束着一条银丝软带,",
    "一头乌黑长发如瀑般垂至腰际,",
    "几缕碎发轻掩着那张清丽中略带苍白的俏脸；",
    "她眉如远山含黛，",
    "眸若秋水映月，",
    "唇边总挂着一丝若有若无的浅笑，",
    "仿佛能化解世间所有伤痛——然而那笑意深处却藏着一抹令人心疼的孤寂，",
    "那是厄难毒体与生俱来的诅咒，",
    "是她以毕生之力抗争的宿命；",
    "玉手纤纤，",
    "指尖时常萦绕着淡淡的七彩毒雾，",
    "却偏偏能用这些夺人性命的毒物炼就救死扶伤的灵药，",
    "一如她矛盾而动人的存在：",
    "既是令人闻风丧胆的毒女，",
    "又是那个在青山镇小医馆里温柔为穷苦百姓诊治的善良姑娘，",
    "待到后来与萧炎并肩而行，",
    "那双素来沉静的眸子终于多了几分生机与暖意，",
    "宛如被春风拂过的寒潭，",
    "泛起粼粼波光。",
    "小医仙身着淡紫色长裙,",
    "纤细腰肢束着一条银丝软带,",
    "一头乌黑长发如瀑般垂至腰际,",
    "几缕碎发轻掩着那张清丽中略带苍白的俏脸；",
    "她眉如远山含黛，",
    "眸若秋水映月，",
    "唇边总挂着一丝若有若无的浅笑，",
    "仿佛能化解世间所有伤痛——然而那笑意深处却藏着一抹令人心疼的孤寂，",
    "那是厄难毒体与生俱来的诅咒，",
    "是她以毕生之力抗争的宿命；",
    "玉手纤纤，",
    "指尖时常萦绕着淡淡的七彩毒雾，",
    "却偏偏能用这些夺人性命的毒物炼就救死扶伤的灵药，",
    "一如她矛盾而动人的存在：",
    "既是令人闻风丧胆的毒女，",
    "又是那个在青山镇小医馆里温柔为穷苦百姓诊治的善良姑娘，",
    "待到后来与萧炎并肩而行，",
    "那双素来沉静的眸子终于多了几分生机与暖意，",
    "宛如被春风拂过的寒潭，",
    "泛起粼粼波光。"
]


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(path, audio, int(sample_rate))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    mean_abs = float(np.mean(np.abs(audio))) if audio.size else 0.0
    print(f"saved={path} samples={audio.size} peak={peak:.6f} mean_abs={mean_abs:.6f}")


def concat_stream_chunks(chunks) -> tuple[np.ndarray, int]:
    audio_parts = []
    sample_rate = 24000
    for chunk in chunks:
        if chunk.audio.size:
            audio_parts.append(np.asarray(chunk.audio, dtype=np.float32).reshape(-1))
            sample_rate = int(chunk.sample_rate)
    if not audio_parts:
        return np.zeros((0,), dtype=np.float32), sample_rate
    return np.concatenate(audio_parts), sample_rate


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = BaseQwen3TTSOnnxModel(
        providers=providers,
        model_path=MODEL_DIR,
        onnx_dir=ONNX_DIR,
        dtype=np.float16,
    )
    start = time.time()
    nonstream = model.generate_clone_audio_from_reference(
        text=NONSTREAM_TEXT,
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        language="chinese"
    )
    save_audio(OUTPUT_DIR / "base_clone_nonstream.wav", nonstream.audio, nonstream.sample_rate)
    print(time.time() - start)
    start = time.time()
    stream_chunks = model.stream_clone_audio_from_reference(
        STREAM_TEXT_DELTAS,
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        language="chinese",
        max_kv_cache_len=256,
        kv_anchor_segment_count=4
    )
    stream_audio, stream_sr = concat_stream_chunks(stream_chunks)
    save_audio(OUTPUT_DIR / "base_clone_stream.wav", stream_audio, stream_sr)
    print(time.time() - start)

if __name__ == "__main__":
    main()
