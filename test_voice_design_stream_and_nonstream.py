from __future__ import annotations

import os
from pathlib import Path
import time
import numpy as np
import soundfile as sf

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

from src.design_model import DesignQwen3TTSOnnxModel


MODEL_DIR_CANDIDATES = [
    Path("/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"),
    Path("/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1___7B-VoiceDesign"),
]
ONNX_DIR = "onnx_design_fp16"
OUTPUT_DIR = Path("outputs")

LANGUAGE = "chinese"
INSTRUCT = "年轻女性，声音温柔清澈，语速适中，情绪坚定。"
STREAM_TEXT_DELTAS = [
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

STREAM_TEXT = "".join(STREAM_TEXT_DELTAS)
DEBUG_STREAM = True
KV_ANCHOR_SEGMENT_COUNT = 3
PINNED_ANCHOR_SEGMENT_COUNT = 1
MIN_TEXT_CHUNK_CHARS = 16
MAX_TEXT_CHUNK_CHARS = 64


def resolve_model_dir() -> str:
    for path in MODEL_DIR_CANDIDATES:
        if path.exists():
            return str(path)
    candidates = ", ".join(str(path) for path in MODEL_DIR_CANDIDATES)
    raise FileNotFoundError(f"VoiceDesign model directory not found. Tried: {candidates}")


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(path, audio, int(sample_rate))
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    mean_abs = float(np.mean(np.abs(audio))) if audio.size else 0.0
    print(f"saved={path} samples={audio.size} peak={peak:.6f} mean_abs={mean_abs:.6f}")


def concat_stream_chunks(chunks, segment_dir: Path | None = None) -> tuple[np.ndarray, int]:
    audio_parts = []
    sample_rate = 24000
    if segment_dir is not None:
        segment_dir.mkdir(parents=True, exist_ok=True)
    for index, chunk in enumerate(chunks):
        if chunk.audio.size:
            audio = np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
            audio_parts.append(audio)
            sample_rate = int(chunk.sample_rate)
            if segment_dir is not None:
                save_audio(segment_dir / f"segment_{index:02d}.wav", audio, sample_rate)
    if not audio_parts:
        return np.zeros((0,), dtype=np.float32), sample_rate
    return np.concatenate(audio_parts), sample_rate


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = DesignQwen3TTSOnnxModel(
        model_path=resolve_model_dir(),
        onnx_dir=ONNX_DIR,
        dtype=np.float16,
    )
    start = time.time()
    nonstream = model.generate_voice_design(
        text=STREAM_TEXT,
        instruct=INSTRUCT,
        language=LANGUAGE
    )
    save_audio(OUTPUT_DIR / "voice_design_nonstream.wav", nonstream.audio, nonstream.sample_rate)
    print(time.time() - start)
    start = time.time()
    stream_chunks = model.stream_voice_design(
        STREAM_TEXT_DELTAS,
        instruct=INSTRUCT,
        language=LANGUAGE,
        max_kv_cache_len=512,
        kv_anchor_segment_count=KV_ANCHOR_SEGMENT_COUNT,
        pinned_anchor_segment_count=PINNED_ANCHOR_SEGMENT_COUNT,
        min_text_chunk_chars=MIN_TEXT_CHUNK_CHARS,
        max_text_chunk_chars=MAX_TEXT_CHUNK_CHARS,
        debug_stream=DEBUG_STREAM,
    )
    stream_audio, stream_sr = concat_stream_chunks(stream_chunks, OUTPUT_DIR / "voice_design_stream_segments")
    save_audio(OUTPUT_DIR / "voice_design_stream.wav", stream_audio, stream_sr)
    print(time.time() - start)

if __name__ == "__main__":
    main()
