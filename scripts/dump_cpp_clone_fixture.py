from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import BaseQwen3TTSOnnxModel


DEFAULT_MODEL_DIR = "/home/zhang/.cache/modelscope/hub/models/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_ONNX_DIR = "onnx_fp16"
DEFAULT_REF_AUDIO = "data/ref_from_mp3_24k_mono.wav"
DEFAULT_REF_TEXT = "告诉自己，不要怕"
DEFAULT_TEXT = "小医仙一袭淡雅白裙立于山巅，青丝如瀑垂落腰间，眉目温婉如水，肌肤胜雪，清澈的眸子里却藏着看透世事的孤寂。"


def _write_meta(path: Path, model: BaseQwen3TTSOnnxModel, args: argparse.Namespace) -> None:
    cfg = model.prompt_builder.talker_config
    values = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "eos_token_id": int(cfg["codec_eos_token_id"]),
        "vocab_size": int(cfg["vocab_size"]),
        "first_codebook_mask_tail": int(cfg.get("first_codebook_mask_tail", 1024) or 1024),
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "num_key_value_heads": int(cfg.get("num_key_value_heads", cfg["num_attention_heads"])),
        "head_dim": int(cfg.get("head_dim", int(cfg["hidden_size"]) // int(cfg["num_attention_heads"]))),
        "num_code_groups": int(cfg["num_code_groups"]),
        "decode_upsample_rate": int(model.decode_upsample_rate),
        "audio_sample_rate": int(model.audio_sample_rate),
        "tokenizer_decode_chunk_frames": int(model.tokenizer_decode_chunk_frames),
        "tokenizer_decode_context_frames": int(model.tokenizer_decode_context_frames),
        "do_sample": int(bool(args.do_sample)),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "temperature": float(args.temperature),
        "repetition_penalty": float(args.repetition_penalty),
        "seed": int(args.seed),
    }
    with path.open("w", encoding="utf-8") as f:
        for key, value in values.items():
            f.write(f"{key}={value}\n")


def dump_fixture(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = BaseQwen3TTSOnnxModel(
        model_path=args.model_dir,
        onnx_dir=args.onnx_dir,
        dtype=np.float16 if args.dtype == "float16" else np.float32,
    )
    prompt = model.prompt_builder.build_from_reference(
        text=args.text,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        language=args.language,
        x_vector_only_mode=False,
        non_streaming_mode=True,
    )
    ref_code = prompt.metadata.get("ref_code")
    if ref_code is None:
        raise RuntimeError("prompt did not contain ref_code; clone fixture requires ICL mode")
    ref_code = np.asarray(ref_code, dtype=np.int64)
    if ref_code.ndim == 2:
        ref_code = ref_code[None, :, :]

    np.save(out_dir / "inputs_embeds.npy", np.ascontiguousarray(prompt.inputs_embeds.astype(np.float32)))
    np.save(out_dir / "attention_mask.npy", np.ascontiguousarray(prompt.attention_mask.astype(np.int64)))
    np.save(out_dir / "trailing_text_hidden.npy", np.ascontiguousarray(prompt.trailing_text_hidden.astype(np.float32)))
    np.save(out_dir / "tts_pad_embed.npy", np.ascontiguousarray(prompt.tts_pad_embed.astype(np.float32)))
    np.save(out_dir / "ref_code.npy", np.ascontiguousarray(ref_code.astype(np.int64)))
    _write_meta(out_dir / "meta.txt", model, args)

    result = model.generate_audio_from_prompt(
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        min_new_tokens=args.min_new_tokens,
        seed=args.seed,
    )
    generated_codes = result.code_generation.codes if result.code_generation is not None else result.codes
    np.save(out_dir / "py_generated_codes.npy", np.ascontiguousarray(generated_codes.astype(np.int64)))
    np.save(out_dir / "py_audio.npy", np.ascontiguousarray(result.audio.astype(np.float32)))
    sf.write(out_dir / "py_clone.wav", result.audio, result.sample_rate)

    print(f"fixture_dir={out_dir}")
    print(f"inputs_embeds={prompt.inputs_embeds.shape} attention_mask={prompt.attention_mask.shape}")
    print(f"trailing_text_hidden={prompt.trailing_text_hidden.shape} ref_code={ref_code.shape}")
    print(f"py_audio_samples={result.audio.shape[0]} sample_rate={result.sample_rate}")
    print(f"py_generated_frames={result.generated_frames} stopped={result.stopped} reason={result.stop_reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--onnx-dir", default=DEFAULT_ONNX_DIR)
    parser.add_argument("--out-dir", default="outputs/cpp_fixture")
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default="chinese")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    dump_fixture(parse_args())
