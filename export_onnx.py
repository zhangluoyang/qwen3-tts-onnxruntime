from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import torch


COMPONENTS = (
    "tokenizer",
    "text_project",
    "codec_embed",
    "talker_core",
    "speaker_encoder",
    "sub_talker_sample",
)

DTYPE_ALIASES = {
    "float": torch.float32,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

COMPONENT_ALIASES = {
    "all": "all",
    "tokenizer": "tokenizer",
    "text_project": "text_project",
    "text-project": "text_project",
    "codec_embed": "codec_embed",
    "codec-embed": "codec_embed",
    "codec": "codec_embed",
    "talker_core": "talker_core",
    "talker-core": "talker_core",
    "core": "talker_core",
    "speaker_encoder": "speaker_encoder",
    "speaker-encoder": "speaker_encoder",
    "speaker": "speaker_encoder",
    "sub_talker_sample": "sub_talker_sample",
    "sub-talker-sample": "sub_talker_sample",
    "sub_talker": "sub_talker_sample",
    "sub-talker": "sub_talker_sample",
    "frame_prepare": "sub_talker_sample",
    "frame-prepare": "sub_talker_sample",
}


def parse_dtype(value: str) -> torch.dtype:
    key = value.strip().lower()
    try:
        return DTYPE_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(DTYPE_ALIASES))
        raise argparse.ArgumentTypeError(f"unsupported dtype {value!r}; choose one of: {valid}") from exc


def parse_components(value: str) -> tuple[str, ...]:
    raw_components = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not raw_components:
        raise argparse.ArgumentTypeError("components cannot be empty")

    normalized = []
    for item in raw_components:
        component = COMPONENT_ALIASES.get(item)
        if component is None:
            valid = ", ".join(COMPONENTS + ("all",))
            raise argparse.ArgumentTypeError(
                f"unsupported component {item!r}; choose from: {valid}"
            )
        if component == "all":
            return COMPONENTS
        normalized.append(component)

    selected = set(normalized)
    return tuple(component for component in COMPONENTS if component in selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export tokenizer, text_project, codec_embed, talker_core, "
            "speaker_encoder, and sub_talker_sample ONNX files."
        )
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        type=parse_dtype,
        help="Model/export dtype: float32, float16, or bfloat16. Aliases fp32/fp16/bf16 are accepted.",
    )
    parser.add_argument(
        "--model-path",
        default="/nfs5/models/Qwen",
        type=Path,
        help="Path to the Qwen3-TTS model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("./onnx"),
        type=Path,
        help="Root ONNX output directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for Qwen model components. Use auto, cpu, cuda, or cuda:0. auto prefers cuda:0.",
    )
    parser.add_argument(
        "--components",
        default=COMPONENTS,
        type=parse_components,
        help=(
            "Comma-separated components to export. "
            "Use all, tokenizer, text_project, codec_embed, talker_core, "
            "speaker_encoder, sub_talker_sample."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run ONNXRuntime verification after exporting each selected component.",
    )
    parser.add_argument(
        "--audio-path",
        default=Path("./data/ref_from_mp3_24k_mono.wav"),
        type=Path,
        help="Reference audio used by tokenizer verification.",
    )
    parser.add_argument(
        "--tokenizer-device",
        default="auto",
        help="Device map for the speech tokenizer export. auto follows --device.",
    )
    parser.add_argument(
        "--tokenizer-chunk-frames",
        default=300,
        type=int,
        help="Generated codec frame count used when tracing tokenizer12hz_decode_chunk.onnx.",
    )
    parser.add_argument(
        "--tokenizer-context-frames",
        default=25,
        type=int,
        help="Left-context codec frame count used when tracing tokenizer12hz_decode_chunk.onnx.",
    )
    parser.add_argument(
        "--decode-residual-do-sample",
        default=True,
        action=argparse.BooleanOptionalAction,
        help=(
            "Use top-k sampling inside sub_talker_sample for residual/sub-talker codebooks."
        ),
    )
    parser.add_argument(
        "--decode-residual-top-k",
        default=50,
        type=int,
        help="Top-k used only when --decode-residual-do-sample is set.",
    )
    parser.add_argument(
        "--decode-residual-temperature",
        default=0.9,
        type=float,
        help="Temperature used only when --decode-residual-do-sample is set.",
    )
    parser.add_argument(
        "--optimize-talker-core-batch1-shapes",
        action="store_true",
        help=(
            "When exporting talker_core, patch Qwen3TTSTalkerAttention to fix "
            "batch/head/hidden reshape dimensions for batch_size=1 while keeping seq_len dynamic."
        ),
    )
    return parser


def print_header(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def maybe_eval(module) -> None:
    if hasattr(module, "eval"):
        module.eval()


def resolve_device(device: str, fallback: str | None = None) -> str:
    value = str(device).strip().lower()
    if value == "auto":
        if fallback and fallback != "auto":
            return fallback
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value == "cuda":
        return "cuda:0"
    return value


def validate_dtype_device(dtype: torch.dtype, device: str, component: str) -> None:
    if dtype == torch.float16 and device.startswith("cpu"):
        raise RuntimeError(
            f"{component} float16 export requires CUDA. "
            "CPU replication_pad1d and some other ops do not support torch.float16. "
            "Use --device cuda:0, --tokenizer-device cuda:0, or export with --dtype float32."
        )


def maybe_to_device(module, device: str):
    if device.startswith("cpu"):
        return module
    if not torch.cuda.is_available():
        raise RuntimeError(f"requested {device}, but CUDA is not available")
    if hasattr(module, "to"):
        return module.to(device)
    return module


def resolve_tokenizer_dtype(dtype: torch.dtype, tokenizer_device: str) -> torch.dtype:
    validate_dtype_device(dtype, tokenizer_device, "tokenizer")
    print(f"  tokenizer dtype: {dtype}")
    return dtype


def load_qwen_model(model_path: Path, dtype: torch.dtype, device: str):
    from qwen_tts import Qwen3TTSModel

    print_header(f"Loading Qwen3-TTS model from {model_path}")
    model = Qwen3TTSModel.from_pretrained(str(model_path), dtype=dtype, attn_implementation="eager")
    model = maybe_to_device(model, device)
    maybe_eval(model)
    maybe_eval(getattr(model, "model", None))
    return model


def load_speech_tokenizer(model_path: Path, dtype: torch.dtype, tokenizer_device: str):
    from export.tokenizer_export import load_tokenizer

    tokenizer_dir = model_path / "speech_tokenizer"
    print_header(f"Loading speech tokenizer from {tokenizer_dir}")
    return load_tokenizer(device=tokenizer_device, tokenizer_dir=tokenizer_dir, dtype=dtype)


def has_speaker_encoder(qwen_model) -> bool:
    speaker_encoder = getattr(qwen_model, "speaker_encoder", None)
    speaker_config = getattr(getattr(qwen_model, "config", None), "speaker_encoder_config", None)
    return speaker_encoder is not None and speaker_config is not None


def export_selected(args: argparse.Namespace) -> dict[str, tuple[Path, ...]]:
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    selected = tuple(args.components)
    exported: dict[str, tuple[Path, ...]] = {}
    device = resolve_device(args.device)
    tokenizer_device = resolve_device(args.tokenizer_device, fallback=device)

    print("ONNX export config:")
    print(f"  dtype: {args.dtype}")
    print(f"  model_path: {args.model_path}")
    print(f"  output_dir: {output_root}")
    print(f"  device: {device}")
    print(f"  tokenizer_device: {tokenizer_device}")
    print(f"  components: {', '.join(selected)}")
    print(f"  verify: {args.verify}")
    print(f"  decode_residual_do_sample: {args.decode_residual_do_sample}")

    if "tokenizer" in selected:
        from export.tokenizer_export import export_tokenizer, verify_tokenizer_exports

        tokenizer_dtype = resolve_tokenizer_dtype(args.dtype, tokenizer_device)
        tokenizer = load_speech_tokenizer(args.model_path, tokenizer_dtype, tokenizer_device)
        tokenizer_dir = output_root / "tokenizer"

        print_header("Exporting tokenizer")
        encoder_path, decoder_chunk_path = export_tokenizer(
            tokenizer=tokenizer,
            output_dir=tokenizer_dir,
            model_dtype=tokenizer_dtype,
            trace_chunk_frames=args.tokenizer_chunk_frames,
            trace_context_frames=args.tokenizer_context_frames,
        )
        exported["tokenizer"] = (Path(encoder_path), Path(decoder_chunk_path))

        if args.verify:
            print_header("Verifying tokenizer")
            verify_tokenizer_exports(
                tokenizer=tokenizer,
                output_dir=tokenizer_dir,
                audio_path=args.audio_path,
            )

        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_components = [component for component in selected if component != "tokenizer"]
    if not model_components:
        return exported

    validate_dtype_device(args.dtype, device, "Qwen model")
    model = load_qwen_model(args.model_path, args.dtype, device)
    qwen_model = getattr(model, "model", model)
    qwen_model = maybe_to_device(qwen_model, device)
    talker = getattr(qwen_model, "talker", None)
    if talker is None and any(component != "speaker_encoder" for component in model_components):
        raise AttributeError("loaded model does not expose model.talker")
    speaker_encoder_available = has_speaker_encoder(qwen_model)

    if "text_project" in selected:
        from export.text_project_export import export_text_project, verify_onnx_text_project

        text_project_dir = output_root / "text_project"
        text_project_path = text_project_dir / "text_project.onnx"

        print_header("Exporting text_project")
        export_text_project(talker, output_dir=text_project_dir)
        exported["text_project"] = (text_project_path,)

        if args.verify:
            print_header("Verifying text_project")
            verify_onnx_text_project(talker, onnx_path=text_project_path)

    if "codec_embed" in selected:
        from export.codec_embed_export import codec_embed_export, verify_onnx_codec_embed

        codec_embed_dir = output_root / "codec_embed"
        codec_embed_path = codec_embed_dir / "codec_embed.onnx"

        print_header("Exporting codec_embed")
        codec_embed_export(talker, output_dir=codec_embed_dir)
        exported["codec_embed"] = (codec_embed_path,)

        if args.verify:
            print_header("Verifying codec_embed")
            verify_onnx_codec_embed(talker, onnx_path=codec_embed_path)

    if "talker_core" in selected:
        from export.talker_core_export import talker_core_export, verify_onnx_talker_core

        talker_dir = output_root / "talker"
        talker_core_path = talker_dir / "talker_core.onnx"

        print_header("Exporting talker_core")
        talker_core_export(
            talker,
            output_dir=talker_dir,
            optimize_batch1_shapes=args.optimize_talker_core_batch1_shapes,
        )
        exported["talker_core"] = (talker_core_path,)

        if args.verify:
            print_header("Verifying talker_core decode-shaped input")
            verify_onnx_talker_core(
                talker,
                onnx_path=talker_core_path,
                past_len=8,
                seq_len=1,
                optimize_batch1_shapes=args.optimize_talker_core_batch1_shapes,
            )
            print_header("Verifying talker_core zero-past prefill-shaped input")
            verify_onnx_talker_core(
                talker,
                onnx_path=talker_core_path,
                past_len=0,
                seq_len=8,
                optimize_batch1_shapes=args.optimize_talker_core_batch1_shapes,
            )

    if "speaker_encoder" in selected:
        from export.speaker_encoder_export import (
            speaker_encoder_export,
            verify_onnx_speaker_encoder,
        )

        speaker_dir = output_root / "speaker_encoder"
        speaker_path = speaker_dir / "speaker_encoder.onnx"

        if not speaker_encoder_available:
            model_type = getattr(getattr(qwen_model, "config", None), "tts_model_type", "unknown")
            print_header("Skipping speaker_encoder")
            print(
                "  loaded model does not provide speaker_encoder; "
                f"tts_model_type={model_type!r}. This is expected for models such as CustomVoice."
            )
        else:
            print_header("Exporting speaker_encoder")
            speaker_encoder_export(model=qwen_model, output_dir=speaker_dir, device=device)
            exported["speaker_encoder"] = (speaker_path,)

            if args.verify:
                print_header("Verifying speaker_encoder")
                verify_onnx_speaker_encoder(qwen_model, onnx_path=speaker_path)

    if "sub_talker_sample" in selected:
        from export.sub_talker_sample_export import sub_talker_sample_export, verify_onnx_sub_talker_sample

        decode_dir = output_root / "decode"
        sub_talker_sample_path = decode_dir / "sub_talker_sample.onnx"

        print_header("Exporting sub_talker_sample")
        sub_talker_sample_export(
            talker,
            output_dir=decode_dir,
            residual_do_sample=args.decode_residual_do_sample,
            residual_top_k=args.decode_residual_top_k,
            residual_temperature=args.decode_residual_temperature,
        )
        exported["sub_talker_sample"] = (sub_talker_sample_path,)

        if args.verify:
            print_header("Verifying sub_talker_sample")
            verify_onnx_sub_talker_sample(
                talker,
                onnx_path=sub_talker_sample_path,
                residual_do_sample=args.decode_residual_do_sample,
                residual_top_k=args.decode_residual_top_k,
                residual_temperature=args.decode_residual_temperature,
            )

    return exported


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    exported = export_selected(args)

    print("\nExport complete:")
    for component, paths in exported.items():
        for path in paths:
            print(f"  {component}: {path}")


if __name__ == "__main__":
    main()
