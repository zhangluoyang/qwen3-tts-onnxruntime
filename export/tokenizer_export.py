import os
from pathlib import Path
from typing import Any
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper
from onnx import numpy_helper
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
MODEL_PATH = "/nfs5/models/Qwen"
TOKENIZER_DIR = Path(MODEL_PATH) / "speech_tokenizer"
OUTPUT_DIR = Path("/home/lyzhang_ldap/New/onnx/tokenizer")
REF_AUDIO = Path("/home/lyzhang_ldap/New/data/ref_from_mp3_24k_mono.wav")
ONNX_ENCODER_PATH = OUTPUT_DIR / "tokenizer12hz_encode.onnx"
ONNX_DECODER_CHUNK_PATH = OUTPUT_DIR / "tokenizer12hz_decode_chunk.onnx"
ONNX_ENCODER_SAMPLE_LENGTH = 24000
SAMPLE_RATE = 24000
def _register_diff_symbolic() -> None:
    def _diff_symbolic(g, x, n, dim, prepend, append):
        from torch.onnx.symbolic_helper import _get_const
        dim_val = _get_const(dim, "i", "dim")
        axes = g.op("Constant", value_t=torch.tensor([dim_val], dtype=torch.long))
        zero = g.op("Constant", value_t=torch.tensor([0], dtype=torch.long))
        one = g.op("Constant", value_t=torch.tensor([1], dtype=torch.long))
        neg1 = g.op("Constant", value_t=torch.tensor([-1], dtype=torch.long))
        big = g.op("Constant", value_t=torch.tensor([9223372036854775807], dtype=torch.long))

        a = g.op("Slice", x, zero, neg1, axes, one)
        b = g.op("Slice", x, one, big, axes, one)
        diff_result = g.op("Sub", b, a)

        first = g.op("Slice", x, zero, one, axes, one)
        zero_pad = g.op("Sub", first, first)

        return g.op("Concat", zero_pad, diff_result, axis_i=dim_val)

    torch.onnx.register_custom_op_symbolic("aten::diff", _diff_symbolic, 18)


def patch_encoder_dynamic_reshape(onnx_path: str | Path) -> int:
    model = onnx.load(onnx_path)
    patched = 0
    new_nodes = []

    for node in model.graph.node:
        if node.name == "/encoder_transformer/Reshape":
            axes_name = "/encoder_transformer/Unsqueeze_axis1_const_output_0"
            unsqueeze_out = "/encoder_transformer/Range_unsqueeze_axis1_output_0"
            new_nodes.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[axes_name],
                    name="/encoder_transformer/Unsqueeze_axis1_const",
                    value=helper.make_tensor("value", TensorProto.INT64, [1], [1]),
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=["/encoder_transformer/Range_output_0", axes_name],
                    outputs=[unsqueeze_out],
                    name="/encoder_transformer/Range_unsqueeze_axis1",
                )
            )
            patched += 1
            continue

        if node.name == "/encoder_transformer/LessOrEqual":
            if node.input[1] == "/encoder_transformer/Reshape_output_0":
                node.input[1] = "/encoder_transformer/Range_unsqueeze_axis1_output_0"

        new_nodes.append(node)

    if patched:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        onnx.checker.check_model(model)
    onnx.save(model, onnx_path)
    print(f"  Patched encoder dynamic reshape nodes: {patched}")
    return patched


def convert_float_initializers_to_float16(onnx_path: str | Path) -> int:
    model = onnx.load(onnx_path)
    converted = 0

    for initializer in model.graph.initializer:
        if initializer.data_type != TensorProto.FLOAT:
            continue
        array = numpy_helper.to_array(initializer).astype(np.float16)
        initializer.CopyFrom(numpy_helper.from_array(array, name=initializer.name))
        converted += 1

    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
                    converted += 1

        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT:
                array = numpy_helper.to_array(attr.t).astype(np.float16)
                attr.t.CopyFrom(numpy_helper.from_array(array, name=attr.t.name))
                converted += 1

    if converted:
        onnx.checker.check_model(model)
        onnx.save(model, onnx_path)
    print(f"  Converted FLOAT initializers/constants to FLOAT16: {converted}")
    return converted


class TokenizerEncoder(nn.Module):
    def __init__(self, tokenizer_model):
        super().__init__()
        self.model = getattr(tokenizer_model, "model", tokenizer_model)

    def forward(self, audio):
        encoded_frames = self.model.encoder.encode(
            input_values=audio.unsqueeze(1),
            return_dict=True,
        )
        codes = encoded_frames.audio_codes[:, : self.model.encoder_valid_num_quantizers]
        return codes.transpose(1, 2)


def ensure_directory_exists(output_dir: str | Path) -> None:
    os.makedirs(output_dir, exist_ok=True)


def tokenizer_encoder_export(tokenizer_model, output_dir: str | Path, model_dtype):
    ensure_directory_exists(output_dir)
    onnx_path = Path(output_dir) / "tokenizer12hz_encode.onnx"
    wrapper = TokenizerEncoder(tokenizer_model)
    wrapper.eval()
    dummy_audio = torch.randn(1, 24000, device=tokenizer_device(tokenizer_model), dtype=model_dtype)
    torch.onnx.export(
        wrapper,
        (dummy_audio,),
        str(onnx_path),
        input_names=["audio"],
        output_names=["codes"],
        dynamic_axes={"audio": {1: "num_samples"}, "codes": {1: "num_frames"}},
        opset_version=18,
    )
    patch_encoder_dynamic_reshape(onnx_path)
    if model_dtype == torch.float16:
        convert_float_initializers_to_float16(onnx_path)
    return onnx_path


class ChunkDecoderForward(nn.Module):
    def __init__(self, decoder, upsample_rate):
        super().__init__()
        self.decoder = decoder
        self.upsample_rate = upsample_rate

    def forward(self, audio_codes: torch.Tensor, context_frames: torch.Tensor):
        wav = self.decoder(audio_codes.transpose(1, 2))
        audio_values = wav.squeeze(1)

        total_frames = (audio_codes[..., 0] >= 0).sum(dim=1)
        current_frames = total_frames - context_frames
        start_sample = context_frames * self.upsample_rate
        valid_samples = current_frames * self.upsample_rate

        audio_values = audio_values[:, start_sample : start_sample + valid_samples]
        return audio_values, valid_samples.unsqueeze(0)


def _fix_bool_cumsum(onnx_path: Any) -> int:
    onnx_model = onnx.load(onnx_path)
    name_to_node = {o: node for node in onnx_model.graph.node for o in node.output}
    cast_added = 0
    for i, node in enumerate(list(onnx_model.graph.node)):
        if node.op_type == "CumSum":
            data_input = node.input[0]
            src = name_to_node.get(data_input)
            if src and src.op_type in ("Not", "Equal", "Less", "Greater", "And", "Or"):
                cast_name = data_input + "_i64"
                cast_node = onnx.helper.make_node(
                    "Cast", inputs=[data_input], outputs=[cast_name], to=7
                )
                node.input[0] = cast_name
                onnx_model.graph.node.insert(i, cast_node)
                cast_added += 1
    onnx.save(onnx_model, onnx_path)
    return cast_added


def tokenizer_decoder_chunk_export(
    tokenizer_model,
    output_dir: str | Path,
    model_dtype=torch.float32,
    trace_chunk_frames: int = 300,
    trace_context_frames: int = 25,
):
    ensure_directory_exists(output_dir)

    _register_diff_symbolic()
    onnx_path = Path(output_dir) / "tokenizer12hz_decode_chunk.onnx"
    speech_model = tokenizer_model.model
    decode_upsample_rate = speech_model.decode_upsample_rate

    trace_chunk_frames = int(trace_chunk_frames)
    trace_context_frames = int(trace_context_frames)
    trace_total_frames = trace_chunk_frames + trace_context_frames
    if trace_chunk_frames <= 0:
        raise ValueError("trace_chunk_frames must be positive")
    if trace_context_frames < 0:
        raise ValueError("trace_context_frames must be non-negative")

    wrapper = ChunkDecoderForward(speech_model.decoder, decode_upsample_rate)
    wrapper.eval()
    model_device = tokenizer_device(tokenizer_model)
    dummy_codes = torch.randint(0, 1024, (1, trace_total_frames, 16), device=model_device)
    dummy_context = torch.tensor(int(trace_context_frames), dtype=torch.long, device=model_device)

    torch.onnx.export(
        wrapper,
        (dummy_codes, dummy_context),
        str(onnx_path),
        input_names=["audio_codes", "context_frames"],
        output_names=["audio_values", "lengths"],
        dynamic_axes={
            "audio_codes": {1: "codes_length"},
            "audio_values": {1: "audio_length"},
        },
        opset_version=18,
        do_constant_folding=False,
        dynamo=False,
    )
    _fix_bool_cumsum(onnx_path)
    if model_dtype == torch.float16:
        convert_float_initializers_to_float16(onnx_path)
    return onnx_path


def load_tokenizer(device=None, tokenizer_dir: str | Path = TOKENIZER_DIR, dtype=None):
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    kwargs = {"device_map": device}
    if dtype is not None:
        kwargs["dtype"] = dtype
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(tokenizer_dir), **kwargs)
    if dtype is not None:
        tokenizer.model.to(device=device, dtype=dtype)
    tokenizer.model.eval()
    return tokenizer


def load_audio(audio_path: str | Path = REF_AUDIO):
    import soundfile as sf

    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz audio, got {sr} Hz: {audio_path}")
    return np.array(audio)


def fit_audio_length(audio, num_samples):
    if audio.shape[0] >= num_samples:
        return audio[:num_samples]
    return np.pad(audio, (0, num_samples - audio.shape[0]))


def prepare_onnx_audio_input(
    tokenizer,
    audio,
    num_samples=ONNX_ENCODER_SAMPLE_LENGTH,
):
    audio = fit_audio_length(audio, num_samples)
    inputs = tokenizer.feature_extractor(
        raw_audio=[audio],
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )
    input_values = inputs["input_values"].squeeze(1)
    return input_values.cpu().numpy().astype(np.float32)


def run_pytorch_encoder(tokenizer, audio_input):
    model = tokenizer.model
    device = tokenizer_device(tokenizer)
    dtype = getattr(model, "dtype", torch.float32)

    input_values = torch.from_numpy(audio_input).to(device=device, dtype=dtype)
    with torch.inference_mode():
        encoded_frames = model.encoder.encode(
            input_values=input_values.unsqueeze(1),
            return_dict=True,
        )
        codes = encoded_frames.audio_codes[:, : model.encoder_valid_num_quantizers]
        codes = codes.transpose(1, 2)
    return codes.detach().cpu().numpy()


def tokenizer_device(tokenizer):
    device = getattr(tokenizer, "device", None)
    if device is not None:
        return torch.device(device)
    model = getattr(tokenizer, "model", tokenizer)
    return next(model.parameters()).device


def verify_encode(tokenizer=None, audio_path: str | Path = REF_AUDIO):
    tokenizer = tokenizer or load_tokenizer()
    audio = load_audio(audio_path)

    with torch.inference_mode():
        encoded = tokenizer.encode(audio, sr=SAMPLE_RATE, return_dict=True)

    codes = encoded["audio_codes"] if isinstance(encoded, dict) else encoded.audio_codes
    if isinstance(codes, (list, tuple)):
        codes = codes[0]
    codes = codes.detach().cpu().numpy() if torch.is_tensor(codes) else np.asarray(codes)

    assert codes.ndim in (2, 3), codes.shape
    assert codes.shape[-1] == 16, codes.shape
    assert codes.shape[-2] > 0, codes.shape
    print(f"encode ok: codes shape={codes.shape}, dtype={codes.dtype}")
    return encoded


def verify_decode(tokenizer=None, encoded=None):
    tokenizer = tokenizer or load_tokenizer()
    if encoded is None:
        encoded = verify_encode(tokenizer)

    with torch.inference_mode():
        wavs, sr = tokenizer.decode(encoded)

    wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    wav = wav.detach().cpu().numpy() if torch.is_tensor(wav) else np.asarray(wav)

    assert sr == SAMPLE_RATE, sr
    assert wav.size > 0, wav.shape
    assert np.isfinite(wav).all()
    print(f"decode ok: wav shape={wav.shape}, sr={sr}")
    return wavs, sr


def print_code_compare_stats(onnx_codes, pytorch_codes):
    mismatches = onnx_codes != pytorch_codes
    mismatch_count = int(mismatches.sum())
    total = int(mismatches.size)
    match_rate = 1.0 - mismatch_count / total
    abs_diff = np.abs(onnx_codes.astype(np.int64) - pytorch_codes.astype(np.int64))

    print(
        "onnx stats: "
        f"shape={onnx_codes.shape}, dtype={onnx_codes.dtype}, "
        f"min={onnx_codes.min()}, max={onnx_codes.max()}, unique={np.unique(onnx_codes).size}"
    )
    print(
        "pytorch stats: "
        f"shape={pytorch_codes.shape}, dtype={pytorch_codes.dtype}, "
        f"min={pytorch_codes.min()}, max={pytorch_codes.max()}, unique={np.unique(pytorch_codes).size}"
    )
    print(
        "compare stats: "
        f"total={total}, mismatches={mismatch_count}, "
        f"mismatch_rate={mismatch_count / total:.6f}, match_rate={match_rate:.6f}, "
        f"max_abs_diff={abs_diff.max()}, mean_abs_diff={abs_diff.mean():.6f}"
    )

    if mismatch_count:
        first_indices = np.argwhere(mismatches)[:5]
        examples = []
        for index in first_indices:
            index_tuple = tuple(index.tolist())
            examples.append(
                f"{index_tuple}: onnx={onnx_codes[index_tuple]}, pytorch={pytorch_codes[index_tuple]}"
            )
        print("first mismatches: " + "; ".join(examples))

    return mismatch_count


def verify_onnx_encode(
    tokenizer,
    audio_path: str | Path = REF_AUDIO,
    onnx_path: str | Path = ONNX_ENCODER_PATH,
    num_samples=ONNX_ENCODER_SAMPLE_LENGTH,
):
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    audio = load_audio(audio_path)
    audio_input = prepare_onnx_audio_input(tokenizer, audio, num_samples=num_samples)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_codes = session.run([output_name], {input_name: audio_input})[0]
    pytorch_codes = run_pytorch_encoder(tokenizer, audio_input)

    assert onnx_codes.ndim == 3, onnx_codes.shape
    assert onnx_codes.shape[0] == 1, onnx_codes.shape
    assert onnx_codes.shape[-1] == 16, onnx_codes.shape
    assert onnx_codes.shape[1] > 0, onnx_codes.shape
    assert onnx_codes.shape == pytorch_codes.shape, (onnx_codes.shape, pytorch_codes.shape)

    mismatch_count = print_code_compare_stats(onnx_codes, pytorch_codes)
    if mismatch_count:
        mismatches = onnx_codes != pytorch_codes
        first_index = np.argwhere(mismatches)[0]
        first_index_tuple = tuple(first_index.tolist())
        raise AssertionError(
            "ONNX encode mismatch: "
            f"mismatches={mismatches.sum()}, "
            f"first_index={first_index_tuple}, "
            f"onnx={onnx_codes[first_index_tuple]}, "
            f"pytorch={pytorch_codes[first_index_tuple]}"
        )

    print(
        f"onnx encode ok: codes shape={onnx_codes.shape}, "
        f"dtype={onnx_codes.dtype}, matches PyTorch"
    )
    return onnx_codes


def verify_onnx_decode_chunk(
    onnx_path: str | Path = ONNX_DECODER_CHUNK_PATH,
    providers: list[str] | None = None,
    cases: tuple[tuple[int, int], ...] = ((80, 5), (120, 60), (325, 25)),
):
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    providers = providers or ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    rng = np.random.default_rng(20260603)

    for code_len, context_frames in cases:
        audio_codes = rng.integers(0, 1024, size=(1, code_len, 16), dtype=np.int64)
        context = np.array(context_frames, dtype=np.int64)
        audio_values, lengths = session.run(
            ["audio_values", "lengths"],
            {
                "audio_codes": audio_codes,
                "context_frames": context,
            },
        )
        expected_samples = (int(code_len) - int(context_frames)) * 1920
        reported_samples = int(np.asarray(lengths).reshape(-1)[0])
        if reported_samples != expected_samples:
            raise AssertionError(
                "decode chunk length mismatch: "
                f"code_len={code_len}, context_frames={context_frames}, "
                f"reported={reported_samples}, expected={expected_samples}"
            )
        if audio_values.shape != (1, expected_samples):
            raise AssertionError(
                "decode chunk audio shape mismatch: "
                f"code_len={code_len}, context_frames={context_frames}, "
                f"audio_shape={audio_values.shape}, expected={(1, expected_samples)}"
            )
        print(
            "onnx decode chunk ok: "
            f"code_len={code_len}, context_frames={context_frames}, "
            f"audio_shape={audio_values.shape}, lengths={lengths.tolist()}"
        )


def verify_tokenizer_exports(
    tokenizer,
    output_dir: str | Path = OUTPUT_DIR,
    audio_path: str | Path = REF_AUDIO,
):
    output_dir = Path(output_dir)
    encoded = verify_encode(tokenizer, audio_path=audio_path)
    verify_decode(tokenizer, encoded)
    verify_onnx_encode(
        tokenizer,
        audio_path=audio_path,
        onnx_path=output_dir / "tokenizer12hz_encode.onnx",
    )
    verify_onnx_decode_chunk(onnx_path=output_dir / "tokenizer12hz_decode_chunk.onnx")


def export_tokenizer(
    tokenizer=None,
    output_dir: str | Path = OUTPUT_DIR,
    model_dtype=torch.float32,
    trace_chunk_frames: int = 300,
    trace_context_frames: int = 25,
):
    tokenizer = tokenizer or load_tokenizer(device="cpu")
    encoder_path = tokenizer_encoder_export(
        tokenizer_model=tokenizer,
        output_dir=output_dir,
        model_dtype=model_dtype,
    )
    decoder_chunk_path = tokenizer_decoder_chunk_export(
        tokenizer_model=tokenizer,
        output_dir=output_dir,
        model_dtype=model_dtype,
        trace_chunk_frames=trace_chunk_frames,
        trace_context_frames=trace_context_frames,
    )
    return encoder_path, decoder_chunk_path
