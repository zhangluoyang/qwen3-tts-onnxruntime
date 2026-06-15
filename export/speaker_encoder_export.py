import argparse
import os
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper
from transformers import AutoConfig, AutoProcessor
from export.tokenizer_export import ensure_directory_exists
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSProcessor
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
from qwen_tts import Qwen3TTSModel
import torch.nn.functional as F


SPEAKER_ENCODER_ONNX_PATH = Path("./onnx/speaker_encoder/speaker_encoder.onnx")
SPEAKER_ENCODER_ATOL = 1e-4
SPEAKER_ENCODER_RTOL = 1e-4


def _speaker_encoder_output(output):
    """兼容 speaker_encoder 返回 tensor 或 tuple/list 的版本差异。"""
    if isinstance(output, (tuple, list)):
        return output[0]
    return output

class SpeakerEncoderWrapper(nn.Module):

    def __init__(self, speaker_encoder):
        super().__init__()
        self.encoder = speaker_encoder

    def forward(self, mel):
        return _speaker_encoder_output(self.encoder(mel))

def speaker_encoder_export(model, output_dir, device=None): 
    if getattr(model, "speaker_encoder", None) is None:
        model_type = getattr(getattr(model, "config", None), "tts_model_type", "unknown")
        raise ValueError(
            "loaded model does not provide speaker_encoder; "
            f"tts_model_type={model_type!r}. Skip the speaker_encoder component for this model."
        )
    if getattr(getattr(model, "config", None), "speaker_encoder_config", None) is None:
        model_type = getattr(getattr(model, "config", None), "tts_model_type", "unknown")
        raise ValueError(
            "loaded model config does not provide speaker_encoder_config; "
            f"tts_model_type={model_type!r}. Skip the speaker_encoder component for this model."
        )
    if device is not None and not str(device).startswith("cpu"):
        model.speaker_encoder.to(device)
    wrapper = SpeakerEncoderWrapper(speaker_encoder=model.speaker_encoder).eval()
    mel_dim = model.config.speaker_encoder_config.mel_dim
    model_dtype = next(model.speaker_encoder.parameters()).dtype
    model_device = _get_module_device(model.speaker_encoder)
    print(f"  speaker_encoder dtype: {model_dtype}, device: {model_device}")
    trace_frames = 100
    dummy_mel = torch.randn(1, trace_frames, mel_dim, device=model_device, dtype=model_dtype)
    ensure_directory_exists(output_dir=output_dir)
    torch.onnx.export(
        wrapper,
        (dummy_mel,),
        os.path.join(output_dir, "speaker_encoder.onnx"),
        input_names=["mel"],
        output_names=["speaker_embedding"],
        dynamic_axes={"mel": {1: "time"}},
        opset_version=18,
        do_constant_folding=False,
        dynamo=False,
    )


def _get_module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_module_dtype(module):
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _as_numpy(tensor):
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def _to_speaker_encoder_tensor(value, device, dtype):
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _prepare_speaker_encoder_input(model, frames=100, batch_size=1, seed=0):
    mel_dim = model.config.speaker_encoder_config.mel_dim
    dtype = next(model.speaker_encoder.parameters()).dtype
    device = _get_module_device(model.speaker_encoder)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(
        batch_size,
        frames,
        mel_dim,
        dtype=torch.float32,
        generator=generator,
    ).to(device=device, dtype=dtype)


def print_speaker_encoder_compare_stats(onnx_output, pytorch_output):
    diff = onnx_output.astype(np.float64) - pytorch_output.astype(np.float64)
    abs_diff = np.abs(diff)

    print(
        "onnx speaker_encoder stats: "
        f"shape={onnx_output.shape}, dtype={onnx_output.dtype}, "
        f"min={onnx_output.min():.6f}, max={onnx_output.max():.6f}, "
        f"mean={onnx_output.mean():.6f}, std={onnx_output.std():.6f}"
    )
    print(
        "pytorch speaker_encoder stats: "
        f"shape={pytorch_output.shape}, dtype={pytorch_output.dtype}, "
        f"min={pytorch_output.min():.6f}, max={pytorch_output.max():.6f}, "
        f"mean={pytorch_output.mean():.6f}, std={pytorch_output.std():.6f}"
    )
    print(
        "speaker_encoder compare stats: "
        f"max_abs_diff={abs_diff.max():.8f}, "
        f"mean_abs_diff={abs_diff.mean():.8f}, "
        f"rmse={np.sqrt(np.mean(diff * diff)):.8f}"
    )


def verify_onnx_speaker_encoder(
    model,
    onnx_path=SPEAKER_ENCODER_ONNX_PATH,
    mel=None,
    frames=100,
    seed=0,
    atol=SPEAKER_ENCODER_ATOL,
    rtol=SPEAKER_ENCODER_RTOL,
):
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if model.speaker_encoder is None:
        raise ValueError("model.speaker_encoder is None")

    wrapper = SpeakerEncoderWrapper(model.speaker_encoder).eval()
    device = _get_module_device(wrapper)
    dtype = _get_module_dtype(wrapper)

    if mel is None:
        mel_torch = _prepare_speaker_encoder_input(model, frames=frames, seed=seed)
    else:
        mel_torch = _to_speaker_encoder_tensor(mel, device=device, dtype=dtype)

    with torch.inference_mode():
        pytorch_output = wrapper(mel_torch)
    pytorch_output = _as_numpy(pytorch_output)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_output = session.run([output_name], {input_name: _as_numpy(mel_torch)})[0]

    assert onnx_output.shape == pytorch_output.shape, (onnx_output.shape, pytorch_output.shape)
    print_speaker_encoder_compare_stats(onnx_output, pytorch_output)

    if not np.allclose(onnx_output, pytorch_output, atol=atol, rtol=rtol):
        abs_diff = np.abs(onnx_output - pytorch_output)
        first_index = np.argwhere(abs_diff > (atol + rtol * np.abs(pytorch_output)))[0]
        first_index_tuple = tuple(first_index.tolist())
        raise AssertionError(
            "ONNX speaker_encoder mismatch: "
            f"first_index={first_index_tuple}, "
            f"onnx={onnx_output[first_index_tuple]}, "
            f"pytorch={pytorch_output[first_index_tuple]}, "
            f"abs_diff={abs_diff[first_index_tuple]}"
        )

    print(
        "onnx speaker_encoder ok: "
        f"output shape={onnx_output.shape}, dtype={onnx_output.dtype}, "
        f"matches PyTorch with atol={atol}, rtol={rtol}"
    )
    return onnx_output
