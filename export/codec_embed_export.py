import argparse
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoProcessor
from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSProcessor
from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import (Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor)


def ensure_directory_exists(output_dir: str | Path) -> None:
    os.makedirs(output_dir, exist_ok=True)

def _text_embedding(talker):
    """兼容不同版本 talker 文本 embedding 层命名。"""
    if hasattr(talker.model, "text_embed_tokens"):
        return talker.model.text_embed_tokens
    if hasattr(talker.model, "text_embedding"):
        return talker.model.text_embedding
    raise AttributeError("talker.model has neither text_embed_tokens nor text_embedding")


def _codec_embedding(talker):
    """兼容不同版本 talker codec embedding 层命名。"""
    if hasattr(talker.model, "embed_tokens"):
        return talker.model.embed_tokens
    if hasattr(talker.model, "codec_embedding"):
        return talker.model.codec_embedding
    raise AttributeError("talker.model has neither embed_tokens nor codec_embedding")

class CodecEmbed(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.embed_tokens = _codec_embedding(talker)
        self.residual_embeds = nn.ModuleList(
            list(talker.code_predictor.get_input_embeddings())
        )

    def forward(self, token_ids, ref_code):
        token_embed = self.embed_tokens(token_ids)
        embed_dtype = token_embed.dtype

        ref_code_parts = [self.embed_tokens(ref_code[..., 0])]
        for index, embed in enumerate(self.residual_embeds):
            ref_code_parts.append(embed(ref_code[..., index + 1]).to(embed_dtype))
        ref_code_embed = torch.stack(ref_code_parts, dim=0).sum(dim=0).to(embed_dtype)
        return token_embed, ref_code_embed

def codec_embed_export(talker, output_dir):
    ensure_directory_exists(output_dir=output_dir)
    wrapper = CodecEmbed(talker).eval()
    model_device = next(talker.parameters()).device
    dummy_ids = torch.tensor([[100, 101, 102, 103, 104]], dtype=torch.long, device=model_device)
    dummy_ref_code = torch.randint(0, 1024, (1, 30, 16), dtype=torch.long, device=model_device)
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_ref_code),
        os.path.join(output_dir, "codec_embed.onnx"),
        input_names=["token_ids", "ref_code"],
        output_names=["embed", "ref_code_embed"],
        dynamic_shapes={
            "token_ids": {1: torch.export.Dim("seq_len")},
            "ref_code": {1: torch.export.Dim("num_frames")},
        },
        opset_version=18,
        dynamo=True,
        external_data=True)


def verify_onnx_codec_embed(
    talker,
    onnx_path: str | Path = Path("./onnx/codec_embed/codec_embed.onnx"),
    token_ids=None,
    ref_code=None,
    atol=1e-4,
    rtol=1e-4,
):
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    wrapper = CodecEmbed(talker).eval()
    model_device = next(talker.parameters()).device
    if token_ids is None:
        token_ids = np.array([[100, 101, 102, 103, 104]], dtype=np.int64)
    if ref_code is None:
        rng = np.random.default_rng(20260603)
        ref_code = rng.integers(0, 1024, size=(1, 30, 16), dtype=np.int64)

    token_ids_torch = torch.from_numpy(np.asarray(token_ids, dtype=np.int64)).to(model_device)
    ref_code_torch = torch.from_numpy(np.asarray(ref_code, dtype=np.int64)).to(model_device)
    with torch.inference_mode():
        torch_embed, torch_ref_code_embed = wrapper(token_ids_torch, ref_code_torch)
    torch_outputs = [
        torch_embed.detach().float().cpu().numpy(),
        torch_ref_code_embed.detach().float().cpu().numpy(),
    ]

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {
        "token_ids": np.asarray(token_ids, dtype=np.int64),
        "ref_code": np.asarray(ref_code, dtype=np.int64),
    }
    onnx_outputs = session.run(None, feed)

    names = ["embed", "ref_code_embed"]
    for name, onnx_output, torch_output in zip(names, onnx_outputs, torch_outputs):
        if onnx_output.shape != torch_output.shape:
            raise AssertionError(f"{name} shape mismatch: onnx={onnx_output.shape}, torch={torch_output.shape}")
        diff = onnx_output.astype(np.float64) - torch_output.astype(np.float64)
        abs_diff = np.abs(diff)
        print(
            f"onnx codec_embed {name}: shape={onnx_output.shape}, "
            f"max_abs_diff={abs_diff.max():.8f}, mean_abs_diff={abs_diff.mean():.8f}"
        )
        if not np.allclose(onnx_output, torch_output, atol=atol, rtol=rtol):
            index = tuple(np.unravel_index(np.argmax(abs_diff), abs_diff.shape))
            raise AssertionError(
                f"ONNX codec_embed mismatch for {name}: index={index}, "
                f"onnx={onnx_output[index]}, torch={torch_output[index]}"
            )

    print(f"onnx codec_embed ok: {onnx_path}")
    return onnx_outputs
