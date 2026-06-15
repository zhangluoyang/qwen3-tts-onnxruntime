import os
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

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


TEXT_PROJECT_ONNX_PATH = Path("./onnx/text_project/text_project.onnx")
TEXT_PROJECT_ATOL = 1e-4
TEXT_PROJECT_RTOL = 1e-4


class TextProject(nn.Module):
    def __init__(self, talker):
        super().__init__()
        self.text_embed = talker.model.text_embedding
        self.text_projection = talker.text_projection

    def forward(self, input_ids):
        return self.text_projection(self.text_embed(input_ids))

def export_text_project(talker, output_dir):
    ensure_directory_exists(output_dir=output_dir)
    wrapper = TextProject(talker).eval()
    dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device=talker.device)
    torch.onnx.export(wrapper,(dummy_input_ids,),
        os.path.join(output_dir, "text_project.onnx"),
        input_names=["input_ids"],
        output_names=["text_embed"],
        dynamic_axes={"input_ids": {1: "seq_len"}},
        opset_version=18,
        dynamo=True,
        external_data=True)


def _get_module_device(module):
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def print_text_project_compare_stats(onnx_output, pytorch_output):
    diff = onnx_output.astype(np.float64) - pytorch_output.astype(np.float64)
    abs_diff = np.abs(diff)

    print(
        "onnx text_project stats: "
        f"shape={onnx_output.shape}, dtype={onnx_output.dtype}, "
        f"min={onnx_output.min():.6f}, max={onnx_output.max():.6f}, "
        f"mean={onnx_output.mean():.6f}, std={onnx_output.std():.6f}"
    )
    print(
        "pytorch text_project stats: "
        f"shape={pytorch_output.shape}, dtype={pytorch_output.dtype}, "
        f"min={pytorch_output.min():.6f}, max={pytorch_output.max():.6f}, "
        f"mean={pytorch_output.mean():.6f}, std={pytorch_output.std():.6f}"
    )
    print(
        "text_project compare stats: "
        f"max_abs_diff={abs_diff.max():.8f}, "
        f"mean_abs_diff={abs_diff.mean():.8f}, "
        f"rmse={np.sqrt(np.mean(diff * diff)):.8f}"
    )


def verify_onnx_text_project(
    talker,
    onnx_path=TEXT_PROJECT_ONNX_PATH,
    input_ids=None,
    atol=TEXT_PROJECT_ATOL,
    rtol=TEXT_PROJECT_RTOL,
):
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if input_ids is None:
        input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
    else:
        input_ids = np.asarray(input_ids, dtype=np.int64)

    wrapper = TextProject(talker).eval()
    input_ids_torch = torch.from_numpy(input_ids).to(_get_module_device(wrapper))
    with torch.inference_mode():
        pytorch_output = wrapper(input_ids_torch).detach().cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    onnx_output = session.run([output_name], {input_name: input_ids})[0]

    assert onnx_output.shape == pytorch_output.shape, (onnx_output.shape, pytorch_output.shape)
    print_text_project_compare_stats(onnx_output, pytorch_output)

    if not np.allclose(onnx_output, pytorch_output, atol=atol, rtol=rtol):
        abs_diff = np.abs(onnx_output - pytorch_output)
        first_index = np.argwhere(abs_diff > (atol + rtol * np.abs(pytorch_output)))[0]
        first_index_tuple = tuple(first_index.tolist())
        raise AssertionError(
            "ONNX text_project mismatch: "
            f"first_index={first_index_tuple}, "
            f"onnx={onnx_output[first_index_tuple]}, "
            f"pytorch={pytorch_output[first_index_tuple]}, "
            f"abs_diff={abs_diff[first_index_tuple]}"
        )

    print(
        "onnx text_project ok: "
        f"output shape={onnx_output.shape}, dtype={onnx_output.dtype}, "
        f"matches PyTorch with atol={atol}, rtol={rtol}"
    )
    return onnx_output
