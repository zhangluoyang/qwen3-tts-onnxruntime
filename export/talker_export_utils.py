from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper


RESIDUAL_TOP_K = 50
RESIDUAL_TEMPERATURE = 0.9
RESIDUAL_DO_SAMPLE = False


def resolve_talker(model_or_talker: nn.Module) -> nn.Module:
    if hasattr(model_or_talker, "code_predictor") and hasattr(model_or_talker, "model"):
        return model_or_talker

    if hasattr(model_or_talker, "talker"):
        return model_or_talker.talker

    inner_model = getattr(model_or_talker, "model", None)
    if inner_model is not None and hasattr(inner_model, "talker"):
        return inner_model.talker

    raise AttributeError("expected a talker module, model.talker, or model.model.talker")


def get_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def codec_embedding(talker: nn.Module) -> nn.Module:
    if hasattr(talker.model, "embed_tokens"):
        return talker.model.embed_tokens
    if hasattr(talker.model, "codec_embedding"):
        return talker.model.codec_embedding
    raise AttributeError("talker.model has neither embed_tokens nor codec_embedding")


def inline_tensor_to_array(tensor: onnx.TensorProto) -> np.ndarray | None:
    if tensor.data_location == TensorProto.EXTERNAL:
        return None
    return onnx.numpy_helper.to_array(tensor)


def constant_tensor_values(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for initializer in model.graph.initializer:
        value = inline_tensor_to_array(initializer)
        if value is not None:
            values[initializer.name] = value

    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                value = inline_tensor_to_array(attr.t)
                if value is not None:
                    values[node.output[0]] = value
                break
    return values


def patch_dynamic_range_reshape(onnx_path: str | Path) -> int:
    """Patch traced Range->Reshape([trace_len, 1]) masks into dynamic Unsqueeze(axis=1)."""

    model = onnx.load(str(onnx_path), load_external_data=False)
    if any(node.name.endswith("_range_unsqueeze_axis1") for node in model.graph.node):
        print("dynamic range reshape patch skipped: already patched")
        return 0

    constants = constant_tensor_values(model)
    patched = 0
    new_nodes = []
    replaced_outputs: dict[str, str] = {}

    for node in model.graph.node:
        is_range_reshape = (
            node.op_type == "Reshape"
            and len(node.input) >= 2
            and "Range" in node.input[0]
        )
        shape_value = constants.get(node.input[1]) if len(node.input) >= 2 else None
        if (
            is_range_reshape
            and shape_value is not None
            and shape_value.shape == (2,)
            and int(shape_value[1]) == 1
        ):
            node_prefix = node.name or f"/model/RangeReshapePatch_{patched}"
            axes_name = f"{node_prefix}_unsqueeze_axis1_const_output_0"
            unsqueeze_out = f"{node_prefix}_range_unsqueeze_axis1_output_0"
            new_nodes.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[axes_name],
                    name=f"{node_prefix}_unsqueeze_axis1_const",
                    value=helper.make_tensor("value", TensorProto.INT64, [1], [1]),
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[node.input[0], axes_name],
                    outputs=[unsqueeze_out],
                    name=f"{node_prefix}_range_unsqueeze_axis1",
                )
            )
            replaced_outputs[node.output[0]] = unsqueeze_out
            patched += 1
            continue

        for input_index, input_name in enumerate(node.input):
            if input_name in replaced_outputs:
                node.input[input_index] = replaced_outputs[input_name]
        new_nodes.append(node)

    if patched:
        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        onnx.save(model, str(onnx_path))
        print(f"patched dynamic range reshape nodes: {patched}")
    else:
        print("dynamic range reshape patch skipped: no Range->Reshape([N,1]) node found")
    return patched
