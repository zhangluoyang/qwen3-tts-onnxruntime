from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper
from transformers.cache_utils import DynamicCache

from export.talker_export_utils import constant_tensor_values, patch_dynamic_range_reshape
from export.tokenizer_export import ensure_directory_exists


TALKER_CORE_ATOL = 5.0e-3
TALKER_CORE_RTOL = 5.0e-3
TALKER_CORE_OPSET = 18
TALKER_CORE_ONNX_FILENAME = "talker_core.onnx"
TALKER_CORE_EXTERNAL_DATA_FILENAME = f"{TALKER_CORE_ONNX_FILENAME}.data"
TALKER_CORE_TRACE_PAST_LEN = 8
TALKER_CORE_TRACE_SEQ_LEN = 1
TALKER_CORE_SEED = 20260605


def _batch1_shape_optimized_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from qwen_tts.core.models import modeling_qwen3_tts as qwen3_tts_modeling

    seq_len = hidden_states.shape[1]
    num_heads = int(self.config.num_attention_heads)
    num_kv_heads = int(self.config.num_key_value_heads)
    head_dim = int(self.head_dim)
    hidden_size = int(self.config.hidden_size)

    query_states = self.q_proj(hidden_states).view(1, seq_len, num_heads, head_dim)
    query_states = self.q_norm(query_states).transpose(1, 2)

    key_states = self.k_proj(hidden_states).view(1, seq_len, num_kv_heads, head_dim)
    key_states = self.k_norm(key_states).transpose(1, 2)

    value_states = self.v_proj(hidden_states).view(1, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = qwen3_tts_modeling.apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        self.rope_scaling["mrope_section"],
        self.rope_scaling["interleaved"],
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = qwen3_tts_modeling.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = qwen3_tts_modeling.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(1, seq_len, hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@contextmanager
def patch_talker_attention_batch1_shapes(enabled: bool = True):
    if not enabled:
        yield
        return
    from qwen_tts.core.models import modeling_qwen3_tts as qwen3_tts_modeling

    attention_cls = qwen3_tts_modeling.Qwen3TTSTalkerAttention
    original_forward = attention_cls.forward
    attention_cls.forward = _batch1_shape_optimized_attention_forward
    try:
        print("patched Qwen3TTSTalkerAttention forward: fixed batch/head/hidden shapes, dynamic seq_len")
        yield
    finally:
        attention_cls.forward = original_forward


def _get_talker_head(talker: nn.Module) -> nn.Module:
    if hasattr(talker, "lm_head"):
        return talker.lm_head
    if hasattr(talker, "codec_head"):
        return talker.codec_head
    raise AttributeError("talker has neither lm_head nor codec_head")


def _flatten_cache(cache: Any, num_layers: int) -> tuple[torch.Tensor, ...]:
    if cache is None:
        raise ValueError("past_key_values is None; call the model with use_cache=True.")
    if hasattr(cache, "layers"):
        return tuple(
            tensor
            for i in range(num_layers)
            for tensor in (cache.layers[i].keys, cache.layers[i].values)
        )
    return tuple(tensor for layer in cache for tensor in layer)


def _legacy_cache_from_flat(past_kv_flat: tuple[torch.Tensor, ...], num_layers: int) -> DynamicCache:
    if len(past_kv_flat) != 2 * num_layers:
        raise ValueError(f"expected {2 * num_layers} KV tensors, got {len(past_kv_flat)}")
    legacy_cache = tuple(
        (past_kv_flat[2 * i], past_kv_flat[2 * i + 1])
        for i in range(num_layers)
    )
    return DynamicCache.from_legacy_cache(legacy_cache)


class TalkerCore(nn.Module):
    """Shared talker transformer step used by both prompt prefill and frame decode."""

    def __init__(self, talker: nn.Module) -> None:
        super().__init__()
        self.talker = talker
        self.head = _get_talker_head(talker)
        self.num_layers = int(talker.config.num_hidden_layers)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        *past_kv_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        past_key_values = _legacy_cache_from_flat(past_kv_flat, self.num_layers)
        out = self.talker.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
            return_dict=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        logits = self.head(hidden)
        valid_cache_len = cache_position[-1] + 1
        new_kv = tuple(
            tensor[:, :, :valid_cache_len, :]
            for tensor in _flatten_cache(out.past_key_values, self.num_layers)
        )
        return (logits, hidden) + new_kv


def _get_module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _get_module_dtype(module: nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _prepare_talker_core_inputs(
    talker: nn.Module,
    past_len: int = TALKER_CORE_TRACE_PAST_LEN,
    seq_len: int = TALKER_CORE_TRACE_SEQ_LEN,
    batch_size: int = 1,
    seed: int = TALKER_CORE_SEED,
) -> tuple[torch.Tensor, ...]:
    device = _get_module_device(talker)
    dtype = _get_module_dtype(talker)
    hidden_size = int(talker.config.hidden_size)
    num_layers = int(talker.config.num_hidden_layers)
    num_kv_heads = int(getattr(talker.config, "num_key_value_heads", talker.config.num_attention_heads))
    head_dim = int(getattr(talker.config, "head_dim", hidden_size // int(talker.config.num_attention_heads)))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inputs_embeds = torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        dtype=torch.float32,
        generator=generator,
    ).to(device=device, dtype=dtype)
    attention_mask = torch.ones(batch_size, past_len + seq_len, dtype=torch.long, device=device)
    cache_position = torch.arange(past_len, past_len + seq_len, dtype=torch.long, device=device)
    past_kv = []
    for _ in range(num_layers):
        key = torch.randn(
            batch_size,
            num_kv_heads,
            past_len,
            head_dim,
            dtype=torch.float32,
            generator=generator,
        ).to(device=device, dtype=dtype)
        value = torch.randn(
            batch_size,
            num_kv_heads,
            past_len,
            head_dim,
            dtype=torch.float32,
            generator=generator,
        ).to(device=device, dtype=dtype)
        past_kv.extend([key, value])
    return (inputs_embeds, attention_mask, cache_position, *past_kv)


def _talker_core_io_names(talker: nn.Module) -> tuple[list[str], list[str]]:
    num_layers = int(talker.config.num_hidden_layers)
    input_names = ["inputs_embeds", "attention_mask", "cache_position"]
    output_names = ["logits", "last_hidden"]
    for i in range(num_layers):
        input_names += [f"past_key_{i}", f"past_value_{i}"]
        output_names += [f"new_past_key_{i}", f"new_past_value_{i}"]
    return input_names, output_names


def _talker_core_dynamic_axes(talker: nn.Module) -> dict[str, dict[int, str]]:
    num_layers = int(talker.config.num_hidden_layers)
    dynamic_axes = {
        "inputs_embeds": {1: "seq_len"},
        "attention_mask": {1: "full_len"},
        "cache_position": {0: "seq_len"},
    }
    for i in range(num_layers):
        dynamic_axes[f"past_key_{i}"] = {2: "past_len"}
        dynamic_axes[f"past_value_{i}"] = {2: "past_len"}
        dynamic_axes[f"new_past_key_{i}"] = {2: "new_len"}
        dynamic_axes[f"new_past_value_{i}"] = {2: "new_len"}
    return dynamic_axes


def _save_onnx_with_single_external_data(
    staged_onnx_path: Path,
    output_path: Path,
    data_file_name: str = TALKER_CORE_EXTERNAL_DATA_FILENAME,
) -> None:
    import onnx

    model = onnx.load(str(staged_onnx_path), load_external_data=True)
    data_path = output_path.with_name(data_file_name)
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_file_name,
        size_threshold=0,
        convert_attribute=False,
    )


def patch_cache_position_dynamic_reshape(onnx_path: str | Path) -> int:
    """Patch traced cache_position.reshape([1, 1]) into dynamic Unsqueeze(axis=1)."""
    model = onnx.load(str(onnx_path), load_external_data=False)
    if any(node.name.endswith("_cache_position_unsqueeze_axis1") for node in model.graph.node):
        print("talker_core cache_position reshape patch skipped: already patched")
        return 0

    constants = constant_tensor_values(model)
    patched = 0
    new_nodes = []
    replaced_outputs: dict[str, str] = {}

    for node in model.graph.node:
        shape_value = constants.get(node.input[1]) if node.op_type == "Reshape" and len(node.input) >= 2 else None
        if (
            node.op_type == "Reshape"
            and len(node.input) >= 2
            and node.input[0] == "cache_position"
            and shape_value is not None
            and shape_value.shape == (2,)
            and int(shape_value[0]) == 1
            and int(shape_value[1]) == 1
        ):
            node_prefix = node.name or f"/model/CachePositionReshapePatch_{patched}"
            axes_name = f"{node_prefix}_cache_position_unsqueeze_axis1_const_output_0"
            unsqueeze_out = f"{node_prefix}_cache_position_unsqueeze_axis1_output_0"
            new_nodes.append(
                helper.make_node(
                    "Constant",
                    inputs=[],
                    outputs=[axes_name],
                    name=f"{node_prefix}_cache_position_unsqueeze_axis1_const",
                    value=helper.make_tensor("value", TensorProto.INT64, [1], [1]),
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Unsqueeze",
                    inputs=[node.input[0], axes_name],
                    outputs=[unsqueeze_out],
                    name=f"{node_prefix}_cache_position_unsqueeze_axis1",
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
        print(f"patched talker_core cache_position reshape nodes: {patched}")
    else:
        print("talker_core cache_position reshape patch skipped: no cache_position Reshape([1,1]) node found")
    return patched


def _torch_export_talker_core(
    wrapper: nn.Module,
    dummy_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    output_path: Path,
    external_data: bool,
) -> None:
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=TALKER_CORE_OPSET,
        do_constant_folding=False,
        dynamo=False,
        external_data=external_data,
    )


def talker_core_export(
    talker: nn.Module,
    output_dir: str | Path,
    external_data: bool = True,
    merge_external_data: bool = True,
    optimize_batch1_shapes: bool = False,
) -> Path:
    ensure_directory_exists(output_dir=output_dir)
    wrapper = TalkerCore(talker).eval()
    dummy_inputs = _prepare_talker_core_inputs(talker)
    input_names, output_names = _talker_core_io_names(talker)
    dynamic_axes = _talker_core_dynamic_axes(talker)
    output_path = Path(output_dir) / TALKER_CORE_ONNX_FILENAME

    with patch_talker_attention_batch1_shapes(optimize_batch1_shapes):
        if external_data and merge_external_data:
            with tempfile.TemporaryDirectory(prefix="talker_core_onnx_") as tmp_dir:
                staged_path = Path(tmp_dir) / TALKER_CORE_ONNX_FILENAME
                _torch_export_talker_core(
                    wrapper,
                    dummy_inputs,
                    input_names,
                    output_names,
                    dynamic_axes,
                    staged_path,
                    external_data=True,
                )
                _save_onnx_with_single_external_data(staged_path, output_path)
        else:
            _torch_export_talker_core(
                wrapper,
                dummy_inputs,
                input_names,
                output_names,
                dynamic_axes,
                output_path,
                external_data=external_data,
            )
    patch_dynamic_range_reshape(output_path)
    patch_cache_position_dynamic_reshape(output_path)
    return output_path


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    return tensor.detach().cpu().numpy()


def _to_numpy_feed(input_names: list[str], tensors: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    feed = {}
    for name, tensor in zip(input_names, tensors):
        array = _as_numpy(tensor)
        if tensor.dtype == torch.long:
            array = array.astype(np.int64, copy=False)
        feed[name] = array
    return feed


def _print_compare_stats(
    output_names: list[str],
    onnx_outputs: list[np.ndarray],
    pytorch_outputs: list[np.ndarray],
) -> None:
    max_abs_diff = 0.0
    worst_name = None
    for name, onnx_output, pytorch_output in zip(output_names, onnx_outputs, pytorch_outputs):
        diff = onnx_output.astype(np.float64) - pytorch_output.astype(np.float64)
        current_max = float(np.abs(diff).max()) if diff.size else 0.0
        if current_max >= max_abs_diff:
            max_abs_diff = current_max
            worst_name = name
        if name in ("logits", "last_hidden"):
            print(
                f"{name} compare stats: shape={onnx_output.shape}, "
                f"max_abs_diff={current_max:.8f}, mean_abs_diff={np.abs(diff).mean():.8f}"
            )
    print(f"talker_core compare stats: outputs={len(output_names)}, max_abs_diff={max_abs_diff:.8f}, worst={worst_name}")


def verify_onnx_talker_core(
    talker: nn.Module,
    onnx_path: Path | str,
    inputs: tuple[torch.Tensor, ...] | None = None,
    past_len: int = TALKER_CORE_TRACE_PAST_LEN,
    seq_len: int = TALKER_CORE_TRACE_SEQ_LEN,
    batch_size: int = 1,
    seed: int = TALKER_CORE_SEED,
    atol: float = TALKER_CORE_ATOL,
    rtol: float = TALKER_CORE_RTOL,
    providers: list[str] | None = None,
    optimize_batch1_shapes: bool = False,
) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    wrapper = TalkerCore(talker).eval()
    if inputs is None:
        inputs = _prepare_talker_core_inputs(
            talker,
            past_len=past_len,
            seq_len=seq_len,
            batch_size=batch_size,
            seed=seed,
        )

    input_names, output_names = _talker_core_io_names(talker)
    with patch_talker_attention_batch1_shapes(optimize_batch1_shapes), torch.inference_mode():
        pytorch_outputs = wrapper(*inputs)
    pytorch_outputs_np = [_as_numpy(output) for output in pytorch_outputs]

    if providers is None:
        available = ort.get_available_providers()
        providers = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    onnx_outputs = session.run(output_names, _to_numpy_feed(input_names, inputs))

    for name, onnx_output, pytorch_output in zip(output_names, onnx_outputs, pytorch_outputs_np):
        if onnx_output.shape != pytorch_output.shape:
            raise AssertionError(f"{name} shape mismatch: onnx={onnx_output.shape}, pytorch={pytorch_output.shape}")

    _print_compare_stats(output_names, onnx_outputs, pytorch_outputs_np)
    for name, onnx_output, pytorch_output in zip(output_names, onnx_outputs, pytorch_outputs_np):
        if not np.allclose(onnx_output, pytorch_output, atol=atol, rtol=rtol):
            raise AssertionError(f"ONNX talker_core output mismatch: {name}")
    print(
        "onnx talker_core ok: "
        f"past_len={past_len}, seq_len={seq_len}, outputs={len(output_names)}, providers={providers}"
    )
    return dict(zip(output_names, onnx_outputs))
