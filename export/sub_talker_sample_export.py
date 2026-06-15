from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from export.talker_export_utils import (
    RESIDUAL_DO_SAMPLE,
    RESIDUAL_TEMPERATURE,
    RESIDUAL_TOP_K,
    codec_embedding,
    get_module_device,
    get_module_dtype,
    resolve_talker,
)
from export.tokenizer_export import ensure_directory_exists


SUB_TALKER_SAMPLE_ATOL = 5.0e-3
SUB_TALKER_SAMPLE_RTOL = 5.0e-3
SUB_TALKER_SAMPLE_OPSET = 18
SUB_TALKER_SAMPLE_ONNX_FILENAME = "sub_talker_sample.onnx"
SUB_TALKER_SAMPLE_EXTERNAL_DATA_FILENAME = f"{SUB_TALKER_SAMPLE_ONNX_FILENAME}.data"
SUB_TALKER_SAMPLE_SEED = 20260605


class FixedResidualSampler(nn.Module):
    def __init__(
        self,
        top_k: int = RESIDUAL_TOP_K,
        temperature: float = RESIDUAL_TEMPERATURE,
        do_sample: bool = RESIDUAL_DO_SAMPLE,
        filter_value: float = -1.0e9,
    ) -> None:
        super().__init__()
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.do_sample = bool(do_sample)
        self.filter_value = float(filter_value)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        scores = logits[:, -1, :].float()
        if not self.do_sample:
            return torch.topk(scores, k=1, dim=-1).indices.squeeze(-1)

        scores = scores / self.temperature
        k = min(self.top_k, scores.shape[-1])
        threshold = torch.topk(scores, k=k, dim=-1).values[:, -1:]
        filtered = torch.where(scores < threshold, scores.new_full((), self.filter_value), scores)
        probs = F.softmax(filtered, dim=-1)
        random_u = torch.rand_like(probs[:, :1])
        cdf = torch.cumsum(probs, dim=-1)
        return torch.sum((cdf < random_u).to(torch.int64), dim=-1)


class SubTalkerSample(nn.Module):
    """Generate residual codebooks and the one-frame talker input embedding."""

    def __init__(
        self,
        talker: nn.Module,
        residual_top_k: int = RESIDUAL_TOP_K,
        residual_temperature: float = RESIDUAL_TEMPERATURE,
        residual_do_sample: bool = RESIDUAL_DO_SAMPLE,
    ) -> None:
        super().__init__()
        self.talker = resolve_talker(talker)
        self.codec_embed = codec_embedding(self.talker)
        self.code_predictor = self.talker.code_predictor
        self.residual_embeds = nn.ModuleList(list(self.code_predictor.get_input_embeddings()))
        self.residual_sampler = FixedResidualSampler(
            top_k=residual_top_k,
            temperature=residual_temperature,
            do_sample=residual_do_sample,
        )
        self.num_code_groups = int(
            getattr(
                self.code_predictor.config,
                "num_code_groups",
                len(self.residual_embeds) + 1,
            )
        )
        expected_residuals = self.num_code_groups - 1
        if expected_residuals != len(self.residual_embeds):
            raise ValueError(
                "num_code_groups does not match code_predictor residual embeddings: "
                f"{self.num_code_groups=} residual_embeds={len(self.residual_embeds)}"
            )

    def _code_predictor_logits(self, context: torch.Tensor, gen_step: int):
        hidden_in = self.code_predictor.small_to_mtp_projection(context)
        out = self.code_predictor.model(
            input_ids=None,
            inputs_embeds=hidden_in,
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state[:, -1:, :]
        return self.code_predictor.lm_head[gen_step](hidden)

    def _sample_and_embed_residual(
        self,
        logits: torch.Tensor,
        gen_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual_token = self.residual_sampler(logits).reshape(logits.shape[0], 1)
        residual_embed = self.residual_embeds[gen_step](residual_token)
        return residual_token, residual_embed

    def forward(
        self,
        first_token: torch.Tensor,
        last_hidden: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first_token = first_token.to(torch.long).reshape(1, 1)
        main_embed = self.codec_embed(first_token)
        context = torch.cat([last_hidden.to(main_embed.dtype), main_embed], dim=1)

        token_columns = [first_token]
        frame_parts = [main_embed]
        for gen_step in range(self.num_code_groups - 1):
            logits = self._code_predictor_logits(context, gen_step)
            residual_token, residual_embed = self._sample_and_embed_residual(logits, gen_step)
            residual_embed = residual_embed.to(context.dtype)
            token_columns.append(residual_token)
            frame_parts.append(residual_embed)
            context = torch.cat([context, residual_embed], dim=1)

        codebook_tokens = torch.cat(token_columns, dim=1)
        all_embeds = torch.cat(frame_parts, dim=1)
        frame_embed = torch.sum(all_embeds, dim=1, keepdim=True)
        decode_embed = frame_embed + text_embed.to(frame_embed.dtype)
        return codebook_tokens, frame_embed, decode_embed


def _prepare_sub_talker_sample_inputs(
    talker: nn.Module,
    batch_size: int = 1,
    seed: int = SUB_TALKER_SAMPLE_SEED,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    talker = resolve_talker(talker)
    device = get_module_device(talker)
    dtype = get_module_dtype(talker)
    hidden_size = int(talker.config.hidden_size)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    first_token = torch.full((batch_size,), 7, dtype=torch.long, device=device)
    last_hidden = torch.randn(
        batch_size,
        1,
        hidden_size,
        dtype=torch.float32,
        generator=generator,
    ).to(device=device, dtype=dtype)
    text_embed = torch.randn(
        batch_size,
        1,
        hidden_size,
        dtype=torch.float32,
        generator=generator,
    ).to(device=device, dtype=dtype)
    return first_token, last_hidden, text_embed


def _sub_talker_sample_io_names() -> tuple[list[str], list[str]]:
    return (
        ["first_token", "last_hidden", "text_embed"],
        ["codebook_tokens", "frame_embed", "decode_embed"],
    )


def _sub_talker_sample_dynamic_axes() -> dict[str, dict[int, str]]:
    return {
        "first_token": {0: "batch"},
        "last_hidden": {0: "batch"},
        "text_embed": {0: "batch"},
        "codebook_tokens": {0: "batch"},
        "frame_embed": {0: "batch"},
        "decode_embed": {0: "batch"},
    }


def _save_onnx_with_single_external_data(
    staged_onnx_path: Path,
    output_path: Path,
    data_file_name: str = SUB_TALKER_SAMPLE_EXTERNAL_DATA_FILENAME,
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


def _torch_export_sub_talker_sample(
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
        opset_version=SUB_TALKER_SAMPLE_OPSET,
        do_constant_folding=False,
        dynamo=False,
        external_data=external_data,
    )


def sub_talker_sample_export(
    talker: nn.Module,
    output_dir: str | Path,
    external_data: bool = True,
    merge_external_data: bool = True,
    residual_do_sample: bool = RESIDUAL_DO_SAMPLE,
    residual_top_k: int = RESIDUAL_TOP_K,
    residual_temperature: float = RESIDUAL_TEMPERATURE,
) -> Path:
    ensure_directory_exists(output_dir=output_dir)
    wrapper = SubTalkerSample(
        talker,
        residual_top_k=residual_top_k,
        residual_temperature=residual_temperature,
        residual_do_sample=residual_do_sample,
    ).eval()
    dummy_inputs = _prepare_sub_talker_sample_inputs(talker)
    input_names, output_names = _sub_talker_sample_io_names()
    dynamic_axes = _sub_talker_sample_dynamic_axes()
    output_path = Path(output_dir) / SUB_TALKER_SAMPLE_ONNX_FILENAME

    if external_data and merge_external_data:
        with tempfile.TemporaryDirectory(prefix="sub_talker_sample_onnx_") as tmp_dir:
            staged_path = Path(tmp_dir) / SUB_TALKER_SAMPLE_ONNX_FILENAME
            _torch_export_sub_talker_sample(
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
        _torch_export_sub_talker_sample(
            wrapper,
            dummy_inputs,
            input_names,
            output_names,
            dynamic_axes,
            output_path,
            external_data=external_data,
        )
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


def verify_onnx_sub_talker_sample(
    talker: nn.Module,
    onnx_path: Path | str,
    inputs: tuple[torch.Tensor, ...] | None = None,
    batch_size: int = 1,
    seed: int = SUB_TALKER_SAMPLE_SEED,
    atol: float = SUB_TALKER_SAMPLE_ATOL,
    rtol: float = SUB_TALKER_SAMPLE_RTOL,
    residual_do_sample: bool = RESIDUAL_DO_SAMPLE,
    residual_top_k: int = RESIDUAL_TOP_K,
    residual_temperature: float = RESIDUAL_TEMPERATURE,
    providers: list[str] | None = None,
) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    talker = resolve_talker(talker)
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    wrapper = SubTalkerSample(
        talker,
        residual_top_k=residual_top_k,
        residual_temperature=residual_temperature,
        residual_do_sample=residual_do_sample,
    ).eval()
    if inputs is None:
        inputs = _prepare_sub_talker_sample_inputs(talker, batch_size=batch_size, seed=seed)

    input_names, output_names = _sub_talker_sample_io_names()
    with torch.inference_mode():
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

    first_token = _as_numpy(inputs[0]).reshape(-1)
    codebook_tokens = np.asarray(onnx_outputs[0])
    first_ok = np.array_equal(codebook_tokens[:, 0], first_token)
    residual_range_ok = bool(
        ((0 <= codebook_tokens[:, 1:]) & (codebook_tokens[:, 1:] < int(talker.code_predictor.config.vocab_size))).all()
    )
    if not (first_ok and residual_range_ok):
        raise AssertionError(
            "sub_talker_sample codebook token validity failed: "
            f"first_ok={first_ok}, residual_range_ok={residual_range_ok}"
        )

    if not residual_do_sample:
        for name, onnx_output, pytorch_output in zip(output_names, onnx_outputs, pytorch_outputs_np):
            if onnx_output.dtype.kind in "iu":
                if not np.array_equal(onnx_output, pytorch_output):
                    raise AssertionError(f"ONNX sub_talker_sample output mismatch: {name}")
            elif not np.allclose(onnx_output, pytorch_output, atol=atol, rtol=rtol):
                raise AssertionError(f"ONNX sub_talker_sample output mismatch: {name}")

    print(
        "onnx sub_talker_sample ok: "
        f"outputs={len(onnx_outputs)}, residual_do_sample={residual_do_sample}, "
        f"first_token={first_ok}, residual_range={residual_range_ok}, providers={providers}"
    )
    return dict(zip(output_names, onnx_outputs))
