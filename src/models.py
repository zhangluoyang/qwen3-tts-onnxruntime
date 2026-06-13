from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
import time
import numpy as np

from src.OnnxSessionRunner import ORT_INPUT_DTYPES, OnnxSessionRunner, is_ortvalue

from src.builders import (
    DEFAULT_MODEL_PATH,
    DEFAULT_ONNX_DIR,
    BaseVoiceClonePromptBuilder,
    CustomVoicePromptBuilder,
    PromptInputs,
    VoiceClonePromptItem,
    normalize_compute_dtype,
)
from src.sampling import apply_repetition_penalty, sample_token


DEFAULT_PROMPT_PROVIDERS = ["CPUExecutionProvider"]
DEFAULT_RUNTIME_PROVIDERS = ["CUDAExecutionProvider"]


@dataclass
class TalkerPrefillOutputs:
    prompt: PromptInputs
    logits: np.ndarray
    last_hidden: Any
    past_key_values: tuple[tuple[Any, Any], ...]
    raw_outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def next_token_logits(self) -> np.ndarray:
        return self.logits[:, -1, :]

    @property
    def past_kv(self) -> tuple[tuple[Any, Any], ...]:
        return self.past_key_values


@dataclass
class FrameDecodeStepOutputs:
    logits: np.ndarray
    last_hidden: Any
    codebook_tokens: np.ndarray
    frame_embed: Any | None
    past_key_values: tuple[tuple[Any, Any], ...]
    raw_outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def next_token_logits(self) -> np.ndarray:
        return self.logits[:, -1, :]

    @property
    def past_kv(self) -> tuple[tuple[Any, Any], ...]:
        return self.past_key_values


@dataclass
class CodeGenerationOutputs:
    codes: np.ndarray
    stopped: bool
    stop_reason: str
    generated_frames: int
    prefill: TalkerPrefillOutputs
    last_step: FrameDecodeStepOutputs | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioGenerationOutputs:
    audio: np.ndarray
    sample_rate: int
    lengths: np.ndarray
    codes: np.ndarray
    context_frames: int
    generated_frames: int
    stopped: bool
    stop_reason: str
    code_generation: CodeGenerationOutputs | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class GenerationTimer:
    def __init__(self) -> None:
        self._started_at = time.perf_counter()
        self._records: dict[str, list[float]] = {}

    def add(self, name: str, seconds: float) -> None:
        self._records.setdefault(str(name), []).append(float(seconds))

    def summary(self) -> dict[str, dict[str, float | int]]:
        total_elapsed = time.perf_counter() - self._started_at
        summary: dict[str, dict[str, float | int]] = {
            "total": {
                "count": 1,
                "total_seconds": float(total_elapsed),
                "avg_seconds": float(total_elapsed),
                "min_seconds": float(total_elapsed),
                "max_seconds": float(total_elapsed),
            }
        }
        for name, values in self._records.items():
            if not values:
                continue
            total = float(sum(values))
            count = int(len(values))
            summary[name] = {
                "count": count,
                "total_seconds": total,
                "avg_seconds": total / count,
                "min_seconds": float(min(values)),
                "max_seconds": float(max(values)),
            }
        return summary


class Qwen3TTSOnnxModelBase:
    """Shared ONNX runtime layer for prompt -> codes -> audio."""

    prompt_builder_cls = None

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        onnx_dir: str | Path = DEFAULT_ONNX_DIR,
        providers: Optional[list[str]] = None,
        prompt_providers: Optional[list[str]] = None,
        prefill_providers: Optional[list[str]] = None,
        dtype: np.dtype | str = np.float32,
        prefill_use_iobinding: Optional[bool] = True,
        decode_providers: Optional[list[str]] = None,
        decode_use_iobinding: Optional[bool] = True,
        tokenizer_decode_providers: Optional[list[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.onnx_dir = Path(onnx_dir)
        self.providers = list(providers) if providers is not None else list(DEFAULT_RUNTIME_PROVIDERS)
        self.prompt_providers = list(prompt_providers) if prompt_providers is not None else list(DEFAULT_PROMPT_PROVIDERS)
        self.prefill_providers = list(prefill_providers) if prefill_providers is not None else list(self.providers)
        self.decode_providers = list(decode_providers) if decode_providers is not None else list(self.providers)
        self.tokenizer_decode_providers = (
            list(tokenizer_decode_providers) if tokenizer_decode_providers is not None else list(self.providers)
        )
        self.dtype = normalize_compute_dtype(dtype)
        self.prefill_use_iobinding = prefill_use_iobinding
        self.talker_core_onnx_path = self.onnx_dir / "talker" / "talker_core.onnx"
        self.sub_talker_sample_onnx_path = self.onnx_dir / "decode" / "sub_talker_sample.onnx"
        self.decode_use_iobinding = decode_use_iobinding
        self.tokenizer_decode_chunk_frames = 300
        self.tokenizer_decode_context_frames = 25
        self._talker_core_runner: Optional[OnnxSessionRunner] = None
        self._sub_talker_sample_runner: Optional[OnnxSessionRunner] = None
        self._tokenizer_decode_runner: Optional[OnnxSessionRunner] = None
        self._talker_core_output_names: list[str] | None = None
        self._talker_core_kv_output_names: tuple[tuple[str, str], ...] | None = None
        self._decode_attention_mask_cache: dict[tuple[int, int], np.ndarray] = {}
        self._decode_cache_position_cache: dict[int, np.ndarray] = {}
        self._suppressed_main_codec_tokens_cache: dict[int, np.ndarray] = {}
        self.audio_sample_rate, self.decode_upsample_rate = self._load_tokenizer_audio_config()

        self.prompt_builder = self._make_prompt_builder()
        _ = self.talker_core_runner
        self._validate_talker_core_session()
        self._preload_sessions()

    @property
    def talker_core_runner(self) -> OnnxSessionRunner:
        if self._talker_core_runner is None:
            self._talker_core_runner = OnnxSessionRunner(
                self.talker_core_onnx_path,
                providers=self.prefill_providers,
                name="talker_core",
                use_iobinding=self.prefill_use_iobinding,
            )
        return self._talker_core_runner

    @property
    def sub_talker_sample_runner(self) -> OnnxSessionRunner:
        if self._sub_talker_sample_runner is None:
            self._sub_talker_sample_runner = OnnxSessionRunner(
                self.sub_talker_sample_onnx_path,
                providers=self.decode_providers,
                name="sub_talker_sample",
                use_iobinding=self.decode_use_iobinding,
                log_severity_level=None,
            )
            self._validate_sub_talker_sample_session()
        return self._sub_talker_sample_runner

    @property
    def tokenizer_decode_runner(self) -> OnnxSessionRunner:
        if self._tokenizer_decode_runner is None:
            self._tokenizer_decode_runner = OnnxSessionRunner(
                self._tokenizer_decode_path(),
                providers=self.tokenizer_decode_providers,
                name="tokenizer_decode",
                use_iobinding=True,
            )
            self._validate_tokenizer_decode_session()
        return self._tokenizer_decode_runner

    def _tokenizer_decode_path(self) -> Path:
        return self.onnx_dir / "tokenizer" / "tokenizer12hz_decode_chunk.onnx"

    def _make_prompt_builder(self):
        if self.prompt_builder_cls is None:
            return None
        return self.prompt_builder_cls(
            model_path=self.model_path,
            onnx_dir=self.onnx_dir,
            providers=self.prompt_providers,
            dtype=self.dtype,
        )

    def _preload_sessions(self) -> None:
        """Load mode-specific ONNX sessions during model construction."""

    def _preload_decode_sessions(self) -> None:
        _ = self.sub_talker_sample_runner
        _ = self.talker_core_runner

    def _validate_talker_core_session(self) -> None:
        runner = self.talker_core_runner
        required_inputs = {"inputs_embeds", "attention_mask", "cache_position", "past_key_0", "past_value_0"}
        required_outputs = {"logits", "last_hidden", "new_past_key_0", "new_past_value_0"}
        missing_inputs = sorted(required_inputs - set(runner.input_names))
        missing_outputs = sorted(required_outputs - set(runner.output_names))
        if missing_inputs or missing_outputs:
            raise ValueError(
                "talker_core.onnx must accept inputs_embeds/attention_mask/cache_position/past_kv "
                "and return logits/last_hidden/new_past_kv. "
                f"missing_inputs={missing_inputs}, missing_outputs={missing_outputs}, "
                f"path={runner.path}"
            )
        self._validate_runner_input_dtype(runner, "inputs_embeds")

    def _validate_sub_talker_sample_session(self) -> None:
        runner = self.sub_talker_sample_runner
        required_inputs = {"first_token", "last_hidden", "text_embed"}
        required_outputs = {"codebook_tokens", "frame_embed", "decode_embed"}
        missing_inputs = sorted(required_inputs - set(runner.input_names))
        missing_outputs = sorted(required_outputs - set(runner.output_names))
        if missing_inputs or missing_outputs:
            raise ValueError(
                "sub_talker_sample.onnx must accept first_token/last_hidden/text_embed "
                "and return codebook_tokens/frame_embed/decode_embed. "
                f"missing_inputs={missing_inputs}, missing_outputs={missing_outputs}, "
                f"path={runner.path}"
            )
        self._validate_runner_input_dtype(runner, "last_hidden")
        self._validate_runner_input_dtype(runner, "text_embed")

    def _validate_runner_input_dtype(self, runner: OnnxSessionRunner, input_name: str) -> None:
        input_meta = runner.input_metas.get(input_name)
        input_dtype = ORT_INPUT_DTYPES.get(input_meta.type) if input_meta is not None else None
        if input_dtype is None:
            return
        input_dtype = np.dtype(input_dtype)
        if input_dtype != self.dtype:
            raise ValueError(
                f"{runner.name}.{input_name} dtype mismatch: ONNX expects {input_dtype}, "
                f"but requested dtype is {self.dtype}. Use matching ONNX exports or pass dtype={input_dtype}."
            )

    def _validate_tokenizer_decode_session(self) -> None:
        required_inputs = {"audio_codes", "context_frames"}
        required_outputs = {"audio_values", "lengths"}
        runner = self.tokenizer_decode_runner
        missing_inputs = sorted(required_inputs - set(runner.input_names))
        missing_outputs = sorted(required_outputs - set(runner.output_names))
        if missing_inputs or missing_outputs:
            raise ValueError(
                "tokenizer12hz_decode_chunk.onnx must accept audio_codes/context_frames "
                "and return audio_values/lengths. "
                f"missing_inputs={missing_inputs}, missing_outputs={missing_outputs}, "
                f"path={runner.path}"
            )

    def _load_tokenizer_audio_config(self) -> tuple[int, int]:
        config_path = self.model_path / "speech_tokenizer" / "config.json"
        if not config_path.exists():
            return 24000, 1920
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        return int(config.get("output_sample_rate", 24000)), int(config.get("decode_upsample_rate", 1920))

    def run_prefill(self, prompt: PromptInputs, timing: GenerationTimer | None = None) -> TalkerPrefillOutputs:
        runner = self.talker_core_runner
        output_names = list(runner.output_names)
        feed = self._make_talker_core_prefill_feed(prompt)
        start = time.perf_counter()
        outputs = runner.run(
            output_names=output_names,
            feed=feed,
            use_iobinding=self.prefill_use_iobinding,
            copy_outputs_to_cpu=not runner.use_iobinding,
        )
        if timing is not None:
            timing.add("talker_core_prefill", time.perf_counter() - start)
        raw_outputs = {
            name: output
            for name, output in zip(output_names, outputs)
        }
        logits = self._to_cpu_numpy(raw_outputs["logits"], "logits")
        raw_outputs["logits"] = logits
        return TalkerPrefillOutputs(
            prompt=prompt,
            logits=logits,
            last_hidden=raw_outputs["last_hidden"],
            past_key_values=self._collect_talker_core_past_key_values(raw_outputs),
            raw_outputs=raw_outputs,
            metadata={
                "prefill_output_names": output_names,
                "prefill_use_iobinding": bool(runner.use_iobinding),
                "prompt_metadata": dict(prompt.metadata),
                "split_talker": True,
            },
        )

    def _get_talker_core_output_names(self) -> list[str]:
        if self._talker_core_output_names is None:
            self._talker_core_output_names = list(self.talker_core_runner.output_names)
        return self._talker_core_output_names

    def _get_talker_core_kv_output_names(self) -> tuple[tuple[str, str], ...]:
        if self._talker_core_kv_output_names is None:
            key_prefix = "new_past_key_"
            value_prefix = "new_past_value_"
            layer_ids = sorted(
                int(name.removeprefix(key_prefix))
                for name in self.talker_core_runner.output_names
                if name.startswith(key_prefix)
            )
            self._talker_core_kv_output_names = tuple(
                (f"{key_prefix}{layer_id}", f"{value_prefix}{layer_id}")
                for layer_id in layer_ids
            )
        return self._talker_core_kv_output_names

    def _collect_talker_core_past_key_values(self, raw_outputs: dict[str, Any]) -> tuple[tuple[Any, Any], ...]:
        return tuple(
            (raw_outputs[key_name], raw_outputs[value_name])
            for key_name, value_name in self._get_talker_core_kv_output_names()
        )

    @staticmethod
    def _to_cpu_numpy(value: Any, name: str) -> np.ndarray:
        if is_ortvalue(value):
            try:
                return value.numpy()
            except Exception as exc:
                raise RuntimeError(f"{name} must be CPU-bound to convert to numpy") from exc
        return np.asarray(value)

    def _make_talker_core_prefill_feed(self, prompt: PromptInputs) -> dict[str, Any]:
        feed = prompt.to_prefill_feed()
        inputs_embeds = np.asarray(feed["inputs_embeds"], dtype=self.dtype)
        batch_size = int(inputs_embeds.shape[0])
        seq_len = int(inputs_embeds.shape[1])
        core_feed: dict[str, Any] = {
            "inputs_embeds": np.ascontiguousarray(inputs_embeds),
            "attention_mask": np.ascontiguousarray(feed["attention_mask"]),
            "cache_position": np.arange(seq_len, dtype=np.int64),
        }
        for layer_id, (key, value) in enumerate(self._empty_past_key_values(batch_size=batch_size)):
            core_feed[f"past_key_{layer_id}"] = key
            core_feed[f"past_value_{layer_id}"] = value
        return core_feed

    def text_embed_for_decode_step(self, prompt: PromptInputs, generation_step: int) -> np.ndarray:
        generation_step = int(generation_step)
        if generation_step < 0:
            raise ValueError("generation_step must be non-negative")
        if generation_step < prompt.trailing_text_hidden.shape[1]:
            text_embed = prompt.trailing_text_hidden[:, generation_step:generation_step + 1, :]
        else:
            text_embed = prompt.tts_pad_embed
        return np.ascontiguousarray(text_embed.astype(self.dtype, copy=False))

    def run_frame_decode_step(
        self,
        first_token: int | np.ndarray,
        last_hidden: Any,
        past_key_values: tuple[tuple[Any, Any], ...],
        text_embed: np.ndarray,
        generation_step: int,
        cache_position: int | None = None,
        return_frame_embed: bool = False,
        timing: GenerationTimer | None = None,
    ) -> FrameDecodeStepOutputs:
        return self._run_split_frame_decode_step(
            first_token=first_token,
            last_hidden=last_hidden,
            past_key_values=past_key_values,
            text_embed=text_embed,
            generation_step=generation_step,
            cache_position=cache_position,
            return_frame_embed=return_frame_embed,
            timing=timing,
        )

    def _run_split_frame_decode_step(
        self,
        first_token: int | np.ndarray,
        last_hidden: Any,
        past_key_values: tuple[tuple[Any, Any], ...],
        text_embed: np.ndarray,
        generation_step: int,
        cache_position: int | None = None,
        return_frame_embed: bool = False,
        timing: GenerationTimer | None = None,
    ) -> FrameDecodeStepOutputs:
        sample_runner = self.sub_talker_sample_runner
        sample_output_names = ["codebook_tokens", "decode_embed"]
        if return_frame_embed:
            sample_output_names.insert(1, "frame_embed")
        start = time.perf_counter()
        sample_outputs = sample_runner.run(
            output_names=sample_output_names,
            feed=self._make_sub_talker_sample_feed(
                first_token=first_token,
                last_hidden=last_hidden,
                text_embed=text_embed,
            ),
            use_iobinding=self.decode_use_iobinding,
            copy_outputs_to_cpu=not sample_runner.use_iobinding,
        )
        if timing is not None:
            timing.add("sub_talker_sample", time.perf_counter() - start)
        sample_raw = {
            name: output
            for name, output in zip(sample_output_names, sample_outputs)
        }

        core_runner = self.talker_core_runner
        core_output_names = self._get_talker_core_output_names()
        start = time.perf_counter()
        core_outputs = core_runner.run(
            output_names=core_output_names,
            feed=self._make_talker_core_decode_feed(
                decode_embed=sample_raw["decode_embed"],
                past_key_values=past_key_values,
                cache_position=cache_position,
            ),
            use_iobinding=self.decode_use_iobinding,
            copy_outputs_to_cpu=not core_runner.use_iobinding,
        )
        if timing is not None:
            timing.add("talker_core_decode", time.perf_counter() - start)
        core_raw = {
            name: output
            for name, output in zip(core_output_names, core_outputs)
        }

        logits = self._to_cpu_numpy(core_raw["logits"], "logits")
        codebook_tokens = self._to_cpu_numpy(sample_raw["codebook_tokens"], "codebook_tokens").astype(
            np.int64,
            copy=False,
        )
        raw_outputs = {
            "logits": logits,
            "last_hidden_out": core_raw["last_hidden"],
            "codebook_tokens": codebook_tokens,
            **{f"sample.{key}": value for key, value in sample_raw.items()},
            **{f"core.{key}": value for key, value in core_raw.items()},
        }
        if return_frame_embed:
            raw_outputs["frame_embed"] = sample_raw.get("frame_embed")

        return FrameDecodeStepOutputs(
            logits=logits,
            last_hidden=core_raw["last_hidden"],
            codebook_tokens=codebook_tokens,
            frame_embed=sample_raw.get("frame_embed") if return_frame_embed else None,
            past_key_values=self._collect_talker_core_past_key_values(core_raw),
            raw_outputs=raw_outputs,
            metadata={
                "sample_output_names": sample_output_names,
                "core_output_names": core_output_names,
                "decode_use_iobinding": bool(sample_runner.use_iobinding and core_runner.use_iobinding),
                "generation_step": int(generation_step),
                "split_talker": True,
            },
        )

    def run_talker_core_decode_step(
        self,
        decode_embed: Any,
        past_key_values: tuple[tuple[Any, Any], ...],
        cache_position: int | None = None,
        timing: GenerationTimer | None = None,
    ) -> tuple[np.ndarray, Any, tuple[tuple[Any, Any], ...], dict[str, Any]]:
        core_runner = self.talker_core_runner
        core_output_names = self._get_talker_core_output_names()
        start = time.perf_counter()
        core_outputs = core_runner.run(
            output_names=core_output_names,
            feed=self._make_talker_core_decode_feed(
                decode_embed=decode_embed,
                past_key_values=past_key_values,
                cache_position=cache_position,
            ),
            use_iobinding=self.decode_use_iobinding,
            copy_outputs_to_cpu=not core_runner.use_iobinding,
        )
        if timing is not None:
            timing.add("talker_core_decode", time.perf_counter() - start)
        core_raw = {
            name: output
            for name, output in zip(core_output_names, core_outputs)
        }
        logits = self._to_cpu_numpy(core_raw["logits"], "logits")
        return logits, core_raw["last_hidden"], self._collect_talker_core_past_key_values(core_raw), core_raw

    def run_sub_talker_sample_step(
        self,
        first_token: int | np.ndarray,
        last_hidden: Any,
        text_embed: np.ndarray,
        timing: GenerationTimer | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        sample_runner = self.sub_talker_sample_runner
        sample_output_names = ["codebook_tokens", "frame_embed"]
        start = time.perf_counter()
        sample_outputs = sample_runner.run(
            output_names=sample_output_names,
            feed=self._make_sub_talker_sample_feed(
                first_token=first_token,
                last_hidden=last_hidden,
                text_embed=text_embed,
            ),
            use_iobinding=self.decode_use_iobinding,
            copy_outputs_to_cpu=True,
        )
        if timing is not None:
            timing.add("sub_talker_sample", time.perf_counter() - start)
        sample_raw = {
            name: output
            for name, output in zip(sample_output_names, sample_outputs)
        }
        codebook_tokens = self._to_cpu_numpy(sample_raw["codebook_tokens"], "codebook_tokens").astype(
            np.int64,
            copy=False,
        )
        frame_embed = self._to_cpu_numpy(sample_raw["frame_embed"], "frame_embed").astype(self.dtype, copy=False)
        return codebook_tokens, np.ascontiguousarray(frame_embed), sample_raw

    def run_frame_decode_from_prefill(
        self,
        prefill: TalkerPrefillOutputs,
        first_token: int | np.ndarray,
        generation_step: int = 0,
        return_frame_embed: bool = False,
        timing: GenerationTimer | None = None,
    ) -> FrameDecodeStepOutputs:
        text_embed = self.text_embed_for_decode_step(prefill.prompt, generation_step)
        return self.run_frame_decode_step(
            first_token=first_token,
            last_hidden=prefill.last_hidden,
            past_key_values=prefill.past_key_values,
            text_embed=text_embed,
            generation_step=generation_step,
            return_frame_embed=return_frame_embed,
            timing=timing,
        )

    def run_next_frame_decode(
        self,
        previous: FrameDecodeStepOutputs,
        prompt: PromptInputs,
        first_token: int | np.ndarray,
        generation_step: int,
        return_frame_embed: bool = False,
        timing: GenerationTimer | None = None,
    ) -> FrameDecodeStepOutputs:
        text_embed = self.text_embed_for_decode_step(prompt, generation_step)
        return self.run_frame_decode_step(
            first_token=first_token,
            last_hidden=previous.last_hidden,
            past_key_values=previous.past_key_values,
            text_embed=text_embed,
            generation_step=generation_step,
            return_frame_embed=return_frame_embed,
            timing=timing,
        )

    def generate_codes_from_prompt(
        self,
        prompt: PromptInputs,
        max_new_tokens: int = 2048,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int = 0,
        eos_token_id: int | None = None,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        return_frame_embed: bool = False,
        timing: GenerationTimer | None = None,
    ) -> CodeGenerationOutputs:
        rng = self._resolve_rng(rng=rng, seed=seed)
        eos_token_id = self._codec_eos_token_id(eos_token_id)
        start = time.perf_counter()
        prefill = self.run_prefill(prompt, timing=timing)
        if timing is not None:
            timing.add("prefill", time.perf_counter() - start)
        generated = self._generate_code_frames(
            prefill=prefill,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            eos_token_id=eos_token_id,
            rng=rng,
            return_frame_embed=return_frame_embed,
            chunk_frames=None,
            timing=timing,
        )
        start = time.perf_counter()
        result = next(generated)
        if timing is not None:
            timing.add("decode_loop", time.perf_counter() - start)
        return result

    def iter_code_chunks_from_prompt(
        self,
        prompt: PromptInputs,
        max_new_tokens: int = 2048,
        chunk_frames: int = 12,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int = 0,
        eos_token_id: int | None = None,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        return_frame_embed: bool = False,
        timing: GenerationTimer | None = None,
    ):
        rng = self._resolve_rng(rng=rng, seed=seed)
        eos_token_id = self._codec_eos_token_id(eos_token_id)
        prefill = self.run_prefill(prompt)
        yield from self._generate_code_frames(
            prefill=prefill,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            eos_token_id=eos_token_id,
            rng=rng,
            return_frame_embed=return_frame_embed,
            chunk_frames=chunk_frames,
            timing=timing,
        )

    def decode_codes_to_audio(
        self,
        codes: np.ndarray,
        context_frames: int = 0,
        stopped: bool = False,
        stop_reason: str = "",
        code_generation: CodeGenerationOutputs | None = None,
        metadata: dict[str, Any] | None = None,
        timing: GenerationTimer | None = None,
    ) -> AudioGenerationOutputs:
        audio_codes = self._prepare_audio_codes(codes)
        context_frames = int(context_frames)
        if context_frames < 0:
            raise ValueError("context_frames must be non-negative")
        if context_frames > audio_codes.shape[1]:
            raise ValueError(
                f"context_frames={context_frames} is larger than codes_length={audio_codes.shape[1]}"
            )

        return self._decode_codes_to_audio_chunked(
            audio_codes=audio_codes,
            context_frames=context_frames,
            stopped=stopped,
            stop_reason=stop_reason,
            code_generation=code_generation,
            metadata=metadata,
            timing=timing,
        )

    def _decode_code_chunk_to_audio(
        self,
        audio_codes: np.ndarray,
        context_frames: int,
        stopped: bool,
        stop_reason: str,
        code_generation: CodeGenerationOutputs | None = None,
        metadata: dict[str, Any] | None = None,
        timing: GenerationTimer | None = None,
    ) -> AudioGenerationOutputs:
        generated_frames = int(audio_codes.shape[1] - context_frames)
        if generated_frames == 0:
            return self._empty_audio_generation_output(
                codes=audio_codes,
                context_frames=context_frames,
                stopped=stopped,
                stop_reason=stop_reason,
                code_generation=code_generation,
                metadata=metadata,
            )

        start = time.perf_counter()
        outputs = self.tokenizer_decode_runner.run(
            output_names=["audio_values", "lengths"],
            feed={
                "audio_codes": audio_codes,
                "context_frames": np.array(context_frames, dtype=np.int64),
            },
            use_iobinding=self.tokenizer_decode_runner.use_iobinding,
        )
        if timing is not None:
            timing.add("tokenizer_decode", time.perf_counter() - start)
        audio_values = np.asarray(outputs[0], dtype=np.float32)
        lengths = np.asarray(outputs[1], dtype=np.int64)
        valid_length = int(lengths.reshape(-1)[0]) if lengths.size else int(audio_values.shape[-1])
        available_length = int(audio_values.shape[-1])
        if valid_length > available_length:
            raise RuntimeError(
                "tokenizer decode output is shorter than its reported length. "
                f"audio_values.shape={audio_values.shape}, lengths={lengths.tolist()}, "
                f"codes_length={audio_codes.shape[1]}, context_frames={context_frames}, "
                f"path={self.tokenizer_decode_runner.path}. "
                "Re-export tokenizer12hz_decode_chunk.onnx with a large enough chunk trace length."
            )
        valid_length = max(0, valid_length)
        audio = np.ascontiguousarray(audio_values.reshape(audio_values.shape[0], -1)[0, :valid_length])
        return AudioGenerationOutputs(
            audio=audio,
            sample_rate=int(self.audio_sample_rate),
            lengths=lengths,
            codes=audio_codes,
            context_frames=context_frames,
            generated_frames=generated_frames,
            stopped=bool(stopped),
            stop_reason=str(stop_reason),
            code_generation=code_generation,
            metadata={
                "tokenizer_decode_path": str(self.tokenizer_decode_runner.path),
                "decode_upsample_rate": int(self.decode_upsample_rate),
                **(metadata or {}),
            },
        )

    def _decode_codes_to_audio_chunked(
        self,
        audio_codes: np.ndarray,
        context_frames: int,
        stopped: bool,
        stop_reason: str,
        code_generation: CodeGenerationOutputs | None = None,
        metadata: dict[str, Any] | None = None,
        timing: GenerationTimer | None = None,
    ) -> AudioGenerationOutputs:
        generated_frames = int(audio_codes.shape[1] - context_frames)
        if generated_frames == 0:
            return self._empty_audio_generation_output(
                codes=audio_codes,
                context_frames=context_frames,
                stopped=stopped,
                stop_reason=stop_reason,
                code_generation=code_generation,
                metadata=metadata,
            )

        chunks: list[np.ndarray] = []
        start_frame = int(context_frames)
        total_frames = int(audio_codes.shape[1])
        while start_frame < total_frames:
            end_frame = min(start_frame + int(self.tokenizer_decode_chunk_frames), total_frames)
            left_context = min(int(self.tokenizer_decode_context_frames), start_frame)
            input_start = start_frame - left_context
            code_chunk = audio_codes[:, input_start:end_frame, :]
            decoded = self._decode_code_chunk_to_audio(
                audio_codes=code_chunk,
                context_frames=left_context,
                stopped=False,
                stop_reason="chunk",
                code_generation=None,
                metadata={
                    "chunk_input_start": input_start,
                    "chunk_start_frame": start_frame,
                    "chunk_end_frame": end_frame,
                },
                timing=timing,
            )
            expected_samples = (end_frame - start_frame) * int(self.decode_upsample_rate)
            if decoded.lengths.reshape(-1)[0] < expected_samples:
                raise RuntimeError(
                    "tokenizer decode chunk reported fewer samples than expected: "
                    f"reported={int(decoded.lengths.reshape(-1)[0])}, expected={expected_samples}, "
                    f"frames={start_frame}:{end_frame}, context={left_context}, "
                    f"path={self.tokenizer_decode_runner.path}"
                )
            if decoded.audio.shape[0] < expected_samples:
                raise RuntimeError(
                    "tokenizer decode chunk output is shorter than expected: "
                    f"audio_samples={decoded.audio.shape[0]}, expected_samples={expected_samples}, "
                    f"frames={start_frame}:{end_frame}, context={left_context}, "
                    f"path={self.tokenizer_decode_runner.path}"
                )
            chunks.append(decoded.audio[-expected_samples:].astype(np.float32, copy=False))
            start_frame = end_frame

        audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.float32)
        return AudioGenerationOutputs(
            audio=np.ascontiguousarray(audio.astype(np.float32, copy=False)),
            sample_rate=int(self.audio_sample_rate),
            lengths=np.array([[audio.shape[0]]], dtype=np.int64),
            codes=audio_codes,
            context_frames=int(context_frames),
            generated_frames=generated_frames,
            stopped=bool(stopped),
            stop_reason=str(stop_reason),
            code_generation=code_generation,
            metadata={
                "tokenizer_decode_path": str(self.tokenizer_decode_runner.path),
                "decode_upsample_rate": int(self.decode_upsample_rate),
                "chunk_frames": int(self.tokenizer_decode_chunk_frames),
                "left_context_frames": int(self.tokenizer_decode_context_frames),
                "num_audio_chunks": len(chunks),
                **(metadata or {}),
            },
        )

    def decode_generation_to_audio(
        self,
        generation: CodeGenerationOutputs,
        ref_code: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
        timing: GenerationTimer | None = None,
    ) -> AudioGenerationOutputs:
        audio_codes, context_frames = self._codes_for_audio_decode(
            generated_codes=generation.codes,
            prompt=generation.prefill.prompt,
            ref_code=ref_code,
        )
        return self.decode_codes_to_audio(
            codes=audio_codes,
            context_frames=context_frames,
            stopped=generation.stopped,
            stop_reason=generation.stop_reason,
            code_generation=generation,
            metadata={
                "generated_codes_shape": tuple(int(dim) for dim in generation.codes.shape),
                **(metadata or {}),
            },
            timing=timing,
        )

    def generate_audio_from_prompt(
        self,
        prompt: PromptInputs,
        max_new_tokens: int = 2048,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int = 0,
        eos_token_id: int | None = None,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        timing: GenerationTimer | None = None,
    ) -> AudioGenerationOutputs:
        start = time.perf_counter()
        generation = self.generate_codes_from_prompt(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            eos_token_id=eos_token_id,
            rng=rng,
            seed=seed,
            timing=timing,
        )
        if timing is not None:
            timing.add("code_generation", time.perf_counter() - start)
        start = time.perf_counter()
        audio = self.decode_generation_to_audio(generation, timing=timing)
        if timing is not None:
            timing.add("audio_decode", time.perf_counter() - start)
        return audio

    def iter_audio_chunks_from_prompt(
        self,
        prompt: PromptInputs,
        max_new_tokens: int = 2048,
        chunk_frames: int = 12,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int = 0,
        eos_token_id: int | None = None,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ):
        generated_so_far = np.zeros((1, 0, self._num_code_groups()), dtype=np.int64)
        previous_generated_frames = 0
        for code_chunk in self.iter_code_chunks_from_prompt(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            chunk_frames=chunk_frames,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            eos_token_id=eos_token_id,
            rng=rng,
            seed=seed,
        ):
            if code_chunk.codes.shape[1] > 0:
                generated_so_far = np.concatenate([generated_so_far, code_chunk.codes], axis=1)

            audio_codes, reference_context_frames = self._codes_for_audio_decode(
                generated_codes=generated_so_far,
                prompt=prompt,
            )
            context_frames = reference_context_frames + previous_generated_frames
            audio = self.decode_codes_to_audio(
                codes=audio_codes,
                context_frames=context_frames,
                stopped=code_chunk.stopped,
                stop_reason=code_chunk.stop_reason,
                code_generation=code_chunk,
                metadata={
                    "chunk_code_frames": int(code_chunk.codes.shape[1]),
                    "generated_frames_total": int(generated_so_far.shape[1]),
                    "reference_context_frames": int(reference_context_frames),
                },
            )
            previous_generated_frames = int(generated_so_far.shape[1])
            yield audio

    def _generate_code_frames(
        self,
        prefill: TalkerPrefillOutputs,
        max_new_tokens: int,
        do_sample: bool | None,
        top_k: int | None,
        top_p: float | None,
        temperature: float | None,
        repetition_penalty: float | None,
        min_new_tokens: int,
        eos_token_id: int,
        rng: np.random.Generator,
        return_frame_embed: bool,
        chunk_frames: int | None,
        timing: GenerationTimer | None = None,
    ):
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if chunk_frames is not None and chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")

        do_sample, top_k, top_p, temperature, repetition_penalty = self._resolve_generation_options(
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        max_codec_frames = max(int(max_new_tokens) - 1, 1)
        generated_first_tokens: list[int] = []
        all_frames: list[np.ndarray] = []
        chunk_buffer: list[np.ndarray] = []
        logits = prefill.logits
        last_step: FrameDecodeStepOutputs | None = None
        stopped = False
        stop_reason = "max_new_tokens"

        for generation_step in range(max_codec_frames):
            start = time.perf_counter()
            first_token = self._sample_first_token(
                logits=logits,
                generated_first_tokens=generated_first_tokens,
                eos_token_id=eos_token_id,
                min_new_tokens=min_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                rng=rng,
            )
            if timing is not None:
                timing.add("main_token_sample", time.perf_counter() - start)

            if first_token == eos_token_id:
                stopped = True
                stop_reason = "eos"
                break

            start = time.perf_counter()
            if generation_step == 0:
                step = self.run_frame_decode_from_prefill(
                    prefill=prefill,
                    first_token=first_token,
                    generation_step=generation_step,
                    return_frame_embed=return_frame_embed,
                    timing=timing,
                )
            else:
                if last_step is None:
                    raise RuntimeError("missing previous frame decode state")
                step = self.run_next_frame_decode(
                    previous=last_step,
                    prompt=prefill.prompt, 
                    first_token=first_token,
                    generation_step=generation_step,
                    return_frame_embed=return_frame_embed,
                    timing=timing,
                )
            if timing is not None:
                timing.add("frame_decode", time.perf_counter() - start)

            frame = np.asarray(step.codebook_tokens, dtype=np.int64)
            all_frames.append(frame)
            chunk_buffer.append(frame)
            generated_first_tokens.append(first_token)
            logits = step.logits
            last_step = step

            if chunk_frames is not None and len(chunk_buffer) >= chunk_frames:
                yield self._code_generation_output(
                    frames=chunk_buffer,
                    stopped=False,
                    stop_reason="chunk",
                    prefill=prefill,
                    last_step=last_step,
                    metadata={
                        "generated_frames_total": len(all_frames),
                        "chunk_frames": len(chunk_buffer),
                        "max_codec_frames": int(max_codec_frames),
                    },
                )
                chunk_buffer = []

        if chunk_frames is not None:
            if chunk_buffer or stopped or not all_frames or stop_reason == "max_new_tokens":
                yield self._code_generation_output(
                    frames=chunk_buffer,
                    stopped=stopped,
                    stop_reason=stop_reason,
                    prefill=prefill,
                    last_step=last_step,
                    metadata={
                        "generated_frames_total": len(all_frames),
                        "chunk_frames": len(chunk_buffer),
                        "max_codec_frames": int(max_codec_frames),
                    },
                )
            return

        yield self._code_generation_output(
            frames=all_frames,
            stopped=stopped,
            stop_reason=stop_reason,
            prefill=prefill,
            last_step=last_step,
            metadata={
                "generated_first_tokens": np.asarray(generated_first_tokens, dtype=np.int64),
                "eos_token_id": int(eos_token_id),
                "min_new_tokens": int(min_new_tokens),
                "max_codec_frames": int(max_codec_frames),
            },
        )

    def _sample_first_token(
        self,
        logits: np.ndarray,
        generated_first_tokens: list[int],
        eos_token_id: int,
        min_new_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        rng: np.random.Generator,
        suppress_eos: bool = False,
    ) -> int:
        scores = np.asarray(logits, dtype=np.float32).reshape(-1).copy()
        if repetition_penalty is not None and repetition_penalty != 1.0:
            scores = apply_repetition_penalty(scores, generated_first_tokens, repetition_penalty)
        suppress_tokens = self._suppressed_main_codec_tokens(eos_token_id)
        scores[suppress_tokens] = -np.inf
        if suppress_eos or len(generated_first_tokens) < int(min_new_tokens):
            scores[eos_token_id] = -np.inf
        return sample_token(
            scores,
            rng,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )

    def _code_generation_output(
        self,
        frames: list[np.ndarray],
        stopped: bool,
        stop_reason: str,
        prefill: TalkerPrefillOutputs,
        last_step: FrameDecodeStepOutputs | None,
        metadata: dict[str, Any] | None = None,
    ) -> CodeGenerationOutputs:
        if frames:
            codes = np.concatenate(frames, axis=0)
            if codes.ndim == 2:
                codes = codes[None, :, :]
        else:
            codes = np.zeros((1, 0, int(self.prompt_builder.talker_config["num_code_groups"])), dtype=np.int64)
        return CodeGenerationOutputs(
            codes=codes.astype(np.int64, copy=False),
            stopped=bool(stopped),
            stop_reason=str(stop_reason),
            generated_frames=int(codes.shape[1]),
            prefill=prefill,
            last_step=last_step,
            metadata=metadata or {},
        )

    def _empty_audio_generation_output(
        self,
        codes: np.ndarray,
        context_frames: int,
        stopped: bool,
        stop_reason: str,
        code_generation: CodeGenerationOutputs | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AudioGenerationOutputs:
        return AudioGenerationOutputs(
            audio=np.zeros((0,), dtype=np.float32),
            sample_rate=int(self.audio_sample_rate),
            lengths=np.array([[0]], dtype=np.int64),
            codes=self._prepare_audio_codes(codes),
            context_frames=int(context_frames),
            generated_frames=0,
            stopped=bool(stopped),
            stop_reason=str(stop_reason),
            code_generation=code_generation,
            metadata={
                "tokenizer_decode_path": str(self._tokenizer_decode_path()),
                "decode_upsample_rate": int(self.decode_upsample_rate),
                **(metadata or {}),
            },
        )

    def _codes_for_audio_decode(
        self,
        generated_codes: np.ndarray,
        prompt: PromptInputs | None = None,
        ref_code: np.ndarray | None = None,
    ) -> tuple[np.ndarray, int]:
        generated_codes = self._prepare_audio_codes(generated_codes)
        reference_codes = self._reference_codes_for_audio_decode(prompt=prompt, ref_code=ref_code)
        if reference_codes is None:
            return generated_codes, 0

        reference_codes = self._prepare_audio_codes(reference_codes)
        if reference_codes.shape[0] != generated_codes.shape[0]:
            raise ValueError(
                "reference code batch mismatch: "
                f"ref batch={reference_codes.shape[0]}, generated batch={generated_codes.shape[0]}"
            )
        audio_codes = np.concatenate([reference_codes, generated_codes], axis=1)
        return audio_codes, int(reference_codes.shape[1])

    @staticmethod
    def _reference_codes_for_audio_decode(
        prompt: PromptInputs | None = None,
        ref_code: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if ref_code is not None:
            return np.asarray(ref_code, dtype=np.int64)
        if prompt is None:
            return None
        metadata_ref_code = prompt.metadata.get("ref_code")
        if metadata_ref_code is None:
            return None
        return np.asarray(metadata_ref_code, dtype=np.int64)

    def _prepare_audio_codes(self, codes: np.ndarray | CodeGenerationOutputs) -> np.ndarray:
        if isinstance(codes, CodeGenerationOutputs):
            codes = codes.codes
        codes = np.asarray(codes, dtype=np.int64)
        if codes.ndim == 2:
            codes = codes[None, :, :]
        if codes.ndim != 3:
            raise ValueError(f"audio codes must have shape [T,16] or [1,T,16], got {codes.shape}")
        if codes.shape[0] != 1:
            raise NotImplementedError(f"audio decode currently supports batch_size=1, got {codes.shape[0]}")
        expected_groups = self._num_code_groups()
        if codes.shape[2] != expected_groups:
            raise ValueError(f"audio codes codebook count mismatch: expected {expected_groups}, got {codes.shape[2]}")
        return np.ascontiguousarray(codes)

    def _num_code_groups(self) -> int:
        if self.prompt_builder is not None:
            return int(self.prompt_builder.talker_config["num_code_groups"])
        return 16

    @staticmethod
    def _resolve_rng(rng: np.random.Generator | None, seed: int | None) -> np.random.Generator:
        if rng is not None:
            return rng
        return np.random.default_rng(seed)

    def _codec_eos_token_id(self, eos_token_id: int | None = None) -> int:
        if eos_token_id is not None:
            return int(eos_token_id)
        return int(self.prompt_builder.talker_config["codec_eos_token_id"])

    def _suppressed_main_codec_tokens(self, eos_token_id: int) -> np.ndarray:
        cached = self._suppressed_main_codec_tokens_cache.get(int(eos_token_id))
        if cached is not None:
            return cached
        vocab_size = int(self.prompt_builder.talker_config["vocab_size"])
        mask_tail = int(self.prompt_builder.talker_config.get("first_codebook_mask_tail", 1024))
        start = max(0, vocab_size - mask_tail)
        tokens = np.arange(start, vocab_size, dtype=np.int64)
        suppressed = tokens[tokens != int(eos_token_id)]
        self._suppressed_main_codec_tokens_cache[int(eos_token_id)] = suppressed
        return suppressed

    def _generation_config_value(self, key: str, default: Any) -> Any:
        if self.prompt_builder is None:
            return default
        return self.prompt_builder.config.get(key, default)

    def _resolve_generation_options(
        self,
        do_sample: bool | None,
        top_k: int | None,
        top_p: float | None,
        temperature: float | None,
        repetition_penalty: float | None,
    ) -> tuple[bool, int, float, float, float]:
        return (
            bool(self._generation_config_value("do_sample", True) if do_sample is None else do_sample),
            int(self._generation_config_value("top_k", 50) if top_k is None else top_k),
            float(self._generation_config_value("top_p", 1.0) if top_p is None else top_p),
            float(self._generation_config_value("temperature", 0.9) if temperature is None else temperature),
            float(
                self._generation_config_value("repetition_penalty", 1.05)
                if repetition_penalty is None
                else repetition_penalty
            ),
        )

    def _make_sub_talker_sample_feed(
        self,
        first_token: int | np.ndarray,
        last_hidden: Any,
        text_embed: np.ndarray,
    ) -> dict[str, Any]:
        first_token_array = np.asarray(first_token, dtype=np.int64).reshape(-1)
        last_hidden_shape = self._value_shape(last_hidden)
        if len(last_hidden_shape) != 3 or last_hidden_shape[1] != 1:
            raise ValueError(
                "sub_talker_sample expects last_hidden to be the final token only, "
                f"shape [batch, 1, hidden], got {last_hidden_shape}."
            )
        batch_size = last_hidden_shape[0]
        if first_token_array.shape[0] == 1 and batch_size != 1:
            first_token_array = np.full((batch_size,), int(first_token_array[0]), dtype=np.int64)
        if first_token_array.shape[0] != batch_size:
            raise ValueError(f"first_token batch mismatch: got {first_token_array.shape[0]}, expected {batch_size}")
        return {
            "first_token": np.ascontiguousarray(first_token_array),
            "last_hidden": last_hidden,
            "text_embed": np.ascontiguousarray(np.asarray(text_embed, dtype=self.dtype)),
        }

    def _make_talker_core_decode_feed(
        self,
        decode_embed: Any,
        past_key_values: tuple[tuple[Any, Any], ...],
        cache_position: int | None = None,
    ) -> dict[str, Any]:
        decode_embed_shape = self._value_shape(decode_embed)
        if len(decode_embed_shape) != 3 or decode_embed_shape[1] != 1:
            raise ValueError(
                "talker_core decode expects decode_embed shape [batch, 1, hidden], "
                f"got {decode_embed_shape}."
            )
        batch_size = decode_embed_shape[0]
        past_len = self._past_kv_length(past_key_values)
        position = past_len if cache_position is None else int(cache_position)
        feed: dict[str, Any] = {
            "inputs_embeds": decode_embed,
            "attention_mask": self._decode_attention_mask(batch_size=batch_size, past_len=past_len),
            "cache_position": self._decode_cache_position(position),
        }
        for layer_id, (key, value) in enumerate(past_key_values):
            feed[f"past_key_{layer_id}"] = key
            feed[f"past_value_{layer_id}"] = value
        return feed

    def _decode_attention_mask(self, batch_size: int, past_len: int) -> np.ndarray:
        key = (int(batch_size), int(past_len))
        cached = self._decode_attention_mask_cache.get(key)
        if cached is None:
            cached = np.ones((int(batch_size), int(past_len) + 1), dtype=np.int64)
            self._decode_attention_mask_cache[key] = cached
        return cached

    def _decode_cache_position(self, past_len: int) -> np.ndarray:
        past_len = int(past_len)
        cached = self._decode_cache_position_cache.get(past_len)
        if cached is None:
            cached = np.array([past_len], dtype=np.int64)
            self._decode_cache_position_cache[past_len] = cached
        return cached

    @staticmethod
    def _value_shape(value: Any) -> tuple[int, ...]:
        shape = getattr(value, "shape", None)
        if callable(shape):
            return tuple(int(dim) for dim in shape())
        if shape is not None:
            return tuple(int(dim) for dim in shape)
        return tuple(int(dim) for dim in np.asarray(value).shape)

    @classmethod
    def _past_kv_length(cls, past_key_values: tuple[tuple[Any, Any], ...]) -> int:
        if not past_key_values:
            raise ValueError("past_key_values must not be empty")
        return int(cls._value_shape(past_key_values[0][0])[2])

    def _empty_past_key_values(self, batch_size: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        if self.prompt_builder is None:
            raise RuntimeError("prompt_builder is required to build empty talker_core KV inputs")
        config = self.prompt_builder.talker_config
        num_layers = int(config["num_hidden_layers"])
        num_kv_heads = int(config.get("num_key_value_heads", config["num_attention_heads"]))
        head_dim = int(config.get("head_dim", int(config["hidden_size"]) // int(config["num_attention_heads"])))
        shape = (int(batch_size), num_kv_heads, 0, head_dim)
        return tuple(
            (
                np.zeros(shape, dtype=self.dtype),
                np.zeros(shape, dtype=self.dtype),
            )
            for _ in range(num_layers)
        )

    @staticmethod
    def _collect_past_key_values(
        raw_outputs: dict[str, Any],
        key_prefix: str = "past_key_",
        value_prefix: str = "past_value_",
    ) -> tuple[tuple[Any, Any], ...]:
        layer_ids = sorted(
            int(name.removeprefix(key_prefix))
            for name in raw_outputs
            if name.startswith(key_prefix)
        )
        past_key_values = []
        for layer_id in layer_ids:
            key_name = f"{key_prefix}{layer_id}"
            value_name = f"{value_prefix}{layer_id}"
            if value_name not in raw_outputs:
                raise ValueError(f"Missing {value_name} in talker outputs")
            past_key_values.append((raw_outputs[key_name], raw_outputs[value_name]))
        return tuple(past_key_values)


class StreamingTextBuffer:
    _split_pattern = re.compile(r"[。！？!?\.\u2026]\s*|[;；]\s*|\n")

    def __init__(self, min_chars: int = 20, max_chars: int = 80) -> None:
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self._cache = ""

    def push(self, text_fragment: str) -> list[str]:
        self._cache += str(text_fragment or "")
        return self._extract(force=False)

    def finish(self) -> list[str]:
        return self._extract(force=True)

    def _extract(self, force: bool) -> list[str]:
        if force:
            tail = self._cache
            self._cache = ""
            return [tail] if tail else []

        segments: list[str] = []
        while self._cache:
            cut_idx = None
            if len(self._cache) >= self.min_chars:
                for match in self._split_pattern.finditer(self._cache):
                    if match.end() >= self.min_chars:
                        cut_idx = match.end()
                        break
            if cut_idx is None and len(self._cache) >= self.max_chars:
                space_idx = self._cache.rfind(" ")
                cut_idx = space_idx + 1 if space_idx > 0 else self.max_chars
            if cut_idx is None:
                break
            segments.append(self._cache[:cut_idx])
            self._cache = self._cache[cut_idx:]
        return segments

class StreamingTextEmbedQueue:
    def __init__(self, prompt_builder: BaseVoiceClonePromptBuilder) -> None:
        self.prompt_builder = prompt_builder
        self._items: deque[np.ndarray] = deque()

    def __bool__(self) -> bool:
        return bool(self._items)

    def append_segment(self, segment: str) -> None:
        embeds = self.prompt_builder.target_text_embeds(segment)
        self.extend_sequence(embeds)

    def append_embed(self, embed: np.ndarray) -> None:
        embed = np.asarray(embed, dtype=self.prompt_builder.dtype)
        if embed.ndim != 3 or embed.shape[1] != 1:
            raise ValueError(f"streaming text embed must have shape [batch, 1, hidden], got {embed.shape}")
        self._items.append(np.ascontiguousarray(embed))

    def extend_sequence(self, embeds: np.ndarray) -> None:
        embeds = np.asarray(embeds, dtype=self.prompt_builder.dtype)
        if embeds.ndim != 3:
            raise ValueError(f"streaming text embeds must have shape [batch, seq, hidden], got {embeds.shape}")
        for index in range(embeds.shape[1]):
            self.append_embed(embeds[:, index:index + 1, :])

    def pop(self) -> np.ndarray:
        return self._items.popleft()


class SegmentKVWindowMixin:
    def _init_kv_window(
        self,
        kv_window_frames: int | None,
        kv_window_max_frames: int | None,
    ) -> None:
        self.prefix_kv_len = self.model._past_kv_length(self.past_key_values)
        self.kv_window_frames = None if kv_window_frames is None or int(kv_window_frames) <= 0 else int(kv_window_frames)
        if self.kv_window_frames is None:
            self.kv_window_max_frames = None
        elif kv_window_max_frames is None:
            self.kv_window_max_frames = self.kv_window_frames + max(64, self.kv_window_frames // 4)
        else:
            self.kv_window_max_frames = max(self.kv_window_frames, int(kv_window_max_frames))
        self.kv_dropped_frames = 0
        self.segment_boundaries: list[int] = []

    def record_segment_boundary(self) -> None:
        boundary = int(self.frames_generated)
        if boundary <= 0:
            return
        if self.segment_boundaries and self.segment_boundaries[-1] == boundary:
            return
        self.segment_boundaries.append(boundary)
        self._trim_kv_to_segment_window()

    def _trim_kv_to_segment_window(self) -> None:
        if self.kv_window_frames is None or self.kv_window_max_frames is None:
            return

        retained_generated = int(self.frames_generated) - int(self.kv_dropped_frames)
        if retained_generated <= int(self.kv_window_max_frames):
            return

        threshold = int(self.frames_generated) - int(self.kv_window_frames)
        candidates = [
            boundary
            for boundary in self.segment_boundaries
            if int(self.kv_dropped_frames) < boundary <= threshold
        ]
        if not candidates:
            return

        cut_boundary = max(candidates)
        drop_frames = int(cut_boundary) - int(self.kv_dropped_frames)
        if drop_frames <= 0:
            return

        before_kv_len = self.model._past_kv_length(self.past_key_values)
        self.past_key_values = self._trim_past_key_values(drop_frames)
        after_kv_len = self.model._past_kv_length(self.past_key_values)
        self.kv_dropped_frames = int(cut_boundary)
        self.segment_boundaries = [
            boundary
            for boundary in self.segment_boundaries
            if boundary > self.kv_dropped_frames
        ]
        print(
            "kv_window_trim",
            {
                "frames_generated": int(self.frames_generated),
                "cut_boundary": int(cut_boundary),
                "drop_frames": int(drop_frames),
                "kv_dropped_frames": int(self.kv_dropped_frames),
                "before_kv_len": int(before_kv_len),
                "after_kv_len": int(after_kv_len),
                "prefix_kv_len": int(self.prefix_kv_len),
                "kv_window_frames": int(self.kv_window_frames),
                "kv_window_max_frames": int(self.kv_window_max_frames),
                "remaining_segment_boundaries": [int(boundary) for boundary in self.segment_boundaries],
            },
        )
        if self.last_step is not None:
            self.last_step.past_key_values = self.past_key_values
            self.last_step.metadata = {
                **self.last_step.metadata,
                "kv_window_frames": int(self.kv_window_frames),
                "kv_window_max_frames": int(self.kv_window_max_frames),
                "kv_dropped_frames": int(self.kv_dropped_frames),
            }

    def _trim_past_key_values(self, drop_generated_frames: int):
        trimmed = []
        for layer_id, (key, value) in enumerate(self.past_key_values):
            trimmed.append(
                (
                    self._trim_kv_tensor(key, drop_generated_frames, f"past_key_{layer_id}"),
                    self._trim_kv_tensor(value, drop_generated_frames, f"past_value_{layer_id}"),
                )
            )
        return tuple(trimmed)

    def _trim_kv_tensor(self, value, drop_generated_frames: int, name: str):
        array = self._kv_to_numpy(value, name)
        if array.ndim != 4:
            raise ValueError(f"{name} must have shape [batch, heads, seq, dim], got {array.shape}")
        generated_len = int(array.shape[2]) - int(self.prefix_kv_len)
        if generated_len <= int(drop_generated_frames):
            raise ValueError(
                f"{name} cannot drop {drop_generated_frames} generated frames from "
                f"generated_len={generated_len}, prefix_kv_len={self.prefix_kv_len}"
            )
        prefix = array[:, :, : self.prefix_kv_len, :]
        recent = array[:, :, self.prefix_kv_len + int(drop_generated_frames):, :]
        trimmed = np.ascontiguousarray(np.concatenate([prefix, recent], axis=2))
        return self._kv_from_numpy_like(trimmed, value)

    @staticmethod
    def _kv_to_numpy(value, name: str) -> np.ndarray:
        if is_ortvalue(value):
            try:
                return value.numpy()
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot trim {name}: ONNX Runtime did not expose this OrtValue as numpy. "
                    "Run with decode_use_iobinding=False, or make KV outputs CPU-bound before trimming."
                ) from exc
        return np.asarray(value)

    def _kv_from_numpy_like(self, array: np.ndarray, like):
        if not is_ortvalue(like):
            return array
        device_name = str(like.device_name()).lower()
        if device_name == "cpu":
            return array
        return self.model.talker_core_runner.to_device_ortvalue(
            array,
            device=device_name,
            device_id=self.model.talker_core_runner.output_device_id,
        )

    def _kv_window_metadata(self) -> dict[str, int | None]:
        return {
            "kv_window_frames": int(self.kv_window_frames) if self.kv_window_frames is not None else None,
            "kv_window_max_frames": int(self.kv_window_max_frames) if self.kv_window_max_frames is not None else None,
            "kv_dropped_frames": int(self.kv_dropped_frames),
            "prefix_kv_len": int(self.prefix_kv_len),
        }

    def _next_cache_position(self) -> int:
        return int(self.prefix_kv_len) + int(self.frames_generated)


class StreamingDecodeState(SegmentKVWindowMixin):
    def __init__(
        self,
        model: Qwen3TTSOnnxModelBase,
        prompt: PromptInputs,
        max_new_tokens: int,
        do_sample: bool | None,
        top_k: int | None,
        top_p: float | None,
        temperature: float | None,
        repetition_penalty: float | None,
        min_new_tokens: int,
        eos_token_id: int | None,
        rng: np.random.Generator,
        kv_window_frames: int | None = 512,
        kv_window_max_frames: int | None = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.max_new_tokens = int(max_new_tokens)
        self.max_codec_frames = max(self.max_new_tokens - 1, 1)
        self.do_sample, self.top_k, self.top_p, self.temperature, self.repetition_penalty = (
            model._resolve_generation_options(
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
        )
        self.min_new_tokens = int(min_new_tokens)
        self.eos_token_id = model._codec_eos_token_id(eos_token_id)
        self.rng = rng

        self.prefill = model.run_prefill(prompt)
        self.logits = self.prefill.logits
        self.last_hidden = self.prefill.last_hidden
        self.past_key_values = self.prefill.past_key_values
        self._init_kv_window(kv_window_frames=kv_window_frames, kv_window_max_frames=kv_window_max_frames)
        self.generated_first_tokens: list[int] = []
        self.generated_codes: list[np.ndarray] = []
        self.frames_generated = 0
        self.public_code_start_frame = 0
        self.finished = False
        self.stop_reason = "running"
        self.last_step: FrameDecodeStepOutputs | None = None

    def step(
        self,
        text_embed: np.ndarray,
        allow_eos: bool = True,
    ) -> np.ndarray | None:
        if self.finished:
            return None
        if self.frames_generated >= self.max_codec_frames:
            self.finished = True
            self.stop_reason = "max_new_tokens"
            return None

        text_embed = np.ascontiguousarray(np.asarray(text_embed, dtype=self.model.dtype))
        first_token = self.model._sample_first_token(
            logits=self.logits,
            generated_first_tokens=self.generated_first_tokens,
            eos_token_id=self.eos_token_id,
            min_new_tokens=self.min_new_tokens,
            do_sample=self.do_sample,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            rng=self.rng,
            suppress_eos=not allow_eos,
        )
        if first_token == self.eos_token_id:
            self.finished = True
            self.stop_reason = "eos"
            return None

        step = self.model.run_frame_decode_step(
            first_token=first_token,
            last_hidden=self.last_hidden,
            past_key_values=self.past_key_values,
            text_embed=text_embed,
            generation_step=self.frames_generated,
            cache_position=self._next_cache_position(),
            return_frame_embed=False,
        )

        code_row = np.asarray(step.codebook_tokens, dtype=np.int64).reshape(-1)
        self.generated_first_tokens.append(int(first_token))
        self.generated_codes.append(code_row)
        self.frames_generated += 1
        self.logits = step.logits
        self.last_hidden = step.last_hidden
        self.past_key_values = step.past_key_values
        step.metadata = {
            **step.metadata,
            "streaming_decode_order": "sample_then_advance",
            "generation_step": int(self.frames_generated - 1),
        }
        self.last_step = step
        return code_row

    def mark_generated_prefix(self, prefix_frames: int) -> None:
        self.public_code_start_frame = max(0, int(prefix_frames))

    def codes_array(self, include_discarded: bool = False) -> np.ndarray:
        if not self.generated_codes:
            return np.zeros((1, 0, self.model._num_code_groups()), dtype=np.int64)
        start = 0 if include_discarded else int(self.public_code_start_frame)
        visible_codes = self.generated_codes[start:]
        if not visible_codes:
            return np.zeros((1, 0, self.model._num_code_groups()), dtype=np.int64)
        return np.stack(visible_codes, axis=0).astype(np.int64)[None, :, :]

    def code_generation_output(self) -> CodeGenerationOutputs:
        visible_frames = max(0, int(self.frames_generated) - int(self.public_code_start_frame))
        return CodeGenerationOutputs(
            codes=self.codes_array(),
            stopped=bool(self.finished),
            stop_reason=str(self.stop_reason),
            generated_frames=int(visible_frames),
            prefill=self.prefill,
            last_step=self.last_step,
            metadata={
                "streaming": True,
                "total_frames_generated": int(self.frames_generated),
                "discarded_prefix_frames": int(self.public_code_start_frame),
                "generated_first_tokens": np.asarray(self.generated_first_tokens, dtype=np.int64),
                "eos_token_id": int(self.eos_token_id),
                "min_new_tokens": int(self.min_new_tokens),
                "max_codec_frames": int(self.max_codec_frames),
                **self._kv_window_metadata(),
            },
        )


class StreamingAudioChunkBuffer:
    def __init__(
        self,
        model: Qwen3TTSOnnxModelBase,
        prompt: PromptInputs,
        audio_chunk_frames: int = 300,
        left_context_frames: int = 25,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.audio_chunk_frames = int(audio_chunk_frames)
        self.left_context_frames = int(left_context_frames)
        if self.audio_chunk_frames <= 0:
            raise ValueError("audio_chunk_frames must be positive")
        if self.left_context_frames < 0:
            raise ValueError("left_context_frames must be non-negative")
        self.first_audio_chunk_frames = self.audio_chunk_frames

        ref_code = model._reference_codes_for_audio_decode(prompt=prompt)
        if ref_code is None:
            self.ref_code = np.zeros((0, model._num_code_groups()), dtype=np.int64)
        else:
            self.ref_code = model._prepare_audio_codes(ref_code)[0]
        self.generated_codes: list[np.ndarray] = []
        self.decoded_generated_frames = 0
        self.public_code_start_frame = 0
        self.sample_rate = int(model.audio_sample_rate)

    def append_code(self, code_row: np.ndarray) -> None:
        row = np.asarray(code_row, dtype=np.int64).reshape(-1)
        expected_groups = self.model._num_code_groups()
        if row.shape != (expected_groups,):
            raise ValueError(f"code row must have shape [{expected_groups}], got {row.shape}")
        self.generated_codes.append(row)

    def push_code(self, code_row: np.ndarray, state: StreamingDecodeState) -> list[AudioGenerationOutputs]:
        self.append_code(code_row)
        chunk_frames = (
            self.first_audio_chunk_frames
            if self.decoded_generated_frames == 0
            else self.audio_chunk_frames
        )
        if self.available_generated_frames() - self.decoded_generated_frames >= chunk_frames:
            return [self._decode_available(final=False, state=state)]
        return []

    def mark_prefix_codes(self, prefix_frames: int) -> None:
        prefix_frames = max(0, int(prefix_frames))
        self.public_code_start_frame = max(int(self.public_code_start_frame), prefix_frames)
        self.decoded_generated_frames = 0

    def flush(self, state: StreamingDecodeState | None = None) -> AudioGenerationOutputs | None:
        if self.available_generated_frames() <= self.decoded_generated_frames:
            return None
        return self._decode_available(final=True, state=state)

    def available_generated_frames(self) -> int:
        return max(0, len(self.generated_codes) - int(self.public_code_start_frame))

    def codes_array(self, include_prefix: bool = False) -> np.ndarray:
        start = 0 if include_prefix else int(self.public_code_start_frame)
        visible_codes = self.generated_codes[start:]
        if visible_codes:
            generated = np.stack(visible_codes, axis=0).astype(np.int64)
        else:
            generated = np.zeros((0, self.model._num_code_groups()), dtype=np.int64)
        return np.concatenate([self.ref_code, generated], axis=0)

    def _decode_available(
        self,
        final: bool,
        state: StreamingDecodeState | None,
    ) -> AudioGenerationOutputs:
        visible_codes = self.codes_array()
        decode_codes = self.codes_array(include_prefix=True)
        ref_len = int(self.ref_code.shape[0])
        prefix_frames = int(self.public_code_start_frame)
        visible_start = int(self.decoded_generated_frames)
        visible_end = int(self.available_generated_frames())
        decode_start = ref_len + prefix_frames + visible_start
        decode_end = ref_len + prefix_frames + visible_end
        context = min(self.left_context_frames, decode_start)
        input_start = decode_start - context
        code_chunk = decode_codes[None, input_start:decode_end, :]
        generation = state.code_generation_output() if state is not None else None
        decoded = self.model._decode_code_chunk_to_audio(
            audio_codes=code_chunk,
            context_frames=context,
            stopped=bool(final and (state.finished if state is not None else False)),
            stop_reason=str(state.stop_reason if state is not None else ("final" if final else "chunk")),
            code_generation=generation,
            metadata={
                "streaming": True,
                "final": bool(final),
                "chunk_start_frame": int(ref_len + visible_start),
                "chunk_end_frame": int(ref_len + visible_end),
                "chunk_input_start": int(input_start),
                "left_context_frames": int(context),
                "decode_start_frame": int(decode_start),
                "decode_end_frame": int(decode_end),
                "decode_prefix_frames": int(prefix_frames),
                "decoded_generated_frames_before": int(self.decoded_generated_frames),
            },
        )
        expected_samples = (decode_end - decode_start) * int(self.model.decode_upsample_rate)
        if decoded.lengths.reshape(-1)[0] < expected_samples:
            raise RuntimeError(
                "tokenizer decode chunk reported fewer samples than expected: "
                f"reported={int(decoded.lengths.reshape(-1)[0])}, expected={expected_samples}, "
                f"frames={decode_start}:{decode_end}, context={context}, path={self.model.tokenizer_decode_runner.path}"
            )
        if decoded.audio.shape[0] < expected_samples:
            raise RuntimeError(
                "tokenizer decode chunk output is shorter than expected: "
                f"audio_samples={decoded.audio.shape[0]}, expected={expected_samples}, "
                f"frames={decode_start}:{decode_end}, context={context}, path={self.model.tokenizer_decode_runner.path}"
            )
        audio = np.ascontiguousarray(decoded.audio[-expected_samples:].astype(np.float32, copy=False))
        self.decoded_generated_frames = visible_end
        self.sample_rate = int(decoded.sample_rate)
        return AudioGenerationOutputs(
            audio=audio,
            sample_rate=int(decoded.sample_rate),
            lengths=np.array([[audio.shape[0]]], dtype=np.int64),
            codes=visible_codes[None, :, :],
            context_frames=int(ref_len + visible_start),
            generated_frames=int(visible_end - visible_start),
            stopped=bool(final and (state.finished if state is not None else False)),
            stop_reason=str(state.stop_reason if state is not None else ("final" if final else "chunk")),
            code_generation=generation,
            metadata={
                **decoded.metadata,
                "streaming": True,
                "final": bool(final),
                "audio_chunk_frames": int(self.audio_chunk_frames),
                "first_audio_chunk_frames": int(self.first_audio_chunk_frames),
                "decoded_generated_frames": int(self.decoded_generated_frames),
                "discarded_prefix_frames": int(self.public_code_start_frame),
                "total_buffer_generated_frames": int(len(self.generated_codes)),
            },
        )


class StreamingSessionBase:
    state_cls = StreamingDecodeState

    def __init__(
        self,
        model: Qwen3TTSOnnxModelBase,
        language: str = "auto",
        max_new_tokens: int = 2048,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int = 0,
        eos_token_id: int | None = None,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        audio_chunk_frames: int = 300,
        left_context_frames: int = 25,
        min_text_chunk_chars: int = 20,
        max_text_chunk_chars: int = 80,
        kv_window_frames: int | None = 512,
        kv_window_max_frames: int | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample, self.top_k, self.top_p, self.temperature, self.repetition_penalty = (
            model._resolve_generation_options(
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
            )
        )
        self.min_new_tokens = int(min_new_tokens)
        self.eos_token_id = eos_token_id
        self.rng = model._resolve_rng(rng=rng, seed=seed)
        self.audio_chunk_frames = int(audio_chunk_frames)
        self.left_context_frames = int(left_context_frames)
        self.kv_window_frames = kv_window_frames
        self.kv_window_max_frames = kv_window_max_frames

        self.text_buffer = StreamingTextBuffer(min_chars=min_text_chunk_chars, max_chars=max_text_chunk_chars)
        self.text_embeds = StreamingTextEmbedQueue(model.prompt_builder)
        self.state: StreamingDecodeState | None = None
        self.audio_buffer: StreamingAudioChunkBuffer | None = None
        self.tts_pad_embed: np.ndarray | None = None
        self.tts_eos_embed: np.ndarray | None = None
        self._ended = False

    @property
    def sample_rate(self) -> int:
        if self.audio_buffer is None:
            return int(self.model.audio_sample_rate)
        return int(self.audio_buffer.sample_rate)

    @property
    def is_finished(self) -> bool:
        return self.state is not None and self.state.finished

    def push_text(self, text_fragment: str) -> list[AudioGenerationOutputs]:
        return list(self.push_text_iter(text_fragment))

    def push_text_iter(self, text_fragment: str):
        if self._ended:
            raise RuntimeError("Cannot push text after end_text() has been called.")
        for segment in self.text_buffer.push(text_fragment):
            if self.state is None:
                yield from self._start_iter(segment, include_initial_eos=False)
            else:
                yield from self._consume_text_segment_iter(segment, allow_eos=False)

    def end_text(self) -> list[AudioGenerationOutputs]:
        return list(self.end_text_iter())

    def end_text_iter(self):
        if not self._ended:
            tail = "".join(self.text_buffer.finish())
            if self.state is None:
                yield from self._start_iter(tail, include_initial_eos=True)
            else:
                if tail:
                    yield from self._consume_text_segment_iter(tail, allow_eos=False)
                if self.tts_eos_embed is None:
                    raise RuntimeError("streaming session is missing tts_eos_embed")
                self.text_embeds.append_embed(self.tts_eos_embed)
            self._ended = True
        yield from self._consume_pending_text_iter(allow_eos=True)
        yield from self._drain_to_eos_iter()
        final = self.flush()
        if final is not None and final.audio.size:
            yield final

    def drain(self, max_steps: int | None = None) -> list[AudioGenerationOutputs]:
        return list(self.drain_iter(max_steps=max_steps))

    def drain_iter(self, max_steps: int | None = None):
        yield from self._drain_to_eos_iter(max_steps=max_steps)

    def flush(self) -> AudioGenerationOutputs | None:
        if self.audio_buffer is None:
            return None
        return self.audio_buffer.flush(self.state)

    def generated_codes(self) -> np.ndarray:
        if self.state is None:
            return np.zeros((1, 0, self.model._num_code_groups()), dtype=np.int64)
        return self.state.codes_array()

    def _start_iter(self, initial_text: str, include_initial_eos: bool):
        self._start_state(initial_text=initial_text, include_initial_eos=include_initial_eos)
        yield from self._consume_pending_text_iter(allow_eos=include_initial_eos)
        if initial_text and self.state is not None:
            self.state.record_segment_boundary()

    def _consume_text_segment_iter(self, segment: str, discard: bool = False, allow_eos: bool = True):
        self.text_embeds.append_segment(segment)
        yield from self._consume_pending_text_iter(discard=discard, allow_eos=allow_eos)
        if self.state is not None and not discard:
            self.state.record_segment_boundary()

    def _build_streaming_prompt(self, initial_text: str, include_initial_eos: bool) -> PromptInputs:
        raise NotImplementedError

    def _resolve_tts_eos_embed(self, prompt: PromptInputs) -> np.ndarray:
        tts_eos_embed = prompt.metadata.get("tts_eos_embed")
        if tts_eos_embed is None:
            tts_eos_embed = self.model.prompt_builder._tts_special_embeds()[1]
        return tts_eos_embed

    def _start_state(self, initial_text: str, include_initial_eos: bool) -> None:
        prompt = self._build_streaming_prompt(initial_text=initial_text, include_initial_eos=include_initial_eos)
        self.tts_pad_embed = prompt.tts_pad_embed
        self.tts_eos_embed = self._resolve_tts_eos_embed(prompt)
        self.state = self.state_cls(
            model=self.model,
            prompt=prompt,
            max_new_tokens=self.max_new_tokens + int(prompt.trailing_text_hidden.shape[1]),
            do_sample=self.do_sample,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            min_new_tokens=self.min_new_tokens,
            eos_token_id=self.eos_token_id,
            rng=self.rng,
            kv_window_frames=self.kv_window_frames,
            kv_window_max_frames=self.kv_window_max_frames,
        )
        self.audio_buffer = StreamingAudioChunkBuffer(
            model=self.model,
            prompt=prompt,
            audio_chunk_frames=self.audio_chunk_frames,
            left_context_frames=self.left_context_frames,
        )
        self.text_embeds.extend_sequence(prompt.trailing_text_hidden)

    def _consume_pending_text_iter(self, discard: bool = False, allow_eos: bool = True):
        while self.state is not None and self.text_embeds and not self.state.finished:
            text_embed = self.text_embeds.pop()
            step_allow_eos = bool(allow_eos) and not discard
            code_row = self.state.step(text_embed, allow_eos=step_allow_eos)
            if code_row is not None and not discard:
                yield from self._push_code_iter(code_row)
            elif code_row is not None and discard and self.audio_buffer is not None:
                self.audio_buffer.append_code(code_row)

    def _drain_to_eos_iter(self, max_steps: int | None = None):
        if self.state is None:
            return
        if self.tts_pad_embed is None:
            raise RuntimeError("streaming session is missing tts_pad_embed")
        steps_left = self.state.max_codec_frames if max_steps is None else int(max_steps)
        while steps_left > 0 and not self.state.finished:
            code_row = self.state.step(self.tts_pad_embed)
            if code_row is not None:
                yield from self._push_code_iter(code_row)
            steps_left -= 1

    def _push_code_iter(self, code_row: np.ndarray):
        if self.audio_buffer is None or self.state is None:
            raise RuntimeError("streaming session has not been started")
        for chunk in self.audio_buffer.push_code(code_row, self.state):
            if chunk.audio.size:
                yield chunk


class VoiceCloneStreamingSession(StreamingSessionBase):
    def __init__(
        self,
        model: "BaseQwen3TTSOnnxModel",
        ref_audio: str | Path | tuple[np.ndarray, int],
        ref_text: Optional[str],
        language: str = "auto",
        x_vector_only_mode: bool = True,
        **kwargs,
    ) -> None:
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.x_vector_only_mode = bool(x_vector_only_mode)
        super().__init__(model=model, language=language, **kwargs)

    def _build_streaming_prompt(self, initial_text: str, include_initial_eos: bool) -> PromptInputs:
        return self.model.prompt_builder.build_streaming_from_reference(
            initial_text=initial_text,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
            language=self.language,
            x_vector_only_mode=self.x_vector_only_mode,
            include_initial_eos=include_initial_eos,
        )

class BaseQwen3TTSOnnxModel(Qwen3TTSOnnxModelBase):
    """Base voice-clone ONNX model path: prompt construction + talker prefill."""

    prompt_builder_cls = BaseVoiceClonePromptBuilder

    def _preload_sessions(self) -> None:
        self.prompt_builder._get_tokenizer_encoder_runner()
        self.prompt_builder._get_speaker_encoder_runner()
        self._preload_decode_sessions()
        _ = self.tokenizer_decode_runner

    def build_clone_prompt(
        self,
        text: str,
        language: str = "auto",
        ref_text: Optional[str] = None,
        ref_code: Optional[np.ndarray] = None,
        ref_spk_embedding: Optional[np.ndarray] = None,
        x_vector_only_mode: bool = False,
        non_streaming_mode: bool = False,
    ) -> PromptInputs:
        return self.prompt_builder.build(
            text=text,
            language=language,
            ref_text=ref_text,
            ref_code=ref_code,
            ref_spk_embedding=ref_spk_embedding,
            x_vector_only_mode=x_vector_only_mode,
            non_streaming_mode=non_streaming_mode,
        )

    def clone_prefill(
        self,
        text: str,
        language: str = "auto",
        ref_text: Optional[str] = None,
        ref_code: Optional[np.ndarray] = None,
        ref_spk_embedding: Optional[np.ndarray] = None,
        x_vector_only_mode: bool = False,
        non_streaming_mode: bool = False,
    ) -> TalkerPrefillOutputs:
        prompt = self.build_clone_prompt(
            text=text,
            language=language,
            ref_text=ref_text,
            ref_code=ref_code,
            ref_spk_embedding=ref_spk_embedding,
            x_vector_only_mode=x_vector_only_mode,
            non_streaming_mode=non_streaming_mode,
        )
        return self.run_prefill(prompt)

    def clone_prefill_from_prompt_item(
        self,
        text: str,
        prompt_item: VoiceClonePromptItem,
        language: str = "auto",
        non_streaming_mode: bool = False,
    ) -> TalkerPrefillOutputs:
        prompt = self.prompt_builder.build_from_prompt_item(
            text=text,
            prompt_item=prompt_item,
            language=language,
            non_streaming_mode=non_streaming_mode,
        )
        return self.run_prefill(prompt)

    def clone_prefill_from_reference(
        self,
        text: str,
        ref_audio: str | Path | tuple[np.ndarray, int],
        ref_text: Optional[str],
        language: str = "auto",
        x_vector_only_mode: bool = False,
        non_streaming_mode: bool = False,
    ) -> TalkerPrefillOutputs:
        prompt = self.prompt_builder.build_from_reference(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            x_vector_only_mode=x_vector_only_mode,
            non_streaming_mode=non_streaming_mode,
        )
        return self.run_prefill(prompt)

    def generate_clone_audio_from_reference(
        self,
        text: str,
        ref_audio: str | Path | tuple[np.ndarray, int],
        ref_text: Optional[str],
        language: str = "auto",
        x_vector_only_mode: bool = False,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 2048,
        do_sample: bool | None = True,
        top_k: int | None = 50,
        top_p: float | None = 1.0,
        temperature: float | None = 0.9,
        repetition_penalty: float | None = 1.05,
        min_new_tokens: int = 2,
        eos_token_id: int | None = None,
        rng: np.random.Generator | None = None,
        seed: int | None = 1234,
        return_timings: bool = False,
    ) -> AudioGenerationOutputs:
        timing = GenerationTimer() if return_timings else None
        start = time.perf_counter()
        prompt = self.prompt_builder.build_from_reference(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            x_vector_only_mode=x_vector_only_mode,
            non_streaming_mode=non_streaming_mode,
        )
        if timing is not None:
            timing.add("build_prompt", time.perf_counter() - start)
        result = self.generate_audio_from_prompt(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            eos_token_id=eos_token_id,
            rng=rng,
            seed=seed,
            timing=timing,
        )
        if timing is not None:
            result.metadata["timings"] = timing.summary()
        return result

    @staticmethod
    def _resolve_stream_context_mode(context_mode: str | None, x_vector_only_mode: bool) -> str:
        if context_mode is None:
            return "speaker_embedding" if x_vector_only_mode else "ref_code_icl"

        mode = str(context_mode or "speaker_embedding").lower()
        if mode in {"speaker", "speaker_embedding", "xvector", "x_vector"}:
            return "speaker_embedding"
        if mode in {"icl", "ref_code_icl", "in_context", "non_streaming_icl"}:
            return "ref_code_icl"
        raise ValueError("context_mode must be 'speaker_embedding' or 'ref_code_icl'")

    def create_clone_stream_from_reference(
        self,
        ref_audio: str | Path | tuple[np.ndarray, int],
        ref_text: Optional[str],
        language: str = "auto",
        x_vector_only_mode: bool = True,
        context_mode: str | None = None,
        max_new_tokens: int = 2048,
        do_sample: bool | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        temperature: float | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int = 0,
        eos_token_id: int | None = None,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        audio_chunk_frames: int = 300,
        left_context_frames: int = 25,
        min_text_chunk_chars: int = 20,
        max_text_chunk_chars: int = 80,
        kv_window_frames: int | None = 512,
        kv_window_max_frames: int | None = None,
    ) -> VoiceCloneStreamingSession:
        resolved_mode = self._resolve_stream_context_mode(context_mode, x_vector_only_mode)
        if resolved_mode == "ref_code_icl":
            return VoiceCloneStreamingSession(
                model=self,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=language,
                x_vector_only_mode=False,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_new_tokens=min_new_tokens,
                eos_token_id=eos_token_id,
                rng=rng,
                seed=seed,
                audio_chunk_frames=audio_chunk_frames,
                left_context_frames=left_context_frames,
                min_text_chunk_chars=min_text_chunk_chars,
                max_text_chunk_chars=max_text_chunk_chars,
                kv_window_frames=kv_window_frames,
                kv_window_max_frames=kv_window_max_frames,
            )
        return VoiceCloneStreamingSession(
            model=self,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            x_vector_only_mode=True,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            eos_token_id=eos_token_id,
            rng=rng,
            seed=seed,
            audio_chunk_frames=audio_chunk_frames,
            left_context_frames=left_context_frames,
            min_text_chunk_chars=min_text_chunk_chars,
            max_text_chunk_chars=max_text_chunk_chars,
            kv_window_frames=kv_window_frames,
            kv_window_max_frames=kv_window_max_frames,
        )

    def stream_clone_audio_from_reference(
        self,
        text_deltas,
        ref_audio: str | Path | tuple[np.ndarray, int],
        ref_text: Optional[str],
        language: str = "auto",
        x_vector_only_mode: bool = False,
        context_mode: str | None = "ref_code_icl",
        max_new_tokens: int = 2048,
        do_sample: bool | None = True,
        top_k: int | None = 50,
        top_p: float | None = 1.0,
        temperature: float | None = 0.9,
        repetition_penalty: float | None = 1,
        min_new_tokens: int = 2,
        eos_token_id: int | None = None,
        rng: np.random.Generator | None = None,
        seed: int | None = 1234,
        audio_chunk_frames: int = 300,
        left_context_frames: int = 25,
        min_text_chunk_chars: int = 20,
        max_text_chunk_chars: int = 80,
        kv_window_frames: int | None = 512,
        kv_window_max_frames: int | None = None,
    ):
        session = self.create_clone_stream_from_reference(
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            x_vector_only_mode=x_vector_only_mode,
            context_mode=context_mode,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
            eos_token_id=eos_token_id,
            rng=rng,
            seed=seed,
            audio_chunk_frames=audio_chunk_frames,
            left_context_frames=left_context_frames,
            min_text_chunk_chars=min_text_chunk_chars,
            max_text_chunk_chars=max_text_chunk_chars,
            kv_window_frames=kv_window_frames,
            kv_window_max_frames=kv_window_max_frames,
        )
        deltas = (text_deltas,) if isinstance(text_deltas, str) else text_deltas
        for delta in deltas:
            yield from session.push_text_iter(delta)
        yield from session.end_text_iter()
