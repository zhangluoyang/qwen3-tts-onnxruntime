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


@dataclass
class StreamingDecodeStateSnapshot:
    """一个合法的流式 decode 检查点，用于恢复到早期连续上下文后继续生成。"""

    logits: np.ndarray
    last_hidden: Any
    past_key_values: tuple[tuple[Any, Any], ...]
    frames_generated: int
    generated_first_tokens: tuple[int, ...]
    finished: bool
    stop_reason: str
    last_step: FrameDecodeStepOutputs | None


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

    def run_prefill(
        self,
        prompt: PromptInputs,
        timing: GenerationTimer | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> TalkerPrefillOutputs:
        runner = self.talker_core_runner
        output_names = list(runner.output_names)
        feed = self._make_talker_core_prefill_feed(prompt)
        self._print_stream_debug_feed("talker_core_prefill", feed, debug_context)
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

    def _debug_value_summary(self, value: Any) -> dict[str, Any]:
        summary: dict[str, Any] = {"shape": self._value_shape(value)}
        if is_ortvalue(value):
            summary["dtype"] = "OrtValue"
            return summary
        array = np.asarray(value)
        summary["dtype"] = str(array.dtype)
        return summary

    def _debug_feed_summary(self, feed: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {name: self._debug_value_summary(value) for name, value in feed.items()}

    def _print_stream_debug_feed(
        self,
        stage: str,
        feed: dict[str, Any],
        debug_context: dict[str, Any] | None,
    ) -> None:
        if not debug_context or not debug_context.get("debug_stream", False):
            return
        cache_position = feed.get("cache_position")
        cache_position_repr = None
        if cache_position is not None:
            cache_position_repr = repr(np.asarray(cache_position, dtype=np.int64))
        print(
            "[stream-debug][feed] "
            f"stage={stage} "
            f"segment_index={debug_context.get('segment_index')} "
            f"segment_source={debug_context.get('segment_source')} "
            f"segment_text={debug_context.get('segment_text')!r} "
            f"step_in_segment={debug_context.get('step_in_segment')} "
            f"global_frame={debug_context.get('global_frame')} "
            f"cache_position={cache_position_repr} "
            f"inputs={self._debug_feed_summary(feed)}",
            flush=True,
        )

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
        return_frame_embed: bool = False,
        timing: GenerationTimer | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> FrameDecodeStepOutputs:
        return self._run_split_frame_decode_step(
            first_token=first_token,
            last_hidden=last_hidden,
            past_key_values=past_key_values,
            text_embed=text_embed,
            generation_step=generation_step,
            return_frame_embed=return_frame_embed,
            timing=timing,
            debug_context=debug_context,
        )

    def _run_split_frame_decode_step(
        self,
        first_token: int | np.ndarray,
        last_hidden: Any,
        past_key_values: tuple[tuple[Any, Any], ...],
        text_embed: np.ndarray,
        generation_step: int,
        return_frame_embed: bool = False,
        timing: GenerationTimer | None = None,
        debug_context: dict[str, Any] | None = None,
    ) -> FrameDecodeStepOutputs:
        sample_runner = self.sub_talker_sample_runner
        sample_output_names = ["codebook_tokens", "decode_embed"]
        if return_frame_embed:
            sample_output_names.insert(1, "frame_embed")
        sample_feed = self._make_sub_talker_sample_feed(
            first_token=first_token,
            last_hidden=last_hidden,
            text_embed=text_embed,
        )
        self._print_stream_debug_feed("sub_talker_sample", sample_feed, debug_context)
        start = time.perf_counter()
        sample_outputs = sample_runner.run(
            output_names=sample_output_names,
            feed=sample_feed,
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
        core_feed = self._make_talker_core_decode_feed(
            decode_embed=sample_raw["decode_embed"],
            past_key_values=past_key_values,
        )
        self._print_stream_debug_feed("talker_core_decode", core_feed, debug_context)
        start = time.perf_counter()
        core_outputs = core_runner.run(
            output_names=core_output_names,
            feed=core_feed,
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
        # 固定完整 prompt 的“伪流式”codec 生成入口。
        # 这里文本已经全部在 prompt 里了，只是把自回归生成的 codec frame
        # 按 chunk_frames 分批 yield 出去，便于下游边生成边解码音频。
        rng = self._resolve_rng(rng=rng, seed=seed)
        eos_token_id = self._codec_eos_token_id(eos_token_id)
        # prefill 会把完整 prompt 先跑进 talker_core，并得到第一步采样所需的
        # logits、last_hidden 以及 past_key_values。后续每生成一帧，KV cache 都会增长。
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
        # tokenizer12hz_decode_chunk.onnx 的最小调用单元。
        # audio_codes 可以包含左上下文，context_frames 表示前面多少帧只是上下文，
        # 不应该被当成本次新生成的音频长度来统计。
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
        # ONNX trace 时有固定的最大 chunk 长度；如果模型报告的有效长度超过实际输出，
        # 基本说明导出的 tokenizer decode chunk 太短，需要重新导出。
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
        # 非 streaming API 最终也走分块 decode，避免一次把很长的 codec 序列喂给
        # tokenizer decoder。这里的分块是“音频解码分块”，不是文本 delta 流式。
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
            # 每块额外带一点左上下文，让 vocoder/tokenizer decoder 在块边界更平滑。
            # 后面只取 decoded.audio 的尾部 expected_samples，把上下文对应的音频裁掉。
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
            # decoded.audio 包含左上下文产生的音频；只保留本块新增 frame 对应的尾部。
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
        # 完整 prompt 的音频分块入口：先分块生成 codec，再把“尚未解码过”的新增
        # codec frame 解成音频块。注意这不是文本增量流式，文本在进入本函数前已经完整。
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
            # reference_context_frames 是参考音频/ref_code 的上下文长度；
            # previous_generated_frames 是前面已经吐给用户的生成帧。
            # 两者相加后，decode_codes_to_audio 只会返回这次新增 code_chunk 对应的音频。
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

        # _generate_code_frames 是完整 prompt 路径的自回归主循环。
        # chunk_frames=None 时一次性返回所有 codec；否则每攒够 chunk_frames 帧就 yield。
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
            # 主 codec codebook 的第一个 token 由 talker logits 采样得到；
            # 其它 codebook token 由 sub_talker_sample 根据 first_token/hidden/text_embed 补齐。
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
            # 第一帧使用 prefill 的 hidden/KV；之后每一帧都复用上一帧返回的 KV cache。
            # 当前实现没有 sliding window，past_key_values 会随着生成帧数持续增长。
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
                # 这里只切 codec，不做音频 decode；音频分块由 iter_audio_chunks_from_prompt
                # 或 StreamingAudioChunkBuffer 负责。
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

    @staticmethod
    def _iter_text_delta_stream(session, text_deltas):
        # 三类流式入口共享同一段调度：逐段推入文本 delta，最后统一 end_text。
        deltas = (text_deltas,) if isinstance(text_deltas, str) else text_deltas
        for delta in deltas:
            yield from session.push_text_iter(delta)
        yield from session.end_text_iter()

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
    ) -> dict[str, Any]:
        decode_embed_shape = self._value_shape(decode_embed)
        if len(decode_embed_shape) != 3 or decode_embed_shape[1] != 1:
            raise ValueError(
                "talker_core decode expects decode_embed shape [batch, 1, hidden], "
                f"got {decode_embed_shape}."
        )
        batch_size = decode_embed_shape[0]
        past_len = self._past_kv_length(past_key_values)
        feed: dict[str, Any] = {
            "inputs_embeds": decode_embed,
            "attention_mask": self._decode_attention_mask(batch_size=batch_size, past_len=past_len),
            "cache_position": self._decode_cache_position(past_len),
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
    """把外部传入的文本 delta 合并成适合 TTS 消费的小段。

    流式接口可能每次只收到几个字，也可能一次收到一长段。TTS 侧不适合逐字启动
    prompt/embedding，因此这里先缓存文本：达到 min_chars 后优先在标点处切分，
    如果一直没有标点，则超过 max_chars 后强制切一段，避免首包等待太久。
    """

    _split_pattern = re.compile(r"[。！？!?\.\u2026]\s*|[;；,，]\s*|\n")

    def __init__(self, min_chars: int = 20, max_chars: int = 80) -> None:
        self.min_chars = int(min_chars)
        self.max_chars = int(max_chars)
        self._cache = ""

    def push(self, text_fragment: str) -> list[str]:
        # 普通 push 不强制吐出尾巴，只有达到标点/长度阈值才返回 segment。
        self._cache += str(text_fragment or "")
        return self._extract(force=False)

    def finish(self) -> list[str]:
        # 输入结束时必须把缓存里的尾巴全部吐出来，即使长度不到 min_chars。
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
                # 优先在中文/英文句末标点、分号、换行处切，尽量保持语义完整。
                for match in self._split_pattern.finditer(self._cache):
                    if match.end() >= self.min_chars:
                        cut_idx = match.end()
                        break
            if cut_idx is None and len(self._cache) >= self.max_chars:
                # 没有合适标点时兜底切分；英文文本优先在空格处切，中文则按长度硬切。
                space_idx = self._cache.rfind(" ")
                cut_idx = space_idx + 1 if space_idx > 0 else self.max_chars
            if cut_idx is None:
                break
            segments.append(self._cache[:cut_idx])
            self._cache = self._cache[cut_idx:]
        return segments

class StreamingTextEmbedQueue:
    """流式文本 embedding 队列。

    每个 text segment 会先转成 [batch, seq, hidden]，再拆成一个个
    [batch, 1, hidden] 的 step embedding。后续每生成一帧 codec，就消费一个
    text_embed；文本耗尽后 session 会改用 tts_pad_embed 继续 drain 到 EOS。
    """

    def __init__(self, prompt_builder: BaseVoiceClonePromptBuilder) -> None:
        self.prompt_builder = prompt_builder
        self._items: deque[np.ndarray] = deque()

    def __bool__(self) -> bool:
        return bool(self._items)

    def append_segment(self, segment: str) -> None:
        # target_text_embeds 只处理新来的文本段，不会重算之前的 delta。
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
        # 拆成单步 embedding，方便 state.step 每次推进一个 codec frame。
        for index in range(embeds.shape[1]):
            self.append_embed(embeds[:, index:index + 1, :])

    def pop(self) -> np.ndarray:
        return self._items.popleft()


class StreamingDecodeState:
    """sample_then_advance 的通用流式 codec 生成状态。

    这个状态保存 talker 的自回归上下文：logits、last_hidden、past_key_values、
    已生成的 first tokens 和完整 codebook tokens。每调用一次 step，就消费一个
    text_embed 并生成一帧 codec code。注意 past_key_values 会随 step 数增长，
    音频 chunk flush 不会清掉这份 KV cache。
    """

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
        debug_context: dict[str, Any] | None = None,
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

        # prefill 只在 session 启动时执行一次，把参考音频/说话人条件/初始文本
        # 写入 talker 的 KV cache，并给出第一帧采样用的 logits。
        self.prefill = model.run_prefill(prompt, debug_context=debug_context)
        self.logits = self.prefill.logits
        self.last_hidden = self.prefill.last_hidden
        self.past_key_values = self.prefill.past_key_values
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
        debug_context: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        if self.finished:
            return None
        # max_new_tokens 仍然按对外已经生成的总 codec 帧数限制；即使后面为了
        # 控制 KV 恢复到 anchor，已经吐给用户的帧也不会从这个计数里消失。
        if len(self.generated_codes) >= self.max_codec_frames:
            self.finished = True
            self.stop_reason = "max_new_tokens"
            return None

        text_embed = np.ascontiguousarray(np.asarray(text_embed, dtype=self.model.dtype))
        # 统一采用 sample_then_advance：
        # 先用当前 logits 采样主 codebook 的 first_token，再调用 split frame decode
        # 生成完整 codebook，并推进 talker_core 得到下一步 logits/KV。
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

        # run_frame_decode_step 内部会先跑 sub_talker_sample，再跑 talker_core decode。
        # 返回的 step.past_key_values 比 self.past_key_values 多一帧上下文。
        step = self.model.run_frame_decode_step(
            first_token=first_token,
            last_hidden=self.last_hidden,
            past_key_values=self.past_key_values,
            text_embed=text_embed,
            generation_step=self.frames_generated,
            return_frame_embed=False,
            debug_context=debug_context,
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

    def kv_cache_length(self) -> int:
        return self.model._past_kv_length(self.past_key_values)

    def snapshot_decode_state(self) -> StreamingDecodeStateSnapshot:
        # 只保存模型继续自回归所需的内部状态。generated_codes/audio_buffer 不在这里保存，
        # 因为它们代表已经对外输出的音频时间线，恢复 anchor 时不能回滚。
        return StreamingDecodeStateSnapshot(
            logits=np.array(self.logits, copy=True),
            last_hidden=self.last_hidden,
            past_key_values=self.past_key_values,
            frames_generated=int(self.frames_generated),
            generated_first_tokens=tuple(int(token) for token in self.generated_first_tokens),
            finished=bool(self.finished),
            stop_reason=str(self.stop_reason),
            last_step=self.last_step,
        )

    def restore_decode_state(
        self,
        snapshot: StreamingDecodeStateSnapshot,
        reset_output_codes: bool = False,
    ) -> None:
        # 恢复的是“模型下一步该从哪里继续算”的状态。
        # reset_output_codes=True 用于段边界重启：旧分支音频已 flush 给用户，
        # 后续新分支 codes 不能再和旧分支 codes 拼在同一个 tokenizer decode 输入里。
        self.logits = np.array(snapshot.logits, copy=True)
        self.last_hidden = snapshot.last_hidden
        self.past_key_values = snapshot.past_key_values
        self.frames_generated = int(snapshot.frames_generated)
        self.generated_first_tokens = list(snapshot.generated_first_tokens)
        self.finished = bool(snapshot.finished)
        self.stop_reason = str(snapshot.stop_reason)
        self.last_step = snapshot.last_step
        if reset_output_codes:
            self.generated_codes = []
            self.public_code_start_frame = 0

    def mark_generated_prefix(self, prefix_frames: int) -> None:
        # 只影响对外可见的 codes，不会裁剪已经增长的 talker KV cache。
        self.public_code_start_frame = max(0, int(prefix_frames))

    def codes_array(self, include_discarded: bool = False) -> np.ndarray:
        # include_discarded=True 时用于内部诊断/对齐；默认只返回对外可见的新生成 codes。
        if not self.generated_codes:
            return np.zeros((1, 0, self.model._num_code_groups()), dtype=np.int64)
        start = 0 if include_discarded else int(self.public_code_start_frame)
        visible_codes = self.generated_codes[start:]
        if not visible_codes:
            return np.zeros((1, 0, self.model._num_code_groups()), dtype=np.int64)
        return np.stack(visible_codes, axis=0).astype(np.int64)[None, :, :]

    def code_generation_output(self) -> CodeGenerationOutputs:
        visible_frames = max(0, len(self.generated_codes) - int(self.public_code_start_frame))
        return CodeGenerationOutputs(
            codes=self.codes_array(),
            stopped=bool(self.finished),
            stop_reason=str(self.stop_reason),
            generated_frames=int(visible_frames),
            prefill=self.prefill,
            last_step=self.last_step,
            metadata={
                "streaming": True,
                "total_frames_generated": int(len(self.generated_codes)),
                "inference_frames_generated": int(self.frames_generated),
                "discarded_prefix_frames": int(self.public_code_start_frame),
                "generated_first_tokens": np.asarray(self.generated_first_tokens, dtype=np.int64),
                "eos_token_id": int(self.eos_token_id),
                "min_new_tokens": int(self.min_new_tokens),
                "max_codec_frames": int(self.max_codec_frames),
            },
        )

class StreamingAudioChunkBuffer:
    """把流式生成的 codec frame 缓冲并分块解码成 PCM 音频。

    生成端是一帧一帧吐 codec code，音频端不适合每帧都跑 tokenizer decoder，
    因此这里按 audio_chunk_frames 攒块。解码时会额外带 ref_code 和 left_context，
    但返回给调用方的 audio 只包含本次新增 frame 对应的尾部 samples。
    """

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

        # 如果 prompt 里有参考音频 codec，则音频 decode 时把它作为前缀上下文。
        # speaker_embedding 模式可能没有 ref_code，此时用空前缀。
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
        # code_row 是单帧 codec，形状应为 [num_code_groups]，通常是 16 个 codebook。
        row = np.asarray(code_row, dtype=np.int64).reshape(-1)
        expected_groups = self.model._num_code_groups()
        if row.shape != (expected_groups,):
            raise ValueError(f"code row must have shape [{expected_groups}], got {row.shape}")
        self.generated_codes.append(row)

    def push_code(self, code_row: np.ndarray, state: StreamingDecodeState) -> list[AudioGenerationOutputs]:
        self.append_code(code_row)
        # 首包和后续包目前使用同一个 chunk 大小；保留 first_audio_chunk_frames 字段，
        # 方便以后做“首包更小、后续更大”的低延迟策略。
        chunk_frames = (
            self.first_audio_chunk_frames
            if self.decoded_generated_frames == 0
            else self.audio_chunk_frames
        )
        if self.available_generated_frames() - self.decoded_generated_frames >= chunk_frames:
            return [self._decode_available(final=False, state=state)]
        return []

    def mark_prefix_codes(self, prefix_frames: int) -> None:
        # 标记一段 generated_codes 为已丢弃前缀。这里会让音频 decode 重新从可见区起点
        # 计算，但不会删除列表里的旧 code，也不会影响 talker KV。
        prefix_frames = max(0, int(prefix_frames))
        self.public_code_start_frame = max(int(self.public_code_start_frame), prefix_frames)
        self.decoded_generated_frames = 0

    def flush(self, state: StreamingDecodeState | None = None) -> AudioGenerationOutputs | None:
        # 文本结束或手动 flush 时，把不足一个 chunk 的剩余 codec 也解出来。
        if self.available_generated_frames() <= self.decoded_generated_frames:
            return None
        return self._decode_available(final=True, state=state)

    def available_generated_frames(self) -> int:
        return max(0, len(self.generated_codes) - int(self.public_code_start_frame))

    def codes_array(self, include_prefix: bool = False) -> np.ndarray:
        # 返回给 tokenizer decoder 的 code 序列始终包含 ref_code；
        # include_prefix=True 时还会包含被标记为 discarded 的生成前缀，
        # 这样左上下文索引仍然能对齐原始生成时间线。
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
        # visible_codes 是对外可见的完整 code；decode_codes 则包含可用于左上下文的前缀。
        visible_codes = self.codes_array()
        decode_codes = self.codes_array(include_prefix=True)
        ref_len = int(self.ref_code.shape[0])
        prefix_frames = int(self.public_code_start_frame)
        visible_start = int(self.decoded_generated_frames)
        visible_end = int(self.available_generated_frames())
        # decode_start/decode_end 使用“ref_code + discarded_prefix + visible_codes”的全局坐标。
        # visible_start 之前的生成帧已经转成音频吐出去了，本次只解 [visible_start, visible_end)。
        decode_start = ref_len + prefix_frames + visible_start
        decode_end = ref_len + prefix_frames + visible_end
        context = min(self.left_context_frames, decode_start)
        input_start = decode_start - context
        code_chunk = decode_codes[None, input_start:decode_end, :]
        generation = state.code_generation_output() if state is not None else None
        # _decode_code_chunk_to_audio 会输出包含左上下文的音频；下面再裁掉上下文 samples。
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
        # 每个 codec frame 对应 decode_upsample_rate 个 PCM sample。
        # expected_samples 只计算本次新增区间，不包含 left_context。
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
        # 只保留新增区间的尾部音频，避免把 left_context 对应音频重复吐给用户。
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
    """文本 delta -> codec frame -> 音频 chunk 的通用流式会话状态机。

    Base clone、CustomVoice、VoiceDesign 共享这套调度逻辑，只替换 prompt
    构建方式和少量条件字段。外部调用 push_text_iter 持续送入文本片段，
    session 内部负责切段、转 embedding、逐帧生成 codec，并在攒够
    audio_chunk_frames 后 yield 音频。
    """

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
        debug_stream: bool = False,
        max_kv_cache_len: int | None = None,
        kv_anchor_segment_count: int = 3,
        max_tts_pad_steps: int = 512,
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
        self.debug_stream = bool(debug_stream)
        self.max_kv_cache_len = None if max_kv_cache_len is None else int(max_kv_cache_len)
        self.kv_anchor_segment_count = int(kv_anchor_segment_count)
        self.max_tts_pad_steps = int(max_tts_pad_steps)
        if self.max_kv_cache_len is not None and self.max_kv_cache_len <= 0:
            raise ValueError("max_kv_cache_len must be positive when set")
        if self.kv_anchor_segment_count <= 0:
            raise ValueError("kv_anchor_segment_count must be positive")
        if self.max_tts_pad_steps < 0:
            raise ValueError("max_tts_pad_steps must be non-negative")

        # text_buffer 管“什么时候把文本 delta 切成段”，text_embeds 管“每帧消费哪个文本 embedding”。
        self.text_buffer = StreamingTextBuffer(min_chars=min_text_chunk_chars, max_chars=max_text_chunk_chars)
        self.text_embeds = StreamingTextEmbedQueue(model.prompt_builder)
        # state/audio_buffer 都是懒启动：第一段足够成句的文本到来时才构建 prompt 和 prefill。
        self.state: StreamingDecodeState | None = None
        self.audio_buffer: StreamingAudioChunkBuffer | None = None
        self.tts_pad_embed: np.ndarray | None = None
        self.tts_eos_embed: np.ndarray | None = None
        self._ended = False
        self._debug_segment_index = 0
        self._completed_text_segments = 0
        self._kv_anchor_checkpoint: StreamingDecodeStateSnapshot | None = None

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
        # 追加一段文本 delta。只有当 StreamingTextBuffer 切出完整 segment 时，
        # 才会真正推进模型；否则本次可能不 yield 任何音频。
        if self._ended:
            raise RuntimeError("Cannot push text after end_text() has been called.")
        for segment in self.text_buffer.push(text_fragment):
            if self.state is None:
                # 第一段文本会触发 prompt 构建和 prefill。
                yield from self._start_iter(segment, include_initial_eos=False)
            else:
                yield from self._consume_text_segment_iter(segment, allow_eos=False)

    def end_text(self) -> list[AudioGenerationOutputs]:
        return list(self.end_text_iter())

    def end_text_iter(self):
        # 通知 session 文本已经全部输入完毕。这里会：
        # 1. flush 文本 buffer 的尾巴；
        # 2. 追加 tts_eos_embed 允许模型看到文本结束；
        # 3. 用 tts_pad_embed 继续生成，直到 EOS 或 max_new_tokens；
        # 4. flush 最后一块不足 audio_chunk_frames 的音频。
        if not self._ended:
            tail = "".join(self.text_buffer.finish())
            if self.state is None:
                yield from self._start_iter(tail, include_initial_eos=True)
            else:
                if tail:
                    yield from self._consume_text_segment_iter(tail, allow_eos=False, source="tail")
                if self.tts_eos_embed is None:
                    raise RuntimeError("streaming session is missing tts_eos_embed")
                context = self._make_segment_debug_context(segment_text="<tts_eos>", source="tts_eos")
                self._print_stream_segment_debug("begin", context)
                self.text_embeds.append_embed(self.tts_eos_embed)
                yield from self._consume_pending_text_iter(allow_eos=True, debug_context=context)
                self._print_stream_segment_debug("end", context)
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

    def _debug_kv_summary(self) -> dict[str, Any]:
        # KV cache 的序列长度在 key/value 的第 3 维：[batch, heads, seq_len, head_dim]。
        # 这里只打印第一层形状和总层数，避免日志被完整 KV 内容淹没。
        if self.state is None:
            return {
                "length": None,
                "layers": 0,
                "first_key_shape": None,
                "first_value_shape": None,
                "frames_generated": 0,
                "total_output_frames": 0,
            }
        past_key_values = self.state.past_key_values
        if not past_key_values:
            return {
                "length": 0,
                "layers": 0,
                "first_key_shape": None,
                "first_value_shape": None,
                "frames_generated": int(self.state.frames_generated),
                "total_output_frames": int(len(self.state.generated_codes)),
            }
        first_key, first_value = past_key_values[0]
        return {
            "length": int(self.model._past_kv_length(past_key_values)),
            "layers": int(len(past_key_values)),
            "first_key_shape": self.model._value_shape(first_key),
            "first_value_shape": self.model._value_shape(first_value),
            "frames_generated": int(self.state.frames_generated),
            "total_output_frames": int(len(self.state.generated_codes)),
        }

    def _is_text_anchor_source(self, source: str | None) -> bool:
        return source in {"initial", "initial+tts_eos", "text", "tail"}

    def _is_kv_anchor_restore_source(self, source: str | None) -> bool:
        # 只在后续仍可能继续到来的普通 text segment 前恢复 anchor。
        # tail/tts_eos/tts_pad 已经处在收尾阶段，此时恢复会把最后一句从较早
        # 的上下文分支重新生成，容易在音色和语气上产生突兀跳变。
        return source in {"text"}

    def _print_stream_anchor_debug(
        self,
        kind: str,
        debug_context: dict[str, Any] | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.debug_stream:
            return
        print(
            f"[stream-debug][anchor.{kind}] "
            f"segment_index={None if debug_context is None else debug_context.get('segment_index')} "
            f"segment_source={None if debug_context is None else debug_context.get('segment_source')} "
            f"segment_text={None if debug_context is None else debug_context.get('segment_text')!r} "
            f"completed_text_segments={self._completed_text_segments} "
            f"anchor_segment_count={self.kv_anchor_segment_count} "
            f"kv_cache={self._debug_kv_summary()} "
            f"extra={extra or {}}",
            flush=True,
        )

    def _maybe_save_kv_anchor(self, debug_context: dict[str, Any] | None) -> None:
        if self.state is None or self._kv_anchor_checkpoint is not None:
            return
        source = None if debug_context is None else debug_context.get("segment_source")
        if not self._is_text_anchor_source(source):
            return
        if self.state.frames_generated <= 0:
            return
        self._completed_text_segments += 1
        if self._completed_text_segments != self.kv_anchor_segment_count:
            return
        self._kv_anchor_checkpoint = self.state.snapshot_decode_state()
        self._print_stream_anchor_debug(
            "save",
            debug_context,
            extra={
                "anchor_kv_cache_len": int(self.state.kv_cache_length()),
                "anchor_inference_frames": int(self.state.frames_generated),
                "total_output_frames": int(len(self.state.generated_codes)),
            },
        )

    def _should_restore_kv_anchor(self, debug_context: dict[str, Any] | None) -> bool:
        if self.state is None or self._kv_anchor_checkpoint is None or self.max_kv_cache_len is None:
            return False
        source = None if debug_context is None else debug_context.get("segment_source")
        if not self._is_kv_anchor_restore_source(source):
            return False
        current_len = int(self.state.kv_cache_length())
        return current_len > int(self.max_kv_cache_len)

    def _restore_kv_anchor(self, debug_context: dict[str, Any] | None, reset_output_codes: bool = False) -> bool:
        if self.state is None or self._kv_anchor_checkpoint is None:
            return False
        before = self._debug_kv_summary()
        self.state.restore_decode_state(
            self._kv_anchor_checkpoint,
            reset_output_codes=reset_output_codes,
        )
        if reset_output_codes and self.audio_buffer is not None:
            self.audio_buffer = StreamingAudioChunkBuffer(
                model=self.model,
                prompt=self.state.prompt,
                audio_chunk_frames=self.audio_chunk_frames,
                left_context_frames=self.left_context_frames,
            )
        self._print_stream_anchor_debug(
            "restore",
            debug_context,
            extra={
                "reason": "kv_cache_len_exceeded",
                "max_kv_cache_len": None if self.max_kv_cache_len is None else int(self.max_kv_cache_len),
                "reset_output_codes": bool(reset_output_codes),
                "kv_cache_before_restore": before,
                "kv_cache_after_restore": self._debug_kv_summary(),
                "branch_output_frames_after_restore": int(len(self.state.generated_codes)),
            },
        )
        return True

    def _make_segment_debug_context(self, segment_text: str, source: str) -> dict[str, Any] | None:
        context = {
            "segment_index": int(self._debug_segment_index),
            "segment_source": str(source),
            "segment_text": str(segment_text),
            "step_in_segment": None,
            "global_frame": None if self.state is None else int(self.state.frames_generated),
            "debug_stream": bool(self.debug_stream),
        }
        self._debug_segment_index += 1
        return context

    def _with_step_debug_context(
        self,
        debug_context: dict[str, Any] | None,
        step_in_segment: int,
    ) -> dict[str, Any] | None:
        if not debug_context:
            return None
        context = dict(debug_context)
        context["step_in_segment"] = int(step_in_segment)
        context["global_frame"] = None if self.state is None else int(self.state.frames_generated)
        return context

    def _print_stream_segment_debug(
        self,
        kind: str,
        debug_context: dict[str, Any] | None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not debug_context or not self.debug_stream:
            return
        kv_label = "kv_cache"
        if kind == "begin":
            kv_label = "kv_cache_before"
        elif kind == "end":
            kv_label = "kv_cache_after"
        print(
            f"[stream-debug][segment.{kind}] "
            f"segment_index={debug_context.get('segment_index')} "
            f"segment_source={debug_context.get('segment_source')} "
            f"segment_text={debug_context.get('segment_text')!r} "
            f"{kv_label}={self._debug_kv_summary()} "
            f"extra={extra or {}}",
            flush=True,
        )

    def _start_iter(self, initial_text: str, include_initial_eos: bool):
        # 初次启动：构建 streaming prompt、执行 prefill、初始化音频缓冲，
        # 然后立刻消费 prompt.trailing_text_hidden。
        source = "initial+tts_eos" if include_initial_eos else "initial"
        context = self._make_segment_debug_context(segment_text=initial_text, source=source)
        self._print_stream_segment_debug("begin", context)
        self._start_state(
            initial_text=initial_text,
            include_initial_eos=include_initial_eos,
            debug_context=context,
        )
        yield from self._consume_pending_text_iter(allow_eos=include_initial_eos, debug_context=context)
        self._print_stream_segment_debug("end", context)
        self._maybe_save_kv_anchor(context)

    def _consume_text_segment_iter(
        self,
        segment: str,
        discard: bool = False,
        allow_eos: bool = True,
        source: str = "text",
    ):
        # 将新文本段转成 embedding 后排队；真正的模型推进在 _consume_pending_text_iter。
        context = self._make_segment_debug_context(segment_text=segment, source=source)
        if self._should_restore_kv_anchor(context):
            pending = self.flush()
            if pending is not None and pending.audio.size:
                yield pending
            self._restore_kv_anchor(context, reset_output_codes=True)
        self._print_stream_segment_debug("begin", context)
        self.text_embeds.append_segment(segment)
        yield from self._consume_pending_text_iter(discard=discard, allow_eos=allow_eos, debug_context=context)
        self._print_stream_segment_debug("end", context)
        if not discard:
            self._maybe_save_kv_anchor(context)

    def _build_streaming_prompt(self, initial_text: str, include_initial_eos: bool) -> PromptInputs:
        raise NotImplementedError

    def _resolve_tts_eos_embed(self, prompt: PromptInputs) -> np.ndarray:
        tts_eos_embed = prompt.metadata.get("tts_eos_embed")
        if tts_eos_embed is None:
            tts_eos_embed = self.model.prompt_builder._tts_special_embeds()[1]
        return tts_eos_embed

    def _start_state(
        self,
        initial_text: str,
        include_initial_eos: bool,
        debug_context: dict[str, Any] | None = None,
    ) -> None:
        prompt = self._build_streaming_prompt(initial_text=initial_text, include_initial_eos=include_initial_eos)
        self.tts_pad_embed = prompt.tts_pad_embed
        self.tts_eos_embed = self._resolve_tts_eos_embed(prompt)
        # max_new_tokens 额外加上 trailing_text_hidden 长度，是为了给初始文本 embedding
        # 留出对应的 codec 生成步数，避免刚启动就触发 max_new_tokens。
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
            debug_context=debug_context,
        )
        self.audio_buffer = StreamingAudioChunkBuffer(
            model=self.model,
            prompt=prompt,
            audio_chunk_frames=self.audio_chunk_frames,
            left_context_frames=self.left_context_frames,
        )
        # prompt 构建时已经为 initial_text 准备了剩余待消费的 trailing_text_hidden；
        # 某些 prompt builder 会把第一帧 text/codec 条件提前放进 prefill。
        # 后续 push 进来的文本段则通过 append_segment 单独转 embedding。
        self.text_embeds.extend_sequence(prompt.trailing_text_hidden)

    def _consume_pending_text_iter(
        self,
        discard: bool = False,
        allow_eos: bool = True,
        debug_context: dict[str, Any] | None = None,
    ):
        # 每消费一个 text_embed，就生成一帧 codec。allow_eos=False 用于文本尚未结束时
        # 抑制 EOS，避免模型在后续 delta 到来前提前停掉。
        step_in_segment = 0
        while self.state is not None and self.text_embeds and not self.state.finished:
            text_embed = self.text_embeds.pop()
            step_allow_eos = bool(allow_eos) and not discard
            step_context = self._with_step_debug_context(debug_context, step_in_segment=step_in_segment)
            code_row = self.state.step(text_embed, allow_eos=step_allow_eos, debug_context=step_context)
            if code_row is not None and not discard:
                yield from self._push_code_iter(code_row)
            elif code_row is not None and discard and self.audio_buffer is not None:
                self.audio_buffer.append_code(code_row)
            step_in_segment += 1

    def _drain_to_eos_iter(self, max_steps: int | None = None):
        # 文本 embedding 消费完后，用 tts_pad_embed 继续喂模型，让声学序列完整展开。
        # 对 TTS 来说，文本 token 数远少于声学 frame 数，tts_pad 不是“多余尾巴”，
        # 而是模型在读完文本后继续吐出语音帧的主要阶段；这里只设置兜底上限，
        # 防止极端情况下迟迟不出 EOS 时无限生成。
        if self.state is None:
            return
        if self.tts_pad_embed is None:
            raise RuntimeError("streaming session is missing tts_pad_embed")
        remaining_steps = max(0, int(self.state.max_codec_frames) - int(len(self.state.generated_codes)))
        if max_steps is None:
            steps_left = min(remaining_steps, int(self.max_tts_pad_steps))
        else:
            steps_left = min(remaining_steps, int(max_steps))
        context = self._make_segment_debug_context(segment_text="<tts_pad>", source="tts_pad")
        self._print_stream_segment_debug(
            "begin",
            context,
            extra={
                "max_steps": steps_left,
                "remaining_steps": remaining_steps,
                "max_tts_pad_steps": int(self.max_tts_pad_steps),
            },
        )
        step_in_segment = 0
        while steps_left > 0 and not self.state.finished:
            step_context = self._with_step_debug_context(context, step_in_segment=step_in_segment)
            code_row = self.state.step(self.tts_pad_embed, debug_context=step_context)
            if code_row is not None:
                yield from self._push_code_iter(code_row)
            steps_left -= 1
            step_in_segment += 1
        self._print_stream_segment_debug("end", context, extra={"steps_consumed": step_in_segment})

    def _push_code_iter(self, code_row: np.ndarray):
        # codec frame 进入音频缓冲；只有攒够 chunk 或最终 flush 时才会 yield 音频。
        if self.audio_buffer is None or self.state is None:
            raise RuntimeError("streaming session has not been started")
        for chunk in self.audio_buffer.push_code(code_row, self.state):
            if chunk.audio.size:
                yield chunk


class VoiceCloneStreamingSession(StreamingSessionBase):
    """Base voice clone 的流式会话，只负责把参考音频/参考文本带进 prompt_builder。"""

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
        # x_vector_only_mode=True 时主要用 speaker embedding 条件；
        # False/ref_code_icl 时会把参考音频 codec 作为 in-context 前缀。
        return self.model.prompt_builder.build_streaming_from_reference(
            initial_text=initial_text,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
            language=self.language,
            x_vector_only_mode=self.x_vector_only_mode,
            include_initial_eos=include_initial_eos,
            defer_target_text=True,
        )


class ConditionedSegmentStreamingSession:
    """Base ICL、Custom、Design 共用的分段重组 prompt 流式会话。

    每个文本 segment 单独构建一个完整的 non-streaming prompt，不复用
    上一段的 KV cache。为了让音色和语气连续，把最近若干段已经生成的
    codec codes 作为 anchor 放进下一段 prompt 的 in-context 前缀里。
    """

    stream_debug_prefix = "conditioned_segment"
    stream_debug_mode = "conditioned_segment_prompt"
    init_debug_prefix: str | None = None

    def __init__(
        self,
        model: "Qwen3TTSOnnxModelBase",
        language: str = "auto",
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
        min_text_chunk_chars: int = 20,
        max_text_chunk_chars: int = 80,
        debug_stream: bool = False,
        kv_anchor_segment_count: int = 3,
        pinned_anchor_segment_count: int = 0,
        **_: Any,
    ) -> None:
        self.model = model
        self.language = language
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = do_sample
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.min_new_tokens = int(min_new_tokens)
        self.eos_token_id = eos_token_id
        self.rng = model._resolve_rng(rng=rng, seed=seed)
        self.text_buffer = StreamingTextBuffer(min_chars=min_text_chunk_chars, max_chars=max_text_chunk_chars)
        self.debug_stream = bool(debug_stream)
        self.kv_anchor_segment_count = max(0, int(kv_anchor_segment_count))
        self.pinned_anchor_segment_count = max(0, int(pinned_anchor_segment_count))
        self._pinned_anchor_segments: list[tuple[str, np.ndarray]] = []
        self._anchor_segments: deque[tuple[str, np.ndarray]] = deque(maxlen=self.kv_anchor_segment_count)
        self._rolling_anchor_evicted = False
        self._ended = False
        self._segment_index = 0

        if self.debug_stream:
            init_extra = self._init_debug_extra()
            init_extra_text = "".join(f" {key}={value!r}" for key, value in init_extra.items())
            debug_prefix = self.init_debug_prefix or self.stream_debug_prefix
            print(
                f"[stream-debug][{debug_prefix}.init] "
                f"mode={self.stream_debug_mode} "
                f"{init_extra_text} "
                f"kv_anchor_segment_count={int(self.kv_anchor_segment_count)} "
                f"pinned_anchor_segment_count={int(self.pinned_anchor_segment_count)} "
                "max_kv_cache_len_behavior='not_used_because_each_segment_builds_a_new_prompt'",
                flush=True,
            )

    @property
    def sample_rate(self) -> int:
        return int(self.model.audio_sample_rate)

    @property
    def is_finished(self) -> bool:
        return self._ended

    def push_text(self, text_fragment: str) -> list[AudioGenerationOutputs]:
        return list(self.push_text_iter(text_fragment))

    def push_text_iter(self, text_fragment: str):
        if self._ended:
            raise RuntimeError("Cannot push text after end_text() has been called.")
        for segment in self.text_buffer.push(text_fragment):
            yield from self._generate_segment_iter(segment=segment, source="text")

    def end_text(self) -> list[AudioGenerationOutputs]:
        return list(self.end_text_iter())

    def end_text_iter(self):
        if not self._ended:
            tail = "".join(self.text_buffer.finish())
            if tail:
                yield from self._generate_segment_iter(segment=tail, source="tail")
            self._ended = True

    def drain(self, max_steps: int | None = None) -> list[AudioGenerationOutputs]:
        return []

    def drain_iter(self, max_steps: int | None = None):
        if False:
            yield None

    def flush(self) -> AudioGenerationOutputs | None:
        return None

    def generated_codes(self) -> np.ndarray:
        anchor_segments = self._all_anchor_segments()
        if not anchor_segments:
            return np.zeros((1, 0, self.model._num_code_groups()), dtype=np.int64)
        return np.concatenate([codes for _, codes in anchor_segments], axis=1)

    def _build_segment_prompt(
        self,
        segment: str,
        anchor_text: str,
        anchor_code: np.ndarray | None,
    ) -> PromptInputs:
        raise NotImplementedError

    def _combined_anchor(self) -> tuple[str, np.ndarray | None]:
        anchor_segments = self._all_anchor_segments()
        if not anchor_segments:
            return "", None
        anchor_text = "".join(text for text, _ in anchor_segments)
        anchor_code = np.concatenate(
            [codes.reshape(1, -1, self.model._num_code_groups()) for _, codes in anchor_segments],
            axis=1,
        )
        return anchor_text, anchor_code

    def _init_debug_extra(self) -> dict[str, Any]:
        return {}

    def _anchor_code_len(self, anchor_code: np.ndarray | None) -> int:
        if anchor_code is None:
            return 0
        if anchor_code.ndim == 3:
            return int(anchor_code.shape[1])
        if anchor_code.ndim == 2:
            return int(anchor_code.shape[0])
        return 0

    def _begin_debug_extra(
        self,
        anchor_text: str,
        anchor_code: np.ndarray | None,
        anchor_summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "new_prompt": True,
            "kv_cache_reused": False,
            "anchor_text_chars": int(len(anchor_text)),
            "anchor_code_len": self._anchor_code_len(anchor_code),
            **anchor_summary,
        }

    def _prompt_debug_extra(self, prompt: PromptInputs) -> dict[str, Any]:
        return {
            "prompt_anchor_code_len": int(prompt.metadata.get("anchor_code_len", 0)),
        }

    def _all_anchor_segments(self) -> list[tuple[str, np.ndarray]]:
        if self._rolling_anchor_evicted:
            return [*self._anchor_segments]
        return [*self._pinned_anchor_segments, *self._anchor_segments]

    def _save_anchor_segment(self, segment: str, generated_codes: np.ndarray) -> bool:
        if generated_codes.shape[1] <= 0:
            return False
        if len(self._pinned_anchor_segments) < self.pinned_anchor_segment_count:
            self._pinned_anchor_segments.append((segment, generated_codes))
            return True
        if self.kv_anchor_segment_count > 0:
            if len(self._anchor_segments) == self.kv_anchor_segment_count:
                self._rolling_anchor_evicted = True
            self._anchor_segments.append((segment, generated_codes))
            return True
        return False

    def _anchor_debug_summary(self) -> dict[str, Any]:
        anchor_frames = [
            int(codes.reshape(1, -1, self.model._num_code_groups()).shape[1])
            for _, codes in self._anchor_segments
        ]
        pinned_anchor_frames = [
            int(codes.reshape(1, -1, self.model._num_code_groups()).shape[1])
            for _, codes in self._pinned_anchor_segments
        ]
        active_pinned_anchor_frames = [] if self._rolling_anchor_evicted else pinned_anchor_frames
        return {
            "pinned_anchor_segment_count": int(len(self._pinned_anchor_segments)),
            "pinned_anchor_active": bool(self._pinned_anchor_segments and not self._rolling_anchor_evicted),
            "rolling_anchor_evicted": bool(self._rolling_anchor_evicted),
            "pinned_anchor_segment_frames": pinned_anchor_frames,
            "pinned_anchor_texts": [text for text, _ in self._pinned_anchor_segments],
            "anchor_segment_count": int(len(self._anchor_segments)),
            "anchor_segment_frames": anchor_frames,
            "anchor_total_frames": int(sum(active_pinned_anchor_frames) + sum(anchor_frames)),
            "anchor_texts": [text for text, _ in self._anchor_segments],
        }

    def _print_segment_debug(
        self,
        kind: str,
        segment_index: int,
        source: str,
        segment: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.debug_stream:
            return
        print(
            f"[stream-debug][{self.stream_debug_prefix}.{kind}] "
            f"segment_index={segment_index} "
            f"segment_source={source} "
            f"segment_text={segment!r} "
            f"anchor_segments={len(self._all_anchor_segments())} "
            f"extra={extra or {}}",
            flush=True,
        )

    def _generate_segment_iter(self, segment: str, source: str):
        segment = str(segment or "")
        if not segment:
            return
        segment_index = self._segment_index
        self._segment_index += 1
        anchor_text, anchor_code = self._combined_anchor()
        anchor_summary = self._anchor_debug_summary()
        self._print_segment_debug(
            "begin",
            segment_index,
            source,
            segment,
            extra=self._begin_debug_extra(
                anchor_text=anchor_text,
                anchor_code=anchor_code,
                anchor_summary=anchor_summary,
            ),
        )

        prompt = self._build_segment_prompt(
            segment=segment,
            anchor_text=anchor_text,
            anchor_code=anchor_code,
        )
        self._print_segment_debug(
            "prompt",
            segment_index,
            source,
            segment,
            extra={
                "inputs_embeds_shape": tuple(int(dim) for dim in prompt.inputs_embeds.shape),
                "attention_mask_shape": tuple(int(dim) for dim in prompt.attention_mask.shape),
                "trailing_text_hidden_shape": tuple(int(dim) for dim in prompt.trailing_text_hidden.shape),
                "trailing_text_hidden_is_tts_pad": bool(prompt.trailing_text_hidden.shape[1] == 1),
                "non_streaming_mode": bool(prompt.metadata.get("non_streaming_mode")),
                "streaming_strategy": "segment_reprompt_with_anchor_codes",
                **self._prompt_debug_extra(prompt),
            },
        )

        output = self.model.generate_audio_from_prompt(
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            min_new_tokens=self.min_new_tokens,
            eos_token_id=self.eos_token_id,
            rng=self.rng,
            seed=None,
        )
        generated_codes = output.code_generation.codes if output.code_generation is not None else output.codes
        generated_codes = np.asarray(generated_codes, dtype=np.int64)
        if generated_codes.ndim == 2:
            generated_codes = generated_codes[None, :, :]
        saved_as_next_anchor = self._save_anchor_segment(segment, generated_codes)

        self._print_segment_debug(
            "end",
            segment_index,
            source,
            segment,
            extra={
                "generated_frames": int(generated_codes.shape[1]),
                "audio_samples": int(output.audio.shape[0]),
                "stopped": bool(output.stopped),
                "stop_reason": str(output.stop_reason),
                "saved_as_next_anchor": bool(saved_as_next_anchor),
                "next_segment_will_reprompt": True,
                **self._anchor_debug_summary(),
            },
        )
        yield output


class VoiceCloneIclSegmentStreamingSession(ConditionedSegmentStreamingSession):
    """Base ref-code ICL 的分段重组 prompt 流式会话。"""

    stream_debug_prefix = "icl_segment"
    init_debug_prefix = "icl_session"
    stream_debug_mode = "base_ref_code_icl_segment_prompt"

    def __init__(
        self,
        model: "BaseQwen3TTSOnnxModel",
        ref_audio: str | Path | tuple[np.ndarray, int],
        ref_text: Optional[str],
        language: str = "auto",
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
        min_text_chunk_chars: int = 20,
        max_text_chunk_chars: int = 80,
        debug_stream: bool = False,
        kv_anchor_segment_count: int = 1,
        **kwargs: Any,
    ) -> None:
        self.ref_audio = ref_audio
        self.ref_text = ref_text or ""
        prompt_builder = model.prompt_builder
        audio_key = prompt_builder._reference_audio_cache_key(ref_audio)
        audio, sr = prompt_builder._load_reference_audio_cached(ref_audio, audio_key)
        self._ref_code = prompt_builder._encode_reference_audio_cached(audio_key, audio, sr)
        self._speaker_embedding = prompt_builder._encode_speaker_embedding_cached(audio_key, audio, sr)
        super().__init__(
            model=model,
            language=language,
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
            min_text_chunk_chars=min_text_chunk_chars,
            max_text_chunk_chars=max_text_chunk_chars,
            debug_stream=debug_stream,
            kv_anchor_segment_count=kv_anchor_segment_count,
            **kwargs,
        )

    def _init_debug_extra(self) -> dict[str, Any]:
        return {
            "ref_text": self.ref_text,
            "ref_code_len": int(self._ref_code.shape[0]),
        }

    def _combined_anchor(self) -> tuple[str, np.ndarray]:
        anchor_segments = self._all_anchor_segments()
        if not anchor_segments:
            return "", self._ref_code
        anchor_text = "".join(text for text, _ in anchor_segments)
        anchor_codes = np.concatenate(
            [codes.reshape(1, -1, self.model._num_code_groups())[0] for _, codes in anchor_segments],
            axis=0,
        )
        return anchor_text, np.concatenate([self._ref_code, anchor_codes], axis=0)

    def _anchor_debug_summary(self) -> dict[str, Any]:
        return {
            "base_ref_code_len": int(self._ref_code.shape[0]),
            **super()._anchor_debug_summary(),
        }

    def _begin_debug_extra(
        self,
        anchor_text: str,
        anchor_code: np.ndarray | None,
        anchor_summary: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "new_prompt": True,
            "kv_cache_reused": False,
            "ref_code_len_with_anchor": self._anchor_code_len(anchor_code),
            "ref_code_len_formula": "base_ref_code_len + anchor_total_frames",
            "anchor_text_chars": int(len(anchor_text)),
            **anchor_summary,
        }

    def _prompt_debug_extra(self, prompt: PromptInputs) -> dict[str, Any]:
        return {
            "prompt_ref_code_len": int(prompt.metadata.get("ref_code_len", -1)),
        }

    def _build_segment_prompt(
        self,
        segment: str,
        anchor_text: str,
        anchor_code: np.ndarray | None,
    ) -> PromptInputs:
        if anchor_code is None:
            anchor_code = self._ref_code
        return self.model.prompt_builder.build(
            text=segment,
            language=self.language,
            ref_text=self.ref_text + anchor_text,
            ref_code=anchor_code,
            ref_spk_embedding=self._speaker_embedding,
            x_vector_only_mode=False,
            non_streaming_mode=True,
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
        debug_stream: bool = False,
        max_kv_cache_len: int | None = None,
        kv_anchor_segment_count: int = 3,
        max_tts_pad_steps: int = 512,
    ) -> VoiceCloneStreamingSession:
        # Base clone 支持两种流式上下文：
        # - speaker_embedding：只用说话人向量，音频 decode 没有 ref_code 前缀；
        # - ref_code_icl：把参考音频 codec 放进 prompt/decoder 上下文，音色更贴近参考。
        resolved_mode = self._resolve_stream_context_mode(context_mode, x_vector_only_mode)
        if resolved_mode == "ref_code_icl":
            return VoiceCloneIclSegmentStreamingSession(
                model=self,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=language,
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
                debug_stream=debug_stream,
                max_kv_cache_len=max_kv_cache_len,
                kv_anchor_segment_count=kv_anchor_segment_count,
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
            debug_stream=debug_stream,
            max_kv_cache_len=max_kv_cache_len,
            kv_anchor_segment_count=kv_anchor_segment_count,
            max_tts_pad_steps=max_tts_pad_steps,
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
        debug_stream: bool = False,
        max_kv_cache_len: int | None = None,
        kv_anchor_segment_count: int = 3,
        max_tts_pad_steps: int = 512,
    ):
        # 对外最常用的 Base clone 流式入口。text_deltas 可以是字符串，也可以是
        # 逐段文本的可迭代对象；函数本身是 generator，会持续 yield AudioGenerationOutputs。
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
            debug_stream=debug_stream,
            max_kv_cache_len=max_kv_cache_len,
            kv_anchor_segment_count=kv_anchor_segment_count,
            max_tts_pad_steps=max_tts_pad_steps,
        )
        yield from self._iter_text_delta_stream(session, text_deltas)
