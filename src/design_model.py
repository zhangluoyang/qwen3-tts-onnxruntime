from __future__ import annotations

import time
import numpy as np

from src.builders import PromptInputs, VoiceDesignPromptBuilder
from src.models import (
    AudioGenerationOutputs,
    CodeGenerationOutputs,
    FrameDecodeStepOutputs,
    GenerationTimer,
    Qwen3TTSOnnxModelBase,
    SegmentKVWindowMixin,
    StreamingSessionBase,
    TalkerPrefillOutputs,
)


class VoiceDesignStreamingDecodeState(SegmentKVWindowMixin):
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
        initial_codec_embed = prompt.metadata.get("initial_codec_embed")
        if initial_codec_embed is None:
            raise ValueError("streaming prompt metadata must include initial_codec_embed")
        initial_codec_embed = np.asarray(initial_codec_embed, dtype=model.dtype)
        if initial_codec_embed.ndim != 3 or initial_codec_embed.shape[1] != 1:
            raise ValueError(f"initial_codec_embed must have shape [batch, 1, hidden], got {initial_codec_embed.shape}")
        self.next_codec_embed = np.ascontiguousarray(initial_codec_embed)
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
        decode_embed = np.ascontiguousarray((self.next_codec_embed + text_embed).astype(self.model.dtype, copy=False))
        logits, last_hidden, past_key_values, core_raw = self.model.run_talker_core_decode_step(
            decode_embed=decode_embed,
            past_key_values=self.past_key_values,
            cache_position=self._next_cache_position(),
        )
        self.logits = logits
        self.last_hidden = last_hidden
        self.past_key_values = past_key_values

        first_token = self.model._sample_first_token(
            logits=logits,
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

        codebook_tokens, frame_embed, sample_raw = self.model.run_sub_talker_sample_step(
            first_token=first_token,
            last_hidden=last_hidden,
            text_embed=text_embed,
        )
        code_row = np.asarray(codebook_tokens, dtype=np.int64).reshape(-1)
        self.generated_first_tokens.append(int(first_token))
        self.generated_codes.append(code_row)
        self.frames_generated += 1
        self.next_codec_embed = frame_embed
        self.last_step = FrameDecodeStepOutputs(
            logits=logits,
            last_hidden=last_hidden,
            codebook_tokens=codebook_tokens,
            frame_embed=frame_embed,
            past_key_values=past_key_values,
            raw_outputs={
                "logits": logits,
                "last_hidden_out": last_hidden,
                "codebook_tokens": codebook_tokens,
                "frame_embed": frame_embed,
                **{f"sample.{key}": value for key, value in sample_raw.items()},
                **{f"core.{key}": value for key, value in core_raw.items()},
            },
            metadata={
                "streaming_decode_order": "advance_then_sample",
                "generation_step": int(self.frames_generated - 1),
                "split_talker": True,
            },
        )
        return code_row

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

class VoiceDesignStreamingSession(StreamingSessionBase):
    """Streaming session for VoiceDesign prompt builders."""

    state_cls = VoiceDesignStreamingDecodeState

    def __init__(
        self,
        model: "DesignQwen3TTSOnnxModel",
        language: str = "auto",
        instruct: str | None = None,
        audio_chunk_frames: int = 50,
        **kwargs,
    ) -> None:
        self.instruct = instruct
        super().__init__(
            model=model,
            language=language,
            audio_chunk_frames=audio_chunk_frames,
            **kwargs,
        )

    def _build_streaming_prompt(self, initial_text: str, include_initial_eos: bool) -> PromptInputs:
        return self.model.prompt_builder.build_streaming(
            instruct=self.instruct or "",
            language=self.language,
            initial_text=initial_text,
            include_initial_eos=include_initial_eos,
        )

class DesignQwen3TTSOnnxModel(Qwen3TTSOnnxModelBase):
    prompt_builder_cls = VoiceDesignPromptBuilder

    def _preload_sessions(self) -> None:
        self._preload_decode_sessions()
        _ = self.tokenizer_decode_runner

    def build_prompt(
        self,
        text: str,
        instruct: str,
        language: str = "auto",
        non_streaming_mode: bool = True,
    ) -> PromptInputs:
        return self.prompt_builder.build(
            text=text,
            instruct=instruct,
            language=language,
            non_streaming_mode=non_streaming_mode,
        )

    def prefill(self, *args, **kwargs) -> TalkerPrefillOutputs:
        return self.run_prefill(self.build_prompt(*args, **kwargs))

    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str = "auto",
        non_streaming_mode: bool = True,
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
        prompt = self.build_prompt(
            text=text,
            instruct=instruct,
            language=language,
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

    def iter_voice_design_audio_chunks(self, text: str, instruct: str, language: str = "auto", **kwargs):
        prompt = self.build_prompt(text=text, instruct=instruct, language=language, non_streaming_mode=True)
        yield from self.iter_audio_chunks_from_prompt(prompt=prompt, **kwargs)

    def create_voice_design_stream(
        self,
        instruct: str,
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
        seed: int | None = None,
        audio_chunk_frames: int = 50,
        left_context_frames: int = 25,
        min_text_chunk_chars: int = 20,
        max_text_chunk_chars: int = 80,
        kv_window_frames: int | None = 512,
        kv_window_max_frames: int | None = None,
    ):
        return VoiceDesignStreamingSession(
            model=self,
            language=language,
            instruct=instruct,
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

    def stream_voice_design(
        self,
        text_deltas,
        instruct: str,
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
        audio_chunk_frames: int = 50,
        left_context_frames: int = 25,
        min_text_chunk_chars: int = 20,
        max_text_chunk_chars: int = 80,
        kv_window_frames: int | None = 512,
        kv_window_max_frames: int | None = None,
    ):
        session = self.create_voice_design_stream(
            instruct=instruct,
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
            kv_window_frames=kv_window_frames,
            kv_window_max_frames=kv_window_max_frames,
        )
        deltas = (text_deltas,) if isinstance(text_deltas, str) else text_deltas
        for delta in deltas:
            yield from session.push_text_iter(delta)
        yield from session.end_text_iter()
