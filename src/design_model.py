from __future__ import annotations

import time
import numpy as np
from src.builders import PromptInputs, VoiceDesignPromptBuilder
from src.models import (
    AudioGenerationOutputs,
    ConditionedSegmentStreamingSession,
    GenerationTimer,
    Qwen3TTSOnnxModelBase,
    TalkerPrefillOutputs,
)


class VoiceDesignStreamingSession(ConditionedSegmentStreamingSession):
    """VoiceDesign 的分段重组 prompt 流式会话。

    每个 segment 都重新构建完整的 VoiceDesign non-streaming prompt；
    最近若干段已生成的 codec codes 会作为 anchor 放进下一段 prompt 前缀，
    因此不会再沿用旧流式路径里不断增长的 KV cache。
    """

    stream_debug_prefix = "design_segment"
    stream_debug_mode = "voice_design_segment_prompt"

    def __init__(
        self,
        model: "DesignQwen3TTSOnnxModel",
        language: str = "auto",
        instruct: str | None = None,
        audio_chunk_frames: int = 50,
        pinned_anchor_segment_count: int = 1,
        **kwargs,
    ) -> None:
        self.instruct = instruct
        super().__init__(
            model=model,
            language=language,
            audio_chunk_frames=audio_chunk_frames,
            pinned_anchor_segment_count=pinned_anchor_segment_count,
            **kwargs,
        )

    def _build_segment_prompt(
        self,
        segment: str,
        anchor_text: str,
        anchor_code: np.ndarray | None,
    ) -> PromptInputs:
        # VoiceDesign 的 instruct 每段都会重新放进 prompt；
        # anchor_text/anchor_code 只负责携带最近几段生成过的上下文音频。
        return self.model.prompt_builder.build(
            text=segment,
            instruct=self.instruct or "",
            language=self.language,
            non_streaming_mode=True,
            anchor_text=anchor_text,
            anchor_code=anchor_code,
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
        audio_chunk_frames: int = 300,
        left_context_frames: int = 25,
        min_text_chunk_chars: int = 20,
        max_text_chunk_chars: int = 80,
        debug_stream: bool = False,
        max_kv_cache_len: int | None = None,
        kv_anchor_segment_count: int = 3,
        pinned_anchor_segment_count: int = 1,
    ):
        # 返回可手动 push_text/end_text 的会话；默认 audio_chunk_frames=300，
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
            debug_stream=debug_stream,
            max_kv_cache_len=max_kv_cache_len,
            kv_anchor_segment_count=kv_anchor_segment_count,
            pinned_anchor_segment_count=pinned_anchor_segment_count,
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
        audio_chunk_frames: int = 300,
        left_context_frames: int = 25,
        min_text_chunk_chars: int = 20,
        max_text_chunk_chars: int = 80,
        debug_stream: bool = False,
        max_kv_cache_len: int | None = None,
        kv_anchor_segment_count: int = 3,
        pinned_anchor_segment_count: int = 1,
    ):
        # 便捷 generator：自动创建 session、推入所有文本 delta，并在最后收尾 flush。
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
            debug_stream=debug_stream,
            max_kv_cache_len=max_kv_cache_len,
            kv_anchor_segment_count=kv_anchor_segment_count,
            pinned_anchor_segment_count=pinned_anchor_segment_count,
        )
        yield from self._iter_text_delta_stream(session, text_deltas)
