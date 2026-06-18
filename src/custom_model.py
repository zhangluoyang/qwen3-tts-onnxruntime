from src.models import *

class ConditionedStreamingSession(ConditionedSegmentStreamingSession):
    """CustomVoice 的分段重组 prompt 流式会话。

    每个 segment 都重新构建完整的 CustomVoice non-streaming prompt；
    最近若干段已生成的 codec codes 会作为 anchor 放进下一段 prompt 前缀，
    因此不会再沿用旧流式路径里不断增长的 KV cache。
    """

    stream_debug_prefix = "custom_segment"
    stream_debug_mode = "custom_voice_segment_prompt"

    def __init__(
        self,
        model: "CustomQwen3TTSOnnxModel",
        language: str = "auto",
        speaker: str | None = None,
        instruct: str | None = None,
        **kwargs,
    ) -> None:
        self.speaker = speaker
        self.instruct = instruct
        super().__init__(
            model=model,
            language=language,
            **kwargs,
        )

    def _build_segment_prompt(
        self,
        segment: str,
        anchor_text: str,
        anchor_code: np.ndarray | None,
    ) -> PromptInputs:
        # CustomVoice 的 speaker/language/instruct 条件每段都会重新放进 prompt；
        # anchor_text/anchor_code 只负责携带最近几段生成过的上下文音频。
        return self.model.prompt_builder.build(
            text=segment,
            speaker=self.speaker or "",
            language=self.language,
            instruct=self.instruct,
            non_streaming_mode=True,
            anchor_text=anchor_text,
            anchor_code=anchor_code,
        )


class CustomQwen3TTSOnnxModel(Qwen3TTSOnnxModelBase):
    prompt_builder_cls = CustomVoicePromptBuilder

    def _preload_sessions(self) -> None:
        self._preload_decode_sessions()
        _ = self.tokenizer_decode_runner

    def supported_speakers(self) -> list[str]:
        return self.prompt_builder.supported_speakers()

    def build_prompt(
        self,
        text: str,
        speaker: str,
        language: str = "auto",
        instruct: Optional[str] = None,
        non_streaming_mode: bool = True,
    ) -> PromptInputs:
        return self.prompt_builder.build(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct,
            non_streaming_mode=non_streaming_mode,
        )

    def prefill(self, *args, **kwargs) -> TalkerPrefillOutputs:
        return self.run_prefill(self.build_prompt(*args, **kwargs))

    def generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str = "auto",
        instruct: Optional[str] = None,
        non_streaming_mode: bool = False,
        max_new_tokens: int = 1024,
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
            speaker=speaker,
            language=language,
            instruct=instruct,
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

    def iter_custom_audio_chunks(self, text: str, speaker: str, language: str = "auto", instruct: Optional[str] = None, **kwargs):
        prompt = self.build_prompt(text=text, speaker=speaker, language=language, instruct=instruct, non_streaming_mode=True)
        yield from self.iter_audio_chunks_from_prompt(prompt=prompt, **kwargs)

    def create_custom_voice_stream(
        self,
        speaker: str,
        language: str = "auto",
        instruct: Optional[str] = None,
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
    ):
        # 创建可手动 push_text/end_text 的会话，适合接 WebSocket/SSE 这种增量文本源。
        return ConditionedStreamingSession(
            model=self,
            speaker=speaker,
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
        )

    def stream_custom_voice(
        self,
        text_deltas,
        speaker: str,
        language: str = "auto",
        instruct: Optional[str] = None,
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
    ):
        # 便捷 generator：把 text_deltas 逐段推入 session，并在末尾自动 end_text。
        # 返回的每个 chunk 都是 AudioGenerationOutputs，chunk.audio 是本次新增 PCM。
        session = self.create_custom_voice_stream(
            speaker=speaker,
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
        )
        yield from self._iter_text_delta_stream(session, text_deltas)
