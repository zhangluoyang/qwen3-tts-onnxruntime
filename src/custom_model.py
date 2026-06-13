from src.models import *

class ConditionedStreamingSession(StreamingSessionBase):
    """Streaming session for CustomVoice prompt builders."""

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
        super().__init__(model=model, language=language, **kwargs)

    def _build_streaming_prompt(self, initial_text: str, include_initial_eos: bool) -> PromptInputs:
        return self.model.prompt_builder.build_streaming(
            speaker=self.speaker or "",
            language=self.language,
            instruct=self.instruct,
            initial_text=initial_text,
            include_initial_eos=include_initial_eos,
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
        kv_window_frames: int | None = 512,
        kv_window_max_frames: int | None = None,
    ):
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
            kv_window_frames=kv_window_frames,
            kv_window_max_frames=kv_window_max_frames,
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
        kv_window_frames: int | None = 512,
        kv_window_max_frames: int | None = None,
    ):
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
            kv_window_frames=kv_window_frames,
            kv_window_max_frames=kv_window_max_frames,
        )
        deltas = (text_deltas,) if isinstance(text_deltas, str) else text_deltas
        for delta in deltas:
            yield from session.push_text_iter(delta)
        yield from session.end_text_iter()
