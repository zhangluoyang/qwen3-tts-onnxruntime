from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from transformers import AutoTokenizer

from src.OnnxSessionRunner import ORT_INPUT_DTYPES, OnnxSessionRunner
from src.audio import load_audio


DEFAULT_MODEL_PATH = Path("/nfs5/models/Qwen")
DEFAULT_ONNX_DIR = Path("./onnx")
SUPPORTED_COMPUTE_DTYPES = {np.dtype(np.float16), np.dtype(np.float32)}
DEFAULT_PROMPT_CACHE_ENTRIES = 128
DEFAULT_REFERENCE_CACHE_ENTRIES = 16


def _array_cache_key(array: np.ndarray) -> tuple[tuple[int, ...], str, str]:
    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.blake2b(array.tobytes(), digest_size=16).hexdigest()
    return tuple(int(dim) for dim in array.shape), array.dtype.str, digest


def _cache_get(cache: OrderedDict, key: Any) -> Any:
    try:
        value = cache[key]
    except KeyError:
        return None
    cache.move_to_end(key)
    return value


def _cache_put(cache: OrderedDict, key: Any, value: Any, max_entries: int) -> Any:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > int(max_entries):
        cache.popitem(last=False)
    return value


def normalize_compute_dtype(dtype: np.dtype | str) -> np.dtype:
    dtype = np.dtype(dtype)
    if dtype not in SUPPORTED_COMPUTE_DTYPES:
        raise ValueError(f"dtype must be np.float16 or np.float32, got {dtype}")
    return dtype


@dataclass
class VoiceClonePromptItem:
    ref_code: Optional[np.ndarray]
    ref_spk_embedding: np.ndarray
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


@dataclass
class PromptInputs:
    inputs_embeds: np.ndarray
    attention_mask: np.ndarray
    trailing_text_hidden: np.ndarray
    tts_pad_embed: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prefill_feed(self) -> dict[str, np.ndarray]:
        return {
            "inputs_embeds": np.ascontiguousarray(self.inputs_embeds),
            "attention_mask": np.ascontiguousarray(self.attention_mask),
        }


class Qwen3TTSPromptBuilder:
    """Shared prompt construction utilities for Qwen3-TTS talker prefill inputs."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        onnx_dir: str | Path = DEFAULT_ONNX_DIR,
        providers: Optional[list[str]] = None,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        self.model_path = Path(model_path)
        self.onnx_dir = Path(onnx_dir)
        self.providers = providers or ["CPUExecutionProvider"]
        self.dtype = normalize_compute_dtype(dtype)
        self._tokenize_cache: OrderedDict[Any, np.ndarray] = OrderedDict()
        self._text_project_cache: OrderedDict[Any, np.ndarray] = OrderedDict()
        self._codec_embed_cache: OrderedDict[Any, np.ndarray] = OrderedDict()
        self._condition_cache: OrderedDict[Any, np.ndarray] = OrderedDict()
        self._ref_code_embed_cache: OrderedDict[Any, np.ndarray] = OrderedDict()
        self._tts_special_embed_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        self.config = self._load_config(self.model_path / "config.json")
        self.talker_config = self.config["talker_config"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        self.text_project_runner = OnnxSessionRunner(
            self.onnx_dir / "text_project" / "text_project.onnx",
            providers=self.providers,
            name="text_project",
        )
        self.codec_embed_runner = OnnxSessionRunner(
            self.onnx_dir / "codec_embed" / "codec_embed.onnx",
            providers=self.providers,
            name="codec_embed",
        )
        self._validate_codec_embed_session()
        self._validate_runner_output_dtype(self.text_project_runner, "text_embed")
        self._validate_runner_output_dtype(self.codec_embed_runner, "embed")
        self._validate_runner_output_dtype(self.codec_embed_runner, "ref_code_embed")

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def build_assistant_text(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"

    @staticmethod
    def build_ref_text(text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n"

    @staticmethod
    def build_instruct_text(instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def tokenize(self, text: str) -> np.ndarray:
        cached = _cache_get(self._tokenize_cache, text)
        if cached is not None:
            return cached.copy()
        encoded = self.tokenizer(text, return_tensors="np", padding=True)
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        _cache_put(self._tokenize_cache, text, input_ids, DEFAULT_PROMPT_CACHE_ENTRIES)
        return input_ids.copy()

    def text_project(self, input_ids: np.ndarray) -> np.ndarray:
        input_ids = np.asarray(input_ids, dtype=np.int64)
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        cache_key = _array_cache_key(input_ids)
        cached = _cache_get(self._text_project_cache, cache_key)
        if cached is not None:
            return cached.copy()
        output = self.text_project_runner.run(
            output_names=["text_embed"],
            feed={"input_ids": input_ids},
        )[0]
        output = np.asarray(output, dtype=self.dtype)
        _cache_put(self._text_project_cache, cache_key, output, DEFAULT_PROMPT_CACHE_ENTRIES)
        return output.copy()

    def codec_embed(self, token_ids: np.ndarray | list[int]) -> np.ndarray:
        token_ids = np.asarray(token_ids, dtype=np.int64)
        cache_key = _array_cache_key(token_ids)
        cached = _cache_get(self._codec_embed_cache, cache_key)
        if cached is not None:
            return cached.copy()
        output = self.codec_embed_runner.run(
            output_names=["embed"],
            feed={
                "token_ids": token_ids,
                "ref_code": self._dummy_ref_code(),
            },
        )[0]
        output = np.asarray(output, dtype=self.dtype)
        _cache_put(self._codec_embed_cache, cache_key, output, DEFAULT_PROMPT_CACHE_ENTRIES)
        return output.copy()

    def ref_code_embed(self, ref_code: np.ndarray) -> np.ndarray:
        ref_code = np.asarray(ref_code, dtype=np.int64)
        if ref_code.ndim == 2:
            ref_code = ref_code[None, :, :]
        if ref_code.ndim != 3:
            raise ValueError(f"ref_code must have shape [T,16] or [B,T,16], got {ref_code.shape}")
        if ref_code.shape[-1] != int(self.talker_config["num_code_groups"]):
            raise ValueError(
                "ref_code codebook count mismatch: "
                f"expected {self.talker_config['num_code_groups']}, got {ref_code.shape[-1]}"
            )
        cache_key = _array_cache_key(ref_code)
        cached = _cache_get(self._ref_code_embed_cache, cache_key)
        if cached is not None:
            return cached.copy()
        output = self.codec_embed_runner.run(
            output_names=["ref_code_embed"],
            feed={
                "token_ids": self._dummy_token_ids(),
                "ref_code": ref_code,
            },
        )[0]
        output = np.asarray(output, dtype=self.dtype)
        _cache_put(self._ref_code_embed_cache, cache_key, output, DEFAULT_REFERENCE_CACHE_ENTRIES)
        return output.copy()

    def _dummy_token_ids(self) -> np.ndarray:
        return np.array([[int(self.talker_config["codec_bos_id"])]], dtype=np.int64)

    def _dummy_ref_code(self) -> np.ndarray:
        return np.zeros((1, 1, int(self.talker_config["num_code_groups"])), dtype=np.int64)

    def _validate_codec_embed_session(self) -> None:
        required_inputs = {"token_ids", "ref_code"}
        required_outputs = {"embed", "ref_code_embed"}
        missing_inputs = sorted(required_inputs - set(self.codec_embed_runner.input_names))
        missing_outputs = sorted(required_outputs - set(self.codec_embed_runner.output_names))
        if missing_inputs or missing_outputs:
            raise ValueError(
                "codec_embed.onnx must be the new prompt-builder version with "
                "inputs token_ids/ref_code and outputs embed/ref_code_embed. "
                f"missing_inputs={missing_inputs}, missing_outputs={missing_outputs}, "
                f"path={self.onnx_dir / 'codec_embed' / 'codec_embed.onnx'}"
            )

    def _validate_runner_output_dtype(self, runner: OnnxSessionRunner, output_name: str) -> None:
        output_meta = runner.output_metas.get(output_name)
        output_dtype = ORT_INPUT_DTYPES.get(output_meta.type) if output_meta is not None else None
        if output_dtype is None:
            return
        output_dtype = np.dtype(output_dtype)
        if output_dtype != self.dtype:
            raise ValueError(
                f"{runner.name}.{output_name} dtype mismatch: ONNX outputs {output_dtype}, "
                f"but requested dtype is {self.dtype}. Use matching ONNX exports or pass dtype={output_dtype}."
            )

    def _tts_special_embeds(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._tts_special_embed_cache is not None:
            return tuple(item.copy() for item in self._tts_special_embed_cache)
        ids = np.array(
            [[
                self.config["tts_bos_token_id"],
                self.config["tts_eos_token_id"],
                self.config["tts_pad_token_id"],
            ]],
            dtype=np.int64,
        )
        embeds = self.text_project(ids)
        self._tts_special_embed_cache = (embeds[:, 0:1], embeds[:, 1:2], embeds[:, 2:3])
        return tuple(item.copy() for item in self._tts_special_embed_cache)

    def _cached_condition(self, key: Any, value_fn) -> np.ndarray:
        cached = _cache_get(self._condition_cache, key)
        if cached is not None:
            return cached.copy()
        value = np.asarray(value_fn(), dtype=self.dtype)
        _cache_put(self._condition_cache, key, value, DEFAULT_PROMPT_CACHE_ENTRIES)
        return value.copy()

    def _language_id(self, language: str) -> Optional[int]:
        if language is None:
            raise ValueError("language must not be None")
        language = str(language).lower()
        if language == "auto":
            return None
        codec_language_id = self.talker_config["codec_language_id"]
        if language not in codec_language_id:
            raise NotImplementedError(f"Language {language!r} is not implemented in this model")
        return int(codec_language_id[language])

    def _codec_prefill_ids(self, language: str) -> list[int]:
        language_id = self._language_id(language)
        if language_id is None:
            return [
                int(self.talker_config["codec_nothink_id"]),
                int(self.talker_config["codec_think_bos_id"]),
                int(self.talker_config["codec_think_eos_id"]),
            ]
        return [
            int(self.talker_config["codec_think_id"]),
            int(self.talker_config["codec_think_bos_id"]),
            language_id,
            int(self.talker_config["codec_think_eos_id"]),
        ]

    def _build_talker_prefix(
        self,
        input_id: np.ndarray,
        language: str,
        speaker_embed: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._tts_special_embeds()

        codec_input_embedding_0 = self.codec_embed(np.array([self._codec_prefill_ids(language)], dtype=np.int64))
        codec_input_embedding_1 = self.codec_embed(
            np.array(
                [[
                    int(self.talker_config["codec_pad_id"]),
                    int(self.talker_config["codec_bos_id"]),
                ]],
                dtype=np.int64,
            )
        )

        if speaker_embed is None:
            codec_input_embedding = np.concatenate(
                [codec_input_embedding_0, codec_input_embedding_1],
                axis=1,
            )
        else:
            speaker_embed = np.asarray(speaker_embed, dtype=self.dtype).reshape(1, 1, -1)
            codec_input_embedding = np.concatenate(
                [codec_input_embedding_0, speaker_embed, codec_input_embedding_1],
                axis=1,
            )

        role_embed = self.text_project(input_id[:, :3])
        pad_prefix = np.broadcast_to(
            tts_pad_embed,
            (1, codec_input_embedding.shape[1] - 2, tts_pad_embed.shape[-1]),
        ).copy()
        text_side = np.concatenate([pad_prefix, tts_bos_embed], axis=1)
        codec_side = codec_input_embedding[:, :-1]
        talker_input_embed = np.concatenate([role_embed, text_side + codec_side], axis=1)
        return talker_input_embed, codec_input_embedding, tts_bos_embed, tts_eos_embed, tts_pad_embed

    def _codec_tail_embed(self) -> np.ndarray:
        return self.codec_embed(
            np.array(
                [[
                    int(self.talker_config["codec_pad_id"]),
                    int(self.talker_config["codec_bos_id"]),
                ]],
                dtype=np.int64,
            )
        )

    def target_text_ids(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros((1, 0), dtype=np.int64)
        return self.tokenize(self.build_assistant_text(text))[:, 3:-5]

    def target_text_embeds(self, text: str) -> np.ndarray:
        ids = self.target_text_ids(text)
        if ids.shape[1] == 0:
            return np.zeros((1, 0, int(self.talker_config["hidden_size"])), dtype=self.dtype)
        return self.text_project(ids)

    def _normalize_anchor_code(self, anchor_code: Optional[np.ndarray]) -> tuple[Optional[np.ndarray], int]:
        if anchor_code is None:
            return None, 0
        anchor_code = np.asarray(anchor_code, dtype=np.int64)
        if anchor_code.ndim == 3:
            if anchor_code.shape[0] != 1:
                raise NotImplementedError(f"anchor_code batch_size must be 1, got {anchor_code.shape[0]}")
            anchor_code = anchor_code[0]
        if anchor_code.ndim != 2:
            raise ValueError(f"anchor_code must have shape [frames, groups], got {anchor_code.shape}")
        if anchor_code.shape[1] != int(self.talker_config["num_code_groups"]):
            raise ValueError(
                "anchor_code codebook count mismatch: "
                f"expected {self.talker_config['num_code_groups']}, got {anchor_code.shape[1]}"
            )
        return anchor_code, int(anchor_code.shape[0])

    def _instruct_embed(self, instruct: Optional[str]) -> Optional[np.ndarray]:
        if instruct is None or instruct == "":
            return None
        instruct_ids = self.tokenize(self.build_instruct_text(str(instruct)))
        return self.text_project(instruct_ids)

    def _role_aligned_codec_part(
        self,
        codec_input_embedding: np.ndarray,
        tts_bos_embed: np.ndarray,
        tts_pad_embed: np.ndarray,
    ) -> np.ndarray:
        pad_prefix = np.broadcast_to(
            tts_pad_embed,
            (1, codec_input_embedding.shape[1] - 2, tts_pad_embed.shape[-1]),
        ).copy()
        text_side = np.concatenate([pad_prefix, tts_bos_embed], axis=1)
        return text_side + codec_input_embedding[:, :-1]

    def _append_non_streaming_text(
        self,
        talker_input_embed: np.ndarray,
        input_id: np.ndarray,
        tts_eos_embed: np.ndarray,
        tts_pad_embed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        text_with_eos = np.concatenate([self.text_project(input_id[:, 3:-5]), tts_eos_embed], axis=1)
        codec_pad = self.codec_embed(
            np.full(
                (1, text_with_eos.shape[1]),
                int(self.talker_config["codec_pad_id"]),
                dtype=np.int64,
            )
        )
        codec_bos = self.codec_embed(np.array([[int(self.talker_config["codec_bos_id"])]], dtype=np.int64))
        talker_input_embed = np.concatenate(
            [
                talker_input_embed[:, :-1],
                text_with_eos + codec_pad,
                tts_pad_embed + codec_bos,
            ],
            axis=1,
        )
        return talker_input_embed, tts_pad_embed

    def _build_conditioned_prompt(
        self,
        text: str,
        codec_input: np.ndarray,
        instruct: Optional[str] = None,
        non_streaming_mode: bool = True,
        anchor_text: str = "",
        anchor_code: Optional[np.ndarray] = None,
    ) -> PromptInputs:
        input_id = self.tokenize(self.build_assistant_text(text))
        instruct_embed = self._instruct_embed(instruct)
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._tts_special_embeds()

        role_embed = self.text_project(input_id[:, :3])
        codec_part = self._role_aligned_codec_part(codec_input, tts_bos_embed, tts_pad_embed)
        talker_input_embed = np.concatenate([role_embed, codec_part], axis=1)

        anchor_code, anchor_code_len = self._normalize_anchor_code(anchor_code)

        if non_streaming_mode and anchor_code_len > 0:
            anchor_text_ids = self.target_text_ids(anchor_text or "")
            anchor_text_embed = (
                self.text_project(anchor_text_ids)
                if anchor_text_ids.shape[1]
                else np.zeros((1, 0, int(self.talker_config["hidden_size"])), dtype=self.dtype)
            )
            target_text_embed = self.text_project(input_id[:, 3:-5]) if input_id[:, 3:-5].shape[1] else np.zeros(
                (1, 0, int(self.talker_config["hidden_size"])),
                dtype=self.dtype,
            )
            text_with_eos = np.concatenate([anchor_text_embed, target_text_embed, tts_eos_embed], axis=1)
            codec_pad = self.codec_embed(
                np.full((1, text_with_eos.shape[1]), int(self.talker_config["codec_pad_id"]), dtype=np.int64)
            )
            text_side = text_with_eos + codec_pad
            codec_bos = self.codec_embed(np.array([[int(self.talker_config["codec_bos_id"])]], dtype=np.int64))
            codec_side = np.concatenate([codec_bos, self.ref_code_embed(anchor_code)], axis=1)
            codec_side = codec_side + np.broadcast_to(
                tts_pad_embed,
                (1, codec_side.shape[1], tts_pad_embed.shape[-1]),
            ).copy()
            talker_input_embed = np.concatenate([talker_input_embed, text_side, codec_side], axis=1)
            trailing_text_hidden = tts_pad_embed
        else:
            first_text = self.text_project(input_id[:, 3:4]) + codec_input[:, -1:]
            talker_input_embed = np.concatenate([talker_input_embed, first_text], axis=1)
            if non_streaming_mode:
                talker_input_embed, trailing_text_hidden = self._append_non_streaming_text(
                    talker_input_embed=talker_input_embed,
                    input_id=input_id,
                    tts_eos_embed=tts_eos_embed,
                    tts_pad_embed=tts_pad_embed,
                )
            else:
                trailing_text_hidden = np.concatenate([self.text_project(input_id[:, 4:-5]), tts_eos_embed], axis=1)

        if instruct_embed is not None:
            talker_input_embed = np.concatenate([instruct_embed, talker_input_embed], axis=1)

        prompt = self._left_pad_prompt_batch([talker_input_embed], [trailing_text_hidden], tts_pad_embed)
        prompt.metadata.update(
            {
                "input_ids": input_id,
                "anchor_text": str(anchor_text or ""),
                "anchor_code_len": int(anchor_code_len),
                "decode_context_code_len": int(anchor_code_len),
                # anchor_code 不只给 talker 当 in-context，也要给 tokenizer decoder
                # 当左上下文；context_frames 会裁掉它对应的音频，避免边界处从零上下文起声。
                "ref_code": anchor_code if anchor_code_len > 0 else None,
            }
        )
        return prompt

    def _build_streaming_conditioned_prompt(
        self,
        codec_input: np.ndarray,
        instruct: Optional[str] = None,
        initial_text: str = "",
        include_initial_eos: bool = False,
    ) -> PromptInputs:
        input_id = self.tokenize(self.build_assistant_text(initial_text or ""))
        instruct_embed = self._instruct_embed(instruct)
        tts_bos_embed, tts_eos_embed, tts_pad_embed = self._tts_special_embeds()

        role_embed = self.text_project(input_id[:, :3])
        codec_part = self._role_aligned_codec_part(codec_input, tts_bos_embed, tts_pad_embed)
        talker_input_embed = np.concatenate([role_embed, codec_part], axis=1)

        target_text_hidden = (
            self.text_project(input_id[:, 3:-5])
            if input_id[:, 3:-5].shape[1]
            else np.zeros((1, 0, int(self.talker_config["hidden_size"])), dtype=self.dtype)
        )
        if include_initial_eos:
            target_text_hidden = np.concatenate([target_text_hidden, tts_eos_embed], axis=1)
        if target_text_hidden.shape[1] == 0:
            raise ValueError("streaming conditioned prompt requires initial_text or include_initial_eos=True")

        if instruct_embed is not None:
            talker_input_embed = np.concatenate([instruct_embed, talker_input_embed], axis=1)

        # 和官方 generate 的相位对齐：第一帧的 codec_bos/speaker 尾部 embedding
        # 与第一个目标文本 embedding 相加后放进 prefill。这样 prefill 输出的 logits
        # 已经可以直接 sample 第 0 帧 codec，后续统一走 sample_then_advance。
        first_text_embed = target_text_hidden[:, :1]
        initial_codec_embed = codec_input[:, -1:].astype(self.dtype, copy=False)
        first_decode_embed = first_text_embed + initial_codec_embed
        talker_input_embed = np.concatenate([talker_input_embed, first_decode_embed], axis=1)
        trailing_text_hidden = target_text_hidden[:, 1:]

        prompt = self._left_pad_prompt_batch([talker_input_embed], [trailing_text_hidden], tts_pad_embed)
        prompt.metadata.update(
            {
                "input_ids": input_id,
                "tts_eos_embed": tts_eos_embed,
                "initial_codec_embed": np.ascontiguousarray(initial_codec_embed),
                "first_text_embed_in_prefill": True,
                "initial_text": initial_text or "",
            }
        )
        return prompt

    def _generate_icl_prompt(
        self,
        text_id: np.ndarray,
        ref_id: np.ndarray,
        ref_code: np.ndarray,
        tts_pad_embed: np.ndarray,
        tts_eos_embed: np.ndarray,
        non_streaming_mode: bool,
        include_text_eos: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        # text_project is token-local embedding + projection, so ref text can be
        # cached independently from the changing target text.
        hidden_size = int(tts_eos_embed.shape[-1])
        if ref_id.shape[1] > 0:
            ref_text_embed = self.text_project(ref_id)
        else:
            ref_text_embed = np.zeros((ref_id.shape[0], 0, hidden_size), dtype=self.dtype)
        if text_id.shape[1] > 0:
            target_text_embed = self.text_project(text_id)
        else:
            target_text_embed = np.zeros((text_id.shape[0], 0, hidden_size), dtype=self.dtype)
        text_embed = np.concatenate([ref_text_embed, target_text_embed], axis=1)
        if include_text_eos:
            text_embed = np.concatenate([text_embed, tts_eos_embed], axis=1)

        ref_code = np.asarray(ref_code, dtype=np.int64)
        if ref_code.ndim != 2:
            raise ValueError(f"ref_code must have shape [num_frames, num_code_groups], got {ref_code.shape}")
        if ref_code.shape[1] != int(self.talker_config["num_code_groups"]):
            raise ValueError(
                "ref_code codebook count mismatch: "
                f"expected {self.talker_config['num_code_groups']}, got {ref_code.shape[1]}"
            )

        codec_embed = self.ref_code_embed(ref_code)
        codec_bos_embed = self.codec_embed(np.array([[int(self.talker_config["codec_bos_id"])]], dtype=np.int64))
        codec_embed = np.concatenate([codec_bos_embed, codec_embed], axis=1)

        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        if non_streaming_mode:
            codec_pad = self.codec_embed(
                np.full((1, text_lens), int(self.talker_config["codec_pad_id"]), dtype=np.int64)
            )
            icl_input_embed = text_embed + codec_pad
            icl_input_embed = np.concatenate([icl_input_embed, codec_embed + tts_pad_embed], axis=1)
            return icl_input_embed, tts_pad_embed

        if text_lens > codec_lens:
            return text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]

        pads = np.broadcast_to(
            tts_pad_embed,
            (1, codec_lens - text_lens, tts_pad_embed.shape[-1]),
        ).copy()
        text_embed = np.concatenate([text_embed, pads], axis=1)
        return text_embed + codec_embed, tts_pad_embed

    @staticmethod
    def _left_pad_prompt_batch(
        prompt_items: list[np.ndarray],
        trailing_items: list[np.ndarray],
        tts_pad_embed: np.ndarray,
    ) -> PromptInputs:
        original_lengths = np.array([item.shape[1] for item in prompt_items], dtype=np.int64)
        max_len = int(original_lengths.max())
        batch_size = len(prompt_items)
        hidden_size = prompt_items[0].shape[-1]

        inputs_embeds = np.zeros((batch_size, max_len, hidden_size), dtype=prompt_items[0].dtype)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
        for i, item in enumerate(prompt_items):
            length = item.shape[1]
            inputs_embeds[i, max_len - length:] = item[0]
            attention_mask[i, max_len - length:] = 1

        trailing_lengths = np.array([item.shape[1] for item in trailing_items], dtype=np.int64)
        max_trailing = int(trailing_lengths.max())
        trailing_text_hidden = np.broadcast_to(
            tts_pad_embed.reshape(1, 1, -1),
            (batch_size, max_trailing, hidden_size),
        ).copy()
        for i, item in enumerate(trailing_items):
            trailing_text_hidden[i, : item.shape[1]] = item[0]

        return PromptInputs(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed.astype(prompt_items[0].dtype, copy=False),
            metadata={
                "original_lengths": original_lengths,
                "trailing_lengths": trailing_lengths,
            },
        )


class BaseVoiceClonePromptBuilder(Qwen3TTSPromptBuilder):
    """Prompt builder for the base Qwen3-TTS voice-clone model."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        onnx_dir: str | Path = DEFAULT_ONNX_DIR,
        providers: Optional[list[str]] = None,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        super().__init__(model_path=model_path, onnx_dir=onnx_dir, providers=providers, dtype=dtype)
        self._tokenizer_encoder_runner: Optional[OnnxSessionRunner] = None
        self._speaker_encoder_runner: Optional[OnnxSessionRunner] = None
        self._reference_audio_cache: OrderedDict[Any, tuple[np.ndarray, int]] = OrderedDict()
        self._reference_code_cache: OrderedDict[Any, np.ndarray] = OrderedDict()
        self._speaker_embedding_cache: OrderedDict[Any, np.ndarray] = OrderedDict()

    def build(
        self,
        text: str,
        language: str = "auto",
        ref_text: Optional[str] = None,
        ref_code: Optional[np.ndarray] = None,
        ref_spk_embedding: Optional[np.ndarray] = None,
        x_vector_only_mode: bool = False,
        non_streaming_mode: bool = False,
    ) -> PromptInputs:
        if not x_vector_only_mode and (ref_text is None or ref_text == ""):
            raise ValueError("ref_text is required when x_vector_only_mode=False (ICL mode).")
        if not x_vector_only_mode and ref_code is None:
            raise ValueError("ref_code is required when x_vector_only_mode=False (ICL mode).")
        if ref_spk_embedding is None:
            raise ValueError("ref_spk_embedding is required for base voice-clone prompt construction.")

        input_id = self.tokenize(self.build_assistant_text(text))
        speaker_embed = np.asarray(ref_spk_embedding, dtype=self.dtype).reshape(-1)
        talker_input_embed, codec_input_embedding, _, tts_eos_embed, tts_pad_embed = self._build_talker_prefix(
            input_id=input_id,
            language=language,
            speaker_embed=speaker_embed,
        )

        ref_code_for_metadata = None
        if not x_vector_only_mode:
            ref_id = self.tokenize(self.build_ref_text(ref_text or ""))
            ref_code_for_metadata = np.asarray(ref_code, dtype=np.int64)
            icl_input_embed, trailing_text_hidden = self._generate_icl_prompt(
                text_id=input_id[:, 3:-5],
                ref_id=ref_id[:, 3:-2],
                ref_code=ref_code_for_metadata,
                tts_pad_embed=tts_pad_embed,
                tts_eos_embed=tts_eos_embed,
                non_streaming_mode=non_streaming_mode,
            )
            talker_input_embed = np.concatenate([talker_input_embed, icl_input_embed], axis=1)
        else:
            first_text = self.text_project(input_id[:, 3:4]) + codec_input_embedding[:, -1:]
            talker_input_embed = np.concatenate([talker_input_embed, first_text], axis=1)
            if non_streaming_mode:
                talker_input_embed = talker_input_embed[:, :-1]
                text_with_eos = np.concatenate([self.text_project(input_id[:, 3:-5]), tts_eos_embed], axis=1)
                codec_pad = self.codec_embed(
                    np.full(
                        (1, input_id[:, 3:-5].shape[1] + 1),
                        int(self.talker_config["codec_pad_id"]),
                        dtype=np.int64,
                    )
                )
                codec_bos = self.codec_embed(np.array([[int(self.talker_config["codec_bos_id"])]], dtype=np.int64))
                talker_input_embed = np.concatenate(
                    [
                        talker_input_embed,
                        text_with_eos + codec_pad,
                        tts_pad_embed + codec_bos,
                    ],
                    axis=1,
                )
                trailing_text_hidden = tts_pad_embed
            else:
                trailing_text_hidden = np.concatenate([self.text_project(input_id[:, 4:-5]), tts_eos_embed], axis=1)

        prompt = self._left_pad_prompt_batch([talker_input_embed], [trailing_text_hidden], tts_pad_embed)
        prompt.metadata.update(
            {
                "mode": "base",
                "language": language,
                "x_vector_only_mode": bool(x_vector_only_mode),
                "icl_mode": bool(not x_vector_only_mode),
                "non_streaming_mode": bool(non_streaming_mode),
                "input_ids": input_id,
                "ref_code": ref_code_for_metadata,
                "ref_code_len": int(ref_code_for_metadata.shape[0]) if ref_code_for_metadata is not None else 0,
            }
        )
        return prompt

    def build_from_prompt_item(
        self,
        text: str,
        prompt_item: VoiceClonePromptItem,
        language: str = "auto",
        non_streaming_mode: bool = False,
    ) -> PromptInputs:
        return self.build(
            text=text,
            language=language,
            ref_text=prompt_item.ref_text,
            ref_code=prompt_item.ref_code,
            ref_spk_embedding=prompt_item.ref_spk_embedding,
            x_vector_only_mode=prompt_item.x_vector_only_mode,
            non_streaming_mode=non_streaming_mode,
        )

    def target_text_ids(self, text: str) -> np.ndarray:
        if not text:
            return np.zeros((1, 0), dtype=np.int64)
        return self.tokenize(self.build_assistant_text(text))[:, 3:-5]

    def target_text_embeds(self, text: str) -> np.ndarray:
        ids = self.target_text_ids(text)
        if ids.shape[1] == 0:
            return np.zeros((1, 0, int(self.talker_config["hidden_size"])), dtype=self.dtype)
        return self.text_project(ids)

    def build_streaming(
        self,
        initial_text: str,
        language: str = "auto",
        ref_text: Optional[str] = None,
        ref_code: Optional[np.ndarray] = None,
        ref_spk_embedding: Optional[np.ndarray] = None,
        x_vector_only_mode: bool = False,
        include_initial_eos: bool = False,
        defer_target_text: bool = False,
        anchor_text: str = "",
        anchor_code: Optional[np.ndarray] = None,
    ) -> PromptInputs:
        if not x_vector_only_mode and (ref_text is None or ref_text == ""):
            raise ValueError("ref_text is required when x_vector_only_mode=False (ICL mode).")
        if not x_vector_only_mode and ref_code is None:
            raise ValueError("ref_code is required when x_vector_only_mode=False (ICL mode).")
        if ref_spk_embedding is None:
            raise ValueError("ref_spk_embedding is required for base voice-clone streaming prompt construction.")

        input_id = self.tokenize(self.build_assistant_text(initial_text or ""))
        speaker_embed = np.asarray(ref_spk_embedding, dtype=self.dtype).reshape(-1)
        talker_input_embed, codec_input_embedding, _, tts_eos_embed, tts_pad_embed = self._build_talker_prefix(
            input_id=input_id,
            language=language,
            speaker_embed=speaker_embed,
        )

        ref_code_for_metadata = None
        codec_bos_embed = self.codec_embed(np.array([[int(self.talker_config["codec_bos_id"])]], dtype=np.int64))
        initial_codec_embed = codec_bos_embed
        if not x_vector_only_mode:
            ref_id = self.tokenize(self.build_ref_text(ref_text or ""))
            ref_code_for_metadata = np.asarray(ref_code, dtype=np.int64)
            anchor_code, anchor_code_len = self._normalize_anchor_code(anchor_code)
            if anchor_code is not None:
                ref_code_for_metadata = np.concatenate([ref_code_for_metadata, anchor_code], axis=0)
            ref_text_embed = self.text_project(ref_id[:, 3:-2]) if ref_id[:, 3:-2].shape[1] else np.zeros(
                (1, 0, int(self.talker_config["hidden_size"])),
                dtype=self.dtype,
            )
            anchor_text_ids = self.target_text_ids(anchor_text or "")
            anchor_text_embed = self.text_project(anchor_text_ids) if anchor_text_ids.shape[1] else np.zeros(
                (1, 0, int(self.talker_config["hidden_size"])),
                dtype=self.dtype,
            )
            target_text_hidden = self.text_project(input_id[:, 3:-5]) if input_id[:, 3:-5].shape[1] else np.zeros(
                (1, 0, int(self.talker_config["hidden_size"])),
                dtype=self.dtype,
            )
            prefix_text_embed = np.concatenate([ref_text_embed, anchor_text_embed], axis=1)
            text_embed = prefix_text_embed if defer_target_text else np.concatenate(
                [prefix_text_embed, target_text_hidden],
                axis=1,
            )
            if include_initial_eos:
                if defer_target_text:
                    target_text_hidden = np.concatenate([target_text_hidden, tts_eos_embed], axis=1)
                else:
                    text_embed = np.concatenate([text_embed, tts_eos_embed], axis=1)

            codec_embed = self.ref_code_embed(ref_code_for_metadata)
            initial_codec_embed = codec_embed[:, -1:].astype(self.dtype, copy=False)
            if text_embed.shape[1] > codec_embed.shape[1]:
                trailing_text_hidden = text_embed[:, codec_embed.shape[1]:].astype(self.dtype, copy=False)
                text_embed = text_embed[:, : codec_embed.shape[1]]
            elif text_embed.shape[1] < codec_embed.shape[1]:
                pads = np.broadcast_to(
                    tts_pad_embed,
                    (1, codec_embed.shape[1] - text_embed.shape[1], tts_pad_embed.shape[-1]),
                ).copy()
                text_embed = np.concatenate([text_embed, pads], axis=1)
                trailing_text_hidden = np.zeros((1, 0, int(self.talker_config["hidden_size"])), dtype=self.dtype)
            else:
                trailing_text_hidden = np.zeros((1, 0, int(self.talker_config["hidden_size"])), dtype=self.dtype)
            if defer_target_text:
                trailing_text_hidden = target_text_hidden.astype(self.dtype, copy=False)
            icl_input_embed = text_embed + codec_embed
            talker_input_embed = np.concatenate([talker_input_embed, icl_input_embed], axis=1)
        else:
            target_text_hidden = self.text_project(input_id[:, 3:-5]) if input_id[:, 3:-5].shape[1] else np.zeros(
                (1, 0, int(self.talker_config["hidden_size"])),
                dtype=self.dtype,
            )
            if include_initial_eos:
                target_text_hidden = np.concatenate([target_text_hidden, tts_eos_embed], axis=1)
            if target_text_hidden.shape[1] == 0:
                raise ValueError("streaming prompt requires initial_text or include_initial_eos=True")
            trailing_text_hidden = target_text_hidden

        prompt = self._left_pad_prompt_batch([talker_input_embed], [trailing_text_hidden], tts_pad_embed)
        prompt.metadata.update(
            {
                "mode": "base",
                "language": language,
                "x_vector_only_mode": bool(x_vector_only_mode),
                "icl_mode": bool(not x_vector_only_mode),
                "streaming_mode": True,
                "include_initial_eos": bool(include_initial_eos),
                "defer_target_text": bool(defer_target_text),
                "input_ids": input_id,
                "initial_codec_embed": np.ascontiguousarray(initial_codec_embed.astype(self.dtype, copy=False)),
                "ref_code": ref_code_for_metadata,
                "ref_code_len": int(ref_code_for_metadata.shape[0]) if ref_code_for_metadata is not None else 0,
                "anchor_text": str(anchor_text or ""),
                "anchor_code_len": int(anchor_code_len) if not x_vector_only_mode else 0,
            }
        )
        return prompt

    def build_from_reference(
        self,
        text: str,
        ref_audio: str | Path | tuple[np.ndarray, int],
        ref_text: Optional[str],
        language: str = "auto",
        x_vector_only_mode: bool = False,
        non_streaming_mode: bool = False,
    ) -> PromptInputs:
        audio_key = self._reference_audio_cache_key(ref_audio)
        audio, sr = self._load_reference_audio_cached(ref_audio, audio_key)
        ref_code = None
        if not x_vector_only_mode:
            ref_code = self._encode_reference_audio_cached(audio_key, audio, sr)
        spk_embedding = self._encode_speaker_embedding_cached(audio_key, audio, sr)
        return self.build(
            text=text,
            language=language,
            ref_text=ref_text,
            ref_code=ref_code,
            ref_spk_embedding=spk_embedding,
            x_vector_only_mode=x_vector_only_mode,
            non_streaming_mode=non_streaming_mode,
        )

    def build_streaming_from_reference(
        self,
        initial_text: str,
        ref_audio: str | Path | tuple[np.ndarray, int],
        ref_text: Optional[str],
        language: str = "auto",
        x_vector_only_mode: bool = False,
        include_initial_eos: bool = False,
        defer_target_text: bool = False,
        anchor_text: str = "",
        anchor_code: Optional[np.ndarray] = None,
    ) -> PromptInputs:
        audio_key = self._reference_audio_cache_key(ref_audio)
        audio, sr = self._load_reference_audio_cached(ref_audio, audio_key)
        ref_code = None
        if not x_vector_only_mode:
            ref_code = self._encode_reference_audio_cached(audio_key, audio, sr)
        spk_embedding = self._encode_speaker_embedding_cached(audio_key, audio, sr)
        return self.build_streaming(
            initial_text=initial_text,
            language=language,
            ref_text=ref_text,
            ref_code=ref_code,
            ref_spk_embedding=spk_embedding,
            x_vector_only_mode=x_vector_only_mode,
            include_initial_eos=include_initial_eos,
            defer_target_text=defer_target_text,
            anchor_text=anchor_text,
            anchor_code=anchor_code,
        )

    def _reference_audio_cache_key(self, ref_audio: str | Path | tuple[np.ndarray, int]) -> tuple[Any, ...]:
        if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
            wav, sr = ref_audio
            return ("array", int(sr), _array_cache_key(np.asarray(wav, dtype=np.float32)))

        path = Path(ref_audio).expanduser()
        try:
            stat = path.stat()
            resolved = path.resolve()
            return ("path", str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
        except OSError:
            return ("path", str(path))

    def _load_reference_audio_cached(
        self,
        ref_audio: str | Path | tuple[np.ndarray, int],
        audio_key: tuple[Any, ...],
    ) -> tuple[np.ndarray, int]:
        cached = _cache_get(self._reference_audio_cache, audio_key)
        if cached is not None:
            audio, sr = cached
            return audio.copy(), int(sr)
        audio, sr = load_audio(ref_audio)
        audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
        cached = (audio, int(sr))
        _cache_put(self._reference_audio_cache, audio_key, cached, DEFAULT_REFERENCE_CACHE_ENTRIES)
        return audio.copy(), int(sr)

    def _encode_reference_audio_cached(
        self,
        audio_key: tuple[Any, ...],
        audio: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        cache_key = ("ref_code", audio_key)
        cached = _cache_get(self._reference_code_cache, cache_key)
        if cached is not None:
            return cached.copy()

        ref_code = np.ascontiguousarray(self.encode_reference_audio(audio, sr).astype(np.int64, copy=False))
        _cache_put(self._reference_code_cache, cache_key, ref_code, DEFAULT_REFERENCE_CACHE_ENTRIES)
        return ref_code.copy()

    def _encode_speaker_embedding_cached(
        self,
        audio_key: tuple[Any, ...],
        audio: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        cache_key = ("speaker_embedding", audio_key)
        cached = _cache_get(self._speaker_embedding_cache, cache_key)
        if cached is not None:
            return cached.copy()

        from .audio import make_speaker_mel

        mel = make_speaker_mel(audio, sr)
        speaker_embedding = np.ascontiguousarray(self.encode_speaker_mel(mel).astype(self.dtype, copy=False))
        _cache_put(self._speaker_embedding_cache, cache_key, speaker_embedding, DEFAULT_REFERENCE_CACHE_ENTRIES)
        return speaker_embedding.copy()

    def encode_reference_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        from .audio import resample_audio

        if sr != 24000:
            audio = resample_audio(audio, sr, 24000)
        audio = np.asarray(audio, dtype=np.float32).reshape(1, -1)
        runner = self._get_tokenizer_encoder_runner()
        codes = runner.run(output_names=["codes"], feed={"audio": audio})[0]
        return np.asarray(codes[0], dtype=np.int64)

    def encode_speaker_mel(self, mel: np.ndarray) -> np.ndarray:
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim == 2:
            mel = mel[None, :, :]
        runner = self._get_speaker_encoder_runner()
        speaker_embedding = runner.run(
            output_names=["speaker_embedding"],
            feed={"mel": mel},
        )[0]
        return np.asarray(speaker_embedding[0], dtype=self.dtype)

    def _get_tokenizer_encoder_runner(self) -> OnnxSessionRunner:
        if self._tokenizer_encoder_runner is None:
            self._tokenizer_encoder_runner = OnnxSessionRunner(
                self.onnx_dir / "tokenizer" / "tokenizer12hz_encode.onnx",
                providers=self.providers,
                name="tokenizer_encode",
            )
        return self._tokenizer_encoder_runner

    def _get_speaker_encoder_runner(self) -> OnnxSessionRunner:
        if self._speaker_encoder_runner is None:
            self._speaker_encoder_runner = OnnxSessionRunner(
                self.onnx_dir / "speaker_encoder" / "speaker_encoder.onnx",
                providers=self.providers,
                name="speaker_encoder",
            )
            self._validate_runner_output_dtype(self._speaker_encoder_runner, "speaker_embedding")
        return self._speaker_encoder_runner


class CustomVoicePromptBuilder(Qwen3TTSPromptBuilder):
    """Prompt builder for CustomVoice models: preset speaker id plus optional instruction."""

    def supported_speakers(self) -> list[str]:
        return sorted(str(item) for item in self.talker_config.get("spk_id", {}).keys())

    def is_0p6b_model(self) -> bool:
        return str(self.config.get("tts_model_size", "")).lower() == "0b6"

    def speaker_embed(self, speaker: str) -> np.ndarray:
        speaker_key = self._normalize_speaker(speaker)
        spk_id = self.talker_config.get("spk_id", {})
        if speaker_key not in spk_id:
            raise ValueError(f"Unsupported speaker={speaker!r}; supported={self.supported_speakers()}")
        return self.codec_embed(np.array([[int(spk_id[speaker_key])]], dtype=np.int64))

    def build(
        self,
        text: str,
        speaker: str,
        language: str = "auto",
        instruct: Optional[str] = None,
        non_streaming_mode: bool = True,
        anchor_text: str = "",
        anchor_code: Optional[np.ndarray] = None,
    ) -> PromptInputs:
        if instruct and self.is_0p6b_model():
            raise ValueError("0.6B CustomVoice does not support instruct; only 1.7B CustomVoice allows instruct.")
        codec_input = self._codec_conditioning(language=language, speaker=speaker)
        prompt = self._build_conditioned_prompt(
            text=text,
            codec_input=codec_input,
            instruct=instruct,
            non_streaming_mode=non_streaming_mode,
            anchor_text=anchor_text,
            anchor_code=anchor_code,
        )
        prompt.metadata.update(
            {
                "mode": "custom_voice",
                "language": language,
                "speaker": self._normalize_speaker(speaker),
                "instruct": instruct or "",
                "non_streaming_mode": bool(non_streaming_mode),
            }
        )
        return prompt

    def build_streaming(
        self,
        speaker: str,
        language: str = "auto",
        instruct: Optional[str] = None,
        initial_text: str = "",
        include_initial_eos: bool = False,
    ) -> PromptInputs:
        if instruct and self.is_0p6b_model():
            raise ValueError("0.6B CustomVoice does not support instruct; only 1.7B CustomVoice allows instruct.")
        codec_input = self._codec_conditioning(language=language, speaker=speaker)
        prompt = self._build_streaming_conditioned_prompt(
            codec_input=codec_input,
            instruct=instruct,
            initial_text=initial_text,
            include_initial_eos=include_initial_eos,
        )
        prompt.metadata.update(
            {
                "mode": "custom_voice",
                "language": language,
                "speaker": self._normalize_speaker(speaker),
                "instruct": instruct or "",
                "streaming_mode": True,
                "include_initial_eos": bool(include_initial_eos),
            }
        )
        return prompt

    def _effective_language(self, language: str, speaker: str) -> str:
        language_key = str(language or "auto").lower()
        speaker_key = self._normalize_speaker(speaker)
        dialect = self.talker_config.get("spk_is_dialect", {}).get(speaker_key)
        if language_key in {"chinese", "auto"} and dialect:
            return str(dialect).lower()
        return language_key

    def _codec_conditioning(self, language: str, speaker: str) -> np.ndarray:
        effective_language = self._effective_language(language, speaker)
        speaker_key = self._normalize_speaker(speaker)

        def build_condition() -> np.ndarray:
            codec_prefill = self.codec_embed(np.array([self._codec_prefill_ids(effective_language)], dtype=np.int64))
            return np.concatenate([codec_prefill, self.speaker_embed(speaker_key), self._codec_tail_embed()], axis=1)

        return self._cached_condition(("custom_voice", effective_language, speaker_key), build_condition)

    @staticmethod
    def _normalize_speaker(speaker: str) -> str:
        speaker = str(speaker or "").strip().lower()
        if not speaker:
            raise ValueError("speaker must not be empty for CustomVoice")
        return speaker


class VoiceDesignPromptBuilder(Qwen3TTSPromptBuilder):
    """Prompt builder for VoiceDesign models: natural-language instruction defines the voice."""

    def build(
        self,
        text: str,
        instruct: str,
        language: str = "auto",
        non_streaming_mode: bool = True,
        anchor_text: str = "",
        anchor_code: Optional[np.ndarray] = None,
    ) -> PromptInputs:
        codec_input = self._codec_conditioning(language=language)
        prompt = self._build_conditioned_prompt(
            text=text,
            codec_input=codec_input,
            instruct=instruct,
            non_streaming_mode=non_streaming_mode,
            anchor_text=anchor_text,
            anchor_code=anchor_code,
        )
        prompt.metadata.update(
            {
                "mode": "voice_design",
                "language": language,
                "instruct": instruct or "",
                "non_streaming_mode": bool(non_streaming_mode),
            }
        )
        return prompt

    def build_streaming(
        self,
        instruct: str,
        language: str = "auto",
        initial_text: str = "",
        include_initial_eos: bool = False,
    ) -> PromptInputs:
        codec_input = self._codec_conditioning(language=language)
        prompt = self._build_streaming_conditioned_prompt(
            codec_input=codec_input,
            instruct=instruct,
            initial_text=initial_text,
            include_initial_eos=include_initial_eos,
        )
        prompt.metadata.update(
            {
                "mode": "voice_design",
                "language": language,
                "instruct": instruct or "",
                "streaming_mode": True,
                "include_initial_eos": bool(include_initial_eos),
            }
        )
        return prompt

    def _codec_conditioning(self, language: str) -> np.ndarray:
        language_key = str(language or "auto").lower()

        def build_condition() -> np.ndarray:
            codec_prefill = self.codec_embed(np.array([self._codec_prefill_ids(language_key)], dtype=np.int64))
            return np.concatenate([codec_prefill, self._codec_tail_embed()], axis=1)

        return self._cached_condition(("voice_design", language_key), build_condition)
