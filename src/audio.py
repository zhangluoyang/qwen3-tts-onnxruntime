from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")


AudioInput = Union[str, Path, tuple[np.ndarray, int]]


def load_audio(audio: AudioInput) -> tuple[np.ndarray, int]:
    if isinstance(audio, tuple) and len(audio) == 2:
        wav, sr = audio
        return _mono(np.asarray(wav, dtype=np.float32)), int(sr)

    import soundfile as sf

    wav, sr = sf.read(str(audio), dtype="float32", always_2d=False)
    return _mono(wav), int(sr)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if int(orig_sr) == int(target_sr):
        return np.asarray(audio, dtype=np.float32)

    import librosa

    return librosa.resample(
        y=np.asarray(audio, dtype=np.float32),
        orig_sr=int(orig_sr),
        target_sr=int(target_sr),
    ).astype(np.float32)


def make_speaker_mel(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = resample_audio(_mono(audio), sr, 24000)
    mel = _mel_spectrogram(
        audio,
        n_fft=1024,
        num_mels=128,
        sampling_rate=24000,
        hop_size=256,
        win_size=1024,
        fmin=0,
        fmax=12000,
    )
    return mel.T.astype(np.float32)


def _mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    return audio.astype(np.float32, copy=False)


def _mel_spectrogram(
    audio: np.ndarray,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: int,
    fmax: int | None = None,
    center: bool = False,
) -> np.ndarray:
    import torch
    from librosa.filters import mel as librosa_mel_fn

    y = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    mel_basis = torch.from_numpy(
        librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
    ).float()
    hann_window = torch.hann_window(win_size)

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1.0e-9)
    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1.0e-5))
    return mel_spec.squeeze(0).numpy()
