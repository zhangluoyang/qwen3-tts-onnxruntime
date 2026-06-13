from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {"samples": 0, "rms": 0.0, "peak": 0.0, "mean_abs": 0.0}
    return {
        "samples": int(x.size),
        "rms": float(np.sqrt(np.mean(x * x))),
        "peak": float(np.max(np.abs(x))),
        "mean_abs": float(np.mean(np.abs(x))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", default="outputs/cpp_fixture")
    args = parser.parse_args()

    root = Path(args.fixture_dir)
    py_audio = np.load(root / "py_audio.npy")
    cpp_audio = np.load(root / "cpp_audio.npy")
    py_codes = np.load(root / "py_generated_codes.npy")
    cpp_codes = np.load(root / "cpp_generated_codes.npy")

    py_stats = _stats(py_audio)
    cpp_stats = _stats(cpp_audio)
    min_len = min(py_audio.size, cpp_audio.size)
    diff = np.asarray(py_audio[:min_len], dtype=np.float32) - np.asarray(cpp_audio[:min_len], dtype=np.float32)
    diff_stats = _stats(diff)

    print(f"py_audio={py_stats}")
    print(f"cpp_audio={cpp_stats}")
    print(f"audio_diff_first_{min_len}={diff_stats}")
    print(f"py_codes_shape={py_codes.shape} cpp_codes_shape={cpp_codes.shape}")
    if py_codes.shape == cpp_codes.shape:
        print(f"codes_equal={bool(np.array_equal(py_codes, cpp_codes))}")
        print(f"first_token_equal={bool(np.array_equal(py_codes[..., 0], cpp_codes[..., 0]))}")
    print(f"cpp_non_silent={bool(cpp_stats['peak'] > 1e-4 and cpp_stats['rms'] > 1e-5)}")


if __name__ == "__main__":
    main()
