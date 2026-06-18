from __future__ import annotations

import subprocess
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    cmake_args = [
        "cmake",
        "-S",
        str(ROOT / "cpp"),
        "-B",
        str(ROOT / "cpp" / "build"),
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    onnxruntime_root = (
        ROOT
        / "cpp"
        / "third_party"
        / "onnxruntime-local"
        / "onnxruntime-linux-x64-gpu-1.26.0"
    )
    fftw_root = ROOT / "cpp" / "third_party" / "fftw3-local"
    if (onnxruntime_root / "lib" / "libonnxruntime.so").exists():
        cmake_args.append(f"-DONNXRUNTIME_ROOT={onnxruntime_root}")
    if (fftw_root / "lib" / "libfftw3f.a").exists():
        cmake_args.append(f"-DFFTW3F_ROOT={fftw_root}")

    subprocess.check_call(cmake_args)
    subprocess.check_call(
        [
            "cmake",
            "--build",
            str(ROOT / "cpp" / "build"),
            "-j",
            str(os.cpu_count() or 8),
        ]
    )


if __name__ == "__main__":
    main()
