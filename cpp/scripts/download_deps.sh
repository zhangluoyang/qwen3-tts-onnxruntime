#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY_DIR="${QWEN3TTS_THIRD_PARTY_DIR:-${CPP_DIR}/third_party}"
DOWNLOAD_DIR="${THIRD_PARTY_DIR}/_downloads"

ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION:-1.26.0}"
ONNXRUNTIME_PACKAGE="onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}"
ONNXRUNTIME_ROOT="${THIRD_PARTY_DIR}/onnxruntime-local"
ONNXRUNTIME_DIR="${ONNXRUNTIME_ROOT}/${ONNXRUNTIME_PACKAGE}"
ONNXRUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/${ONNXRUNTIME_PACKAGE}.tgz"

FFTW_VERSION="${FFTW_VERSION:-3.3.11}"
FFTW_PREFIX="${THIRD_PARTY_DIR}/fftw3-local"
FFTW_URL="https://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

download_file() {
  local url="$1"
  local output="$2"
  if [[ -f "${output}" ]]; then
    echo "已存在: ${output}"
    return
  fi
  echo "下载: ${url}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "${output}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${output}" "${url}"
  else
    echo "错误: 需要 curl 或 wget 来下载依赖。" >&2
    exit 1
  fi
}

install_onnxruntime() {
  if [[ -f "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h" && -f "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" ]]; then
    echo "ONNX Runtime 已安装: ${ONNXRUNTIME_DIR}"
    return
  fi

  mkdir -p "${DOWNLOAD_DIR}" "${ONNXRUNTIME_ROOT}"
  local archive="${DOWNLOAD_DIR}/${ONNXRUNTIME_PACKAGE}.tgz"
  download_file "${ONNXRUNTIME_URL}" "${archive}"

  echo "解压 ONNX Runtime 到: ${ONNXRUNTIME_ROOT}"
  rm -rf "${ONNXRUNTIME_DIR}"
  tar -xzf "${archive}" -C "${ONNXRUNTIME_ROOT}"

  if [[ ! -f "${ONNXRUNTIME_DIR}/include/onnxruntime_cxx_api.h" || ! -f "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" ]]; then
    echo "错误: ONNX Runtime 解压后没有找到 include/ 或 lib/。" >&2
    exit 1
  fi
}

install_fftw() {
  if [[ -f "${FFTW_PREFIX}/include/fftw3.h" && -f "${FFTW_PREFIX}/lib/libfftw3f.a" ]]; then
    echo "FFTW3f 已安装: ${FFTW_PREFIX}"
    return
  fi

  for tool in gcc make; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
      echo "错误: 编译 FFTW 需要 ${tool}。" >&2
      exit 1
    fi
  done

  mkdir -p "${DOWNLOAD_DIR}"
  local archive="${DOWNLOAD_DIR}/fftw-${FFTW_VERSION}.tar.gz"
  local build_dir="${THIRD_PARTY_DIR}/_build_fftw"
  download_file "${FFTW_URL}" "${archive}"

  echo "编译 FFTW ${FFTW_VERSION}，安装到: ${FFTW_PREFIX}"
  rm -rf "${build_dir}" "${FFTW_PREFIX}"
  mkdir -p "${build_dir}"
  tar -xzf "${archive}" -C "${build_dir}"

  pushd "${build_dir}/fftw-${FFTW_VERSION}" >/dev/null
  ./configure \
    --prefix="${FFTW_PREFIX}" \
    --enable-single \
    --enable-static \
    --disable-shared \
    --disable-fortran
  make -j "${JOBS}"
  make install
  popd >/dev/null

  rm -rf "${build_dir}"
}

mkdir -p "${THIRD_PARTY_DIR}"
install_onnxruntime
install_fftw

cat <<EOF

依赖准备完成:
  ONNX Runtime: ${ONNXRUNTIME_DIR}
  FFTW3f:       ${FFTW_PREFIX}

下一步编译:
  cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build cpp/build -j
EOF
