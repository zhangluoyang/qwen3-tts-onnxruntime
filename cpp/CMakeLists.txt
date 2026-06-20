cmake_minimum_required(VERSION 3.20)
project(qwen3_tts_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(cmake/OnnxRuntime.cmake)
find_package(Threads REQUIRED)

set(_PROJECT_FFTW3F_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/fftw3-local")
set(_FALLBACK_FFTW3F_ROOT "/home/zhang/github/Qwen3-Audio-bar/cpp/third_party/fftw3-local")
if(NOT DEFINED FFTW3F_ROOT OR FFTW3F_ROOT STREQUAL "" OR NOT EXISTS "${FFTW3F_ROOT}/lib/libfftw3f.a")
  if(EXISTS "${_PROJECT_FFTW3F_ROOT}/lib/libfftw3f.a")
    set(FFTW3F_ROOT "${_PROJECT_FFTW3F_ROOT}" CACHE PATH "Local FFTW3f root" FORCE)
  elseif(EXISTS "${_FALLBACK_FFTW3F_ROOT}/lib/libfftw3f.a")
    set(FFTW3F_ROOT "${_FALLBACK_FFTW3F_ROOT}" CACHE PATH "Local FFTW3f root" FORCE)
  else()
    set(FFTW3F_ROOT "${_PROJECT_FFTW3F_ROOT}" CACHE PATH "Local FFTW3f root" FORCE)
  endif()
endif()
set(FFTW3F_INCLUDE_DIR "${FFTW3F_ROOT}/include")
set(FFTW3F_LIBRARY "${FFTW3F_ROOT}/lib/libfftw3f.a")

if(NOT EXISTS "${FFTW3F_INCLUDE_DIR}/fftw3.h" OR NOT EXISTS "${FFTW3F_LIBRARY}")
  message(FATAL_ERROR "FFTW3f not found. Run: bash cpp/scripts/download_deps.sh")
endif()

get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
get_filename_component(_ONNXRUNTIME_PACKAGE_DIR "${ONNXRUNTIME_LIBRARY_DIR}" DIRECTORY)
get_filename_component(_PYTHON_SITE_PACKAGES_DIR "${_ONNXRUNTIME_PACKAGE_DIR}" DIRECTORY)

set(_QWEN3TTS_NVIDIA_ROOT_CANDIDATES "${_PYTHON_SITE_PACKAGES_DIR}/nvidia")
if(DEFINED _ORT_PY_DIR)
  get_filename_component(_ORT_PY_PACKAGE_DIR "${_ORT_PY_DIR}" DIRECTORY)
  get_filename_component(_ORT_PY_SITE_PACKAGES_DIR "${_ORT_PY_PACKAGE_DIR}" DIRECTORY)
  list(APPEND _QWEN3TTS_NVIDIA_ROOT_CANDIDATES "${_ORT_PY_SITE_PACKAGES_DIR}/nvidia")
endif()
if(DEFINED ENV{CONDA_PREFIX})
  file(GLOB _CONDA_NVIDIA_ROOTS "$ENV{CONDA_PREFIX}/lib/python*/site-packages/nvidia")
  if(_CONDA_NVIDIA_ROOTS)
    list(SORT _CONDA_NVIDIA_ROOTS COMPARE NATURAL ORDER DESCENDING)
  endif()
  list(APPEND _QWEN3TTS_NVIDIA_ROOT_CANDIDATES ${_CONDA_NVIDIA_ROOTS})
endif()

if(NOT DEFINED QWEN3TTS_NVIDIA_ROOT OR QWEN3TTS_NVIDIA_ROOT STREQUAL "" OR
   NOT EXISTS "${QWEN3TTS_NVIDIA_ROOT}/cuda_runtime/lib")
  foreach(_candidate IN LISTS _QWEN3TTS_NVIDIA_ROOT_CANDIDATES)
    if(EXISTS "${_candidate}/cuda_runtime/lib")
      set(QWEN3TTS_NVIDIA_ROOT "${_candidate}" CACHE PATH
        "Root directory of Python wheel CUDA provider dependencies" FORCE)
      break()
    endif()
  endforeach()
endif()
if(NOT DEFINED QWEN3TTS_NVIDIA_ROOT)
  set(QWEN3TTS_NVIDIA_ROOT "${_PYTHON_SITE_PACKAGES_DIR}/nvidia" CACHE PATH
    "Root directory of Python wheel CUDA provider dependencies")
endif()

set(_QWEN3TTS_DEFAULT_CUDA_PROVIDER_LIB_DIRS
  "${ONNXRUNTIME_LIBRARY_DIR}"
  "${QWEN3TTS_NVIDIA_ROOT}/cublas/lib"
  "${QWEN3TTS_NVIDIA_ROOT}/cuda_runtime/lib"
  "${QWEN3TTS_NVIDIA_ROOT}/cudnn/lib"
  "${QWEN3TTS_NVIDIA_ROOT}/cufft/lib"
  "${QWEN3TTS_NVIDIA_ROOT}/curand/lib"
)
set(_QWEN3TTS_HAS_CUDART_DIR FALSE)
if(DEFINED QWEN3TTS_CUDA_PROVIDER_LIB_DIRS)
  foreach(_dir IN LISTS QWEN3TTS_CUDA_PROVIDER_LIB_DIRS)
    if(EXISTS "${_dir}/libcudart.so.12")
      set(_QWEN3TTS_HAS_CUDART_DIR TRUE)
      break()
    endif()
  endforeach()
endif()
if(NOT DEFINED QWEN3TTS_CUDA_PROVIDER_LIB_DIRS OR QWEN3TTS_CUDA_PROVIDER_LIB_DIRS STREQUAL "" OR
   NOT _QWEN3TTS_HAS_CUDART_DIR)
  set(QWEN3TTS_CUDA_PROVIDER_LIB_DIRS "${_QWEN3TTS_DEFAULT_CUDA_PROVIDER_LIB_DIRS}" CACHE STRING
    "Semicolon-separated CUDA provider library directories" FORCE)
endif()

set(QWEN3TTS_CUDA_PROVIDER_LIB_DIRS_INITIALIZER "")
foreach(_dir IN LISTS QWEN3TTS_CUDA_PROVIDER_LIB_DIRS)
  string(APPEND QWEN3TTS_CUDA_PROVIDER_LIB_DIRS_INITIALIZER "\n    \"${_dir}\",")
endforeach()
list(LENGTH QWEN3TTS_CUDA_PROVIDER_LIB_DIRS QWEN3TTS_CUDA_PROVIDER_LIB_DIRS_COUNT)
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/qwen3tts/build_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/qwen3tts/build_config.h"
  @ONLY
)

add_library(qwen3tts_runtime
  src/audio_frontend.cc
  src/bpe_tokenizer.cc
  src/clone_pipeline.cc
  src/ort_runner.cc
  src/product_clone_runtime.cc
  src/qwen_model.cc
  src/sampler.cc
  src/tensor.cc
  src/wav_writer.cc
)

target_include_directories(qwen3tts_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/generated
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${FFTW3F_INCLUDE_DIR}
)

target_link_libraries(qwen3tts_runtime
  PUBLIC
    ${ONNXRUNTIME_LIBRARY}
    ${FFTW3F_LIBRARY}
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

add_executable(qwen3tts_examples apps/qwen3tts_examples_main.cc)
target_link_libraries(qwen3tts_examples PRIVATE qwen3tts_runtime)

set_target_properties(qwen3tts_examples PROPERTIES
  BUILD_RPATH "${QWEN3TTS_CUDA_PROVIDER_LIB_DIRS}"
)
