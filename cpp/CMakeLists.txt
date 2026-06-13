cmake_minimum_required(VERSION 3.20)
project(qwen3_tts_cpp_clone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(cmake/OnnxRuntime.cmake)
find_package(Threads REQUIRED)

set(FFTW3F_ROOT "/home/zhang/github/Qwen3-Audio-bar/cpp/third_party/fftw3-local" CACHE PATH "Local FFTW3f root")
set(FFTW3F_INCLUDE_DIR "${FFTW3F_ROOT}/include")
set(FFTW3F_LIBRARY "${FFTW3F_ROOT}/lib/libfftw3f.a")

add_library(qwen3tts_clone_runtime
  src/audio_frontend.cc
  src/bpe_tokenizer.cc
  src/clone_pipeline.cc
  src/npy.cc
  src/ort_runner.cc
  src/product_clone_runtime.cc
  src/sampler.cc
  src/tensor.cc
  src/wav_writer.cc
)

target_include_directories(qwen3tts_clone_runtime
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${FFTW3F_INCLUDE_DIR}
)

target_link_libraries(qwen3tts_clone_runtime
  PUBLIC
    ${ONNXRUNTIME_LIBRARY}
    ${FFTW3F_LIBRARY}
    Threads::Threads
    ${CMAKE_DL_LIBS}
)

add_executable(qwen3tts_clone_fixture apps/clone_fixture_main.cc)
target_link_libraries(qwen3tts_clone_fixture PRIVATE qwen3tts_clone_runtime)

add_executable(qwen3tts_clone apps/qwen3tts_clone_main.cc)
target_link_libraries(qwen3tts_clone PRIVATE qwen3tts_clone_runtime)

get_filename_component(ONNXRUNTIME_LIBRARY_DIR "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
set_target_properties(qwen3tts_clone_fixture PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR};/home/zhang/miniconda3/lib/python3.12/site-packages/onnxruntime/capi;/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/cublas/lib;/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib;/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/lib;/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/cufft/lib;/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/curand/lib"
)
set_target_properties(qwen3tts_clone PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_LIBRARY_DIR};/home/zhang/miniconda3/lib/python3.12/site-packages/onnxruntime/capi;/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/cublas/lib;/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib;/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/lib;/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/cufft/lib;/home/zhang/miniconda3/lib/python3.12/site-packages/nvidia/curand/lib"
)
