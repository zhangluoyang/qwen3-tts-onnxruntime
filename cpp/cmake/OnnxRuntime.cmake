get_filename_component(_ORT_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
get_filename_component(_ORT_CPP_DIR "${_ORT_CMAKE_DIR}/.." ABSOLUTE)

set(_ORT_LOCAL_ROOT "${_ORT_CPP_DIR}/third_party/onnxruntime-local/onnxruntime-linux-x64-gpu-1.26.0")
set(_ORT_PY_DIR "/home/zhang/miniconda3/lib/python3.12/site-packages/onnxruntime/capi")
set(_ORT_AUDIO_BAR_ROOT "/home/zhang/github/Qwen3-Audio-bar/cpp/third_party/onnxruntime-local/onnxruntime-linux-x64-gpu-1.26.0")

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime release root containing include/ and lib/")
set(ONNXRUNTIME_LIB "" CACHE FILEPATH "Path to libonnxruntime.so")

if((NOT ONNXRUNTIME_ROOT OR NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so") AND EXISTS "${_ORT_LOCAL_ROOT}/lib/libonnxruntime.so")
  set(ONNXRUNTIME_ROOT "${_ORT_LOCAL_ROOT}" CACHE PATH "ONNX Runtime release root" FORCE)
elseif((NOT ONNXRUNTIME_ROOT OR NOT EXISTS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so") AND EXISTS "${_ORT_AUDIO_BAR_ROOT}/lib/libonnxruntime.so")
  set(ONNXRUNTIME_ROOT "${_ORT_AUDIO_BAR_ROOT}" CACHE PATH "ONNX Runtime release root" FORCE)
endif()

if(ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
  if(NOT ONNXRUNTIME_LIB)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "${ONNXRUNTIME_ROOT}/lib" "${ONNXRUNTIME_ROOT}/lib64" NO_DEFAULT_PATH)
  endif()
endif()

if(ONNXRUNTIME_LIB)
  set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_LIB}")
endif()

if(NOT ONNXRUNTIME_LIBRARY AND EXISTS "${_ORT_PY_DIR}/libonnxruntime.so")
  set(ONNXRUNTIME_LIBRARY "${_ORT_PY_DIR}/libonnxruntime.so")
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR AND EXISTS "/home/zhang/github/Qwen3-Audio-bar/cpp/third_party/onnxruntime/include/onnxruntime_cxx_api.h")
  set(ONNXRUNTIME_INCLUDE_DIR "/home/zhang/github/Qwen3-Audio-bar/cpp/third_party/onnxruntime/include")
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "ONNX Runtime C++ headers not found. Set ONNXRUNTIME_ROOT.")
endif()

if(NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "libonnxruntime.so not found. Set ONNXRUNTIME_ROOT or ONNXRUNTIME_LIB.")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")
