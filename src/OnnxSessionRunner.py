from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")
os.environ.setdefault("ORT_LOG_VERBOSITY_LEVEL", "0")
import onnxruntime as ort

ort.set_default_logger_severity(4)
ort.set_default_logger_verbosity(0)


ORT_INPUT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def provider_uses_cuda(providers):
    return any(
        provider == "CUDAExecutionProvider"
        or (isinstance(provider, tuple) and provider and provider[0] == "CUDAExecutionProvider")
        for provider in (providers or [])
    )


def is_ortvalue(value):
    return isinstance(value, ort.OrtValue)


def normalize_single_value_shape(array, input_shape):
    if input_shape is None or array.size != 1:
        return array

    shape = list(input_shape)
    if len(shape) == 0:
        return array.reshape(())

    if not all(isinstance(dim, (int, np.integer)) for dim in shape):
        return array
    expected_shape = tuple(int(dim) for dim in shape)
    if int(np.prod(expected_shape, dtype=np.int64)) != 1:
        return array
    return array.reshape(expected_shape)


def prepare_cpu_input(array):
    return array if array.ndim == 0 else np.ascontiguousarray(array)


def make_quiet_session_options(session_options=None):
    options = session_options or ort.SessionOptions()
    options.log_severity_level = 4
    options.log_verbosity_level = 0
    return options


class OnnxSessionRunner:
    def __init__(
        self,
        path,
        providers=None,
        timer=None,
        name=None,
        use_iobinding=None,
        session_options=None,
        provider_options=None,
        output_device=None,
        output_device_id=0,
        log_severity_level=None
    ):
        self.path = Path(path)
        self.requested_providers = list(providers or ["CPUExecutionProvider"])
        self.timer = timer
        self.name = name or self.path.stem
        self.output_device_id = int(output_device_id)

        start = time.perf_counter()
        session_options = make_quiet_session_options(session_options)
        if log_severity_level is not None:
            session_options.log_severity_level = log_severity_level
        self.session = ort.InferenceSession(
            str(self.path),
            sess_options=session_options,
            providers=self.requested_providers,
            provider_options=provider_options,
        )
        self.providers = list(self.session.get_providers())
        if use_iobinding is None:
            use_iobinding = provider_uses_cuda(self.providers)
        self.use_iobinding = bool(use_iobinding)
        self.output_device = output_device or ("cuda" if provider_uses_cuda(self.providers) else "cpu")
        self._refresh_metadata()
        if timer is not None:
            timer.add(f"session_load.{self.name}", time.perf_counter() - start)

    @classmethod
    def from_session(
        cls,
        session,
        timer=None,
        name=None,
        path=None,
        use_iobinding=None,
        output_device=None,
        output_device_id=0,
    ):
        runner = cls.__new__(cls)
        runner.path = Path(path) if path is not None else None
        runner.requested_providers = None
        runner.session = session
        runner.providers = list(session.get_providers())
        runner.timer = timer
        runner.name = name or (runner.path.stem if runner.path is not None else "session")
        if use_iobinding is None:
            use_iobinding = provider_uses_cuda(runner.providers)
        runner.use_iobinding = bool(use_iobinding)
        runner.output_device = output_device or ("cuda" if provider_uses_cuda(runner.providers) else "cpu")
        runner.output_device_id = int(output_device_id)
        runner._refresh_metadata()
        return runner

    @property
    def raw_session(self):
        return self.session

    def __getattr__(self, attr):
        return getattr(self.session, attr)

    def _refresh_metadata(self):
        self.input_metas = {item.name: item for item in self.session.get_inputs()}
        self.output_metas = {item.name: item for item in self.session.get_outputs()}
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]

    def prepare_feed_array(self, input_name, value):
        if is_ortvalue(value):
            return value
        array = np.asarray(value)
        input_meta = self.input_metas.get(input_name)
        if input_meta is not None:
            dtype = ORT_INPUT_DTYPES.get(input_meta.type)
            if dtype is not None:
                array = array.astype(dtype, copy=False)
            array = normalize_single_value_shape(array, input_meta.shape)
        return prepare_cpu_input(array)

    def cast_feed(self, feed: Mapping[str, Any] = None):
        return {
            input_name: self.prepare_feed_array(input_name, input_value)
            for input_name, input_value in (feed or {}).items()
        }

    def run(
        self,
        output_names=None,
        feed=None,
        timer=None,
        name=None,
        use_iobinding=None,
        copy_outputs_to_cpu=True,
        output_device_overrides=None,
    ):
        output_names = self._normalize_output_names(output_names)
        feed = self.cast_feed(feed)
        active_timer = timer if timer is not None else self.timer
        timing_name = name or self.name
        should_iobind = self.use_iobinding if use_iobinding is None else bool(use_iobinding)

        start = time.perf_counter()
        if should_iobind:
            outputs = self.run_iobinding(
                output_names,
                feed,
                copy_outputs_to_cpu=copy_outputs_to_cpu,
                feed_is_prepared=True,
                output_device_overrides=output_device_overrides,
            )
        else:
            outputs = self.session.run(output_names, self._plain_run_feed(feed))
        if active_timer is not None:
            active_timer.add(f"onnx.{timing_name}", time.perf_counter() - start)
        return outputs

    def run_iobinding(
        self,
        output_names=None,
        feed=None,
        copy_outputs_to_cpu=True,
        feed_is_prepared=False,
        output_device_overrides=None,
    ):
        output_names = self._binding_output_names(output_names)
        feed = feed if feed_is_prepared else self.cast_feed(feed)

        binding = self.session.io_binding()
        for input_name, input_value in feed.items():
            if is_ortvalue(input_value):
                binding.bind_ortvalue_input(input_name, input_value)
            else:
                binding.bind_cpu_input(input_name, input_value)

        output_device_overrides = output_device_overrides or {}
        for output_name in output_names:
            device_type, device_id = self._output_binding_device(output_name, output_device_overrides)
            if device_type == "cpu":
                binding.bind_output(output_name, device_type="cpu")
            else:
                binding.bind_output(output_name, device_type=device_type, device_id=device_id)

        self.session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu() if copy_outputs_to_cpu else binding.get_outputs()

    def to_device_ortvalue(self, array, device="cuda", device_id=0):
        if is_ortvalue(array):
            return array
        return ort.OrtValue.ortvalue_from_numpy(prepare_cpu_input(np.asarray(array)), device, int(device_id))

    def _normalize_output_names(self, output_names):
        if output_names is None:
            return None
        if isinstance(output_names, str):
            return [output_names]
        return list(output_names)

    def _binding_output_names(self, output_names):
        output_names = self._normalize_output_names(output_names)
        return self.output_names if output_names is None else output_names

    def _output_binding_device(self, output_name, output_device_overrides):
        spec = output_device_overrides.get(output_name)
        if spec is None:
            return self.output_device, self.output_device_id
        if isinstance(spec, str):
            return spec, self.output_device_id
        if isinstance(spec, Mapping):
            device_type = spec.get("device_type", spec.get("device", self.output_device))
            device_id = int(spec.get("device_id", self.output_device_id))
            return device_type, device_id
        if isinstance(spec, (tuple, list)):
            device_type = spec[0]
            device_id = int(spec[1]) if len(spec) > 1 else self.output_device_id
            return device_type, device_id
        raise TypeError(f"Unsupported output device override for {output_name!r}: {spec!r}")

    def _plain_run_feed(self, feed):
        return {
            input_name: input_value.numpy() if is_ortvalue(input_value) else input_value
            for input_name, input_value in feed.items()
        }