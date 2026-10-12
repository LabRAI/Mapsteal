"""Single-GPU execution bootstrap for real CUDA training and local inference.

``model.to("cuda:4")`` does not stop HuggingFace ``Trainer`` from wrapping the
model in ``DataParallel`` across every visible device.  The only reliable barrier
is process-level: resolve the requested torch index to a physical GPU UUID, then
re-exec the command in a clean child whose ``CUDA_VISIBLE_DEVICES`` contains that
UUID alone, so the runtime sees exactly one device at ``cuda:0``.

The child re-verifies the isolation before any weight is loaded and fails closed
rather than falling back to more GPUs.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ISOLATION_MARKER = "MAPSTEAL_SINGLE_GPU"
REQUESTED_DEVICE_ENV = "MAPSTEAL_REQUESTED_DEVICE"
REQUESTED_UUID_ENV = "MAPSTEAL_REQUESTED_UUID"
ISOLATED_DEVICE = "cuda:0"
PARALLELISM_SINGLE = "single_gpu"

#: Commands that load a model and therefore need single-GPU isolation (the MapSteal CLI isolates `train` and `evaluate` itself).
TRAINING_COMMANDS = ("train",)
LOCAL_DEVICE_COMMANDS = ()
ISOLATED_MODEL_COMMANDS = TRAINING_COMMANDS + LOCAL_DEVICE_COMMANDS


class GpuIsolationError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceIdentity:
    """Physical identity of the GPU the operator asked for, resolved by torch index."""

    requested_torch_device: str
    requested_torch_index: int
    requested_physical_uuid: str
    gpu_name: str
    compute_capability: str
    visible_devices_at_launch: tuple[str, ...]
    cuda_visible_devices_env: str | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["visible_devices_at_launch"] = list(self.visible_devices_at_launch)
        return value


def is_cuda_device(device: str) -> bool:
    return str(device).strip().lower().startswith("cuda")


def torch_index(device: str) -> int:
    text = str(device).strip().lower()
    if text == "cuda":
        return 0
    if not text.startswith("cuda:"):
        raise GpuIsolationError(f"not a CUDA device: {device}")
    suffix = text.split(":", 1)[1]
    if not suffix.isdigit():
        raise GpuIsolationError(f"malformed CUDA device: {device}")
    return int(suffix)


def _normalise_uuid(raw: Any) -> str:
    text = str(raw)
    return text if text.startswith("GPU-") else f"GPU-{text}"


def torch_is_initialized() -> bool:
    """True when CUDA is already live, so ``CUDA_VISIBLE_DEVICES`` can no longer bind."""
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:
        return False


def resolve_device_identity(device: str) -> DeviceIdentity:
    """Map the requested torch index to its physical GPU. Never uses nvidia-smi order."""
    import torch

    if not is_cuda_device(device):
        raise GpuIsolationError(f"device isolation only applies to CUDA devices: {device}")
    if not torch.cuda.is_available():
        raise GpuIsolationError("CUDA is not available in this process")
    index = torch_index(device)
    count = torch.cuda.device_count()
    if index >= count:
        raise GpuIsolationError(
            f"requested {device} but only {count} CUDA device(s) are visible")
    visible = []
    for i in range(count):
        visible.append(_normalise_uuid(torch.cuda.get_device_properties(i).uuid))
    props = torch.cuda.get_device_properties(index)
    return DeviceIdentity(
        requested_torch_device=str(device),
        requested_torch_index=index,
        requested_physical_uuid=_normalise_uuid(props.uuid),
        gpu_name=props.name,
        compute_capability=f"{props.major}.{props.minor}",
        visible_devices_at_launch=tuple(visible),
        cuda_visible_devices_env=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def is_isolated_child() -> bool:
    return os.environ.get(ISOLATION_MARKER) == "1"


def child_environment(identity: DeviceIdentity,
                      base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment that pins the child to one physical GPU and blocks a re-exec loop."""
    env = dict(os.environ if base_env is None else base_env)
    env["CUDA_VISIBLE_DEVICES"] = identity.requested_physical_uuid
    env[ISOLATION_MARKER] = "1"
    env[REQUESTED_DEVICE_ENV] = identity.requested_torch_device
    env[REQUESTED_UUID_ENV] = identity.requested_physical_uuid
    env["WORLD_SIZE"] = "1"
    env.pop("RANK", None)
    env.pop("LOCAL_RANK", None)
    return env


def child_argv(argv: list[str], device_flag: str = "--device") -> list[str]:
    """Rewrite the operator's device to the isolated runtime device."""
    out = list(argv)
    for i, token in enumerate(out):
        if token == device_flag and i + 1 < len(out):
            out[i + 1] = ISOLATED_DEVICE
            return out
        if token.startswith(f"{device_flag}="):
            out[i] = f"{device_flag}={ISOLATED_DEVICE}"
            return out
    raise GpuIsolationError(f"cannot isolate: {device_flag} is absent from the command")


def require_single_gpu_runtime(expected_uuid: str | None = None) -> dict[str, Any]:
    """Hard runtime gate. Raises before any model weight is loaded."""
    import torch

    expected = expected_uuid or os.environ.get(REQUESTED_UUID_ENV)
    if not is_isolated_child():
        raise GpuIsolationError(
            "CUDA model execution must run inside the single-GPU bootstrap; "
            "no isolation marker is present")
    if not expected:
        raise GpuIsolationError("single-GPU child is missing the requested physical GPU UUID")
    if not torch.cuda.is_available():
        raise GpuIsolationError("CUDA is not available inside the isolated child")
    count = torch.cuda.device_count()
    if count != 1:
        raise GpuIsolationError(
            f"isolation failed: {count} CUDA devices are visible, expected exactly 1")
    props = torch.cuda.get_device_properties(0)
    runtime_uuid = _normalise_uuid(props.uuid)
    if expected and runtime_uuid != expected:
        raise GpuIsolationError(
            f"isolated GPU {runtime_uuid} is not the requested {expected}")
    world_size = os.environ.get("WORLD_SIZE", "1")
    if world_size not in ("", "1"):
        raise GpuIsolationError(f"WORLD_SIZE={world_size}; distributed training is not permitted")
    for name in ("RANK", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value not in (None, "", "0"):
            raise GpuIsolationError(f"{name}={value}; multi-process training is not permitted")
    return {
        "requested_torch_device": os.environ.get(REQUESTED_DEVICE_ENV),
        "requested_physical_uuid": expected,
        "runtime_visible_devices": [runtime_uuid],
        "runtime_torch_device": ISOLATED_DEVICE,
        "runtime_physical_uuid": runtime_uuid,
        "runtime_device_count": count,
        "runtime_gpu_name": props.name,
        "parallelism": PARALLELISM_SINGLE,
        "world_size": 1,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "isolation_verified": True,
    }


def assert_trainer_is_single_gpu(training_args: Any, model: Any) -> None:
    """Reject a Trainer that would shard, or a model already wrapped for parallelism."""
    n_gpu = getattr(training_args, "n_gpu", None)
    if n_gpu is not None and int(n_gpu) > 1:
        raise GpuIsolationError(f"Trainer reports n_gpu={n_gpu}; refusing to shard training")
    world = getattr(training_args, "world_size", None)
    if world is not None and int(world) > 1:
        raise GpuIsolationError(f"Trainer reports world_size={world}; refusing distributed training")
    wrapper = type(model).__name__
    if wrapper in ("DataParallel", "DistributedDataParallel"):
        raise GpuIsolationError(f"model is wrapped in {wrapper}; refusing multi-GPU training")


def cpu_device_provenance(device: str) -> dict[str, Any]:
    """Provenance block for a non-CUDA run; no isolation is needed or claimed."""
    return {
        "requested_torch_device": str(device), "requested_physical_uuid": None,
        "runtime_visible_devices": [], "runtime_torch_device": str(device),
        "runtime_physical_uuid": None, "runtime_device_count": 0,
        "parallelism": "cpu", "world_size": 1, "isolation_verified": False,
    }


def requested_provenance(device: str) -> dict[str, Any]:
    """What the operator asked for, recorded when the run manifest is first written.

    The verified runtime block is filled in by the backend immediately before
    training; until then ``isolation_verified`` stays false.
    """
    if not is_cuda_device(device):
        return cpu_device_provenance(device)
    return {
        "requested_torch_device": os.environ.get(REQUESTED_DEVICE_ENV, str(device)),
        "requested_physical_uuid": os.environ.get(REQUESTED_UUID_ENV),
        "runtime_torch_device": str(device),
        "runtime_visible_devices": None, "runtime_physical_uuid": None,
        "runtime_device_count": None, "parallelism": PARALLELISM_SINGLE,
        "world_size": 1, "isolation_verified": False,
    }


def device_provenance(device: str) -> dict[str, Any]:
    """Truthful device block for a run manifest, resolved at the moment of training."""
    if not is_cuda_device(device):
        return cpu_device_provenance(device)
    return require_single_gpu_runtime()


# --------------------------------------------------------------------------- bootstrap


def run_isolated_command(identity: DeviceIdentity, argv: list[str],
                         device_flag: str = "--device",
                         python_executable: str | None = None,
                         runner=subprocess.run, module: str = "mapsteal") -> int:
    """Re-exec this CLI (or a named sibling module) in a child pinned to one GPU."""
    command = [python_executable or sys.executable, "-m", module,
               *child_argv(argv, device_flag)]
    completed = runner(command, env=child_environment(identity))
    return int(getattr(completed, "returncode", 1))


def verify_isolation_only(identity: DeviceIdentity, argv: list[str],
                          device_flag: str = "--device",
                          python_executable: str | None = None,
                          runner=subprocess.run, module: str = "mapsteal") -> dict[str, Any]:
    """Resolve, enter the isolated child, confirm one GPU, and stop.

    No training artifact is created and no model is loaded.
    """
    returncode = run_isolated_command(
        identity, argv, device_flag=device_flag,
        python_executable=python_executable, runner=runner, module=module)
    return {
        "check": "device_isolation_only", "requested": identity.to_dict(),
        "child_returncode": returncode, "isolation_verified": returncode == 0,
        "model_loaded": False, "training_started": False, "adapter_written": False,
        "artifact_created": False,
    }


def bootstrap_training_command(args: Any, argv: list[str],
                               python_executable: str | None = None,
                               runner=subprocess.run, module: str = "mapsteal") -> dict[str, Any] | None:
    """Entry hook for a training CLI command.

    Returns ``None`` when the caller should continue in-process (CPU, or already
    inside the verified child).  Returns a result dict when the parent has handed
    the work to an isolated child or completed a dry-run.
    """
    local_inference = getattr(args, "command", None) in LOCAL_DEVICE_COMMANDS
    device_attribute = "local_device" if local_inference else "device"
    device_flag = "--local-device" if local_inference else "--device"
    device = getattr(args, device_attribute, None)
    verify_only = bool(getattr(args, "verify_device_isolation_only", False))

    if device is None or not is_cuda_device(device):
        if verify_only:
            return {"check": "device_isolation_only", "requested": cpu_device_provenance(device),
                    "isolation_verified": False, "reason": "cpu device needs no isolation",
                    "model_loaded": False, "training_started": False,
                    "adapter_written": False, "artifact_created": False}
        return None

    if is_isolated_child():
        runtime = require_single_gpu_runtime()
        if verify_only:
            return {"check": "device_isolation_only", "runtime": runtime,
                    "isolation_verified": True, "model_loaded": False,
                    "training_started": False, "adapter_written": False,
                    "artifact_created": False, "child": True}
        return None

    if torch_is_initialized():
        raise GpuIsolationError(
            "CUDA is already initialized in this process; CUDA_VISIBLE_DEVICES can no longer "
            "bind, so single-GPU isolation cannot be guaranteed")
    identity = resolve_device_identity(device)
    if verify_only:
        return verify_isolation_only(
            identity, argv, device_flag=device_flag,
            python_executable=python_executable, runner=runner, module=module)
    returncode = run_isolated_command(
        identity, argv, device_flag=device_flag,
        python_executable=python_executable, runner=runner, module=module)
    if returncode != 0:
        raise GpuIsolationError(f"isolated training child exited with code {returncode}")
    return {"isolated_child_returncode": returncode, "requested": identity.to_dict(),
            "parallelism": PARALLELISM_SINGLE}


def assert_model_is_single_gpu(model: Any) -> None:
    """Reject explicit parallel wrappers on inference models."""
    wrapper = type(model).__name__
    if wrapper in ("DataParallel", "DistributedDataParallel"):
        raise GpuIsolationError(f"model is wrapped in {wrapper}; refusing multi-GPU inference")


def assert_shared_runtime_device(*models: Any) -> None:
    """A base model and its adapter must end up on the identical runtime device."""
    devices = {next(model.parameters()).device for model in models}
    if len(devices) > 1:
        raise GpuIsolationError(
            f"base and adapter ended up on different devices: {sorted(map(str, devices))}")


def preflight_record(run_dir: Path, identity_or_error: Any, reason: str) -> dict[str, Any]:
    """Minimal, unmistakable failure record. Never looks like a trainable run."""
    record = {
        "schema_version": "mapsteal-training-preflight",
        "status": "preflight_failed", "training_started": False,
        "adapter_written": False, "reason": reason,
        "requested": identity_or_error.to_dict()
        if isinstance(identity_or_error, DeviceIdentity) else str(identity_or_error),
    }
    return record
