"""Read-only GPU inventory and conservative shared-memory admission."""

from __future__ import annotations

import csv
import math
import os
import subprocess
from dataclasses import asdict, dataclass


class GPUInspectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class GPUProcess:
    pid: int
    memory_mib: int
    name: str | None
    user: str | None


@dataclass(frozen=True)
class GPUSnapshot:
    index: int
    uuid: str
    memory_free_mib: int
    memory_used_mib: int
    memory_total_mib: int
    utilization_percent: int | None
    temperature_c: int | None
    processes: tuple[GPUProcess, ...]

    def json(self) -> dict[str, object]:
        return asdict(self)


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GPUInspectionError("nvidia-smi inventory failed; refusing GPU admission") from exc
    return result.stdout


def _number(value: str) -> int | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]"}:
        return None
    return int(float(value))


def _owner(pid: int) -> str | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "user=", "-p", str(pid)],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    owner = result.stdout.strip()
    return owner or None


def inventory() -> tuple[GPUSnapshot, ...]:
    """Capture GPU and compute-process evidence immediately before launch."""
    raw_gpus = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = list(csv.reader(line for line in raw_gpus.splitlines() if line.strip()))
    by_uuid: dict[str, list[GPUProcess]] = {}
    raw_processes = _run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    )
    for row in csv.reader(line for line in raw_processes.splitlines() if line.strip()):
        if len(row) != 4:
            raise GPUInspectionError("unexpected nvidia-smi process CSV shape")
        uuid, raw_pid, name, raw_memory = (item.strip() for item in row)
        try:
            pid = int(raw_pid)
            memory = int(float(raw_memory))
        except ValueError as exc:
            raise GPUInspectionError("invalid nvidia-smi process data") from exc
        by_uuid.setdefault(uuid, []).append(GPUProcess(pid=pid, memory_mib=memory, name=name or None, user=_owner(pid)))
    snapshots: list[GPUSnapshot] = []
    for row in rows:
        if len(row) != 7:
            raise GPUInspectionError("unexpected nvidia-smi GPU CSV shape")
        try:
            index = int(row[0].strip())
            free, used, total = (int(float(item.strip())) for item in row[2:5])
        except ValueError as exc:
            raise GPUInspectionError("invalid nvidia-smi memory data") from exc
        snapshots.append(
            GPUSnapshot(
                index=index,
                uuid=row[1].strip(),
                memory_free_mib=free,
                memory_used_mib=used,
                memory_total_mib=total,
                utilization_percent=_number(row[5]),
                temperature_c=_number(row[6]),
                processes=tuple(by_uuid.get(row[1].strip(), [])),
            )
        )
    if not snapshots:
        raise GPUInspectionError("nvidia-smi reported no GPUs")
    return tuple(snapshots)


def required_free_memory_mib(pilot_peak_reserved_mib: int | None) -> int | None:
    if pilot_peak_reserved_mib is None or pilot_peak_reserved_mib <= 0:
        return None
    return math.ceil(1.25 * pilot_peak_reserved_mib)


@dataclass(frozen=True)
class GPUAdmission:
    allowed: bool
    state: str
    reason: str
    required_free_memory_mib: int | None
    shared_gpu_at_launch: bool


def admit(
    snapshot: GPUSnapshot,
    *,
    external_process_policy: str,
    pilot_peak_reserved_mib: int | None,
    campaign_claimed: bool,
    reserved_by_current_run: bool,
    external_processes_enabled: bool | None = None,
) -> GPUAdmission:
    """Never solve contention by changing a scientific execution setting."""
    if reserved_by_current_run:
        return GPUAdmission(False, "waiting_gpu", "reserved by protected current run", None, False)
    if campaign_claimed:
        return GPUAdmission(False, "waiting_gpu", "already claimed by this campaign", None, False)
    processes = bool(snapshot.processes)
    enabled = (
        external_processes_enabled
        if external_processes_enabled is not None
        else os.environ.get("ARD_CAMPAIGN_ALLOW_EXTERNAL_GPU_PROCESSES") == "1"
    )
    if processes and (external_process_policy != "allow_with_memory_gate" or not enabled):
        return GPUAdmission(False, "waiting_gpu", "external compute process present", None, False)
    required = required_free_memory_mib(pilot_peak_reserved_mib)
    if processes and required is None:
        return GPUAdmission(
            False,
            "waiting_for_memory",
            "pilot peak reserved memory is required for shared GPU admission",
            None,
            True,
        )
    if required is not None and snapshot.memory_free_mib < required:
        return GPUAdmission(
            False,
            "waiting_for_memory",
            f"free memory {snapshot.memory_free_mib} MiB is below required {required} MiB",
            required,
            processes,
        )
    return GPUAdmission(True, "launching", "admitted", required, processes)
