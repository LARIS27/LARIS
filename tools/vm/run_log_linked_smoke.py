from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Callable, Mapping

if TYPE_CHECKING:
    from laris_onpremise.observer.runtime_state_service import (
        RuntimeStateTopologySummary,
    )


CONTROL_ENDPOINTS = ("127.0.0.2", "127.0.0.3")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SAFE_UNIT = re.compile(r"[A-Za-z0-9][A-Za-z0-9@_.:-]{0,255}\.service\Z")


class SmokeError(RuntimeError):
    """Raised when the bounded two-request smoke cannot be trusted."""


@dataclass(frozen=True)
class SmokeConfiguration:
    app_source: Path
    unit: str
    pid: int
    start_time_ticks: int
    expected_executable_sha256: str
    application_log: Path
    launch_receipt: Path
    native_observer: Path
    bpf_object: Path
    model_registration_receipt: Path = Path(
        "/srv/laris-lab/state/model-registrations/qwen2.5-0.5b-q4.json"
    )
    observer_profile_path: Path | None = None
    base_url: str = "http://127.0.0.1:8080"
    run_parent: Path = Path("/var/lib/laris-log-linked")
    control_parent: Path = Path("/run/laris-log-linked")
    observation_duration_seconds: float = 180.0

    def __post_init__(self) -> None:
        for field in (
            "app_source",
            "application_log",
            "launch_receipt",
            "native_observer",
            "bpf_object",
        ):
            value = getattr(self, field)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ValueError(f"{field} must be an absolute pathlib.Path")
        if self.observer_profile_path is not None and (
            not isinstance(self.observer_profile_path, Path)
            or not self.observer_profile_path.is_absolute()
        ):
            raise ValueError(
                "observer_profile_path must be an absolute pathlib.Path"
            )
        if not isinstance(self.unit, str) or _SAFE_UNIT.fullmatch(self.unit) is None:
            raise ValueError("unit must be an exact system service name")
        for field in ("pid", "start_time_ticks"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if (
            not isinstance(self.expected_executable_sha256, str)
            or _DIGEST.fullmatch(self.expected_executable_sha256) is None
        ):
            raise ValueError("expected_executable_sha256 is invalid")

    @property
    def app_root(self) -> Path:
        return self.app_source.parent

    @property
    def observer_profile(self) -> Path:
        if self.observer_profile_path is not None:
            return self.observer_profile_path
        return (
            self.app_root
            / "configs"
            / "runtime"
            / "runtime_state_observer"
            / "x86_64_cpu_simple_kv.json"
        )

    @property
    def adapter_profile(self) -> Path:
        return (
            self.app_root
            / "configs"
            / "runtime"
            / "log_linked"
            / "llama_server_daf6bc9.yaml"
        )

@dataclass(frozen=True)
class _PrivateTruth:
    endpoint_value: str
    slot_id: int
    request_started_monotonic_ns: int
    request_finished_monotonic_ns: int


@dataclass(frozen=True)
class _ObservedRun:
    observation: Mapping[str, object]
    truth: tuple[_PrivateTruth, ...]
    prompts: tuple[bytes, ...]
    responses: tuple[bytes, ...]


def _activate_app_source(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
    previous = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = value if not previous else value + os.pathsep + previous


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _process_start_ticks(pid: int) -> int:
    payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    return int(payload.rsplit(") ", 1)[1].split()[19])


def _process_arguments(pid: int) -> tuple[str, ...]:
    return tuple(
        item.decode("utf-8")
        for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        if item
    )


def _unit_pid(unit: str) -> int:
    completed = subprocess.run(
        ("/usr/bin/systemctl", "show", "--property", "MainPID", "--value", unit),
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        text=True,
    )
    return int(completed.stdout.strip())


def _verify_external_runtime(
    configuration: SmokeConfiguration,
) -> dict[str, object]:
    active = subprocess.run(
        ("/usr/bin/systemctl", "is-active", "--quiet", configuration.unit),
        check=False,
        stdin=subprocess.DEVNULL,
    )
    if active.returncode != 0:
        raise SmokeError("the exact external runtime unit is not active")
    pid = _unit_pid(configuration.unit)
    if (
        pid != configuration.pid
        or _process_start_ticks(pid) != configuration.start_time_ticks
    ):
        raise SmokeError("the exact external runtime identity changed")
    executable_digest = _sha256_file(Path(f"/proc/{pid}/exe"))
    if executable_digest != configuration.expected_executable_sha256:
        raise SmokeError("the external runtime executable digest differs")
    arguments = _process_arguments(pid)
    required = (
        "--log-verbosity",
        "4",
        "--log-timestamps",
        "--log-file",
        str(configuration.application_log),
    )
    if any(value not in arguments for value in required):
        raise SmokeError("the external runtime omits content-free log arguments")
    if any(
        value in arguments
        for value in (
            "-v",
            "--verbose",
            "--log-verbose",
            "--verbose-prompt",
            "--log-disable",
        )
    ):
        raise SmokeError("the external runtime enables a forbidden log option")
    try:
        receipt = json.loads(
            configuration.launch_receipt.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SmokeError("the launch receipt is unavailable") from error
    if (
        not isinstance(receipt, dict)
        or receipt.get("application") != "llama-server"
        or receipt.get("application_source_modified") is not False
        or receipt.get("executable_sha256") != executable_digest
        or tuple(receipt.get("arguments", ())) != arguments
        or not isinstance(receipt.get("version"), str)
        or not str(receipt["version"]).strip()
    ):
        raise SmokeError("the launch receipt differs from the live runtime")
    return {
        "pid": pid,
        "start_time_ticks": configuration.start_time_ticks,
        "digest": executable_digest,
        "arguments": arguments,
        "version": receipt["version"],
    }


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _value_after(arguments: tuple[str, ...], flag: str) -> str:
    if arguments.count(flag) != 1:
        raise SmokeError(f"required launch argument {flag} is unavailable")
    index = arguments.index(flag)
    if index + 1 >= len(arguments):
        raise SmokeError(f"required launch argument {flag} has no value")
    return arguments[index + 1]


def _runtime_metadata(profile: object) -> dict[str, object]:
    arguments = tuple(getattr(profile, "required_launch_arguments"))
    return {
        "schema_version": "1.0",
        "contract_kind": "laris_rehost_runtime",
        "prompt_contract": {
            "api": "llama_cpp_chat",
            "template_policy": "gguf_default",
        },
        "runtime": {
            "source_runtime_profile_digest": getattr(profile, "profile_digest"),
            "llama_cpp_commit": getattr(profile, "llama_cpp_commit"),
            "memory_family": getattr(profile, "memory_family"),
            "n_batch": int(_value_after(arguments, "--batch-size")),
            "n_ubatch": int(_value_after(arguments, "--ubatch-size")),
            "cache_type_k": _value_after(arguments, "--cache-type-k"),
            "cache_type_v": _value_after(arguments, "--cache-type-v"),
            "flash_attention": _value_after(arguments, "--flash-attn") == "on",
            "n_gpu_layers": int(_value_after(arguments, "--n-gpu-layers")),
            "context_shift": bool(getattr(profile, "context_shift")),
            "cache_prompt": bool(getattr(profile, "cache_prompt")),
            "cache_idle_slots": bool(getattr(profile, "cache_idle_slots")),
        },
    }


def _runtime_instance_id(runtime: Mapping[str, object]) -> str:
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
        encoding="ascii"
    ).strip()
    source = _canonical_json(
        {
            "boot_id": boot_id,
            "digest": runtime["digest"],
            "pid": runtime["pid"],
            "start_time_ticks": runtime["start_time_ticks"],
        }
    )
    return "sha256:" + hashlib.sha256(source).hexdigest()


def _prepare_run(
    configuration: SmokeConfiguration,
    runtime: Mapping[str, object],
) -> SimpleNamespace:
    from laris_onpremise.observer.runtime_state_profile import (
        load_runtime_state_observer_profile,
    )

    profile = load_runtime_state_observer_profile(
        configuration.observer_profile
    )
    try:
        registration = json.loads(
            configuration.model_registration_receipt.read_text(encoding="ascii")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SmokeError("the model registration receipt is unavailable") from error
    if not isinstance(registration, dict):
        raise SmokeError("the model registration receipt is invalid")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"run-log-linked-smoke-{stamp}-{os.getpid()}"
    run_root = configuration.run_parent / run_id
    control_root = configuration.control_parent / run_id
    run_root.mkdir(mode=0o700, parents=True)
    control_root.mkdir(mode=0o700, parents=True)
    socket_path = control_root / "runtime-state.sock"
    frame_dir = run_root / "runtime-state-frames"
    evidence_root = run_root / "evidence"
    private_root = run_root / "private"
    runtime_instance_id = _runtime_instance_id(runtime)
    n_ctx = int(_value_after(tuple(runtime["arguments"]), "--ctx-size"))
    workflow_path = run_root / "workflow-profile.json"
    workflow_document = {
        "schema_version": "1.0",
        "adapter_profile": str(configuration.adapter_profile),
        "runtime": {
            "pid": runtime["pid"],
            "start_time_ticks": runtime["start_time_ticks"],
            "executable": str(profile.runtime_binary_path),
            "executable_digest": runtime["digest"],
            "version": runtime["version"],
            "application_log": str(configuration.application_log),
            "canary_url": configuration.base_url + "/v1/chat/completions",
            "canary_authorization_file": None,
        },
        "observer": {
            "socket_path": str(socket_path),
            "profile_path": str(configuration.observer_profile),
            "terminal_seal_path": str(
                frame_dir / "runtime_state_terminal_seal.json"
            ),
            "runtime_instance_id": runtime_instance_id,
            "poll_interval_seconds": 0.05,
            "observation_duration_seconds": configuration.observation_duration_seconds,
        },
        "correlation": {
            "slot_sequence_map": [
                [slot_id, slot_id] for slot_id in range(profile.n_stream)
            ],
            "maximum_clock_error_ns": 100_000_000,
            "maximum_access_delay_ns": 5_000_000_000,
            "maximum_access_pre_terminal_skew_ns": 1_000_000,
            "allowed_path_classes": ["COMPLETION"],
        },
        "rehosting": {
            "model_profile_id": registration.get("model_profile_id"),
            "model_digest": registration.get("gguf_digest"),
            "runtime_profile_digest": profile.profile_digest,
            "n_ctx": n_ctx,
            "n_seq_max": profile.n_seq_max,
            "runtime_metadata": _runtime_metadata(profile),
        },
    }
    _write_new(workflow_path, _canonical_json(workflow_document) + b"\n")
    return SimpleNamespace(
        run_root=run_root,
        control_root=control_root,
        workflow_path=workflow_path,
        evidence_root=evidence_root,
        private_root=private_root,
        socket_path=socket_path,
        frame_dir=frame_dir,
        runtime_instance_id=runtime_instance_id,
        profile=profile,
        runtime=runtime,
    )


def _run_preflight(prepared: SimpleNamespace) -> Mapping[str, object]:
    from laris_onpremise.log_linked.workflow import run_preflight

    return run_preflight(prepared.workflow_path)


def _new_capture_process(
    configuration: SmokeConfiguration,
    prepared: SimpleNamespace,
    *,
    control_timeout_seconds: float = 15.0,
):
    from laris_onpremise.log_linked.capture_process import (
        LogLinkedCaptureConfiguration,
        LogLinkedCaptureProcess,
    )

    capture_configuration = LogLinkedCaptureConfiguration(
        python_executable=Path(sys.executable),
        socket_path=prepared.socket_path,
        frame_dir=prepared.frame_dir,
        profile_path=configuration.observer_profile,
        model_registration_receipt=configuration.model_registration_receipt,
        runtime_instance_id=prepared.runtime_instance_id,
        runtime_pid=configuration.pid,
        runtime_start_time_ticks=configuration.start_time_ticks,
        native_observer=configuration.native_observer,
        runtime_binary=Path(str(prepared.profile.runtime_binary_path)),
        bpf_object=configuration.bpf_object,
        ready_timeout_seconds=45.0,
        control_timeout_seconds=control_timeout_seconds,
        service_log_path=prepared.run_root / "runtime-state-service.log",
    )
    return LogLinkedCaptureProcess(configuration=capture_configuration)


def _send_completion(
    *,
    source_ip: str,
    prompt: str,
    n_predict: int,
    timeout_seconds: float,
) -> tuple[bytes, int]:
    body = _canonical_json(
        {
            "cache_prompt": False,
            "n_predict": n_predict,
            "prompt": prompt,
            "seed": 0,
            "stream": False,
            "temperature": 0,
        }
    )
    connection = http.client.HTTPConnection(
        "127.0.0.1",
        8080,
        timeout=timeout_seconds,
        source_address=(source_ip, 0),
    )
    try:
        connection.request(
            "POST",
            "/completion",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        payload = response.read(4 * 1024 * 1024)
        if response.status != 200:
            raise SmokeError("controlled request did not return HTTP 200")
    finally:
        connection.close()
    try:
        document = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SmokeError("controlled response is invalid") from error
    slot_id = document.get("id_slot") if isinstance(document, dict) else None
    if isinstance(slot_id, bool) or not isinstance(slot_id, int) or slot_id < 0:
        raise SmokeError("controlled response slot is invalid")
    return payload, slot_id


def _topology_bootstrap(_configuration: SmokeConfiguration) -> None:
    _send_completion(
        source_ip="127.0.0.1",
        prompt="LARIS topology initialization.",
        n_predict=1,
        timeout_seconds=30.0,
    )


class _SignallingClient:
    def __init__(
        self,
        socket_path: Path,
        armed: threading.Event,
        emit: Callable[[str], None],
    ) -> None:
        from laris_onpremise.observer.runtime_state_client import RuntimeStateClient

        self.inner = RuntimeStateClient(socket_path, timeout_seconds=15.0)
        self.armed = armed
        self.emit = emit

    def arm(self) -> None:
        self.emit("baseline")
        self.inner.arm()
        self.armed.set()

    def drain(self) -> None:
        self.emit("drain")
        self.inner.drain()

    def stop(self) -> None:
        self.inner.stop()
        self.emit("observer-stop")

    def completed_native_units(self, *, after_cursor: int):
        return self.inner.completed_native_units(after_cursor=after_cursor)

    def read_complete_native_unit(self, descriptor: object) -> bytes:
        return self.inner.read_complete_native_unit(descriptor)


def _controlled_prompt(index: int) -> str:
    marker = os.urandom(16).hex()
    padding = " alpha" * (8 if index == 0 else 37)
    return f"LARIS-CONTROL-{index + 1}-{marker}.{padding}"


def _observe_two_requests(
    configuration: SmokeConfiguration,
    prepared: SimpleNamespace,
    preflight: Mapping[str, object],
    native_topology: RuntimeStateTopologySummary,
    emit: Callable[[str], None],
) -> _ObservedRun:
    from laris_onpremise.log_linked.workflow import ObservationSession

    armed = threading.Event()
    finished = threading.Event()
    failures: list[BaseException] = []
    truth: list[_PrivateTruth] = []
    prompts: list[bytes] = []
    responses: list[bytes] = []
    client = _SignallingClient(prepared.socket_path, armed, emit)

    def traffic() -> None:
        try:
            if not armed.wait(timeout=30.0):
                raise SmokeError("Observer did not arm before controlled traffic")
            for index, endpoint in enumerate(CONTROL_ENDPOINTS):
                prompt = _controlled_prompt(index)
                prompt_bytes = prompt.encode("utf-8")
                prompts.append(prompt_bytes)
                emit(f"request:{endpoint}")
                started = time.monotonic_ns()
                payload, slot_id = _send_completion(
                    source_ip=endpoint,
                    prompt=prompt,
                    n_predict=3 + index * 2,
                    timeout_seconds=90.0,
                )
                finished_at = time.monotonic_ns()
                responses.append(payload)
                truth.append(
                    _PrivateTruth(
                        endpoint_value=endpoint,
                        slot_id=slot_id,
                        request_started_monotonic_ns=started,
                        request_finished_monotonic_ns=finished_at,
                    )
                )
                time.sleep(0.25)
        except BaseException as error:
            failures.append(error)
        finally:
            finished.set()

    worker = threading.Thread(
        target=traffic,
        name="log-linked-two-request-traffic",
        daemon=True,
    )
    worker.start()
    observation = ObservationSession(
        profile=prepared.workflow_path,
        evidence_root=prepared.evidence_root,
        preflight_runner=lambda _path: preflight,
        client_factory=lambda _path: client,
        native_topology=native_topology,
        completion_requested=finished.is_set,
        terminal_drain=client.drain,
        completion_quiescence_polls=2,
    ).run()
    worker.join(timeout=10.0)
    if worker.is_alive():
        raise SmokeError("controlled traffic did not finish")
    if failures:
        raise failures[0]
    if len(truth) != 2:
        raise SmokeError("controlled truth is incomplete")
    prepared.private_root.mkdir(mode=0o700, parents=True)
    _write_new(
        prepared.private_root / "oracle.jsonl",
        b"".join(
            _canonical_json(
                {
                    "endpoint_type": "IPV4",
                    "endpoint_value": item.endpoint_value,
                    "request_started_monotonic_ns": (
                        item.request_started_monotonic_ns
                    ),
                    "request_finished_monotonic_ns": (
                        item.request_finished_monotonic_ns
                    ),
                    "slot_id": item.slot_id,
                }
            )
            + b"\n"
            for item in truth
        ),
    )
    return _ObservedRun(
        observation=observation,
        truth=tuple(truth),
        prompts=tuple(prompts),
        responses=tuple(responses),
    )


def _offline_correlate(
    prepared: SimpleNamespace,
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...], bytes]:
    from laris_onpremise.log_linked.workflow import correlate_observation

    correlation_key = os.urandom(32)
    key_path = prepared.private_root / "correlation.key"
    _write_new(key_path, correlation_key)
    correlation = correlate_observation(
        evidence_root=prepared.evidence_root,
        correlation_key_file=key_path,
        correlation_key_id="log-linked-smoke-key-1",
    )
    result_path = (
        Path(str(correlation["evidence_package"]))
        / "derived"
        / "attribution_results.jsonl"
    )
    documents = tuple(
        json.loads(line)
        for line in result_path.read_text(encoding="utf-8").splitlines()
        if line
    )
    return correlation, documents, correlation_key


def _evaluate_private_oracle(
    results: tuple[Mapping[str, object], ...],
    truth: tuple[_PrivateTruth, ...],
    correlation_key: bytes,
) -> dict[str, int]:
    from laris_onpremise.log_linked.endpoint import derive_endpoint_ref

    unused = list(results)
    correct_attribution = 0
    false_attribution = 0
    certified = 0
    ranked = 0
    ambiguous = 0
    incomplete = 0
    for entry in truth:
        matches = [
            result
            for result in unused
            if result.get("slot_id") == entry.slot_id
            and isinstance(result.get("cutoff_monotonic_ns"), int)
            and entry.request_started_monotonic_ns
            <= int(result["cutoff_monotonic_ns"])
            <= entry.request_finished_monotonic_ns
        ]
        if len(matches) != 1:
            if len(matches) > 1:
                ambiguous += 1
            else:
                incomplete += 1
            continue
        result = matches[0]
        unused.remove(result)
        decision = result.get("final_decision")
        if decision == "AMBIGUOUS":
            ambiguous += 1
        elif decision == "INCOMPLETE":
            incomplete += 1
        else:
            if decision == "CERTIFIED":
                certified += 1
            elif decision == "RANKED":
                ranked += 1
            else:
                incomplete += 1
                continue
            expected = derive_endpoint_ref(
                correlation_key=correlation_key,
                endpoint_type="IPV4",
                endpoint_value=entry.endpoint_value,
            )
            if result.get("endpoint_ref") == expected:
                correct_attribution += 1
            else:
                false_attribution += 1
    incomplete += len(unused)
    return {
        "request_count": len(truth),
        "certified_count": certified,
        "ranked_count": ranked,
        "correct_attribution_count": correct_attribution,
        "false_attribution_count": false_attribution,
        "ambiguous_count": ambiguous,
        "incomplete_count": incomplete,
    }


def _verify_content_privacy(
    prepared: SimpleNamespace,
    prompts: tuple[bytes, ...],
    responses: tuple[bytes, ...],
) -> None:
    raw_log = (
        prepared.evidence_root
        / "application"
        / "raw_llama_server.log"
    ).read_bytes()
    if any(prompt in raw_log for prompt in prompts):
        raise SmokeError("controlled prompt content appeared in the application log")
    if any(response and response in raw_log for response in responses):
        raise SmokeError("controlled response content appeared in the application log")
    try:
        prepared.private_root.relative_to(prepared.evidence_root)
    except ValueError:
        return
    raise SmokeError("private oracle entered the evidence root")


def _verify_writer_receipt(prepared: SimpleNamespace) -> dict[str, object]:
    from laris_onpremise.log_linked.contracts import parse_writer_capture_receipt

    receipt = parse_writer_capture_receipt(
        (prepared.evidence_root / "writer_capture_receipt.json").read_bytes()
    )
    failures = receipt.get("safe_failure_counts")
    if not isinstance(failures, Mapping):
        raise SmokeError("writer safe-failure counters are unavailable")
    safe_failure_count = sum(
        value
        for value in failures.values()
        if isinstance(value, int) and not isinstance(value, bool)
    )
    return {
        "safe_failure_count": safe_failure_count,
        "apply_count": receipt.get("apply_count"),
        "unit_count": receipt.get("unit_count"),
        "copied_byte_count": receipt.get("copied_byte_count"),
        "observer_cpu_time_ns": receipt.get("observer_cpu_time_ns"),
        "native_ring_frame_high_water_mark": receipt.get(
            "native_ring_frame_high_water_mark"
        ),
        "persistence_queue_high_water_mark": receipt.get(
            "persistence_queue_high_water_mark"
        ),
    }


def run_smoke(
    configuration: SmokeConfiguration,
    emit: Callable[[str], None] = lambda _event: None,
) -> dict[str, object]:
    _activate_app_source(configuration.app_source)
    from laris_onpremise.observer.runtime_state_service import (
        RuntimeStateTopologySummary,
    )

    runtime = _verify_external_runtime(configuration)
    emit("identity")
    prepared = _prepare_run(configuration, runtime)
    preflight = _run_preflight(prepared)
    if preflight.get("status") != "SUPPORTED":
        raise SmokeError("content-free Log-linked preflight is unsupported")
    capture = _new_capture_process(configuration, prepared)
    emit("attach")
    try:
        native_topology_document = capture.start(
            lambda: (
                emit("bootstrap"),
                _topology_bootstrap(configuration),
            )
        )
        try:
            native_topology = RuntimeStateTopologySummary(
                **native_topology_document
            )
        except TypeError as error:
            raise SmokeError(
                "runtime-state topology cannot be bound for acquisition"
            ) from error
        observed = _observe_two_requests(
            configuration,
            prepared,
            preflight,
            native_topology,
            emit,
        )
        capture.wait()
    except BaseException:
        capture.abort()
        raise
    retained = _verify_external_runtime(configuration)
    if retained != runtime:
        raise SmokeError("external runtime identity changed during the smoke")
    emit("correlate")
    correlation, results, correlation_key = _offline_correlate(prepared)
    _verify_content_privacy(prepared, observed.prompts, observed.responses)
    emit("oracle-evaluate")
    evaluation = _evaluate_private_oracle(
        results,
        observed.truth,
        correlation_key,
    )
    writer = _verify_writer_receipt(prepared)
    passed = (
        observed.observation.get("status") == "COMPLETE"
        and writer["safe_failure_count"] == 0
        and len(results) == 2
        and correlation.get("certified_count", 0)
        + correlation.get("ranked_count", 0)
        == 2
        and correlation.get("ambiguous_count") == 0
        and correlation.get("incomplete_count") == 0
        and evaluation
        == {
            "request_count": 2,
            "certified_count": 2,
            "ranked_count": 0,
            "correct_attribution_count": 2,
            "false_attribution_count": 0,
            "ambiguous_count": 0,
            "incomplete_count": 0,
        }
    )
    if not passed:
        raise SmokeError("two-request Log-linked attribution gate failed")
    summary = {
        "schema_version": "1.0",
        "status": "PASS",
        "run_root": str(prepared.run_root),
        "runtime_pid": retained["pid"],
        "runtime_start_time_ticks": retained["start_time_ticks"],
        "request_count": 2,
        "correct_attribution_count": 2,
        "false_attribution_count": 0,
        "ambiguous_count": 0,
        "incomplete_count": 0,
        "captured_unit_count": observed.observation.get(
            "captured_unit_count"
        ),
        "external_runtime_retained": True,
        "prompt_or_response_logged": False,
        **writer,
    }
    _write_new(
        prepared.run_root / "two-request-smoke-summary.json",
        _canonical_json(summary) + b"\n",
    )
    return summary


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be an integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded two-request Log-linked smoke"
    )
    parser.add_argument("--app-source", type=Path, required=True)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--pid", type=_positive_int, required=True)
    parser.add_argument("--start-time-ticks", type=_positive_int, required=True)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--application-log", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--native-observer", type=Path, required=True)
    parser.add_argument("--bpf-object", type=Path, required=True)
    parser.add_argument(
        "--model-registration-receipt",
        type=Path,
        default=SmokeConfiguration.model_registration_receipt,
    )
    return parser


def main() -> None:
    arguments = _argument_parser().parse_args()
    os.umask(0o077)
    configuration = SmokeConfiguration(
        app_source=arguments.app_source,
        unit=arguments.unit,
        pid=arguments.pid,
        start_time_ticks=arguments.start_time_ticks,
        expected_executable_sha256=arguments.expected_executable_sha256,
        application_log=arguments.application_log,
        launch_receipt=arguments.launch_receipt,
        native_observer=arguments.native_observer,
        bpf_object=arguments.bpf_object,
        model_registration_receipt=arguments.model_registration_receipt,
    )
    summary = run_smoke(configuration)
    print(
        "LOG_LINKED_TWO_REQUEST_SMOKE_PASS="
        + _canonical_json(summary).decode("ascii"),
        flush=True,
    )


if __name__ == "__main__":
    main()
