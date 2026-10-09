"""Strict configuration contracts shared by non-UI runtime components."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence
from urllib.parse import urlsplit

import yaml

from laris_onpremise.core.records import canonical_json


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_PATH_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_REQUEST_REFERENCE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]{0,127})-T([1-9][0-9]*)$")
_SERVER_FIELDS = frozenset({"schema_version", "server_id", "endpoint", "expected_runtime"})
_ENDPOINT_FIELDS = frozenset({"base_url", "health_path"})
_RUNTIME_FIELDS_10 = frozenset(
    {
        "hosting_application",
        "runtime_library",
        "probe_symbol",
        "slot_count",
        "model_name",
    }
)
_RUNTIME_FIELDS_11 = _RUNTIME_FIELDS_10 | {"n_ctx"}
_SCENARIO_30_BASE_FIELDS = frozenset(
    {
        "schema_version",
        "scenario_id",
        "display_name",
        "execution_mode",
        "runner_profile",
        "server_profile",
        "lifecycle",
        "context",
        "actors",
        "schedule",
        "execution",
    }
)
_SCENARIO_30_RUNTIME_FIELDS = _SCENARIO_30_BASE_FIELDS | {"runtime"}
_SCENARIO_31_BASE_FIELDS = _SCENARIO_30_BASE_FIELDS | {"run_name"}
_SCENARIO_31_RUNTIME_FIELDS = _SCENARIO_31_BASE_FIELDS | {"runtime"}
_SCENARIO_RUNTIME_REQUIRED_FIELDS = frozenset(
    {"required_runtime_state_profile_id", "runtime_slots"}
)
_SCENARIO_RUNTIME_OPTIONAL_FIELDS = frozenset(
    {"default_model_profile_id", "allowed_model_profile_ids"}
)
_LIFECYCLE_FIELDS = frozenset({"runtime_policy", "finalization"})
_CONTEXT_FIELDS = frozenset({"mode", "mechanism"})
_ACTOR_FIELDS = frozenset(
    {"actor_label", "credential_ref", "source_device_ref", "turns"}
)
_TURN_FIELDS = frozenset({"turn_index", "prompt"})
_WAVE_FIELDS = frozenset({"wave_id", "submit_mode", "requests", "expected"})
_EXPECTED_FIELDS = frozenset({"active", "queued_minimum"})
_EXECUTION_FIELDS = frozenset(
    {
        "n_predict",
        "request_timeout_seconds",
        "inter_wave_delay_ms",
        "repetition_count",
        "seed",
    }
)


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is unsafe or ambiguous."""


def _mapping(name: str, value: object, fields: frozenset[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ConfigurationError(f"{name} field set is invalid")
    return value


def _sequence(name: str, value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ConfigurationError(f"{name} must be a non-empty array")
    return value


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{name} must be a non-empty string")
    return value


def _identifier(name: str, value: object) -> str:
    text = _text(name, value)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ConfigurationError(f"{name} is invalid")
    return text


def _path_component(name: str, value: object) -> str:
    text = _text(name, value)
    if _PATH_COMPONENT.fullmatch(text) is None or text in {".", ".."}:
        raise ConfigurationError(f"{name} is invalid")
    return text


def _positive_integer(name: str, value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > maximum:
        raise ConfigurationError(f"{name} must be between 1 and {maximum}")
    return value


def _nonnegative_integer(name: str, value: object, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
        raise ConfigurationError(f"{name} must be between 0 and {maximum}")
    return value


class _StrictSafeLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _StrictSafeLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigurationError("YAML object keys must be strings")
        if key in result:
            raise ConfigurationError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


def load_yaml_bytes(raw: bytes, *, maximum_bytes: int = 1024 * 1024) -> dict[str, object]:
    if not isinstance(raw, bytes):
        raise ConfigurationError("configuration payload must be bytes")
    if len(raw) > maximum_bytes:
        raise ConfigurationError("configuration file exceeds size limit")
    try:
        value = yaml.load(raw.decode("utf-8"), Loader=_StrictSafeLoader)
    except ConfigurationError:
        raise
    except (UnicodeError, yaml.YAMLError) as error:
        raise ConfigurationError("configuration YAML is invalid") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConfigurationError("configuration root must be an object")
    return value


def load_yaml_mapping(path: Path, *, maximum_bytes: int = 1024 * 1024) -> dict[str, object]:
    if not isinstance(path, Path):
        raise ConfigurationError("configuration path must be a Path")
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ConfigurationError("configuration file is unavailable") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ConfigurationError("configuration file must be a regular non-symlink file")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ConfigurationError("configuration file cannot be read") from error
    return load_yaml_bytes(raw, maximum_bytes=maximum_bytes)


@dataclass(frozen=True)
class ServerProfile:
    schema_version: str
    server_id: str
    base_url: str
    health_path: str
    hosting_application: str
    runtime_library: str
    probe_symbol: str
    slot_count: int
    model_name: str
    n_ctx: int | None = None

    @classmethod
    def from_mapping(cls, value: object) -> "ServerProfile":
        root = _mapping("server profile", value, _SERVER_FIELDS)
        schema_version = root["schema_version"]
        if schema_version not in {"1.0", "1.1"}:
            raise ConfigurationError("server profile schema_version must be 1.0 or 1.1")
        endpoint = _mapping("endpoint", root["endpoint"], _ENDPOINT_FIELDS)
        runtime = _mapping(
            "expected_runtime",
            root["expected_runtime"],
            _RUNTIME_FIELDS_10 if schema_version == "1.0" else _RUNTIME_FIELDS_11,
        )
        base_url = _text("base_url", endpoint["base_url"])
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ConfigurationError("base_url must be an absolute HTTP URL")
        if parsed.username is not None or parsed.password is not None:
            raise ConfigurationError("base_url cannot contain credentials")
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ConfigurationError("base_url cannot contain path, query or fragment")
        health_path = _text("health_path", endpoint["health_path"])
        if not health_path.startswith("/") or "?" in health_path or "#" in health_path:
            raise ConfigurationError("health_path is invalid")
        return cls(
            schema_version=schema_version,
            server_id=_identifier("server_id", root["server_id"]),
            base_url=base_url.rstrip("/"),
            health_path=health_path,
            hosting_application=_identifier(
                "hosting_application", runtime["hosting_application"]
            ),
            runtime_library=_text("runtime_library", runtime["runtime_library"]),
            probe_symbol=_identifier("probe_symbol", runtime["probe_symbol"]),
            slot_count=_positive_integer("slot_count", runtime["slot_count"], maximum=1024),
            model_name=_text("model_name", runtime["model_name"]),
            n_ctx=(
                None
                if schema_version == "1.0"
                else _positive_integer("n_ctx", runtime["n_ctx"], maximum=2**31 - 1)
            ),
        )

    @classmethod
    def load(cls, path: Path) -> "ServerProfile":
        return cls.from_mapping(load_yaml_mapping(path))

    def to_mapping(self) -> dict[str, object]:
        runtime: dict[str, object] = {
            "hosting_application": self.hosting_application,
            "runtime_library": self.runtime_library,
            "probe_symbol": self.probe_symbol,
            "slot_count": self.slot_count,
            "model_name": self.model_name,
        }
        if self.schema_version == "1.1":
            runtime["n_ctx"] = self.n_ctx
        return {
            "schema_version": self.schema_version,
            "server_id": self.server_id,
            "endpoint": {"base_url": self.base_url, "health_path": self.health_path},
            "expected_runtime": runtime,
        }


@dataclass(frozen=True)
class ActorTurn:
    turn_index: int
    prompt: str

    def to_mapping(self) -> dict[str, object]:
        return {"turn_index": self.turn_index, "prompt": self.prompt}


@dataclass(frozen=True)
class LiveActor:
    actor_label: str
    credential_ref: str
    source_device_ref: str
    turns: tuple[ActorTurn, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "actor_label": self.actor_label,
            "credential_ref": self.credential_ref,
            "source_device_ref": self.source_device_ref,
            "turns": [turn.to_mapping() for turn in self.turns],
        }


@dataclass(frozen=True)
class ScheduleWave:
    wave_id: str
    submit_mode: str
    request_refs: tuple[str, ...]
    expected_active: int
    expected_queued_minimum: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "wave_id": self.wave_id,
            "submit_mode": self.submit_mode,
            "requests": list(self.request_refs),
            "expected": {
                "active": self.expected_active,
                "queued_minimum": self.expected_queued_minimum,
            },
        }


@dataclass(frozen=True)
class LiveScenario:
    schema_version: str
    scenario_id: str
    run_name: str
    display_name: str
    execution_mode: str
    runner_profile: str
    server_profile: str
    runtime_policy: str
    finalization: str
    context_mode: str
    context_mechanism: str
    required_runtime_state_profile_id: str | None
    default_model_profile_id: str | None
    allowed_model_profile_ids: tuple[str, ...]
    runtime_slots: int | None
    actors: tuple[LiveActor, ...]
    schedule: tuple[ScheduleWave, ...]
    n_predict: int
    request_timeout_seconds: int
    inter_wave_delay_ms: int
    repetition_count: int
    seed: int

    @classmethod
    def from_mapping(cls, value: object) -> "LiveScenario":
        if not isinstance(value, Mapping):
            raise ConfigurationError("scenario field set is invalid")
        schema_version = value.get("schema_version")
        if schema_version == "3.0":
            fields = (
                _SCENARIO_30_RUNTIME_FIELDS
                if "runtime" in value
                else _SCENARIO_30_BASE_FIELDS
            )
        elif schema_version == "3.1":
            fields = (
                _SCENARIO_31_RUNTIME_FIELDS
                if "runtime" in value
                else _SCENARIO_31_BASE_FIELDS
            )
        else:
            raise ConfigurationError("scenario schema_version must be 3.0 or 3.1")
        root = _mapping("scenario", value, fields)
        scenario_id = _identifier("scenario_id", root["scenario_id"])
        run_name = (
            _path_component("derived run_name", scenario_id)
            if schema_version == "3.0"
            else _path_component("run_name", root["run_name"])
        )
        if root["execution_mode"] != "live" or root["runner_profile"] != "live_http":
            raise ConfigurationError("live scenario mode and runner_profile are required")
        lifecycle = _mapping(
            "lifecycle",
            root["lifecycle"],
            _LIFECYCLE_FIELDS,
        )
        if lifecycle["runtime_policy"] != "fresh_per_run":
            raise ConfigurationError("runtime_policy must be fresh_per_run")
        if lifecycle["finalization"] != "automatic":
            raise ConfigurationError("finalization must be automatic")
        context = _mapping("context", root["context"], _CONTEXT_FIELDS)
        context_mode = _text("context mode", context["mode"])
        if context_mode not in {"cumulative_context", "turn_only"}:
            raise ConfigurationError("context mode is unsupported")
        context_mechanism = _text("context mechanism", context["mechanism"])
        if context_mechanism not in {"full_context_recompute", "application_session"}:
            raise ConfigurationError("context mechanism is unsupported")

        required_runtime_state_profile_id: str | None = None
        default_model_profile_id: str | None = None
        allowed_model_profile_ids: tuple[str, ...] = ()
        runtime_slots: int | None = None
        if "runtime" in root:
            runtime_value = root["runtime"]
            if not isinstance(runtime_value, Mapping):
                raise ConfigurationError("scenario runtime field set is invalid")
            runtime_fields = set(runtime_value)
            if (
                not _SCENARIO_RUNTIME_REQUIRED_FIELDS.issubset(runtime_fields)
                or not runtime_fields.issubset(
                    _SCENARIO_RUNTIME_REQUIRED_FIELDS
                    | _SCENARIO_RUNTIME_OPTIONAL_FIELDS
                )
            ):
                raise ConfigurationError("scenario runtime field set is invalid")
            runtime = runtime_value
            required_runtime_state_profile_id = _identifier(
                "required_runtime_state_profile_id",
                runtime["required_runtime_state_profile_id"],
            )
            allowed_value = runtime.get("allowed_model_profile_ids", ())
            if isinstance(allowed_value, (str, bytes)) or not isinstance(
                allowed_value, Sequence
            ):
                raise ConfigurationError("allowed_model_profile_ids must be an array")
            allowed_model_profile_ids = tuple(
                _identifier("allowed model_profile_id", item)
                for item in allowed_value
            )
            if len(set(allowed_model_profile_ids)) != len(allowed_model_profile_ids):
                raise ConfigurationError("allowed_model_profile_ids must be unique")
            if "default_model_profile_id" in runtime:
                default_model_profile_id = _identifier(
                    "default_model_profile_id",
                    runtime["default_model_profile_id"],
                )
                if (
                    allowed_model_profile_ids
                    and allowed_model_profile_ids.count(default_model_profile_id) != 1
                ):
                    raise ConfigurationError(
                        "default_model_profile_id must occur once in allowed_model_profile_ids"
                    )
            runtime_slots = _positive_integer(
                "runtime_slots", runtime["runtime_slots"], maximum=1024
            )

        actors: list[LiveActor] = []
        actor_labels: set[str] = set()
        credential_refs: set[str] = set()
        expected_requests: set[str] = set()
        for actor_value in _sequence("actors", root["actors"]):
            actor = _mapping("actor", actor_value, _ACTOR_FIELDS)
            label = _identifier("actor_label", actor["actor_label"])
            credential_ref = _identifier("credential_ref", actor["credential_ref"])
            if label in actor_labels:
                raise ConfigurationError("actor_label must be unique")
            if credential_ref in credential_refs:
                raise ConfigurationError("credential_ref must be unique")
            actor_labels.add(label)
            credential_refs.add(credential_ref)
            turns: list[ActorTurn] = []
            for expected_index, turn_value in enumerate(
                _sequence("turns", actor["turns"]), start=1
            ):
                turn = _mapping("turn", turn_value, _TURN_FIELDS)
                index = _positive_integer("turn_index", turn["turn_index"], maximum=1000)
                if index != expected_index:
                    raise ConfigurationError("actor turn_index must be continuous from 1")
                prompt = _text("prompt", turn["prompt"])
                if len(prompt.encode("utf-8")) > 64 * 1024:
                    raise ConfigurationError("prompt exceeds size limit")
                turns.append(ActorTurn(index, prompt))
                expected_requests.add(f"{label}-T{index}")
            actors.append(
                LiveActor(
                    actor_label=label,
                    credential_ref=credential_ref,
                    source_device_ref=_identifier(
                        "source_device_ref", actor["source_device_ref"]
                    ),
                    turns=tuple(turns),
                )
            )

        waves: list[ScheduleWave] = []
        observed_requests: list[str] = []
        last_turn = {label: 0 for label in actor_labels}
        for wave_value in _sequence("schedule", root["schedule"]):
            wave = _mapping("schedule wave", wave_value, _WAVE_FIELDS)
            if wave["submit_mode"] != "concurrent":
                raise ConfigurationError("submit_mode must be concurrent")
            request_refs = tuple(
                _text("request reference", item)
                for item in _sequence("wave requests", wave["requests"])
            )
            wave_actors: set[str] = set()
            for request_ref in request_refs:
                match = _REQUEST_REFERENCE.fullmatch(request_ref)
                if match is None or request_ref not in expected_requests:
                    raise ConfigurationError("schedule request is unknown")
                label, raw_turn = match.groups()
                turn_index = int(raw_turn)
                if label in wave_actors:
                    raise ConfigurationError("one actor cannot submit two turns in one wave")
                if turn_index != last_turn[label] + 1:
                    raise ConfigurationError("actor turns must be scheduled in order")
                wave_actors.add(label)
                last_turn[label] = turn_index
                observed_requests.append(request_ref)
            expected = _mapping("wave expected", wave["expected"], _EXPECTED_FIELDS)
            active = _positive_integer("expected active", expected["active"], maximum=1024)
            queued = _nonnegative_integer(
                "expected queued_minimum", expected["queued_minimum"], maximum=1024
            )
            if active + queued > len(request_refs):
                raise ConfigurationError("wave expectation exceeds request count")
            waves.append(
                ScheduleWave(
                    wave_id=_identifier("wave_id", wave["wave_id"]),
                    submit_mode="concurrent",
                    request_refs=request_refs,
                    expected_active=active,
                    expected_queued_minimum=queued,
                )
            )
        if len(set(observed_requests)) != len(observed_requests):
            raise ConfigurationError("schedule requests must be unique")
        if set(observed_requests) != expected_requests:
            raise ConfigurationError("schedule must cover every actor turn")

        execution = _mapping("execution", root["execution"], _EXECUTION_FIELDS)
        seed = execution["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ConfigurationError("seed must be a non-negative integer")
        return cls(
            schema_version=schema_version,
            scenario_id=scenario_id,
            run_name=run_name,
            display_name=_text("display_name", root["display_name"]),
            execution_mode="live",
            runner_profile="live_http",
            server_profile=_identifier("server_profile", root["server_profile"]),
            runtime_policy="fresh_per_run",
            finalization="automatic",
            context_mode=context_mode,
            context_mechanism=context_mechanism,
            required_runtime_state_profile_id=required_runtime_state_profile_id,
            default_model_profile_id=default_model_profile_id,
            allowed_model_profile_ids=allowed_model_profile_ids,
            runtime_slots=runtime_slots,
            actors=tuple(actors),
            schedule=tuple(waves),
            n_predict=_positive_integer("n_predict", execution["n_predict"], maximum=4096),
            request_timeout_seconds=_positive_integer(
                "request_timeout_seconds",
                execution["request_timeout_seconds"],
                maximum=3600,
            ),
            inter_wave_delay_ms=_nonnegative_integer(
                "inter_wave_delay_ms", execution["inter_wave_delay_ms"], maximum=60000
            ),
            repetition_count=_positive_integer(
                "repetition_count", execution["repetition_count"], maximum=100
            ),
            seed=seed,
        )

    @classmethod
    def load(cls, path: Path) -> "LiveScenario":
        return cls.from_mapping(load_yaml_mapping(path))

    @property
    def actor_count(self) -> int:
        return len(self.actors)

    @property
    def turns_per_actor(self) -> int:
        counts = {len(actor.turns) for actor in self.actors}
        if len(counts) != 1:
            raise ConfigurationError("all actors must have the same turn count")
        return next(iter(counts))

    @property
    def request_count(self) -> int:
        return sum(len(wave.request_refs) for wave in self.schedule)

    @property
    def wave_count(self) -> int:
        return len(self.schedule)

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "display_name": self.display_name,
            "execution_mode": self.execution_mode,
            "runner_profile": self.runner_profile,
            "server_profile": self.server_profile,
            "lifecycle": {
                "runtime_policy": self.runtime_policy,
                "finalization": self.finalization,
            },
            "context": {"mode": self.context_mode, "mechanism": self.context_mechanism},
            "actors": [actor.to_mapping() for actor in self.actors],
            "schedule": [wave.to_mapping() for wave in self.schedule],
            "execution": {
                "n_predict": self.n_predict,
                "request_timeout_seconds": self.request_timeout_seconds,
                "inter_wave_delay_ms": self.inter_wave_delay_ms,
                "repetition_count": self.repetition_count,
                "seed": self.seed,
            },
        }
        if self.schema_version == "3.1":
            mapping["run_name"] = self.run_name
        if self.runtime_slots is not None:
            runtime: dict[str, object] = {
                "required_runtime_state_profile_id": (
                    self.required_runtime_state_profile_id
                ),
                "runtime_slots": self.runtime_slots,
            }
            if self.default_model_profile_id is not None:
                runtime["default_model_profile_id"] = self.default_model_profile_id
            if self.allowed_model_profile_ids:
                runtime["allowed_model_profile_ids"] = list(
                    self.allowed_model_profile_ids
                )
            mapping["runtime"] = runtime
        return mapping


@dataclass(frozen=True)
class FrozenConfiguration:
    run_directory: Path
    server_profile_digest: str
    scenario_digest: str
    selected_model_profile_digest: str | None = None


@dataclass(frozen=True)
class ResolvedLiveConfiguration:
    scenario: LiveScenario
    server: ServerProfile
    allowed_models: tuple[object, ...] = ()

    @property
    def dashboard_model_name(self) -> str:
        return self.server.model_name


class LiveConfigurationCatalog:
    def __init__(
        self,
        servers: Mapping[str, ServerProfile],
        scenarios: Mapping[str, ResolvedLiveConfiguration],
    ) -> None:
        self._servers = MappingProxyType(dict(servers))
        self._scenarios = MappingProxyType(dict(scenarios))

    @classmethod
    def load(
        cls,
        server_directory: Path,
        scenario_directory: Path,
        model_registry: object | None = None,
    ) -> "LiveConfigurationCatalog":
        if not server_directory.is_dir() or server_directory.is_symlink():
            raise ConfigurationError("server profile directory is invalid")
        if not scenario_directory.is_dir() or scenario_directory.is_symlink():
            raise ConfigurationError("scenario directory is invalid")
        servers: dict[str, ServerProfile] = {}
        for path in sorted(server_directory.glob("*.yaml"), key=lambda item: item.name):
            profile = ServerProfile.load(path)
            if profile.server_id in servers:
                raise ConfigurationError("duplicate server_id")
            servers[profile.server_id] = profile
        scenarios: dict[str, ResolvedLiveConfiguration] = {}
        for path in sorted(scenario_directory.glob("*.yaml"), key=lambda item: item.name):
            scenario = LiveScenario.load(path)
            if scenario.scenario_id in scenarios:
                raise ConfigurationError("duplicate live scenario_id")
            try:
                server = servers[scenario.server_profile]
            except KeyError as error:
                raise ConfigurationError("live scenario references an unknown server") from error
            allowed_models: tuple[object, ...] = ()
            if scenario.runtime_slots is not None:
                if scenario.runtime_slots != server.slot_count:
                    raise ConfigurationError(
                        "scenario runtime_slots must match server slot_count"
                    )
                if model_registry is None or not hasattr(model_registry, "compatible"):
                    raise ConfigurationError("approved model registry is required")
                try:
                    allowed_models = tuple(
                        model_registry.compatible(
                            scenario.required_runtime_state_profile_id,
                            scenario.allowed_model_profile_ids,
                        )
                    )
                except ValueError as error:
                    raise ConfigurationError(
                        "scenario references an unapproved model_profile_id"
                    ) from error
                if (
                    scenario.default_model_profile_id is not None
                    and scenario.default_model_profile_id
                    not in {
                        getattr(model, "model_profile_id", None)
                        for model in allowed_models
                    }
                ):
                    raise ConfigurationError(
                        "default_model_profile_id is incompatible with the scenario"
                    )
            scenarios[scenario.scenario_id] = ResolvedLiveConfiguration(
                scenario=scenario,
                server=server,
                allowed_models=allowed_models,
            )
        return cls(servers, scenarios)

    def list_servers(self) -> tuple[ServerProfile, ...]:
        return tuple(self._servers[key] for key in sorted(self._servers))

    def list_scenarios(self) -> tuple[ResolvedLiveConfiguration, ...]:
        return tuple(self._scenarios[key] for key in sorted(self._scenarios))

    def get_server(self, server_id: str) -> ServerProfile:
        try:
            return self._servers[server_id]
        except KeyError as error:
            raise ConfigurationError("unknown server_id") from error

    def get_scenario(self, scenario_id: str) -> ResolvedLiveConfiguration:
        try:
            return self._scenarios[scenario_id]
        except KeyError as error:
            raise ConfigurationError("unknown live scenario_id") from error


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise ConfigurationError(f"configuration artifact already exists: {path.name}") from error


def freeze_configuration(
    run_directory: Path,
    server_profile: ServerProfile,
    scenario: LiveScenario,
    selected_model: object | None = None,
    *,
    preallocated: bool = False,
) -> FrozenConfiguration:
    if scenario.server_profile != server_profile.server_id:
        raise ConfigurationError("scenario and server profile do not match")
    if preallocated:
        try:
            metadata = run_directory.lstat()
            children = tuple(run_directory.iterdir())
        except OSError as error:
            raise ConfigurationError("allocated run directory is unavailable") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ConfigurationError("allocated run directory is invalid")
        for child in children:
            child_metadata = child.lstat()
            if (
                child.name not in {"bundle", "evidence", "rehost"}
                or stat.S_ISLNK(child_metadata.st_mode)
                or not stat.S_ISDIR(child_metadata.st_mode)
            ):
                raise ConfigurationError("allocated run directory is not empty")
    else:
        try:
            run_directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise ConfigurationError("run directory already exists") from error
        except OSError as error:
            raise ConfigurationError("run directory cannot be created") from error
    server_bytes = canonical_json(server_profile.to_mapping()) + b"\n"
    scenario_bytes = canonical_json(scenario.to_mapping()) + b"\n"
    server_digest = "sha256:" + hashlib.sha256(server_bytes).hexdigest()
    scenario_digest = "sha256:" + hashlib.sha256(scenario_bytes).hexdigest()
    selected_model_digest: str | None = None
    _write_exclusive(run_directory / "frozen_server_profile.json", server_bytes)
    _write_exclusive(run_directory / "frozen_scenario.json", scenario_bytes)
    if selected_model is not None:
        try:
            selected_mapping = selected_model.to_mapping()
        except AttributeError as error:
            raise ConfigurationError("selected model is invalid") from error
        if (
            not isinstance(selected_mapping, Mapping)
            or set(selected_mapping)
            != {"model_profile_id", "model_name", "model_digest"}
            or selected_mapping["model_profile_id"] is None
            or selected_mapping["model_digest"] is None
        ):
            raise ConfigurationError("selected model identity is incomplete")
        selected_bytes = canonical_json(dict(selected_mapping)) + b"\n"
        selected_model_digest = (
            "sha256:" + hashlib.sha256(selected_bytes).hexdigest()
        )
        _write_exclusive(
            run_directory / "selected_model_profile.json",
            selected_bytes,
        )
    seal: dict[str, object] = {
        "schema_version": "1.0" if selected_model_digest is None else "2.0",
        "server_profile_digest": server_digest,
        "scenario_digest": scenario_digest,
    }
    if selected_model_digest is not None:
        seal["selected_model_profile_digest"] = selected_model_digest
    _write_exclusive(
        run_directory / "configuration_seal.json",
        canonical_json(seal) + b"\n",
    )
    return FrozenConfiguration(
        run_directory,
        server_digest,
        scenario_digest,
        selected_model_digest,
    )
