#!/usr/bin/env python3
from __future__ import annotations

import argparse
import stat
import sys
import threading
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from laris_onpremise.attribution.proxy.binding_ledger import BindingLedger
from laris_onpremise.kv_state.acquisition.acquirer import SelectiveRuntimeStateAcquirer
from laris_onpremise.attribution.proxy.coordinator import AcquisitionCoordinator
from laris_onpremise.kv_state.evidence.evidence_root import (
    validate_prepared_acquisition_evidence_root,
)
from laris_onpremise.kv_state.evidence.store import RuntimeStatePayloadStore
from laris_onpremise.attribution.proxy.lease_controller import SequenceLeaseController
from laris_onpremise.core.records import IdentityKey, IdentityKeyRegistry
from laris_onpremise.kv_state.observation.runtime_state_profile import (
    load_runtime_state_observer_profile,
)
from laris_onpremise.kv_state.observation.runtime_state_client import RuntimeStateClient
from laris_onpremise.kv_state.observation.runtime_state_repository import (
    RuntimeStateClientObservationSource,
)
from laris_onpremise.runtime.llama_server_client import LlamaServerClient
from laris_onpremise.attribution.proxy.core import LarisProxyCore
from laris_onpremise.attribution.proxy.control import (
    ProxyControlProtocol,
    UnixProxyControlServer,
)
from laris_onpremise.attribution.proxy.http import LarisProxyServer
from laris_onpremise.attribution.proxy.health import (
    FileObserverStatusProvider,
    ProxyHealthProvider,
    RuntimeTopologyFileProvider,
)


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8090)
    parser.add_argument("--upstream-url", required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--runtime-state-socket", type=Path, required=True)
    parser.add_argument("--runtime-state-profile", type=Path, required=True)
    parser.add_argument(
        "--runtime-state-timeout-seconds",
        type=float,
        default=45.0,
    )
    parser.add_argument("--acquisition-evidence-root", type=Path, required=True)
    parser.add_argument("--runtime-pid", type=int, required=True)
    parser.add_argument("--runtime-start-time-ticks", type=int, required=True)
    parser.add_argument("--identity-key-file", type=Path, required=True)
    parser.add_argument("--identity-key-id", required=True)
    parser.add_argument("--ledger-mac-key-file", type=Path, required=True)
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--controller-instance-id", required=True)
    parser.add_argument("--runtime-instance-id", required=True)
    parser.add_argument("--context-ref", default="context-main")
    parser.add_argument("--slot-count", type=int, default=4)
    parser.add_argument("--queue-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--slot-idle-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--control-socket", type=Path)
    parser.add_argument("--control-evidence-dir", type=Path)
    parser.add_argument("--run-nonce")
    parser.add_argument("--health-server-id")
    parser.add_argument("--health-hosting-application")
    parser.add_argument("--health-model-name")
    parser.add_argument("--health-runtime-library")
    parser.add_argument("--health-probe-symbol")
    parser.add_argument("--health-observer-status-file", type=Path)
    parser.add_argument("--health-runtime-topology-file", type=Path)
    parser.add_argument("--health-runtime-state-service-pid", type=int)
    parser.add_argument(
        "--health-runtime-state-service-start-time-ticks", type=int
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    return add_arguments(
        argparse.ArgumentParser(
            description="Run the LARIS Sequence Lease Proxy."
        )
    )


def run(args: argparse.Namespace) -> int:
    control_values = (
        args.control_socket,
        args.control_evidence_dir,
        args.run_nonce,
    )
    if any(value is not None for value in control_values) and not all(
        value is not None for value in control_values
    ):
        raise ValueError(
            "--control-socket, --control-evidence-dir and --run-nonce "
            "must be provided together"
        )
    health_values = (
        args.health_server_id,
        args.health_hosting_application,
        args.health_model_name,
        args.health_runtime_library,
        args.health_probe_symbol,
        args.health_observer_status_file,
        args.health_runtime_topology_file,
        args.health_runtime_state_service_pid,
        args.health_runtime_state_service_start_time_ticks,
    )
    if any(value is not None for value in health_values) and not all(
        value is not None for value in health_values
    ):
        raise ValueError("all dashboard health options must be provided together")
    identity_key = args.identity_key_file.read_bytes()
    ledger_mac_key = args.ledger_mac_key_file.read_bytes()
    identity_registry = IdentityKeyRegistry(
        (IdentityKey(args.identity_key_id, identity_key),)
    )
    runtime_profile = load_runtime_state_observer_profile(
        args.runtime_state_profile
    )
    validate_prepared_acquisition_evidence_root(
        args.acquisition_evidence_root,
        runtime_state_profile=args.runtime_state_profile,
    )
    socket_path = args.runtime_state_socket
    if not socket_path.is_absolute() or socket_path.is_symlink():
        raise ValueError("runtime-state socket must be an absolute non-symlink path")
    try:
        socket_mode = socket_path.stat().st_mode
    except OSError as error:
        raise ValueError("runtime-state socket is not available") from error
    if not stat.S_ISSOCK(socket_mode):
        raise ValueError("runtime-state socket path is not a Unix socket")
    observation_repository = RuntimeStateClientObservationSource(
        client=RuntimeStateClient(
            socket_path,
            timeout_seconds=args.runtime_state_timeout_seconds,
        ),
        profile=runtime_profile,
    )
    payload_store = RuntimeStatePayloadStore(args.acquisition_evidence_root)
    with BindingLedger(args.ledger, mac_key=ledger_mac_key) as ledger:
        controller = SequenceLeaseController(
            ledger=ledger,
            controller_instance_id=args.controller_instance_id,
            runtime_instance_id=args.runtime_instance_id,
            context_ref=args.context_ref,
            context_ref_source="configured_single_context",
            slot_count=args.slot_count,
        )
        controller.recover_controller()
        upstream = LlamaServerClient(args.upstream_url)
        acquirer = SelectiveRuntimeStateAcquirer(
            observation_source=observation_repository,
            store=payload_store,
        )
        acquisition_coordinator = AcquisitionCoordinator(
            ledger=ledger,
            acquirer=acquirer,
            runtime_pid=args.runtime_pid,
            runtime_start_time_ticks=args.runtime_start_time_ticks,
        )
        core = LarisProxyCore(
            controller=controller,
            upstream=upstream,
            acquisition_coordinator=acquisition_coordinator,
            identity_key_registry=identity_registry,
            identity_key_id=args.identity_key_id,
            tenant_id=args.tenant_id,
            queue_timeout_seconds=args.queue_timeout_seconds,
            slot_idle_timeout_seconds=args.slot_idle_timeout_seconds,
        )
        health_provider = None
        if all(value is not None for value in health_values):
            observer_status = FileObserverStatusProvider(
                args.health_observer_status_file
            )
            initial_observer = observer_status()
            if initial_observer.probe_symbol != args.health_probe_symbol:
                raise ValueError("observer status probe symbol mismatch")
            if (
                initial_observer.target_pid != args.runtime_pid
                or initial_observer.target_start_time_ticks
                != args.runtime_start_time_ticks
            ):
                raise ValueError("observer target does not match runtime identity")
            health_provider = ProxyHealthProvider(
                server_id=args.health_server_id,
                core=core,
                upstream=upstream,
                hosting_application=args.health_hosting_application,
                model_name=args.health_model_name,
                slot_count=args.slot_count,
                runtime_library=args.health_runtime_library,
                probe_symbol=args.health_probe_symbol,
                runtime_pid=initial_observer.target_pid,
                runtime_start_time_ticks=(
                    initial_observer.target_start_time_ticks
                ),
                observer_status=observer_status,
                runtime_state_service_pid=(
                    args.health_runtime_state_service_pid
                ),
                runtime_state_service_start_time_ticks=(
                    args.health_runtime_state_service_start_time_ticks
                ),
                runtime_state_socket=socket_path,
                ledger_path=args.ledger,
                runtime_state_payload_root=args.acquisition_evidence_root,
                runtime_topology=RuntimeTopologyFileProvider(
                    args.health_runtime_topology_file
                )(),
            )
        server = LarisProxyServer(
            (args.listen_host, args.listen_port),
            core=core,
            health_provider=health_provider,
        )
        try:
            if args.control_socket is None:
                server.serve_forever()
            else:
                protocol = ProxyControlProtocol(
                    core=core,
                    run_nonce=args.run_nonce,
                    evidence_dir=args.control_evidence_dir,
                )
                control_server = UnixProxyControlServer(
                    socket_path=args.control_socket,
                    protocol=protocol,
                )
                http_thread = threading.Thread(target=server.serve_forever)
                http_thread.start()
                try:
                    control_server.serve_until_shutdown()
                finally:
                    server.shutdown()
                    http_thread.join(timeout=10)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
