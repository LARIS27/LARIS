#!/usr/bin/env python3
"""Prepare and finalize an evidence-backed runtime-state qualification candidate.

The outer root orchestrator must stop/confirm the Agent IDLE, acquire the fixed
maintenance lock once, and retain the same inherited descriptor continuously
across ``run-live-probe``, ``finalize-live``, and controlled promotion.  This
tool never starts or stops the Agent; the orchestrator restarts it only after
promotion or rollback and after releasing the lock.
"""

from __future__ import annotations

import argparse
import errno
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import ModuleType


class QualificationCliError(ValueError):
    """The trusted root-only qualification CLI boundary is invalid."""


_MAINTENANCE_LOCK_DISPLAY = (
    "/run/laris-selective-maintenance/maintenance.lock"
)
_MAINTENANCE_LOCK = Path(_MAINTENANCE_LOCK_DISPLAY)


def _require_inherited_maintenance_lock(
    descriptor: int,
    *,
    os_api=os,
    fcntl_api=None,
) -> None:
    """Prove this process inherited the already-held fixed maintenance lock."""

    if (
        isinstance(descriptor, bool)
        or not isinstance(descriptor, int)
        or descriptor < 3
    ):
        raise QualificationCliError(
            "maintenance-lock-fd must be an inherited descriptor"
        )
    if os_api.name != "posix" or os_api.geteuid() != 0:
        raise QualificationCliError(
            "maintenance lock proof requires root on POSIX"
        )
    if fcntl_api is None:
        try:
            import fcntl as fcntl_api
        except ImportError as error:
            raise QualificationCliError(
                "POSIX flock support is unavailable"
            ) from error
    try:
        if not os_api.get_inheritable(descriptor):
            raise QualificationCliError(
                "maintenance-lock-fd was not inherited"
            )
        descriptor_metadata = os_api.fstat(descriptor)
        path_metadata = os_api.lstat(_MAINTENANCE_LOCK)
        parent_metadata = os_api.lstat(_MAINTENANCE_LOCK.parent)
        link_target = os_api.readlink(Path(f"/proc/self/fd/{descriptor}"))
    except QualificationCliError:
        raise
    except OSError as error:
        raise QualificationCliError(
            "maintenance lock identity cannot be verified"
        ) from error
    if (
        not stat.S_ISREG(descriptor_metadata.st_mode)
        or stat.S_IMODE(descriptor_metadata.st_mode) != 0o600
        or descriptor_metadata.st_uid != 0
        or descriptor_metadata.st_gid != 0
        or descriptor_metadata.st_nlink != 1
        or not stat.S_ISREG(path_metadata.st_mode)
        or stat.S_IMODE(path_metadata.st_mode) != 0o600
        or path_metadata.st_uid != 0
        or path_metadata.st_gid != 0
        or path_metadata.st_nlink != 1
        or (descriptor_metadata.st_dev, descriptor_metadata.st_ino)
        != (path_metadata.st_dev, path_metadata.st_ino)
        or not stat.S_ISDIR(parent_metadata.st_mode)
        or stat.S_IMODE(parent_metadata.st_mode) != 0o700
        or parent_metadata.st_uid != 0
        or parent_metadata.st_gid != 0
        or link_target != _MAINTENANCE_LOCK_DISPLAY
    ):
        raise QualificationCliError(
            "maintenance lock must be the fixed root:root 0600 file"
        )
    operation = fcntl_api.LOCK_EX | fcntl_api.LOCK_NB
    challenger = -1
    try:
        challenger = os_api.open(
            _MAINTENANCE_LOCK,
            os_api.O_RDONLY
            | getattr(os_api, "O_CLOEXEC", 0)
            | getattr(os_api, "O_NOFOLLOW", 0),
        )
        try:
            fcntl_api.flock(challenger, operation)
        except BlockingIOError as error:
            if error.errno not in {
                errno.EACCES,
                errno.EAGAIN,
                errno.EWOULDBLOCK,
            }:
                raise
        else:
            raise QualificationCliError(
                "parent orchestrator must already hold the maintenance lock"
            )
        try:
            fcntl_api.flock(descriptor, operation)
        except BlockingIOError as error:
            raise QualificationCliError(
                "maintenance-lock-fd does not own the held lock"
            ) from error
    except QualificationCliError:
        raise
    except OSError as error:
        raise QualificationCliError(
            "maintenance lock proof failed closed"
        ) from error
    finally:
        if challenger >= 0:
            os_api.close(challenger)


def _normalized_directory(path: Path, label: str) -> Path:
    if not isinstance(path, Path) or not path.is_absolute():
        raise QualificationCliError(f"{label} must be an absolute path")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise QualificationCliError(f"{label} is unavailable") from error
    if path != resolved or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(
        metadata.st_mode
    ):
        raise QualificationCliError(
            f"{label} must be a normalized non-symlink directory"
        )
    return resolved


def _load_api(checkout_src: Path) -> ModuleType:
    source = _normalized_directory(checkout_src, "checkout src")
    package = _normalized_directory(
        source / "laris_onpremise/deployment", "checkout deployment package"
    )
    if package.parent.parent != source:
        raise QualificationCliError("checkout deployment package binding is invalid")
    sys.path.insert(0, str(source))
    try:
        from laris_onpremise.deployment import runtime_state_qualification
    except ImportError as error:
        raise QualificationCliError(
            "verified checkout qualification API cannot be imported"
        ) from error
    return runtime_state_qualification


def _require_root() -> None:
    if os.name == "posix" and (
        not hasattr(os, "geteuid") or os.geteuid() != 0
    ):
        raise QualificationCliError("runtime-state qualification requires root")


def _add_static_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pending-profile", required=True, type=Path)
    parser.add_argument("--runtime-input-contract", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--runtime-expected-contract", required=True, type=Path)
    parser.add_argument("--abi-probe", required=True, type=Path)
    parser.add_argument("--llama-source", required=True, type=Path)


def _add_live_probe_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--app-src", required=True, type=Path)
    parser.add_argument("--app-manifest", required=True, type=Path)
    parser.add_argument("--expected-app-manifest-digest", required=True)
    parser.add_argument("--maintenance-lock-fd", required=True, type=int)
    parser.add_argument("--pending-profile", required=True, type=Path)
    parser.add_argument("--static-receipt", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--server-binary", required=True, type=Path)
    parser.add_argument("--libllama", required=True, type=Path)
    parser.add_argument("--qualification-runtime-root", type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--model-registration-receipt", type=Path)
    parser.add_argument("--model-profile-id")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-nonce", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--runtime-state-socket", required=True, type=Path)
    parser.add_argument("--runtime-state-frame-dir", required=True, type=Path)
    parser.add_argument(
        "--runtime-state-native-observer", required=True, type=Path
    )
    parser.add_argument("--runtime-state-bpf-object", required=True, type=Path)
    parser.add_argument(
        "--acquisition-evidence-root", required=True, type=Path
    )
    parser.add_argument("--native-observer-binary", required=True, type=Path)
    parser.add_argument("--bpf-object", required=True, type=Path)
    parser.add_argument("--probe-profile", required=True, type=Path)
    parser.add_argument("--abi-probe-binary", required=True, type=Path)
    parser.add_argument("--llama-header", required=True, type=Path)
    parser.add_argument("--compiler-id", required=True)
    parser.add_argument("--compiler-version", required=True)
    parser.add_argument("--build-command-json", required=True)
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog=(
            "The same parent-held /run/laris-selective-maintenance/"
            "maintenance.lock FD must span live probe, finalize, and "
            "promotion while the Agent remains stopped/IDLE."
        ),
    )
    parser.add_argument("--checkout-src", required=True, type=Path)
    commands = parser.add_subparsers(dest="phase", required=True)

    prepare = commands.add_parser(
        "prepare-static",
        allow_abbrev=False,
        help="emit only a live-probe-pending static receipt",
    )
    _add_static_inputs(prepare)
    prepare.add_argument("--qualified-toolchain-profile", required=True)
    prepare.add_argument("--output", required=True, type=Path)

    render = commands.add_parser(
        "render-live-probe-inputs",
        allow_abbrev=False,
        help="render inert receipt-bound Gate0 inputs without running it",
    )
    render.add_argument("--pending-profile", required=True, type=Path)
    render.add_argument("--static-receipt", required=True, type=Path)

    probe = commands.add_parser(
        "run-live-probe",
        allow_abbrev=False,
        help="run root-private Gate0 without publishing active qualification",
    )
    _add_live_probe_inputs(probe)

    finalize = commands.add_parser(
        "finalize-live",
        allow_abbrev=False,
        help="consume a genuine Gate0 native capture and render candidates",
    )
    _add_static_inputs(finalize)
    finalize.add_argument("--static-receipt", required=True, type=Path)
    finalize.add_argument("--gate-zero-result", required=True, type=Path)
    finalize.add_argument("--output-directory", required=True, type=Path)
    finalize.add_argument("--maintenance-lock-fd", required=True, type=int)
    return parser


def _require_absolute_arguments(arguments: argparse.Namespace) -> None:
    for name, value in vars(arguments).items():
        if isinstance(value, Path) and not value.is_absolute():
            raise QualificationCliError(
                f"{name.replace('_', '-')} must be an absolute path"
            )


def _run_live_probe(
    arguments: argparse.Namespace,
    *,
    api: ModuleType,
    process_runner=subprocess.run,
) -> Path:
    _require_inherited_maintenance_lock(arguments.maintenance_lock_fd)
    try:
        import json

        inputs = json.loads(
            api.render_live_probe_inputs(
                pending_profile_path=arguments.pending_profile,
                static_receipt_path=arguments.static_receipt,
            ).decode("ascii")
        )
    except (ImportError, UnicodeError, ValueError) as error:
        raise QualificationCliError(
            "receipt-bound selective Gate0 inputs are invalid"
        ) from error
    try:
        trusted_app_source = api.validate_live_probe_app_source(
            app_src=arguments.app_src,
            app_manifest_path=arguments.app_manifest,
            expected_manifest_digest=(
                arguments.expected_app_manifest_digest
            ),
        )
    except Exception as error:
        raise QualificationCliError(
            "candidate application source validation failed"
        ) from error
    app_source = _normalized_directory(trusted_app_source, "app src")
    app_package = _normalized_directory(
        app_source / "laris_onpremise", "app package"
    )
    selective_module = app_package / "experiments/selective_gate_zero.py"
    try:
        selective_metadata = selective_module.lstat()
    except OSError as error:
        raise QualificationCliError(
            "app selective Gate0 module is unavailable"
        ) from error
    if stat.S_ISLNK(selective_metadata.st_mode) or not stat.S_ISREG(
        selective_metadata.st_mode
    ):
        raise QualificationCliError(
            "app selective Gate0 module must be a regular non-symlink file"
        )
    qualification_runtime_root = getattr(
        arguments, "qualification_runtime_root", None
    )
    if qualification_runtime_root is None:
        expected_server = Path(str(inputs.get("runtime_binary_path")))
        expected_libllama = Path(str(inputs.get("mapped_libllama_path")))
    else:
        runtime_root = _normalized_directory(
            qualification_runtime_root, "qualification runtime root"
        )

        def physical_path(logical: object) -> Path:
            if not isinstance(logical, str):
                raise QualificationCliError(
                    "static qualification runtime path is invalid"
                )
            prefix = "/opt/laris/llama-runtime/"
            if not logical.startswith(prefix) or ".." in Path(logical).parts:
                raise QualificationCliError(
                    "static qualification runtime path is outside its root"
                )
            return runtime_root / logical.removeprefix(prefix)

        expected_server = physical_path(inputs.get("runtime_binary_path"))
        expected_libllama = physical_path(inputs.get("mapped_libllama_path"))
    expected_paths = {
        "server-binary": (arguments.server_binary, expected_server),
        "libllama": (arguments.libllama, expected_libllama),
    }
    qualified_model_path = inputs.get("qualified_model_path")
    model_registration_receipt = getattr(
        arguments, "model_registration_receipt", None
    )
    if qualified_model_path is None:
        if model_registration_receipt is None:
            raise QualificationCliError(
                "dynamic profile requires a model registration receipt"
            )
        model_profile_id = getattr(arguments, "model_profile_id", None)
        if not isinstance(model_profile_id, str) or not model_profile_id:
            raise QualificationCliError(
                "dynamic profile requires an explicit model profile ID"
            )
    else:
        expected_paths["model"] = (arguments.model, Path(qualified_model_path))
        model_profile_id = inputs["qualified_model_profile_id"]
        supplied_model_profile_id = getattr(arguments, "model_profile_id", None)
        if (
            supplied_model_profile_id is not None
            and supplied_model_profile_id != model_profile_id
        ):
            raise QualificationCliError(
                "model profile ID contradicts the static qualification receipt"
            )
    for label, (observed, expected) in expected_paths.items():
        if observed != expected:
            raise QualificationCliError(
                f"{label} contradicts the static qualification receipt"
            )
    offsets = inputs.get("runtime_state_offsets")
    if not isinstance(offsets, dict) or set(offsets) != {
        "apply",
        "buffer_base",
        "buffer_size",
        "constructor",
        "n_pos_per_embd",
        "process_ubatch",
        "tensor_data",
        "tensor_type",
        "view",
    }:
        raise QualificationCliError("receipt-bound runtime-state offsets are invalid")
    bootstrap = (
        "import runpy,sys;"
        "sys.path.insert(0,sys.argv.pop(1));"
        "runpy.run_module('laris_onpremise.experiments.selective_gate_zero',"
        "run_name='__main__')"
    )
    command = (
        sys.executable,
        "-I",
        "-B",
        "-c",
        bootstrap,
        str(app_source),
        "--source-dir",
        str(arguments.source_dir),
        "--server-binary",
        str(arguments.server_binary),
        "--libllama",
        str(arguments.libllama),
        "--model",
        str(arguments.model),
        "--model-profile-id",
        model_profile_id,
        "--output-dir",
        str(arguments.output_dir),
        "--run-id",
        arguments.run_id,
        "--run-nonce",
        arguments.run_nonce,
        "--config",
        str(arguments.config),
        "--runtime-state-socket",
        str(arguments.runtime_state_socket),
        "--runtime-state-frame-dir",
        str(arguments.runtime_state_frame_dir),
        "--runtime-state-profile",
        str(arguments.pending_profile),
        "--runtime-state-static-qualification-receipt",
        str(arguments.static_receipt),
        "--runtime-state-native-observer",
        str(arguments.runtime_state_native_observer),
        "--runtime-state-bpf-object",
        str(arguments.runtime_state_bpf_object),
        "--runtime-state-constructor-offset",
        str(offsets["constructor"]),
        "--runtime-state-n-pos-per-embd-offset",
        str(offsets["n_pos_per_embd"]),
        "--runtime-state-view-offset",
        str(offsets["view"]),
        "--runtime-state-buffer-base-offset",
        str(offsets["buffer_base"]),
        "--runtime-state-buffer-size-offset",
        str(offsets["buffer_size"]),
        "--runtime-state-apply-offset",
        str(offsets["apply"]),
        "--runtime-state-tensor-data-offset",
        str(offsets["tensor_data"]),
        "--runtime-state-tensor-type-offset",
        str(offsets["tensor_type"]),
        "--acquisition-evidence-root",
        str(arguments.acquisition_evidence_root),
        "--native-observer-binary",
        str(arguments.native_observer_binary),
        "--bpf-object",
        str(arguments.bpf_object),
        "--probe-profile",
        str(arguments.probe_profile),
        "--abi-probe-binary",
        str(arguments.abi_probe_binary),
        "--llama-header",
        str(arguments.llama_header),
        "--compiler-id",
        arguments.compiler_id,
        "--compiler-version",
        arguments.compiler_version,
        "--build-command-json",
        arguments.build_command_json,
        "--qualified-toolchain-profile",
        inputs["qualified_toolchain_profile"],
        "--port",
        str(arguments.port),
        "--timeout-seconds",
        str(arguments.timeout_seconds),
    )
    if model_registration_receipt is not None:
        command += (
            "--model-registration-receipt",
            str(model_registration_receipt),
        )
    if qualification_runtime_root is not None:
        command += (
            "--runtime-state-qualification-runtime-root",
            str(qualification_runtime_root),
        )
    environment = os.environ.copy()
    # The isolated Gate0 parent inserts app_source itself; its Python gate and
    # service children need the same explicit source boundary after exec.
    environment["PYTHONPATH"] = str(app_source)
    environment["LD_LIBRARY_PATH"] = str(arguments.libllama.parent)
    completed = process_runner(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        cwd=app_source,
        env=environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise QualificationCliError(
            "receipt-bound selective Gate0 failed"
            + (f": {detail}" if detail else "")
        )
    result_path = (
        arguments.output_dir
        / arguments.run_id
        / "derived/gate_0_result.json"
    )
    if not result_path.is_file() or result_path.is_symlink():
        raise QualificationCliError(
            "receipt-bound selective Gate0 result is unavailable"
        )
    return result_path


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        _require_root()
        _require_absolute_arguments(arguments)
        api = _load_api(arguments.checkout_src)
        if arguments.phase == "prepare-static":
            result = api.prepare_static_qualification(
                pending_profile_path=arguments.pending_profile,
                runtime_input_contract_path=arguments.runtime_input_contract,
                runtime_root=arguments.runtime_root,
                runtime_expected_contract_path=arguments.runtime_expected_contract,
                abi_probe_path=arguments.abi_probe,
                llama_source=arguments.llama_source,
                qualified_toolchain_profile=(
                    arguments.qualified_toolchain_profile
                ),
                output_path=arguments.output,
            )
            print(str(result))
            return 0
        if arguments.phase == "render-live-probe-inputs":
            sys.stdout.buffer.write(
                api.render_live_probe_inputs(
                    pending_profile_path=arguments.pending_profile,
                    static_receipt_path=arguments.static_receipt,
                )
            )
            return 0
        if arguments.phase == "run-live-probe":
            print(str(_run_live_probe(arguments, api=api)))
            return 0
        _require_inherited_maintenance_lock(arguments.maintenance_lock_fd)
        paths = api.finalize_live_qualification(
            static_receipt_path=arguments.static_receipt,
            pending_profile_path=arguments.pending_profile,
            runtime_input_contract_path=arguments.runtime_input_contract,
            runtime_root=arguments.runtime_root,
            runtime_expected_contract_path=arguments.runtime_expected_contract,
            abi_probe_path=arguments.abi_probe,
            llama_source=arguments.llama_source,
            gate_zero_result_path=arguments.gate_zero_result,
            output_directory=arguments.output_directory,
        )
        print(str(paths.profile))
        print(str(paths.runtime_input_contract))
        print(str(paths.live_receipt))
        return 0
    except Exception as error:
        print(f"runtime-state qualification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["QualificationCliError", "main"]
