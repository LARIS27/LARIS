from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from laris_onpremise.kv_state.observation.runtime_state_profile import (
    RuntimeStateProfileError,
)

from .contracts import InvestigatorSampler, RehostRuntimeProfile
from .catalog import PortableBundleCatalog, PortableBundleError
from .runner import RehostExecutionError, digest_file, invoke_helper
from .runtime_profile import load_rehost_runtime_profile


class RehostCommandError(RuntimeError):
    """Raised when the portable worker request is incomplete or inconsistent."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rehost one verified cumulative LARIS runtime-state payload."
    )
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--bundle-manifest-digest")
    parser.add_argument("--bundle-id")
    parser.add_argument("--destination-seq-id", type=int, default=0)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--helper-digest", required=True)
    parser.add_argument("--libllama", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-profile-id", required=True)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--messages-file", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--predict", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_runtime_profile(path: Path) -> RehostRuntimeProfile:
    return load_rehost_runtime_profile(path)


def _portable_case(
    arguments: argparse.Namespace,
    *,
    runtime_profile: RehostRuntimeProfile,
    model_digest: str,
) -> tuple[object, object]:
    if arguments.bundle_manifest_digest is None:
        raise RehostCommandError("bundle manifest digest is required")
    if arguments.bundle_id is None:
        raise RehostCommandError("bundle ID is required")
    catalog = PortableBundleCatalog.open(
        arguments.bundle_root,
        expected_manifest_digest=arguments.bundle_manifest_digest,
    )
    if catalog.runtime_metadata is None or catalog.prompt_contract != "gguf_default":
        raise RehostCommandError(
            "bundle does not contain the GGUF-default rehost contract"
        )
    matches = [item for item in catalog.entries if item.bundle_id == arguments.bundle_id]
    if len(matches) != 1:
        raise RehostCommandError("selection must resolve to exactly one bundle")
    entry = matches[0]
    if (
        entry.model_profile_id != arguments.model_profile_id
        or entry.model_digest != model_digest
    ):
        raise RehostCommandError("selected model differs from the bundle")
    case = entry.to_rehost_case(
        run_id=catalog.source_run_id,
        destination_seq_id=arguments.destination_seq_id,
    )
    runtime_profile.verify_case(case)
    return case, catalog.runtime_metadata


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        runtime_profile = _load_runtime_profile(arguments.runtime_profile)
        model_digest = digest_file(arguments.model)
        case, runtime_metadata = _portable_case(
            arguments,
            runtime_profile=runtime_profile,
            model_digest=model_digest,
        )
        cases = (case,)
        if len(cases) != 1:
            raise RehostCommandError("selection must resolve to exactly one bundle")
        receipt = invoke_helper(
            arguments.helper,
            arguments.model,
            arguments.messages_file,
            cases[0],
            runtime_profile=runtime_profile,
            runtime=runtime_metadata,
            sampler=InvestigatorSampler(
                temperature=arguments.temperature,
                top_k=arguments.top_k,
                top_p=arguments.top_p,
                min_p=arguments.min_p,
                seed=arguments.seed,
            ),
            libllama=arguments.libllama,
            expected_helper_digest=arguments.helper_digest,
            analysis_root=arguments.analysis_root,
            timeout_seconds=arguments.timeout_seconds,
            n_predict=arguments.predict,
        )
    except (
        OSError,
        PortableBundleError,
        RehostCommandError,
        RehostExecutionError,
        RuntimeStateProfileError,
        ValueError,
    ) as error:
        print(f"laris rehost: {error}", file=sys.stderr)
        return 2
    result = {
        "schema_version": "1.0",
        "case": cases[0].to_mapping(),
        "model_digest": cases[0].model_digest,
        "prompt_contract": "gguf_default",
        "source_runtime": runtime_metadata.to_mapping()["runtime"],
        "effective_runtime": {
            "n_batch": runtime_metadata.n_batch,
            "n_ubatch": runtime_metadata.n_ubatch,
            "n_predict": arguments.predict,
        },
        "inquiry_policy": {
            "sampler": InvestigatorSampler(
                temperature=arguments.temperature,
                top_k=arguments.top_k,
                top_p=arguments.top_p,
                min_p=arguments.min_p,
                seed=arguments.seed,
            ).to_mapping(),
        },
        "receipt": receipt.to_mapping(),
        "status": "COMPLETE" if receipt.exit_code == 0 else "FAILED",
    }
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    return 0 if receipt.exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
