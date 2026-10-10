# Operator guide

This guide separates live LARIS operation from publication-only Evaluation1 replay.
Live acquisition runs on the Linux host that owns the target llama.cpp process. Public
bundle replay may use the Windows/WSL setup described in `REHOSTING_SETUP.md`; that WSL
path does not qualify a machine for acquisition.

## Common live-runtime boundary

Before either attribution mode, register the target runtime with its process ID and
`/proc/<pid>/stat` start-time ticks. The current runtime lifecycle is exposed by:

```text
laris runtime inspect --agent-profile AGENT_PROFILE
laris runtime attach --agent-profile AGENT_PROFILE --scenario SCENARIO
laris runtime status --agent-profile AGENT_PROFILE
laris runtime stop-and-seal --agent-profile AGENT_PROFILE --run-id RUN_ID
laris runtime detach --agent-profile AGENT_PROFILE --run-id RUN_ID
```

The PID/start-ticks pair prevents silent attachment to a reused PID. Profiles, key
files, evidence roots, Unix sockets, and model/runtime identities must be operator-owned
regular paths. A failed preflight, missing watermark, lost event, digest mismatch, or
identity mismatch is a failed acquisition; it must not be converted into partial
evidence.

## Proxy evidence path

Proxy mode places the Sequence Lease Proxy between the client and an already running
model service. The operator supplies an authenticated subject boundary, tenant and key
epoch, runtime-state socket/profile, append-only ledger, and acquisition evidence root.
The complete current interface is available with `laris proxy --help`; its required
inputs include:

```text
laris proxy \
  --upstream-url UPSTREAM_URL \
  --ledger LEDGER \
  --runtime-state-socket RUNTIME_STATE_SOCKET \
  --runtime-state-profile RUNTIME_STATE_PROFILE \
  --acquisition-evidence-root EVIDENCE_ROOT \
  --runtime-pid PID \
  --runtime-start-time-ticks START_TICKS \
  --identity-key-file IDENTITY_KEY \
  --identity-key-id KEY_ID \
  --ledger-mac-key-file LEDGER_MAC_KEY \
  --tenant-id TENANT_ID \
  --controller-instance-id CONTROLLER_ID \
  --runtime-instance-id RUNTIME_ID
```

Requests are admitted through a bounded slot lease. The sealed bundle records the
authenticated subject reference and key ID; raw identity-key bytes are not bundle
metadata. Stop admission and seal the run before exporting or moving evidence.

## Log-linked evidence path

Log-linked mode observes runtime units independently and correlates them with a
content-free, preflighted server-log profile:

```text
laris log-linked preflight --profile PROFILE
laris log-linked observe --profile PROFILE --evidence-root EVIDENCE_ROOT
laris log-linked correlate \
  --evidence-root EVIDENCE_ROOT \
  --correlation-key-file CORRELATION_KEY \
  --correlation-key-id KEY_ID
laris log-linked export \
  --evidence-root EVIDENCE_ROOT \
  --endpoint-ref ENDPOINT_REF \
  --bundle-root BUNDLE_ROOT
```

Use `--session-request-ref` during export when more than one resolved request shares an
endpoint. Correlation must finish as `CERTIFIED` or `RANKED`; unresolved, missing, or
ambiguous selection is not silently exported. The endpoint reference is an HMAC-derived
reference bound to the correlation key epoch, not a substitute for preserving the
underlying authorized access record.

## Sealing, custody, and transfer

Keep the original sealed bundle and its manifest immutable. Verify all recorded sizes
and SHA-256 digests after transfer, retain the key-epoch metadata required to resolve
the subject or endpoint reference, and store secret key bytes separately from evidence.
Rehosting operates on a verified export and writes a new analysis record; it never
modifies the sealed acquisition.

For published bundles, follow `REHOSTING_SETUP.md` and
`reproducibility/evaluation1-release.md`. Those fixtures are synthetic and demonstrate
the replay contract only; they are not live-operator credentials.
