#!/bin/bash
set -euo pipefail
umask 077

usage() {
  echo "usage: run_log_linked_llama_server.sh --server-bin PATH --library-path DIR --model PATH --expected-sha256 HEX --upstream-commit HEX --log-file PATH --launch-receipt PATH [--host HOST] [--port PORT] [--parallel N] [--ctx-size N] [--slot-save-path PATH]" >&2
  exit 64
}

server_bin=
library_path=
model=
expected_sha256=
upstream_commit=
log_file=
launch_receipt=
host=127.0.0.1
port=8080
parallel=4
ctx_size=8192
slot_save_path=

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --server-bin) [[ "$#" -ge 2 ]] || usage; server_bin="$2"; shift 2 ;;
    --library-path) [[ "$#" -ge 2 ]] || usage; library_path="$2"; shift 2 ;;
    --model) [[ "$#" -ge 2 ]] || usage; model="$2"; shift 2 ;;
    --expected-sha256) [[ "$#" -ge 2 ]] || usage; expected_sha256="$2"; shift 2 ;;
    --upstream-commit) [[ "$#" -ge 2 ]] || usage; upstream_commit="$2"; shift 2 ;;
    --log-file) [[ "$#" -ge 2 ]] || usage; log_file="$2"; shift 2 ;;
    --launch-receipt) [[ "$#" -ge 2 ]] || usage; launch_receipt="$2"; shift 2 ;;
    --host) [[ "$#" -ge 2 ]] || usage; host="$2"; shift 2 ;;
    --port) [[ "$#" -ge 2 ]] || usage; port="$2"; shift 2 ;;
    --parallel) [[ "$#" -ge 2 ]] || usage; parallel="$2"; shift 2 ;;
    --ctx-size) [[ "$#" -ge 2 ]] || usage; ctx_size="$2"; shift 2 ;;
    --slot-save-path) [[ "$#" -ge 2 ]] || usage; slot_save_path="$2"; shift 2 ;;
    *) usage ;;
  esac
done

for path in "$server_bin" "$library_path" "$model" "$log_file" "$launch_receipt"; do
  [[ "$path" == /* ]] || usage
done
[[ "$expected_sha256" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "$upstream_commit" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$port" =~ ^[1-9][0-9]{0,4}$ && "$port" -le 65535 ]] || usage
[[ "$parallel" =~ ^[1-9][0-9]*$ ]] || usage
[[ "$ctx_size" =~ ^[1-9][0-9]*$ ]] || usage
if [[ -n "$slot_save_path" && ( "$slot_save_path" != /* || ! -d "$slot_save_path" || -L "$slot_save_path" ) ]]; then
  usage
fi

for path in "$server_bin" "$model"; do
  if [[ ! -f "$path" || -L "$path" ]]; then
    echo "required official runtime input is unavailable: $path" >&2
    exit 66
  fi
done
if [[ ! -d "$library_path" || -L "$library_path" ]]; then
  echo "runtime library directory is unavailable" >&2
  exit 66
fi
if [[ -e "$log_file" || -L "$log_file" || -e "$launch_receipt" || -L "$launch_receipt" ]]; then
  echo "log-linked output already exists" >&2
  exit 73
fi

actual_sha256="$(/usr/bin/sha256sum -- "$server_bin" | /usr/bin/cut -d ' ' -f 1)"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "official llama-server digest differs" >&2
  exit 77
fi

/usr/bin/install -d -m 0700 -- "$(/usr/bin/dirname -- "$log_file")"
/usr/bin/install -d -m 0700 -- "$(/usr/bin/dirname -- "$launch_receipt")"
version="$(LD_LIBRARY_PATH="$library_path" "$server_bin" --version 2>&1)"
command=(
  "$server_bin"
  --model "$model"
  --host "$host"
  --port "$port"
  --parallel "$parallel"
  --ctx-size "$ctx_size"
  --slots
  --no-kv-unified
  --flash-attn off
  --cache-type-k f16
  --cache-type-v f16
  --batch-size 128
  --ubatch-size 128
  --n-gpu-layers 0
  --no-cache-idle-slots
  --no-context-shift
  --log-verbosity 4
  --log-timestamps
  --log-file "$log_file"
)
if [[ -n "$slot_save_path" ]]; then
  command+=(--slot-save-path "$slot_save_path")
fi
for argument in "${command[@]}"; do
  case "$argument" in
    --verbose|--verbose-prompt|--log-disable|--log-verbosity=5)
      echo "content-bearing or disabled logging argument is forbidden" >&2
      exit 77
      ;;
  esac
done

/usr/bin/python3 -I -B -c '
import hashlib, json, os, sys
destination, executable_sha256, upstream_commit, version, *arguments = sys.argv[1:]
document = {
    "schema_version": "1.0",
    "application": "llama-server",
    "application_source_modified": False,
    "executable_sha256": "sha256:" + executable_sha256,
    "upstream_commit": upstream_commit,
    "version": version,
    "arguments": arguments,
    "arguments_sha256": "sha256:" + hashlib.sha256(json.dumps(arguments, separators=(",", ":")).encode("utf-8")).hexdigest(),
    "content_logging_enabled": False,
}
payload = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
try:
    os.write(fd, payload)
    os.fsync(fd)
finally:
    os.close(fd)
' "$launch_receipt" "$actual_sha256" "$upstream_commit" "$version" "${command[@]}"

exec /usr/bin/env -i \
  PATH=/usr/bin:/bin \
  LANG=C.UTF-8 \
  LC_ALL=C.UTF-8 \
  LD_LIBRARY_PATH="$library_path" \
  "${command[@]}"
