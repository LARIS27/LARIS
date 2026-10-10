#!/usr/bin/env sh
set -eu

PINNED_LLAMA_COMMIT=daf6bc9f2d3516d7bda73d191c6a01388aff09c5

if [ "$#" -ne 3 ]; then
    echo "usage: setup_public_rehost_runtime.sh REPOSITORY LLAMA_CPP_SOURCE INSTALL_ROOT" >&2
    exit 2
fi

repository=$1
llama_source=$2
install_root=$3

case "$repository:$llama_source:$install_root" in
    /*:/*:/*) ;;
    *) echo "all paths must be absolute WSL paths" >&2; exit 2 ;;
esac
if [ "$install_root" = / ]; then
    echo "INSTALL_ROOT must not be /" >&2
    exit 2
fi
if [ ! -f "$repository/pyproject.toml" ] || [ ! -f "$repository/native/rehosting/Makefile" ]; then
    echo "REPOSITORY is not a LARIS source root" >&2
    exit 2
fi
if [ ! -d "$llama_source/.git" ]; then
    echo "LLAMA_CPP_SOURCE must be an existing git checkout" >&2
    exit 2
fi

actual_commit=$(cd "$llama_source" && git rev-parse HEAD)
if [ "$actual_commit" != "$PINNED_LLAMA_COMMIT" ]; then
    echo "llama.cpp checkout is not at the pinned commit" >&2
    exit 2
fi

build_root="$install_root/build"
runtime_root="$install_root/runtime"
venv_root="$install_root/venv"
work_root="$install_root/work"
mkdir -p "$build_root" "$runtime_root/bin" "$runtime_root/lib" "$work_root"

cmake -S "$llama_source" -B "$build_root/llama.cpp" \
    -DBUILD_SHARED_LIBS=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_CUDA=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=ON
cmake --build "$build_root/llama.cpp" --target llama-server --parallel

make -C "$repository/native/rehosting" \
    LLAMA_SOURCE="$llama_source" \
    LLAMA_BUILD="$build_root/llama.cpp" \
    BUILD_DIR="$build_root/rehosting"

runtime_source="$build_root/llama.cpp/bin/llama-server"
libllama_source="$build_root/llama.cpp/bin/libllama.so"
helper_source="$build_root/rehosting/llama-runtime-payload-rehost"
if [ ! -f "$runtime_source" ] || [ ! -e "$libllama_source" ] || [ ! -f "$helper_source" ]; then
    echo "expected rehosting build output is missing" >&2
    exit 2
fi

install -m 0755 "$runtime_source" "$runtime_root/bin/llama-server"
install -m 0755 "$helper_source" "$runtime_root/bin/llama-runtime-payload-rehost"
for library in "$build_root"/llama.cpp/bin/lib*.so*; do
    [ -e "$library" ] || continue
    cp -L -- "$library" "$runtime_root/lib/$(basename "$library")"
    chmod 0755 "$runtime_root/lib/$(basename "$library")"
done

python3 -m venv "$venv_root"
"$venv_root/bin/python" -m pip install "$repository"
runtime_profile="$runtime_root/runtime-profile.json"
"$venv_root/bin/python" "$repository/evaluation/scripts/build_public_rehost_profile.py" \
    --runtime "$runtime_root/bin/llama-server" \
    --libllama "$runtime_root/lib/libllama.so" \
    --runtime-binary-path "$runtime_root/bin/llama-server" \
    --output "$runtime_profile"

helper_digest=$(sha256sum "$runtime_root/bin/llama-runtime-payload-rehost" | awk '{print $1}')
receipt="$install_root/setup-receipt.json"
"$venv_root/bin/python" - "$receipt" "$venv_root/bin/python" "$runtime_profile" \
    "$runtime_root/bin/llama-runtime-payload-rehost" "sha256:$helper_digest" \
    "$runtime_root/lib/libllama.so" "$work_root" <<'PY'
import json
from pathlib import Path
import sys

target = Path(sys.argv[1])
document = {
    "schema_version": "1.0",
    "state": "PUBLIC_REHOST_RUNTIME_READY",
    "python": sys.argv[2],
    "runtime_profile": sys.argv[3],
    "helper": sys.argv[4],
    "helper_digest": sys.argv[5],
    "libllama": sys.argv[6],
    "work_root": sys.argv[7],
}
target.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n")
print(target)
PY
