#!/bin/sh
set -eu

PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH
umask 077

fail() {
    /usr/bin/printf '%s\n' "$1" >&2
    exit 73
}

[ "$#" -eq 2 ] || fail "usage: install_rehost_runtime.sh INPUT_ROOT RELEASE_ID"
input_root=$1
release_id=$2
case "$release_id" in
    *[!0-9a-f]*|'') fail "release identifier is invalid" ;;
esac
[ "${#release_id}" -eq 16 ] || fail "release identifier is invalid"
case "$input_root" in
    /*) ;;
    *) fail "input root must be absolute" ;;
esac
[ -d "$input_root" ] || fail "input root is unavailable"
[ ! -L "$input_root" ] || fail "input root is a symbolic link"

manifest=$input_root/manifest.sha256
[ -f "$manifest" ] || fail "input manifest is unavailable"
(
    cd "$input_root"
    /usr/bin/sha256sum -c manifest.sha256
) || fail "rehost runtime input digest differs"

source_archive=$input_root/laris_onpremise-src.tar
[ -f "$source_archive" ] || fail "LARIS source archive is unavailable"

passwd_record=$(getent passwd "$(id -un)")
[ -n "$passwd_record" ] || fail "invoking WSL user is unavailable"
user_home=$(/usr/bin/printf '%s\n' "$passwd_record" | /usr/bin/cut -d: -f6)
case "$user_home" in
    /*) ;;
    *) fail "invoking WSL user home is invalid" ;;
esac
[ -d "$user_home" ] || fail "invoking WSL user home is unavailable"
[ ! -L "$user_home" ] || fail "invoking WSL user home is a symbolic link"
[ "$(/usr/bin/stat -c %u -- "$user_home")" -eq "$(id -u)" ] || \
    fail "invoking WSL user home owner is invalid"

base=$user_home/.local/share/laris-rehost
releases=$base/releases
final=$releases/$release_id
pending=$releases/.pending-$release_id
[ ! -e "$final" ] || fail "rehost runtime release already exists"
[ ! -e "$pending" ] || fail "rehost runtime pending release already exists"

cleanup() {
    if [ -d "$pending" ]; then
        /usr/bin/find "$pending" -depth -type f -delete 2>/dev/null || true
        /usr/bin/find "$pending" -depth -type l -delete 2>/dev/null || true
        /usr/bin/find "$pending" -depth -type d -empty -delete 2>/dev/null || true
    fi
}
trap cleanup EXIT HUP INT TERM

/usr/bin/install -d -m 0700 -- "$base" "$releases" "$base/work" "$pending"
/usr/bin/install -d -m 0700 -- "$pending/native" "$pending/lib" "$pending/site"
/usr/bin/install -m 0700 -- "$input_root/llama-runtime-payload-rehost" \
    "$pending/native/llama-runtime-payload-rehost"
[ -d "$input_root/lib" ] && [ ! -L "$input_root/lib" ] || \
    fail "rehost library input directory is unavailable"
library_count=0
libllama_count=0
for library_path in "$input_root"/lib/*; do
    [ -f "$library_path" ] && [ ! -L "$library_path" ] || \
        fail "rehost library input is invalid"
    [ "$(/usr/bin/stat -c %h -- "$library_path")" -eq 1 ] || \
        fail "rehost library input has multiple links"
    library_name=$(/usr/bin/basename -- "$library_path")
    case "$library_name" in
        libllama.so*) libllama_count=$((libllama_count + 1)) ;;
        libggml*.so*) ;;
        *) fail "rehost library input name is invalid" ;;
    esac
    /usr/bin/install -m 0600 -- "$library_path" "$pending/lib/$library_name"
    library_count=$((library_count + 1))
done
[ "$library_count" -ge 2 ] || fail "rehost library input set is incomplete"
[ "$libllama_count" -eq 1 ] || fail "rehost libllama input is ambiguous"
/usr/bin/install -m 0600 -- "$input_root/runtime-profile.json" \
    "$pending/runtime-profile.json"

/usr/bin/tar -xf "$source_archive" -C "$pending/site" --no-same-owner --no-same-permissions
/usr/bin/python3 -m venv --system-site-packages "$pending/venv"
site_packages=$("$pending/venv/bin/python" -c \
    'import sysconfig; print(sysconfig.get_path("purelib"))')
/usr/bin/printf '%s\n' "$pending/site" > "$site_packages/laris-local.pth"
"$pending/venv/bin/python" -I -B -c \
    'import yaml; import laris_onpremise.rehosting.command; print("LARIS_WSL_IMPORT_OK")'

ldd_output=$(/usr/bin/env LD_LIBRARY_PATH="$pending/lib" \
    /usr/bin/ldd "$pending/native/llama-runtime-payload-rehost")
case "$ldd_output" in
    *'not found'*) fail "rehost helper has an unresolved library dependency" ;;
esac

/usr/bin/printf '%s\n' "$final/site" > "$site_packages/laris-local.pth"
/usr/bin/mv -- "$pending" "$final"
"$final/venv/bin/python" -I -B -c \
    'import yaml; import laris_onpremise.rehosting.command; print("LARIS_WSL_FINAL_IMPORT_OK")'
trap - EXIT HUP INT TERM
/usr/bin/printf 'LARIS_WSL_REHOST_RUNTIME_READY root=%s\n' "$final"
