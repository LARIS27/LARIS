#!/bin/sh
set -eu

PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
umask 077

fail() {
    /usr/bin/printf '%s\n' "$1" >&2
    exit 73
}

[ "$#" -eq 5 ] || fail "usage: export_rehost_profile_inputs.sh EXPORT_ID IDENTITY_KEY_ID INSTALLATION_RECEIPT BUNDLE_REQUIREMENTS PYTHON_HELPER"
EXPORT_ID=$1
IDENTITY_KEY_ID=$2
INSTALLATION_RECEIPT=$3
BUNDLE_REQUIREMENTS=$4
PYTHON_HELPER=$5
# The Python helper validates rehost_components and installation_receipt_sha256.

[ "$(/usr/bin/id -u)" -eq 0 ] || fail "root privileges are required"
case "${SUDO_UID-}:${SUDO_GID-}" in
    *[!0-9:]*|:|*:|:*) fail "sudo caller identity is invalid" ;;
esac
case "$EXPORT_ID" in
    *[!0-9a-f]*|'') fail "export identifier is invalid" ;;
esac
[ "${#EXPORT_ID}" -eq 32 ] || fail "export identifier is invalid"

python_helper=$(/usr/bin/realpath -e -- "$PYTHON_HELPER") || \
    fail "export Python helper cannot be resolved"
[ -f "$python_helper" ] && [ ! -L "$python_helper" ] || \
    fail "export Python helper is unavailable"

/usr/bin/python3 -I -B "$python_helper" \
    "$EXPORT_ID" "$IDENTITY_KEY_ID" "$INSTALLATION_RECEIPT" \
    "$BUNDLE_REQUIREMENTS" "$SUDO_UID" "$SUDO_GID"
