#!/bin/sh
set -eu

PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH
umask 077

fail() {
    /usr/bin/printf '%s\n' "$1" >&2
    exit 73
}

[ "$#" -eq 6 ] || fail "usage: export_rehost_identity.sh EXPORT_ID IDENTITY_KEY_ID INSTALLATION_RECEIPT IDENTITY_REQUIREMENTS PYTHON_HELPER --identity-only"
EXPORT_ID=$1
IDENTITY_KEY_ID=$2
INSTALLATION_RECEIPT=$3
IDENTITY_REQUIREMENTS=$4
PYTHON_HELPER=$5
[ "$6" = "--identity-only" ] || fail "identity-only export mode is required"

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
    "$IDENTITY_REQUIREMENTS" "$SUDO_UID" "$SUDO_GID" --identity-only
