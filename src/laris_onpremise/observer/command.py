"""Compatibility entry point for :mod:`laris_onpremise.runtime.command`."""

from laris_onpremise.runtime.command import (
    _inspect_target,
    _parser,
    _snapshot_quiescent,
    main,
)

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
