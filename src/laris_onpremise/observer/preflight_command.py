"""Compatibility entry point for the runtime host-preflight command."""

from laris_onpremise.runtime.preflight_command import build_parser, main, run

__all__ = ["build_parser", "main", "run"]


if __name__ == "__main__":
    raise SystemExit(main())
