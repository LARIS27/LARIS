"""Compatibility CLI for the relocated Proxy command."""

from laris_onpremise.attribution.proxy.command import *
from laris_onpremise.attribution.proxy.command import main

if __name__ == "__main__":
    raise SystemExit(main())
