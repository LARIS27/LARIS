from __future__ import annotations

from io import StringIO

from laris_onpremise.rehosting.terminal import TerminalPresenter, render_banner


def test_plain_banner_reuses_the_historical_laris_identity() -> None:
    value = render_banner(color=False)
    lines = value.splitlines()

    assert "Locking Onto Runtime Inference State" in value
    assert "Local LLMs Rehost Console for Restored-State Inquiry" in value
    assert "\x1b[" not in value
    assert {len(line) for line in lines} == {64}
    assert lines[5].startswith("| |_____/_/")


def test_plain_presenter_emits_stable_nonanimated_stage_output() -> None:
    stream = StringIO()
    presenter = TerminalPresenter(stream, color=False, motion=False)

    presenter.stage("BUNDLE", "VERIFIED", "one explicit bundle", tone="green")
    with presenter.activity("Restoring preserved KV"):
        pass

    assert stream.getvalue() == (
        "[BUNDLE] VERIFIED - one explicit bundle\n"
        "[....] Restoring preserved KV\n"
    )
    assert "\x1b[" not in stream.getvalue()
    assert "\r" not in stream.getvalue()


def test_colored_motion_activity_writes_and_clears_a_bounded_frame() -> None:
    stream = StringIO()
    presenter = TerminalPresenter(
        stream,
        color=True,
        motion=True,
        interval_seconds=60.0,
    )

    presenter.stage("MODEL", "READY", tone="green")
    with presenter.activity("Running verified rehost"):
        pass

    value = stream.getvalue()
    assert "\x1b[32m[MODEL] READY\x1b[0m\n" in value
    assert "\r\x1b[36m[|]\x1b[0m Running verified rehost" in value
    assert value.endswith("\r")
