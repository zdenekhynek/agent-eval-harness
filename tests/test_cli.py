from pathlib import Path

import pytest

from harness.cli import main


def test_run_report_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "runs.db")
    assert main(["--db", db, "run", "--seeds", "3", "--run-id", "smoke"]) == 0
    assert "ran 6 episodes" in capsys.readouterr().out

    assert main(["--db", db, "report"]) == 0
    assert "scripted" in capsys.readouterr().out

    assert main(["--db", db, "show", "scripted", "doorkey-5x5", "0"]) == 0
    assert "SUCCESS" in capsys.readouterr().out


def test_unknown_agent_is_rejected(tmp_path: Path) -> None:
    assert main(["--db", str(tmp_path / "x.db"), "run", "--agents", "gpt-99"]) == 2


def test_report_without_runs(tmp_path: Path) -> None:
    assert main(["--db", str(tmp_path / "x.db"), "report"]) == 1
