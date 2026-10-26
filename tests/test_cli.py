import os
from pathlib import Path

import pytest

from harness.cli import load_dotenv, main


def test_run_report_show(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "runs.db")
    assert main(["--db", db, "run", "--seeds", "3", "--run-id", "smoke"]) == 0
    assert "ran 6 episodes" in capsys.readouterr().out

    assert main(["--db", db, "report"]) == 0
    assert "scripted" in capsys.readouterr().out

    assert main(["--db", db, "show", "scripted", "doorkey-5x5", "0"]) == 0
    assert "SUCCESS" in capsys.readouterr().out


def test_multi_task_report_and_compare(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(tmp_path / "runs.db")
    assert main(["--db", db, "run", "--seeds", "4", "--tasks", "empty-5x5,lavagap-5x5"]) == 0
    out = capsys.readouterr().out
    assert "norm [95% CI]" in out and "overall" in out

    assert main(["--db", db, "compare", "scripted", "random"]) == 0
    out = capsys.readouterr().out
    assert "A better" in out and "all" in out

    assert main(["--db", db, "compare", "scripted", "react"]) == 1


def test_unknown_agent_is_rejected(tmp_path: Path) -> None:
    assert main(["--db", str(tmp_path / "x.db"), "run", "--agents", "gpt-99"]) == 2


def test_report_without_runs(tmp_path: Path) -> None:
    assert main(["--db", str(tmp_path / "x.db"), "report"]) == 1


def test_dotenv_fills_unset_variables_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        'export HARNESS_T_A="quoted value"\n'
        "HARNESS_T_B=plain\n"
        "HARNESS_T_C=from-file\n"
        "not a line\n"
    )
    for key in ("HARNESS_T_A", "HARNESS_T_B"):
        monkeypatch.setenv(key, "")  # registers the variable so pytest restores it
        monkeypatch.delenv(key)
    monkeypatch.setenv("HARNESS_T_C", "from-shell")

    assert load_dotenv(env) == ["HARNESS_T_A", "HARNESS_T_B"]
    assert os.environ["HARNESS_T_A"] == "quoted value"
    assert os.environ["HARNESS_T_B"] == "plain"
    assert os.environ["HARNESS_T_C"] == "from-shell"
    assert load_dotenv(tmp_path / "missing") == []
