import pytest


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's .env (and its real API key) out of every test.

    ``main()`` loads ``.env`` from the working directory; under test that could
    turn an offline test into a billed API call. The live test reads the key
    from the real environment instead.
    """
    monkeypatch.setattr("harness.cli.load_dotenv", lambda *a, **k: [])
