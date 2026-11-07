import pytest

import main
from config.settings import BotConfig


def test_demo_always_selects_paper_mode():
    cfg = main.build_config_from_args(["live", "--demo"], environ={})
    assert cfg.mode == "paper"


def test_live_requires_explicit_confirmation():
    with pytest.raises(SystemExit, match="explicit 'live' argument and CONFIRM_LIVE=YES"):
        main.build_config_from_args(["live"], environ={})


def test_live_requires_live_argument_even_with_confirmation(monkeypatch):
    cfg = BotConfig()
    cfg.exchange.api_key = "test-key"
    cfg.exchange.api_secret = "test-secret"
    monkeypatch.setattr(main, "BotConfig", lambda: cfg)

    assert main.build_config_from_args([], environ={"CONFIRM_LIVE": "YES"}).mode == "paper"
    assert main.build_config_from_args(["live"], environ={"CONFIRM_LIVE": "YES"}).mode == "live"
