from __future__ import annotations

from subprocess import CompletedProcess

import pytest

from opjax.remote import config


def test_modal_proxy_headers_use_combined_environment_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(config.MODAL_PROXY_TOKEN_ENV, "wk-token.ws-secret")

    assert config.modal_proxy_headers() == {
        "Modal-Key": "wk-token",
        "Modal-Secret": "ws-secret",
    }


def test_modal_proxy_headers_fall_back_to_keychain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(config.MODAL_PROXY_TOKEN_ENV, raising=False)
    monkeypatch.setattr(
        config.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args=args,
            returncode=0,
            stdout="wk-keychain.ws-keychain\n",
        ),
    )

    assert config.modal_proxy_headers() == {
        "Modal-Key": "wk-keychain",
        "Modal-Secret": "ws-keychain",
    }


@pytest.mark.parametrize("token", ["", "wk-only", "wk-token.bad-secret"])
def test_modal_proxy_headers_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
) -> None:
    monkeypatch.setenv(config.MODAL_PROXY_TOKEN_ENV, token)
    monkeypatch.setattr(
        config.subprocess,
        "run",
        lambda *args, **kwargs: CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
        ),
    )

    with pytest.raises(RuntimeError, match="MODAL_PROXY_TOKEN_MISSING"):
        config.modal_proxy_headers()
