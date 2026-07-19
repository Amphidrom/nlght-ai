# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for nlght/main.py bootstrapping helpers.

Verifies:
  - _read_raw returns {} for missing file
  - _read_raw returns parsed YAML content for existing file
  - _gateway_binding returns defaults when no http-gateway subsystem configured
  - _gateway_binding reads host/port from http-gateway subsystem
  - _gateway_binding ignores disabled subsystems
  - serve() calls uvicorn.run with host/port derived from config
  - cli() dispatches 'serve' and 'migrate' subcommands, rejects unknown ones
"""
from __future__ import annotations

import textwrap

import pytest

from nlght.main import _gateway_binding, _read_raw, cli

# ---------------------------------------------------------------------------
# _read_raw
# ---------------------------------------------------------------------------


def test_read_raw_returns_empty_dict_for_missing_file() -> None:
    result = _read_raw("/nonexistent/path/platform.yaml")
    assert result == {}


def test_read_raw_returns_parsed_yaml(tmp_path) -> None:
    config = tmp_path / "platform.yaml"
    config.write_text(
        textwrap.dedent("""\
            gateways:
              - name: gw
                kind: http
                enabled: true
                config:
                  host: 127.0.0.1
                  port: 9000
        """),
        encoding="utf-8",
    )
    result = _read_raw(str(config))
    assert result["gateways"][0]["config"]["port"] == 9000


def test_read_raw_returns_empty_dict_for_empty_file(tmp_path) -> None:
    config = tmp_path / "empty.yaml"
    config.write_text("", encoding="utf-8")
    assert _read_raw(str(config)) == {}


# ---------------------------------------------------------------------------
# _gateway_binding
# ---------------------------------------------------------------------------


def test_gateway_binding_returns_defaults_when_no_subsystem() -> None:
    host, port = _gateway_binding({})
    assert host == "0.0.0.0"
    assert port == 8000


def test_gateway_binding_returns_defaults_when_no_http_gateway_kind() -> None:
    raw = {
        "gateways": [
            {"kind": "grpc", "enabled": True, "config": {"host": "db", "port": 5432}}
        ]
    }
    host, port = _gateway_binding(raw)
    assert host == "0.0.0.0"
    assert port == 8000


def test_gateway_binding_reads_host_and_port() -> None:
    raw = {
        "gateways": [
            {
                "kind": "http",
                "enabled": True,
                "config": {"host": "127.0.0.1", "port": 9090},
            }
        ]
    }
    host, port = _gateway_binding(raw)
    assert host == "127.0.0.1"
    assert port == 9090


def test_gateway_binding_ignores_disabled_subsystem() -> None:
    raw = {
        "gateways": [
            {
                "kind": "http",
                "enabled": False,
                "config": {"host": "192.168.1.1", "port": 7777},
            }
        ]
    }
    host, port = _gateway_binding(raw)
    assert host == "0.0.0.0"
    assert port == 8000


def test_gateway_binding_uses_first_enabled_gateway() -> None:
    raw = {
        "gateways": [
            {"kind": "http", "enabled": False, "config": {"host": "bad", "port": 1}},
            {"kind": "http", "enabled": True, "config": {"host": "good", "port": 2}},
        ]
    }
    host, port = _gateway_binding(raw)
    assert host == "good"
    assert port == 2


# ---------------------------------------------------------------------------
# serve() — verify uvicorn.run is called with correct args
# ---------------------------------------------------------------------------


def test_serve_calls_uvicorn_run_with_config_binding(tmp_path, monkeypatch) -> None:
    config = tmp_path / "platform.yaml"
    config.write_text(
        textwrap.dedent("""\
            gateways:
              - kind: http
                enabled: true
                config:
                  host: 127.0.0.1
                  port: 8765
        """),
        encoding="utf-8",
    )

    import nlght.main as serve_module

    run_calls: list[dict] = []

    def fake_uvicorn_run(app, *, host, port, reload, factory) -> None:
        run_calls.append({"host": host, "port": port, "reload": reload, "factory": factory})

    monkeypatch.setenv("NLGHT_CONFIG", str(config))
    monkeypatch.setenv("NLGHT_RELOAD", "")
    monkeypatch.setattr(serve_module.uvicorn, "run", fake_uvicorn_run)

    serve_module.serve()

    assert len(run_calls) == 1
    assert run_calls[0]["host"] == "127.0.0.1"
    assert run_calls[0]["port"] == 8765
    assert run_calls[0]["reload"] is False
    assert run_calls[0]["factory"] is True


def test_serve_enables_reload_from_env(tmp_path, monkeypatch) -> None:
    config = tmp_path / "platform.yaml"
    config.write_text("{}", encoding="utf-8")

    import nlght.main as serve_module

    run_calls: list[dict] = []

    def fake_uvicorn_run(app, *, host, port, reload, factory) -> None:
        run_calls.append({"reload": reload})

    monkeypatch.setenv("NLGHT_CONFIG", str(config))
    monkeypatch.setenv("NLGHT_RELOAD", "true")
    monkeypatch.setattr(serve_module.uvicorn, "run", fake_uvicorn_run)

    serve_module.serve()

    assert run_calls[0]["reload"] is True


# ---------------------------------------------------------------------------
# _warn_if_admin_on_public_binding — RLS-009
# ---------------------------------------------------------------------------


def test_admin_on_public_binding_warns(monkeypatch, caplog) -> None:
    from nlght.main import _warn_if_admin_on_public_binding

    monkeypatch.setenv("NLGHT_ADMIN", "1")
    with caplog.at_level("WARNING"):
        _warn_if_admin_on_public_binding("0.0.0.0")

    assert any("NO authentication" in r.message for r in caplog.records)


def test_admin_on_loopback_does_not_warn(monkeypatch, caplog) -> None:
    from nlght.main import _warn_if_admin_on_public_binding

    monkeypatch.setenv("NLGHT_ADMIN", "1")
    with caplog.at_level("WARNING"):
        _warn_if_admin_on_public_binding("127.0.0.1")

    assert not caplog.records


def test_no_admin_on_public_binding_does_not_warn(monkeypatch, caplog) -> None:
    from nlght.main import _warn_if_admin_on_public_binding

    monkeypatch.delenv("NLGHT_ADMIN", raising=False)
    with caplog.at_level("WARNING"):
        _warn_if_admin_on_public_binding("0.0.0.0")

    assert not caplog.records


# ---------------------------------------------------------------------------
# py.typed — RLS-010: PEP 561 marker ships with the package
# ---------------------------------------------------------------------------


def test_py_typed_marker_exists_and_is_packaged() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    assert (root / "src" / "nlght" / "py.typed").exists()
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"py.typed"' in pyproject


def test_admin_static_assets_exist_and_are_packaged() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    static = root / "src" / "nlght" / "adapters" / "inbound" / "http" / "admin" / "static"
    for asset in ("pico.classless.min.css", "htmx.min.js", "cytoscape.min.js", "THIRD_PARTY_LICENSES.txt"):
        assert (static / asset).is_file()
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"adapters/inbound/http/admin/static/*"' in pyproject


# ---------------------------------------------------------------------------
# cli() — nlght-ai <command> dispatch
# ---------------------------------------------------------------------------


def test_cli_dispatches_serve(monkeypatch) -> None:
    import nlght.main as main_module

    called: list[str] = []
    monkeypatch.setattr(main_module, "serve", lambda: called.append("serve"))
    monkeypatch.setattr("sys.argv", ["nlght-ai", "serve"])

    cli()

    assert called == ["serve"]


def test_cli_dispatches_migrate_with_args(monkeypatch) -> None:
    import nlght.main as main_module

    received: list[list[str]] = []
    monkeypatch.setattr(main_module, "migrate", lambda argv: received.append(argv))
    monkeypatch.setattr("sys.argv", ["nlght-ai", "migrate", "downgrade", "-1"])

    cli()

    assert received == [["downgrade", "-1"]]


def test_cli_dispatches_migrate_without_args(monkeypatch) -> None:
    import nlght.main as main_module

    received: list[list[str]] = []
    monkeypatch.setattr(main_module, "migrate", lambda argv: received.append(argv))
    monkeypatch.setattr("sys.argv", ["nlght-ai", "migrate"])

    cli()

    assert received == [[]]


def test_cli_without_command_exits_with_usage(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["nlght-ai"])

    with pytest.raises(SystemExit) as excinfo:
        cli()

    assert excinfo.value.code == 2
    assert "usage: nlght-ai" in capsys.readouterr().err


def test_cli_unknown_command_exits_with_usage(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["nlght-ai", "frobnicate"])

    with pytest.raises(SystemExit) as excinfo:
        cli()

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unknown command 'frobnicate'" in err
    assert "usage: nlght-ai" in err
