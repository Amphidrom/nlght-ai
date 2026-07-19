# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for FileSystemBackend and SessionSnapshot serialisation."""
from __future__ import annotations

from datetime import UTC, datetime

from nlght.adapters.outbound.hive_mind.persistence import FileSystemBackend
from nlght.core.hive_mind.models import (
    Directive,
    DirectivePriority,
    PromotionStatus,
    SessionResult,
    SessionSnapshot,
    TurnSummary,
    _de_directive,
    _de_result,
    _de_turn,
    _ser_directive,
    _ser_result,
    _ser_turn,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _snapshot(session_id: str = "s1") -> SessionSnapshot:
    return SessionSnapshot(
        session_id=session_id,
        directives=[],
        turns=[],
        results=[],
        interrupted_task_ids=[],
        created_at=_now(),
        updated_at=_now(),
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def test_ser_de_directive_roundtrip() -> None:
    d = Directive(key="lang", value="de", source="user", priority=DirectivePriority.HIGH)
    data = _ser_directive(d)
    d2   = _de_directive(data)
    assert d2.key      == "lang"
    assert d2.value    == "de"
    assert d2.source   == "user"
    assert d2.priority == DirectivePriority.HIGH
    assert d2.id       == d.id


def test_ser_de_turn_roundtrip() -> None:
    t = TurnSummary(
        turn_nr=3, user_input="hello", intent="greet",
        topic="test", entities=["python"], result_summary="ok",
    )
    data = _ser_turn(t)
    t2   = _de_turn(data)
    assert t2.turn_nr       == 3
    assert t2.user_input    == "hello"
    assert t2.entities      == ["python"]
    assert t2.result_summary == "ok"


def test_ser_de_result_roundtrip() -> None:
    r = SessionResult(content="data", entities=["e1"], tags=["tag"], turn_nr=1)
    data = _ser_result(r)
    r2   = _de_result(data)
    assert r2.id       == r.id
    assert r2.content  == "data"
    assert r2.entities == ["e1"]
    assert r2.tags     == ["tag"]
    assert r2.status   == PromotionStatus.FINAL


def test_ser_result_file_backed_clears_content() -> None:
    r = SessionResult(content="big payload", workspace_path="/tmp/file.txt")
    data = _ser_result(r)
    assert data["content"] == ""
    assert data["workspace_path"] == "/tmp/file.txt"


# ---------------------------------------------------------------------------
# SessionSnapshot to_dict / from_dict
# ---------------------------------------------------------------------------

def test_snapshot_to_dict_from_dict_roundtrip() -> None:
    d  = Directive(key="lang", value="de")
    t  = TurnSummary(turn_nr=1, user_input="hi", intent="greet", topic="t")
    r  = SessionResult(content="data")
    snap = SessionSnapshot(
        session_id="s1",
        directives=[d],
        turns=[t],
        results=[r],
        interrupted_task_ids=["task-1"],
        created_at=_now(),
        updated_at=_now(),
    )
    data  = snap.to_dict()
    snap2 = SessionSnapshot.from_dict(data)
    assert snap2.session_id                == "s1"
    assert len(snap2.directives)           == 1
    assert snap2.directives[0].key         == "lang"
    assert len(snap2.turns)                == 1
    assert snap2.turns[0].turn_nr          == 1
    assert len(snap2.results)              == 1
    assert snap2.interrupted_task_ids      == ["task-1"]


# ---------------------------------------------------------------------------
# FileSystemBackend
# ---------------------------------------------------------------------------

def test_filesystem_backend_save_and_load(tmp_path) -> None:
    backend = FileSystemBackend(base_dir=str(tmp_path))
    snap    = _snapshot("sess-1")
    backend.save(snap)
    loaded = backend.load("sess-1")
    assert loaded is not None
    assert loaded.session_id == "sess-1"


def test_filesystem_backend_load_missing_returns_none(tmp_path) -> None:
    backend = FileSystemBackend(base_dir=str(tmp_path))
    assert backend.load("nonexistent") is None


def test_filesystem_backend_exists(tmp_path) -> None:
    backend = FileSystemBackend(base_dir=str(tmp_path))
    assert backend.exists("s") is False
    backend.save(_snapshot("s"))
    assert backend.exists("s") is True


def test_filesystem_backend_delete(tmp_path) -> None:
    backend = FileSystemBackend(base_dir=str(tmp_path))
    backend.save(_snapshot("s"))
    assert backend.delete("s") is True
    assert backend.exists("s") is False


def test_filesystem_backend_delete_missing(tmp_path) -> None:
    backend = FileSystemBackend(base_dir=str(tmp_path))
    assert backend.delete("nonexistent") is False


def test_filesystem_backend_list_sessions(tmp_path) -> None:
    backend = FileSystemBackend(base_dir=str(tmp_path))
    backend.save(_snapshot("a"))
    backend.save(_snapshot("b"))
    sessions = backend.list_sessions()
    assert set(sessions) == {"a", "b"}


def test_filesystem_backend_sanitises_session_id(tmp_path) -> None:
    backend = FileSystemBackend(base_dir=str(tmp_path))
    snap    = _snapshot("session/with:special*chars")
    backend.save(snap)
    # Should not raise — special chars get replaced
    sessions = backend.list_sessions()
    assert len(sessions) == 1


def test_filesystem_backend_save_with_content(tmp_path) -> None:
    backend = FileSystemBackend(base_dir=str(tmp_path))
    d  = Directive(key="lang", value="fr")
    t  = TurnSummary(turn_nr=1, user_input="bonjour", intent="greet", topic="t")
    r  = SessionResult(content="some result", entities=["x"])
    snap = SessionSnapshot(
        session_id="full",
        directives=[d],
        turns=[t],
        results=[r],
        interrupted_task_ids=[],
        created_at=_now(),
        updated_at=_now(),
    )
    backend.save(snap)
    loaded = backend.load("full")
    assert loaded.directives[0].value == "fr"
    assert loaded.turns[0].user_input == "bonjour"
    assert loaded.results[0].content  == "some result"
