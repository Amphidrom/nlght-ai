# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.adapters.outbound.hive_mind.frozen import FrozenStoreCoordinator
from nlght.core.hive_mind.models import AtomType, Directive, SessionResult, TurnSummary, WorkingAtom, WriteIntent


class _Inner:
    def __init__(self) -> None:
        self.directives = {"lang": "de"}
        self.directive_objects = [Directive(key="lang", value="de")]
        self.turns = [TurnSummary(turn_nr=1, user_input="hi", intent="greet", topic="t", entities=["python"])]
        self.context = {"active_atoms": [], "known_results": []}
        self.slots = {"inner": "value"}
        self.payload = SessionResult(content="inner payload", entities=["python"], tags=["payload"])

    def get_directives(self):
        return self.directives

    def get_directives_list(self):
        return self.directive_objects

    def get_recent_turns(self, n):
        return self.turns[-n:]

    def get_context(self, entities):
        return self.context

    def is_known(self, entities, tags=None):
        return entities == ["python"]

    def resolve_reference(self, ref):
        return self.turns[0] if ref == "python" else None

    def get_payload(self, entities):
        return self.payload if entities == ["python"] else None

    def get_slot(self, key, default_factory):
        return self.slots.get(key, default_factory())


def test_frozen_reads_delegate_to_inner_and_writes_do_not_mutate_inner() -> None:
    inner = _Inner()
    frozen = FrozenStoreCoordinator(inner)

    assert frozen.set_directive("lang", "en") is None
    frozen.record_turn(TurnSummary(turn_nr=2, user_input="x", intent="x", topic="x"))

    assert frozen.get_directives() == {"lang": "de"}
    assert frozen.get_directives_list() == inner.directive_objects
    assert frozen.get_recent_turns(5) == inner.turns
    assert frozen.get_context(["python"]) is inner.context
    assert frozen.is_known(["python"]) is True
    assert frozen.resolve_reference("python") == inner.turns[0]


def test_frozen_branch_promotion_and_store_are_shadow_only() -> None:
    frozen = FrozenStoreCoordinator(_Inner())
    frozen.open_branch("turn")
    frozen.open_branch("phase")
    atom_id, conflict = frozen.write(WriteIntent(
        atom_type=AtomType.RESULT,
        content="shadow result",
        task_id="phase",
        tags=["kind:shadow"],
        promote_immediately=True,
    ))

    assert atom_id
    assert conflict is None
    phase_promoted = frozen.close_branch(promote=True)
    assert [atom.content for atom in phase_promoted] == ["shadow result"]
    assert phase_promoted[0].task_id == "turn"

    turn_promoted = frozen.close_branch(promote=True)
    stored = frozen.store_promoted_atoms(turn_promoted, turn_nr=7, entities=["shadow"])

    assert [result.content for result in stored] == ["shadow result"]
    assert stored[0].entities == ["shadow"]
    assert frozen.close_branch(promote=True) == []


def test_frozen_discard_and_fail_clear_shadow_task_atoms() -> None:
    frozen = FrozenStoreCoordinator(_Inner())
    frozen.start_task("task")
    frozen.write(WriteIntent(atom_type=AtomType.RESULT, content="discard me", task_id="task"))

    assert frozen.finish_task("task", entities=[], turn_nr=1) == []
    assert frozen.close_branch(promote=True) == []

    frozen.write(WriteIntent(atom_type=AtomType.RESULT, content="orphan", task_id="other"))
    frozen.discard_task("other")
    frozen.write(WriteIntent(atom_type=AtomType.RESULT, content="orphan", task_id="other"))
    frozen.fail_task("other")
    assert frozen.close_branch(promote=True) == []


def test_frozen_payload_and_slots_are_shadow_first() -> None:
    inner = _Inner()
    frozen = FrozenStoreCoordinator(inner)

    assert frozen.get_slot("inner", str) == "value"
    assert frozen.get_slot("missing", lambda: "default") == "default"
    frozen.set_slot("inner", "shadow")
    assert frozen.get_slot("inner", str) == "shadow"
    assert inner.slots["inner"] == "value"
    assert frozen.get_payload(["python"]).content == "inner payload"

    result_id, stored = frozen.promote_payload("shadow payload", entities=["python"], tags=["tag"], turn_nr=3)

    assert result_id
    assert stored is True
    assert frozen.get_payload(["python"]).content == "shadow payload"
    assert frozen.get_payload(["missing"]) is None


def test_frozen_store_promoted_atoms_ignores_non_storable_types() -> None:
    frozen = FrozenStoreCoordinator(_Inner())
    stored = frozen.store_promoted_atoms(
        [WorkingAtom(atom_type=AtomType.CONTEXT, content="ctx", task_id="t", promote_to_parent=True)],
        turn_nr=1,
        entities=["x"],
    )

    assert stored == []
