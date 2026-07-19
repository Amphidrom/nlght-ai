# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

SubjectType = Literal["tool", "model", "playbook"]
Effect = Literal["allow", "deny"]


@dataclass(slots=True, frozen=True)
class AccessRule:
    """One access-policy rule (row in ``access_policies``).

    ``subject`` is a glob matched against the subject's name (tool resource
    name, model name, or playbook name). ``conditions`` maps condition keys to
    lists of glob patterns — all keys must match (AND), any value per key
    suffices (OR). Supported keys:

    - ``model`` — the effective model of the request
    - ``header:<name>`` — a request header (name compared case-insensitively)
    - ``client_host`` — the caller's host

    An empty ``conditions`` mapping matches unconditionally.

    Decision semantics (see RuleBasedAccessPolicy): no enabled rule matching a
    subject → access allowed; otherwise the highest-priority rule whose
    conditions match decides (deny sorts before allow on equal priority), and
    if no matching rule's conditions are satisfied access is denied
    (allowlist semantics).
    """

    rule_id: uuid.UUID
    subject_type: SubjectType
    subject: str
    effect: Effect
    conditions: dict[str, list[str]] = field(default_factory=dict)
    priority: int = 0
    enabled: bool = True
