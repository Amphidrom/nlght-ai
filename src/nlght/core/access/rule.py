# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

#: What a subject of each type is named by.
#:
#:     resource   a whole configured ResourceDef, by its address
#:     tool       one callable signature of one such resource
#:     model      a model name
#:     playbook   a playbook name
#:
#: `resource` and `tool` used to be one type doing both jobs: `subject_type`
#: said "tool" and the value compared was `resource.name`, so a rule that read
#: like "this tool may be called" in fact authorized an entire resource — every
#: one of its operations, and its use inside a workflow step where no tool call
#: happens at all. The two are separated because they are separately decidable:
#: a resource may be usable by a workflow while only some of its operations are
#: offered to a model.
SubjectType = Literal["resource", "tool", "model", "playbook"]
Effect = Literal["allow", "deny"]


@dataclass(slots=True, frozen=True)
class AccessRule:
    """One access-policy rule (row in ``access_policies``).

    ``subject`` is a glob matched against the subject's name, which is:

        resource   ``<kind>/<name>``                    — a ResourceDef's address
        tool       ``<kind>/<name>::<signature>``       — one of its signatures
        model      the model name
        playbook   the playbook name

    The two data-side forms are the resource's address (`ResourceDef.address`),
    with the signature appended for a tool. Globbing then falls out of the shape:
    ``data_store/*`` is every data store, ``data_store/*::data-search-keyword``
    is that one operation wherever it is offered.

    A tool subject is *not* the operation name a model calls. The model still
    calls ``data-search``; the policy names ``data_store/data-main::data-search``.
    Two namespaces, deliberately: the short name is an API, the qualified one has
    to distinguish two resources that offer the same operation.

    ``conditions`` maps condition keys to lists of glob patterns — all keys must
    match (AND), any value per key suffices (OR). Supported keys:

    - ``model`` — the effective model of the request
    - ``header:<name>`` — a request header (name compared case-insensitively)
    - ``client_host`` — the caller's host
    - ``workflow`` — the name of the workflow being executed, which restricts a
      subject to specific flows. It is the workflow the executor actually
      resolved and is running, not one a request claimed. A caller outside any
      workflow has no workflow name, so such a rule never matches it — and under
      allowlist semantics that means denied, deliberately, rather than a bypass
      for anything that runs outside a flow.

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


#: Separates a resource's address from the signature it offers.
#:
#: Two colons rather than one, because a kind may not contain "/" but nothing
#: has ever forbidden a ":" in a resource name — and a separator a name can
#: contain is a separator that eventually splits in the wrong place.
TOOL_SEPARATOR = "::"


def tool_subject(resource_address: str, signature_name: str) -> str:
    """The policy name of one callable signature of one resource.

    Formatted here and nowhere else. The catalog builds it to ask, an operator
    writes it in a rule, an error message prints it — three producers of one
    string, which is two more than can be kept in agreement by hand.

    Takes the address rather than a `ResourceDef` so that `core.access` does not
    depend on the runtime's resource type; a rule is about names, and it should
    not need to know what a resource is in order to name one.
    """
    return f"{resource_address}{TOOL_SEPARATOR}{signature_name}"
