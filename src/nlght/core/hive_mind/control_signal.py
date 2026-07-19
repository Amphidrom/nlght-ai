# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""check_control_signals — evaluates control signals from Observe output.

Decision priority:
    1. Stop intent  → ControlSignalDecision(should_stop=True)
    2. Scope change → note scope, continue
    3. Priority boost → note tool boost, continue
    4. No control signal → pass_through

Usage::

    decision = check_control_signals(
        classifications = classifications,  # list of SignalClassification-like objects
        mental_model    = mental_model,
        cid             = cid,
    )
"""

from __future__ import annotations

import logging
from typing import Any

from nlght.core.hive_mind.models import ControlSignalDecision

logger = logging.getLogger(__name__)

_STOP_INTENTS: frozenset[str] = frozenset({
    "stop",
    "cancel",
    "abort",
    "reset",
    "clear",
})

_PRIORITY_INTENT_TO_TOOL: dict[str, str] = {
    "search":   "web_search",
    "retrieve": "knowledge_retrieval",
    "execute":  "code_executor",
    "write":    "file_writer",
}

_CONTROL_SIGNAL_TYPE = "control"


def check_control_signals(
    classifications: list[Any],
    mental_model:    object,
    cid:             str,
) -> ControlSignalDecision:
    """Evaluate control signals from a list of SignalClassification-like objects.

    Parameters:
        classifications: Sub-signals from ObserveStep.
        mental_model:    Current MentalModel (typed as Any to avoid circular imports).
        cid:             Correlation ID for logging.
    """
    stop_reason:    str | None = None
    priority_boost: str | None = None
    scope_change:   str | None = None
    found_control:  bool       = False

    for sc in classifications:
        signal_type = getattr(sc, "type", None)
        # sc is duck-typed Any (avoids a circular import) -- hasattr() guards this
        # at runtime but mypy can't narrow Any | None through it.
        type_value  = signal_type.value if hasattr(signal_type, "value") else str(signal_type)  # type: ignore[union-attr]

        if type_value != _CONTROL_SIGNAL_TYPE and "control" not in type_value:
            continue

        found_control = True
        metadata    = getattr(sc, "metadata", None)
        intent      = getattr(metadata, "intent", None)
        intent_value = intent.value if hasattr(intent, "value") else str(intent)  # type: ignore[union-attr]

        logger.debug("[%s] control_signal: found | intent=%s", cid, intent_value)

        if intent_value in _STOP_INTENTS:
            stop_reason = intent_value
            logger.info("[%s] control_signal: STOP | reason=%s", cid, stop_reason)
            return ControlSignalDecision(
                should_stop  = True,
                stop_reason  = stop_reason,
                pass_through = False,
            )

        scope = _extract_scope_change(sc)
        if scope:
            scope_change = scope
            logger.info("[%s] control_signal: scope_change → %s", cid, scope_change)

        if intent_value in _PRIORITY_INTENT_TO_TOOL:
            priority_boost = _PRIORITY_INTENT_TO_TOOL[intent_value]
            logger.info("[%s] control_signal: priority_boost → %s", cid, priority_boost)

    if not found_control:
        logger.debug("[%s] control_signal: no control signal → pass_through", cid)
        return ControlSignalDecision(pass_through=True)

    if not stop_reason and not priority_boost and not scope_change:
        logger.debug("[%s] control_signal: no actionable directive → pass_through", cid)
        return ControlSignalDecision(pass_through=True)

    decision = ControlSignalDecision(
        should_stop    = False,
        priority_boost = priority_boost,
        scope_change   = scope_change,
        pass_through   = False,
    )
    logger.info("[%s] control_signal: %r", cid, decision)
    return decision


def _extract_scope_change(sc: object) -> str | None:
    entities: list[str] = getattr(sc, "entities", [])
    for e in entities:
        e_lower = e.lower()
        if e_lower.startswith("scope:"):
            return e[len("scope:"):]
        if "scope" in e_lower and "=" in e:
            _, _, val = e.partition("=")
            return val.strip()
    return None
