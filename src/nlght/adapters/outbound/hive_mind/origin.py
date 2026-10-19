# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Preserve content origin when a run writes session memory.

The run ledger supplies origin stamps to remembered turns, results, and atoms.
"""

from __future__ import annotations

from nlght.core.model.trust import PLATFORM_TEXT, ContentLedger, ContentOrigin


class LedgerStamp:
    """Mixed into a coordinator: ``bind_ledger()`` and the stamp it enables."""

    _ledger: ContentLedger | None = None

    def bind_ledger(self, ledger: ContentLedger) -> None:
        """Bind the run's origin ledger to later memory writes.

        Args:
            ledger (ContentLedger): Content ledger for the active run.

        Side Effects:
            Replaces the coordinator's bound ledger.
        """
        self._ledger = ledger

    def _stamp(self, *inherited: ContentOrigin | None) -> ContentOrigin | None:
        parts = [origin for origin in inherited if origin is not None]
        if self._ledger is not None:
            run = self._ledger.origin
            # A run that read nothing wrote from deployment code alone.
            parts.append(run if run.sources else PLATFORM_TEXT)
        return ContentOrigin.combine(*parts) if parts else None
