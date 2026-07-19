# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import ClassVar

from nlght.core.tools.tool import ToolBase, ToolParameter, ToolSignature


class ActivatePlaybookTool(ToolBase):
    """Built-in control tool for playbook selection.

    The step loop intercepts calls to this tool before execute() is reached.
    activate() is a no-op signal — playbook phase execution is driven by the
    step that detects the tool call name.
    """

    KIND: ClassVar[str] = "activate_playbook"
    PROVIDER: ClassVar[str] = "platform"

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [
            ToolSignature(
                name="activate_playbook",
                description=(
                    "Select and activate a playbook before calling any task tool. "
                    "Call this first when the request matches an available playbook. "
                    "Use name='none' if no playbook matches."
                ),
                method_name="activate",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description=(
                            "Playbook name to activate (from the available playbooks in the system prompt), "
                            "or 'none' if no playbook matches."
                        ),
                        required=True,
                    ),
                    ToolParameter(
                        name="reason",
                        type="string",
                        description="Brief reason for the selection.",
                        required=False,
                    ),
                ],
            )
        ]

    async def activate(self, *, name: str, reason: str = "") -> str:
        return f"playbook={name}"
