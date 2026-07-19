# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nlght.core.model.budget import TokenBudget

logger = logging.getLogger(__name__)

_CHARS_PER_TOKEN = 3.5
_CHUNK_THRESHOLD_RATIO = 0.85

OUTPUT_HEADROOM = 2048
CHUNK_TARGET_TOKENS = 3000
CHUNK_ACK_MAX_TOKENS = 32

CHUNK_SYSTEM_INJECTION = """

[INCREMENTAL MODE ACTIVE]
You will now receive the input material in multiple parts.
For each part:
- Store the information internally.
- Respond ONLY with: "Chunk received and integrated."
- Do NOT summarize, analyze, or interpret yet.
- Do NOT produce any structured output or JSON yet.
- Wait for more input.

When you receive the message "FINALIZE", you will receive the actual task.
At that point — and only then — produce your complete response.
"""

FINALIZE_PREFIX = (
    "FINALIZE\n\n"
    "You have now seen all parts of the input material.\n"
    "Ignore all your previous chunk-phase responses.\n"
    "Produce the final output as if all input had been provided at once.\n"
    "Do not reference chunking or parts in your response.\n\n"
)


def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def total_message_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_tokens(m.get("content") or "") for m in messages)


def needs_chunking(messages: list[dict[str, Any]], token_budget: TokenBudget) -> bool:
    total = total_message_tokens(messages)
    limit = token_budget.total
    ratio = total / limit if limit > 0 else 0
    if ratio > _CHUNK_THRESHOLD_RATIO:
        logger.info(
            "llm.auto_chunk.triggered | est_tokens=%d context_window=%d ratio=%.2f",
            total, limit, ratio,
        )
        return True
    return False


def apply_budget_to_messages(
    messages: list[dict[str, Any]],
    token_budget: TokenBudget,
) -> tuple[list[dict[str, Any]], int]:
    if not messages:
        return messages, 0

    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    system_tokens = sum(estimate_tokens(m.get("content") or "") for m in system_msgs)
    available = token_budget.get_available_for_input(system_prompt_tokens=system_tokens)

    if not other_msgs:
        return messages, system_tokens

    last_msg = other_msgs[-1]
    middle = other_msgs[:-1]
    last_tokens = estimate_tokens(last_msg.get("content") or "")
    remaining = available - last_tokens

    kept_middle: list[dict[str, Any]] = []
    skipped = 0
    for msg in middle:
        t = estimate_tokens(msg.get("content") or "")
        if remaining - t >= 0:
            kept_middle.append(msg)
            remaining -= t
        else:
            skipped += 1
            logger.warning(
                "llm.budget_trim | role=%s tokens_est=%d available_left=%d",
                msg.get("role"), t, remaining,
            )

    if skipped:
        logger.info(
            "llm.budget_trim_summary | skipped=%d available=%d last_tokens=%d",
            skipped, available, last_tokens,
        )

    filtered = system_msgs + kept_middle + [last_msg]
    total = system_tokens + sum(estimate_tokens(m.get("content") or "") for m in kept_middle) + last_tokens
    return filtered, total


def _split_message_content(content: str, target_tokens: int) -> list[str]:
    target_chars = int(target_tokens * _CHARS_PER_TOKEN)
    if len(content) <= target_chars:
        return [content]

    chunks: list[str] = []
    current = ""
    for para in content.split("\n\n"):
        if len(current) + len(para) + 2 > target_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n\n" + para if current else para
    if current.strip():
        chunks.append(current.strip())

    refined: list[str] = []
    for chunk in chunks:
        if len(chunk) <= target_chars:
            refined.append(chunk)
        else:
            sub = ""
            for line in chunk.split("\n"):
                if len(sub) + len(line) + 1 > target_chars and sub:
                    refined.append(sub.strip())
                    sub = line
                else:
                    sub = sub + "\n" + line if sub else line
            if sub.strip():
                refined.append(sub.strip())

    final: list[str] = []
    for chunk in refined:
        if len(chunk) <= target_chars:
            final.append(chunk)
        else:
            for start in range(0, len(chunk), target_chars):
                final.append(chunk[start: start + target_chars])
    return final


def prepare_chunked_session(
    messages: list[dict[str, Any]],
    token_budget: TokenBudget,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    if other_msgs:
        task_msg: dict[str, Any] = other_msgs[-1]
        middle = other_msgs[:-1]
    else:
        task_msg = {"role": "user", "content": "FINALIZE"}
        middle = []

    if system_msgs:
        augmented_system = [
            {"role": sm["role"], "content": sm.get("content", "") + CHUNK_SYSTEM_INJECTION}
            for sm in system_msgs
        ]
    else:
        augmented_system = [{"role": "system", "content": CHUNK_SYSTEM_INJECTION.strip()}]

    combined_parts = [
        f"[{m.get('role', 'user').upper()}]\n{m.get('content', '')}"
        for m in middle
        if (m.get("content") or "").strip()
    ]
    combined = "\n\n---\n\n".join(combined_parts)
    content_chunks = _split_message_content(combined, CHUNK_TARGET_TOKENS) if combined.strip() else []

    task_content = task_msg.get("content") or ""
    if estimate_tokens(task_content) > CHUNK_TARGET_TOKENS * 2:
        task_parts = _split_message_content(task_content, CHUNK_TARGET_TOKENS)
        for part in task_parts[:-1]:
            content_chunks.append(f"[TASK CONTEXT]\n{part}")
        task_msg = {"role": task_msg.get("role", "user"), "content": task_parts[-1]}
        logger.info(
            "llm.auto_chunk.task_split | original_tokens=%d task_chunks=%d",
            estimate_tokens(task_content), len(task_parts) - 1,
        )

    logger.info(
        "llm.auto_chunk.prepared | system=%d middle=%d chunks=%d task_tokens=%d",
        len(system_msgs), len(middle), len(content_chunks),
        estimate_tokens(task_msg.get("content") or ""),
    )
    return augmented_system, content_chunks, task_msg
