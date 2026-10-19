# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""One control, one size, wherever it sits.

Pico sizes a checkbox `1.25em` square, which makes it two different things at
once. `em` ties it to whatever font-size it lands in, and a checkbox inside a
flex label is a flex item that `flex-shrink` squeezes narrower than it is tall
as soon as the label's text runs long. Both happen in this UI, so checkboxes on
the step form came out visibly smaller than the ones on the workflow form.

Asserted as structure rather than as the text of a rule: what would bring the
bug back is a template pinning its own size on a checkbox, not the wording of
the stylesheet.
"""

from __future__ import annotations

import re
from pathlib import Path

_ADMIN = Path(__file__).resolve().parents[6] / "src" / "nlght" / "adapters" / "inbound" / "http" / "admin"
_TEMPLATES = _ADMIN / "templates"

_CHECKBOX = re.compile(r"<input[^>]*type=\"(?:checkbox|radio)\"[^>]*>")


def _templates() -> list[Path]:
    found = sorted(_TEMPLATES.rglob("*.html"))
    assert found, "no admin templates found; the path above is wrong"
    return found


def test_the_size_is_fixed_to_the_page_and_not_to_the_neighbourhood() -> None:
    base = (_TEMPLATES / "base.html").read_text(encoding="utf-8")

    rule = re.search(
        r"input\[type=\"checkbox\"\][^{]*\{([^}]*)\}", base, re.S
    )
    assert rule is not None, "no shared size rule for checkboxes"
    body = rule.group(1)

    # `rem`, because `em` is what tied the size to the surrounding font-size.
    assert "rem" in body and re.search(r"\d(?<!r)em\b", body) is None
    # `flex: none`, because a flex label squeezed it narrower than it was tall.
    assert "flex" in body


def test_no_template_pins_its_own_checkbox_size() -> None:
    """Keep checkbox sizing in the shared form style."""
    offenders = [
        (path.relative_to(_TEMPLATES).as_posix(), tag)
        for path in _templates()
        for tag in _CHECKBOX.findall(path.read_text(encoding="utf-8"))
        if re.search(r"(?:width|height|font-size)\s*:", tag)
    ]

    assert offenders == []
