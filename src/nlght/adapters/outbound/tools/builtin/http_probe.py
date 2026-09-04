# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging
import shlex
import uuid
from typing import TYPE_CHECKING, ClassVar

from nlght.adapters.outbound.tools.builtin.action_semantics import (
    HTTP_METHOD_BINDER,
    HTTP_PROBE,
)
from nlght.core.errors.errors import ToolExecutionError
from nlght.core.tools.tool import ToolBase, ToolParameter, ToolSignature

if TYPE_CHECKING:
    from nlght.ports.outbound.os_runtime import OsRuntime

logger = logging.getLogger(__name__)

# Probe script runs inside the pentest container via os_runtime.
# Accepts a JSON params file path as argv[1], outputs a JSON result to stdout.
_PROBE_SCRIPT = """\
import json, sys, urllib.request, urllib.error, ssl

_BINARY_PREFIXES = (
    "image/", "video/", "audio/",
    "application/octet-stream", "application/zip", "application/gzip",
    "application/pdf", "application/wasm", "application/x-bzip",
    "application/x-tar",
)

with open(sys.argv[1]) as _f:
    params = json.load(_f)

url              = params["url"]
method           = params.get("method", "GET").upper()
req_headers      = params.get("headers") or {}
body_str         = params.get("body") or None
max_body         = int(params.get("max_body", 4096))
follow_redirects = bool(params.get("follow_redirects", True))

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):
        return None


handlers = [urllib.request.HTTPSHandler(context=ctx)]
if not follow_redirects:
    handlers.append(_NoRedirect())
opener = urllib.request.build_opener(*handlers)

req = urllib.request.Request(
    url, method=method, headers=req_headers,
    data=body_str.encode() if body_str else None,
)

result = {"url": url}
try:
    with opener.open(req, timeout=15) as resp:
        status = resp.status
        reason = resp.reason
        hdrs   = dict(resp.headers)
        raw    = resp.read(max_body + 1)
except urllib.error.HTTPError as e:
    status = e.code
    reason = e.reason
    hdrs   = dict(e.headers)
    raw    = e.read(max_body + 1) if hasattr(e, "read") else b""
except Exception as exc:
    print(json.dumps({"error": str(exc), "url": url}))
    sys.exit(0)

ct = hdrs.get("Content-Type", hdrs.get("content-type", "")).lower().split(";")[0].strip()
is_binary = ct.startswith(_BINARY_PREFIXES) or (len(raw) >= 16 and raw[:256].count(0) > 5)

result.update({"status_code": status, "reason": reason, "headers": hdrs, "body_bytes": len(raw)})

if is_binary:
    result["body_binary"]  = True
    result["content_type"] = ct
else:
    text = raw.decode("utf-8", errors="replace")
    if len(text) <= max_body:
        result["body"] = text
    else:
        result["body_preview"]     = text[:max_body]
        result["body_truncated"]   = True
        result["body_total_chars"] = len(text)

print(json.dumps(result))
"""


class HttpProbeTool(ToolBase):
    """Structured HTTP endpoint probe that runs inside the pentest container.

    Unlike shell_exec(curl -si ...), binary response bodies are never written
    into the LLM context — they are summarised (status, content-type, size).
    Text bodies beyond max_body_bytes are previewed with a truncation indicator.

    Requires an OsRuntime (runs the probe as a Python script in the container
    so network access uses the same context as all other pentest tools).

    Configuration:
      ``max_body_bytes`` — max body text to include inline (default 4096)
      ``timeout_s``      — outer operation timeout in seconds (default 20)
    """

    KIND: ClassVar[str]     = "http_probe"
    PROVIDER: ClassVar[str] = "local"

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [
            ToolSignature(
                name="http_probe",
                description=(
                    "Probe an HTTP endpoint and return structured result: "
                    "status_code, response headers, and a safe body representation. "
                    "Binary bodies (images, archives, …) are auto-summarised — never "
                    "floods context. Use for baseline probes instead of shell_exec(curl -si …)."
                ),
                method_name="probe",
                parameters=[
                    ToolParameter(
                        name="url",
                        type="string",
                        description="Full URL to probe (http:// or https://).",
                    ),
                    ToolParameter(
                        name="method",
                        type="string",
                        description="HTTP method: GET POST PUT DELETE HEAD OPTIONS PATCH. Default: GET.",
                        required=False,
                    ),
                    ToolParameter(
                        name="headers",
                        type="object",
                        description="Additional request headers as key-value pairs.",
                        required=False,
                    ),
                    ToolParameter(
                        name="body",
                        type="string",
                        description="Request body string (for POST/PUT/PATCH).",
                        required=False,
                    ),
                    ToolParameter(
                        name="follow_redirects",
                        type="boolean",
                        description="Follow HTTP redirects. Default: true.",
                        required=False,
                    ),
                ],
                action=HTTP_PROBE,
                argument_binder=HTTP_METHOD_BINDER,
            ),
        ]

    def _runtime(self) -> OsRuntime:
        if self.os_runtime is None:
            raise ToolExecutionError(
                f"HttpProbeTool '{self.name}' requires an OsRuntime but none was injected."
            )
        return self.os_runtime

    def _max_body(self) -> int:
        return int(self.config.get("max_body_bytes", 4096))

    async def probe(
        self,
        *,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        body: str | None = None,
        follow_redirects: bool = True,
    ) -> str:
        if not url.startswith(("http://", "https://")):
            return json.dumps({"error": "url must start with http:// or https://"})

        runtime  = self._runtime()
        max_body = self._max_body()
        # Unique per call → concurrent probes of the same URL never collide.
        token    = uuid.uuid4().hex

        params = {
            "url":              url,
            "method":           method,
            "headers":          headers or {},
            "body":             body,
            "follow_redirects": follow_redirects,
            "max_body":         max_body,
        }
        # Consistent with the rest of the pipeline: scratch lives under /workspace/tmp.
        params_path = f"/workspace/tmp/nlght/probe_params_{token}.json"
        script_path = f"/workspace/tmp/nlght/probe_{token}.py"

        try:
            await runtime.write_text(params_path, json.dumps(params))
            await runtime.write_text(script_path, _PROBE_SCRIPT)
            # Run through bash so the container BASH_ENV command logger records it.
            _exit, stdout, stderr = await runtime.exec(
                ["bash", "-c", f"python3 {shlex.quote(script_path)} {shlex.quote(params_path)}"],
            )
        except Exception as exc:
            logger.warning("http_probe.exec_failed | url=%s error=%s", url, exc)
            return json.dumps({"error": f"probe execution failed: {exc}", "url": url})
        finally:
            # Best-effort cleanup — scratch files would otherwise pile up on the
            # workspace volume across probes.
            for _p in (script_path, params_path):
                try:
                    await runtime.delete(_p)
                except Exception:
                    pass

        raw = stdout.strip()
        if not raw:
            return json.dumps({
                "error": f"probe script produced no output (exit={_exit})",
                "stderr": stderr[:500],
                "url": url,
            })

        try:
            parsed = json.loads(raw)
            logger.info(
                "http_probe.ok | url=%s status=%s binary=%s",
                url,
                parsed.get("status_code", "?"),
                parsed.get("body_binary", False),
            )
        except json.JSONDecodeError:
            logger.warning("http_probe.parse_error | url=%s raw=%s", url, raw[:200])

        return raw
