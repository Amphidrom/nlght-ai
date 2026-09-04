# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import ast
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from nlght.core.ingestion import DocumentClassification, Enrichment

_FQN = re.compile(r"\b[a-zA-Z_][\w]*(?:\.[A-Za-z_][\w]*)+\b")
_TS_IMPORT = re.compile(r"(?:import|export).*?from\s+[\"']([^\"']+)[\"']|require\(\s*[\"']([^\"']+)[\"']\s*\)")
_TS_SYMBOL = re.compile(r"(?:class|function|interface|type|enum)\s+([A-Za-z_][\w]*)")
_JAVA_PACKAGE = re.compile(r"\bpackage\s+([\w.]+)\s*;")
_JAVA_IMPORT = re.compile(r"\bimport\s+(?:static\s+)?([\w.*]+)\s*;")
_JAVA_SYMBOL = re.compile(r"\b(?:class|interface|enum|record)\s+([A-Za-z_][\w]*)")
_HEADING = re.compile(r"^\s*(?:#{1,6}|={1,6})\s+(.+?)\s*$", re.MULTILINE)
_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)|xref:([^\[]+)\[")
_PROTO_PACKAGE = re.compile(r"\bpackage\s+([\w.]+)\s*;")
_PROTO_IMPORT = re.compile(r"\bimport\s+[\"']([^\"']+)[\"']")
_PROTO_SYMBOL = re.compile(r"\b(?:message|enum|service)\s+([A-Za-z_][\w]*)")
_GRADLE_REF = re.compile(r"[\"']([\w.-]+:[\w.-]+)(?::[^\"']+)?[\"']|project\(\s*[\"'](:[^\"']+)[\"']")
_GRADLE_SYMBOL = re.compile(r"\b(?:task\s+|tasks\.register\(\s*[\"'])([A-Za-z0-9_]+)")
_VUE_COMPONENT = re.compile(r"<([A-Z][A-Za-z0-9_]*)\b")


def _sorted(values: set[str]) -> tuple[str, ...]:
    return tuple(sorted(value for value in values if value))


class BuiltinDocumentEnricher:
    VERSION = "builtin-v1"

    def enrich(
        self,
        *,
        path: Path,
        text: str,
        classification: DocumentClassification,
    ) -> Enrichment:
        language = classification.language
        handler = {
            "asciidoc": self._documentation,
            "gradle": self._gradle,
            "gradle.kts": self._gradle,
            "java": self._java,
            "javascript": self._typescript,
            "js": self._typescript,
            "json": self._json,
            "jsx": self._typescript,
            "markdown": self._documentation,
            "md": self._documentation,
            "protobuf": self._protobuf,
            "python": self._python,
            "py": self._python,
            "text": self._generic,
            "tsx": self._typescript,
            "typescript": self._typescript,
            "vue": self._vue,
            "xml": self._xml,
        }.get(language, self._generic)
        return handler(path, text, classification)

    def _result(
        self,
        name: str,
        *,
        declaration: str | None = None,
        tags: set[str] | None = None,
        symbols: set[str] | None = None,
        references: set[str] | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Enrichment:
        return Enrichment(
            enricher=name,
            version=self.VERSION,
            declaration=declaration,
            tags=_sorted(tags or set()),
            symbols=_sorted(symbols or set()),
            references=_sorted(references or set()),
            attributes=attributes or {},
        )

    def _generic(self, path: Path, text: str, classification: DocumentClassification) -> Enrichment:
        return self._result(
            "generic",
            declaration=".".join(path.with_suffix("").parts),
            tags={classification.kind, classification.language},
            references=set(_FQN.findall(text)),
        )

    def _typescript(self, path: Path, text: str, classification: DocumentClassification) -> Enrichment:
        refs = {left or right for left, right in _TS_IMPORT.findall(text)}
        symbols = set(_TS_SYMBOL.findall(text))
        tags = {"code", "module", "javascript" if classification.language in {"javascript", "js", "jsx"} else "typescript"}
        return self._result("typescript", declaration=".".join(path.with_suffix("").parts), tags=tags, symbols=symbols, references=refs)

    def _python(self, path: Path, text: str, _classification: DocumentClassification) -> Enrichment:
        tags = {"python", "code", "module"}
        symbols: set[str] = set()
        references: set[str] = set()
        parsed = True
        try:
            tree = ast.parse(text)
            symbols = {node.name for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    references.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    references.add(node.module)
        except SyntaxError:
            parsed = False
        return self._result(
            "python", declaration=".".join(path.with_suffix("").parts), tags=tags, symbols=symbols, references=references, attributes={"parsed": parsed}
        )

    def _java(self, _path: Path, text: str, _classification: DocumentClassification) -> Enrichment:
        package_match = _JAVA_PACKAGE.search(text)
        package = package_match.group(1) if package_match else None
        symbols = set(_JAVA_SYMBOL.findall(text))
        declaration = f"{package}.{sorted(symbols)[0]}" if package and symbols else (sorted(symbols)[0] if symbols else None)
        return self._result(
            "java", declaration=declaration, tags={"java", "code"}, symbols=symbols, references=set(_JAVA_IMPORT.findall(text)), attributes={"package": package}
        )

    def _documentation(self, path: Path, text: str, classification: DocumentClassification) -> Enrichment:
        refs = {left or right for left, right in _LINK.findall(text)}
        refs.update(_FQN.findall(text))
        headings = set(_HEADING.findall(text))
        return self._result(
            classification.language, declaration=f"doc.{path.stem}", tags={"doc", "text", classification.language}, symbols=headings, references=refs
        )

    def _protobuf(self, _path: Path, text: str, _classification: DocumentClassification) -> Enrichment:
        package_match = _PROTO_PACKAGE.search(text)
        package = package_match.group(1) if package_match else None
        symbols = set(_PROTO_SYMBOL.findall(text))
        declaration = f"{package}.{sorted(symbols)[0]}" if package and symbols else (sorted(symbols)[0] if symbols else None)
        return self._result(
            "protobuf",
            declaration=declaration,
            tags={"protobuf", "schema", "code"},
            symbols=symbols,
            references=set(_PROTO_IMPORT.findall(text)),
            attributes={"package": package},
        )

    def _vue(self, path: Path, text: str, _classification: DocumentClassification) -> Enrichment:
        symbols = {path.stem, *_VUE_COMPONENT.findall(text)}
        refs = {left or right for left, right in _TS_IMPORT.findall(text)}
        refs.update(_VUE_COMPONENT.findall(text))
        return self._result("vue", declaration=f"vue.{path.stem}", tags={"vue", "component", "ui", "code"}, symbols=symbols, references=refs)

    def _xml(self, path: Path, text: str, _classification: DocumentClassification) -> Enrichment:
        tags = {"xml", "document", "structured"}
        symbols: set[str] = set()
        references: set[str] = set()
        parsed = True
        try:
            root = ET.fromstring(text)
            for element in root.iter():
                symbols.add(element.tag.split("}", 1)[-1])
                symbols.update(element.attrib)
                references.update(_FQN.findall(" ".join(element.attrib.values())))
                if element.text:
                    references.update(_FQN.findall(element.text))
        except ET.ParseError:
            parsed = False
        return self._result("xml", declaration=f"xml.{path.stem}", tags=tags, symbols=symbols, references=references, attributes={"parsed": parsed})

    def _gradle(self, path: Path, text: str, _classification: DocumentClassification) -> Enrichment:
        refs = {left or right for left, right in _GRADLE_REF.findall(text)}
        symbols = {f"task:{value}" for value in _GRADLE_SYMBOL.findall(text)}
        return self._result("gradle", declaration=f"gradle.{path.parent.name}", tags={"gradle", "build"}, symbols=symbols, references=refs)

    def _json(self, path: Path, text: str, _classification: DocumentClassification) -> Enrichment:
        symbols: set[str] = set()
        references: set[str] = set()
        parsed = True

        def walk(value: object) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    symbols.add(str(key))
                    if key in {"class", "service", "module", "component", "ref", "type"} and isinstance(child, str):
                        references.add(child)
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
            elif isinstance(value, str):
                references.update(_FQN.findall(value))

        try:
            walk(json.loads(text))
        except json.JSONDecodeError:
            parsed = False
        return self._result(
            "json", declaration=f"json.{path.stem}", tags={"json", "data", "document"}, symbols=symbols, references=references, attributes={"parsed": parsed}
        )
