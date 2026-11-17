"""Bank registration and the file identity shared by the RAG stores."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


BANK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
BankType = Literal["obsidian", "repo"]


@dataclass(frozen=True, slots=True)
class FileRef:
    bank: str
    path: str

    def __post_init__(self) -> None:
        if not isinstance(self.bank, str) or not BANK_ID.fullmatch(self.bank):
            raise ValueError("invalid bank ID")
        if (
            not isinstance(self.path, str)
            or "\\" in self.path
            or "\x00" in self.path
            or re.match(r"^[A-Za-z]:/", self.path)
            or any(part in {"", ".", ".."} for part in self.path.split("/"))
        ):
            raise ValueError("file path must be a canonical bank-relative path")

    def to_key(self) -> str:
        return json.dumps([self.bank, self.path], ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_key(cls, key: str) -> FileRef:
        pair = json.loads(key)
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("invalid file key")
        return cls(*pair)

    def to_dict(self) -> dict[str, str]:
        return {"bank": self.bank, "path": self.path}


@dataclass(frozen=True, slots=True)
class BankConfig:
    id: str
    name: str
    type: BankType
    path: Path
    include_extensions: tuple[str, ...] = (".md",)
    exclude_globs: tuple[str, ...] = ()
    doc_globs: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    code_globs: tuple[str, ...] = ("*",)
    doc_roles: dict[str, tuple[str, ...]] = field(default_factory=dict)
    watch_enabled: bool = False
    watch_debounce_seconds: float = 5.0


def select_banks(
    registry: Mapping[str, BankConfig], banks: Sequence[str] | None
) -> frozenset[str]:
    if banks is None:
        return frozenset(registry)
    if not banks or isinstance(banks, (str, bytes)) or any(
        not isinstance(bank, str) for bank in banks
    ):
        raise ValueError("banks must contain one or more bank IDs")
    requested = frozenset(banks)
    unknown = requested.difference(registry)
    if unknown:
        raise ValueError(f"unknown bank IDs: {', '.join(sorted(unknown))}")
    return requested
