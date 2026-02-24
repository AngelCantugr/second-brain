from __future__ import annotations

from collections import Counter
from pathlib import Path


def count_notes_by_top_level(vault_root: Path, note_paths: list[Path]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for path in note_paths:
        relative = path.relative_to(vault_root)
        topic = relative.parts[0] if len(relative.parts) > 1 else "root"
        counter[topic] += 1
    return dict(sorted(counter.items(), key=lambda item: item[0].lower()))
