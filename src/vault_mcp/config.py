from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    vault_root: Path = Path("Obsidian")
    index_dir: Path = Path("~/.vault-index")
    collection_name: str = "vault_chunks"
    embedding_model: str = "nomic-embed-text"

    def resolved_vault_root(self) -> Path:
        return self.vault_root.expanduser().resolve()

    def resolved_index_dir(self) -> Path:
        return self.index_dir.expanduser().resolve()
