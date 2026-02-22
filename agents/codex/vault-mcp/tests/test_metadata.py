from vault_mcp.config import Settings


def test_settings_defaults_exist() -> None:
    settings = Settings()
    assert settings.collection_name == "vault_chunks"
