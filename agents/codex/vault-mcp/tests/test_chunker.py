from vault_mcp.chunker import chunk_markdown


def test_chunk_markdown_preserves_heading_and_splits_long_sections() -> None:
    long_body = " ".join([f"word{i}" for i in range(1200)])
    markdown = "# Intro\n" + long_body + "\n## Details\n" + "short section"

    chunks = chunk_markdown(markdown, max_tokens=500)

    assert len(chunks) >= 3
    assert chunks[0].heading == "Intro"
    assert chunks[-1].heading == "Details"
    assert all(chunk.text.startswith(f"{chunk.heading}\n") for chunk in chunks)
