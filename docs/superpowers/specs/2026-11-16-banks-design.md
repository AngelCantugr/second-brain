# Banks as the unit of knowledge in second-brain

## Intent

A single second-brain instance can index several Obsidian vaults and repositories. Each is a named bank with the same public operations for discovery, search, fetch, and graph exploration. Agents can start with a search across everything and narrow subsequent calls to particular banks.

## Approaches

- **Add bank metadata to existing paths.** This preserves existing citations and simplifies rollout. It keeps vault-relative paths and `repo:` paths as separate identity conventions and leaves the original vault special.
- **Use bank IDs and relative paths throughout.** This makes every root addressable in the same way and removes ambiguity between identically named files. It changes the config and tool contracts and requires a fresh index. We chose this approach because the source files can be reindexed at cutover.
- **Maintain a separate index per bank and merge results.** This would make storage isolation possible, but it adds work to compare hybrid search scores and compute relationships across banks. The agreed shared index serves both cross-bank retrieval and cross-bank graph discovery directly.

## Configuration and identity

Use a `[[banks]]` table for every root, including the original vault. A bank has a required unique, stable `id` and root `path`, an optional display `name`, and a source `type` such as `obsidian` or `repo`. Type supplies discovery and parsing defaults. Include and exclude rules, document roles, and enabled parsers are bank settings. Obsidian wikilinks and canvases, Git ignore rules, and symbol extraction remain capabilities of the applicable files and sources. A configuration may consist entirely of vault banks or entirely of repo banks.

```toml
[[banks]]
id = "learning"
path = "~/vaults/learning"
type = "obsidian"

[[banks]]
id = "cerberus-review"
path = "~/GithubRepositories/angelcantugr/cerberus-review"
type = "repo"
```

Every file has a canonical reference such as `{"bank": "learning", "path": "Projects/Roadmap.md"}`. Paths are normalized, case-preserving, root-relative paths with forward slashes. File references, chunk IDs, sync checkpoints, graph endpoints, and citation payloads all derive from this pair. Chunk and symbol identity also includes its position or locator within the file where needed. A bank ID change is an identity change. Changes to display names do not change file identity. Fetch and file sync validate that the resolved path stays inside the registered root, including when it traverses a symlink.

## Tool behavior

Tool names and storage encoding are builder decisions. The public contract uses structured bank and path fields.

- **List banks** returns each configured bank's ID, display name, source type, and availability. It includes unavailable banks and omits absolute filesystem paths.
- **Search and query** default to all banks in the shared vector and keyword indexes. An optional filter accepts one or more bank IDs. Eligibility is applied to both retrieval channels before ranking and limiting. Every hit and citation carries its canonical file reference and the source type. Content kind remains independently filterable.
- **Fetch** accepts a canonical file reference and returns the stored chunks for that file in source order, with provenance. This uses the same text as search, including redactions, heading context, and symbol excerpts. An unknown bank is an input error. A file with no stored chunks returns a structured not-found result. Fetch has no raw filesystem read path.
- **Graph** operations accept canonical file references wherever they take a file as input. Related and map results accept the same bank selector as search. A related query can use any anchor and filter candidate neighbors before applying its result limit. Map uses the selected banks as its node set. Connections may take fixed endpoints from any bank, with an optional bank selector restricting intermediate nodes. Results identify the bank of every returned node.
- **Status and sync** support every registered bank. Status includes per-bank counts or availability. Full and incremental sync default to all banks. File sync takes a bank and relative path. Watching can be configured per bank and uses that bank's file selection rules.

An omitted bank selector means all banks. An explicitly empty selector or unknown bank ID is an input error. Bank selection is retrieval scope for a single trusted second-brain instance.

## Graph behavior

Use a shared graph keyed by canonical references. Automatic semantic relationships can connect eligible documents across banks, including repository Markdown and vault notes. Reuse the existing eligibility and scoring mechanisms where they fit, with repository Markdown eligible for semantic relationships. Preserve typed code mention and import edges. Linking to a symbol uses its bank and file reference plus its symbol locator. Individual code symbols do not join document semantic neighborhoods by default.

Resolve ordinary Obsidian wikilinks within the originating bank, even when an identically named note exists elsewhere. Explicit bank-qualified references can connect any two banks. Repository references that resolve through existing code mention handling retain their typed edge. Graph filters select eligible results from the shared graph without reranking the underlying relationship scores.

## Rebuild and cutover

Build fresh vector, keyword, checkpoint, and graph stores from the bank config. This release can adopt the new canonical contract in one cutover. Existing `vault_path`, `[[code_sources]]`, and `repo:` references in tool calls transition with the bundled config examples, documentation, and consumers. Source files stay in place. Existing links written in Markdown continue to resolve using the Markdown rules above.

Prepare the new stores alongside the existing ones. Run a full sync, check bank inventory and index counts, validate scoped and cross-bank queries, and switch the server configuration after verification. Retain the previous config and stores for rollback until the new instance is accepted. A temporarily unavailable root is reported per bank and must preserve its indexed data during subsequent incremental or full syncs. An intentionally removed bank can be omitted when rebuilding fresh stores.

## Acceptance

- Register two vault banks and a repo bank in one config. List all three and identify each result's bank in a combined search.
- Index matching relative paths in different banks. Fetch each independently, and check that search and citations round-trip to the right indexed file.
- Scope search to one bank and to several banks. Verify both keyword and semantic retrieval respect the scope.
- Build a graph with an automatic relationship between documents in different banks and an ordinary wikilink whose target title exists in two vaults. Verify the wikilink stays local and the inferred relationship appears when its bank is in scope.
- Sync one file with the same relative path in two banks. Verify only the selected identity is updated. Make one root unavailable and verify its previously indexed content survives a sync.
- Rebuild stores from scratch and confirm graph, fetch, and retrieval checks before cutover.
