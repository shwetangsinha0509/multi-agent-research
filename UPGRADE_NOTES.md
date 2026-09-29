## Finding 1 — Broken import path after FastMCP 3 → 4 upgrade

**What I ran:** python mcp_servers/rag_server.py

**Exact error:**
ModuleNotFoundError: No module named 'mcp.server.fastmcp'. This is mcp 2.x,
where FastMCP was renamed to MCPServer (from mcp.server.mcpserver import MCPServer)
and other APIs changed; see the migration guide at
https://py.sdk.modelcontextprotocol.io/v2/migration/#fastmcp-renamed-to-mcpserver
or pin 'mcp<2' to keep running v1 code.

**Root cause:**
FastMCP 4 pulls in mcp 2.x as a dependency. In mcp 2.x, the import path
`from mcp.server.fastmcp import FastMCP` no longer exists. Code written against
FastMCP 3 uses this path and breaks immediately on startup.

**What I expected:**
The FastMCP upgrade guide says "most FastMCP 3 servers upgrade untouched." This
server did not — it broke on the very first import.

**Status:** Investigating fix.

**Fix:** Changed `from mcp.server.fastmcp import FastMCP` to `from fastmcp import FastMCP`
**Doc that helped:** https://gofastmcp.com (direct package import is the correct FastMCP 4 pattern)
**Note for contribution:** The upgrade guide says "most servers upgrade untouched" but does not
explicitly call out this import path change. A migration note or a clearer error message pointing
to the correct import would prevent this confusion.

## Finding 2 — FastMCP() constructor no longer accepts `host` and `port`

**What I ran:** python mcp_servers/rag_server.py (after fixing Finding 1)

**Exact error:**
TypeError: FastMCP() no longer accepts `host`. Pass `host` to `run_http_async()`,
or set FASTMCP_HOST.

**Root cause:**
In FastMCP 4, `host` and `port` were removed from the FastMCP() constructor.
They must now be passed to run_http_async() or set via environment variables
FASTMCP_HOST and FASTMCP_PORT.

**Status:** FIXED.

**Fix:** Moved `host` and `port` out of FastMCP() constructor into mcp.run() call.
Old: mcp = FastMCP("name", host="0.0.0.0", port=10000)
New: mcp = FastMCP("name") ... mcp.run(transport="streamable-http", host="0.0.0.0", port=10000)

**Note for contribution:** This is a silent breaking change for anyone who set host/port
in the constructor (common pattern in FastMCP 3 docs and examples). The error message
is good, but the upgrade guide doesn't list this as a breaking change with a before/after
example. That's a gap worth filing.

## Finding 3 — FastMCP 4 startup banner (new behaviour, not a bug)

**Observation:**
FastMCP 4 prints a large ASCII art banner on startup. FastMCP 3 did not.
In production/cloud deployments this adds noise to logs. There is no documented
way to suppress it.

**Potential contribution:** Add a FASTMCP_QUIET or --no-banner flag, or document
how to suppress the banner for production use.

## Finding 4 — Local port conflict when running all services simultaneously

**What happened:** web_search_agent.py and document_agent.py both bind to port 10000.
On Render (cloud) this works because each service is isolated in its own container.
Locally, the second service to start gets: [Errno 10048] only one usage of each socket
address permitted.

**Not a FastMCP bug.** This is a local development limitation of the Render free-tier
port strategy (all services use port 10000 per Render's requirement).

**Workaround for local testing:** Run each agent in isolation, or override the port
with an env var before starting each one.