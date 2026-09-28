import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from starlette.applications import Starlette

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.helpers import new_text_message
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from fastmcp import Client as MCPClient
from dotenv import load_dotenv

load_dotenv()

RAG_MCP_URL = os.getenv("RAG_MCP_URL", "http://127.0.0.1:8004/mcp")


# ─── Agent Executor ───────────────────────────────────────────────────────────

class DocumentAgentExecutor(AgentExecutor):
    """
    Core logic of the Document Agent.
    Receives a task from Coordinator via A2A,
    calls RAG MCP server, returns results.
    """

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # Extract the task message
        user_message = context.get_user_input()

        # Parse query and collection name
        # Expected format: "QUERY: <query> | COLLECTION: <collection_name>"
        query = user_message
        collection_name = ""

        if "| COLLECTION:" in user_message:
            parts = user_message.split("| COLLECTION:")
            query = parts[0].replace("QUERY:", "").strip()
            collection_name = parts[1].strip()

        try:
            # Connect to RAG MCP server and call search_document tool
            async with MCPClient(RAG_MCP_URL) as mcp:
                result = await mcp.call_tool(
                    "search_document",
                    {
                        "query": query,
                        "collection_name": collection_name
                    }
                )
                answer = result.content[0].text if result.content else "No results found."

        except Exception as e:
            answer = f"Document search failed: {str(e)}"

        # Return result to Coordinator via event queue
        await event_queue.enqueue_event(
            new_text_message(answer)
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise NotImplementedError("Cancel not supported")


# ─── Agent Card ───────────────────────────────────────────────────────────────

def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Document Search Agent",
        description=(
            "Searches uploaded PDF documents for relevant information. "
            "Connects to a RAG MCP server backed by ChromaDB and "
            "sentence-transformer embeddings."
        ),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="search_document",
                name="Search Document",
                description="Search an uploaded PDF for relevant chunks",
                tags=["rag", "document", "pdf", "search"],
            )
        ],
    )


# ─── Server ───────────────────────────────────────────────────────────────────

def main():
    agent_card = build_agent_card()

    request_handler = DefaultRequestHandler(
        agent_executor=DocumentAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card=agent_card))
    routes.extend(create_jsonrpc_routes(request_handler=request_handler, rpc_url="/", enable_v0_3_compat=True))

    app = Starlette(routes=routes)

    print("Starting Document Agent on port 8002...")
    uvicorn.run(app, host="0.0.0.0", port=10000)


if __name__ == "__main__":
    main()