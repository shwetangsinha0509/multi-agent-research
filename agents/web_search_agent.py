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
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# ─── Agent Executor ───────────────────────────────────────────────────────────

class WebSearchAgentExecutor(AgentExecutor):
    """
    Core logic of the Web Search Agent.
    Receives search tasks from Coordinator via A2A,
    calls Tavily, returns formatted results.
    """

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        user_message = context.get_user_input()

        # Parse queries
        # Expected format: "QUERIES: query1 | query2 | query3"
        queries = []
        if "QUERIES:" in user_message:
            raw = user_message.replace("QUERIES:", "").strip()
            queries = [q.strip() for q in raw.split("|") if q.strip()]
        else:
            queries = [user_message.strip()]

        all_results = []

        for query in queries:
            try:
                response = tavily_client.search(
                    query=query,
                    max_results=5,
                    search_depth="basic"
                )
                results = response.get("results", [])
                for r in results:
                    all_results.append(
                        f"Source: {r['url']}\n"
                        f"Title: {r['title']}\n"
                        f"Content: {r['content']}"
                    )
            except Exception as e:
                all_results.append(f"Search failed for '{query}': {str(e)}")

        if not all_results:
            answer = "No results found for the given queries."
        else:
            answer = "\n\n---\n\n".join(all_results)

        await event_queue.enqueue_event(new_text_message(answer))

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise NotImplementedError("Cancel not supported")


# ─── Agent Card ───────────────────────────────────────────────────────────────

def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Web Search Agent",
        description=(
            "Searches the web for current information using Tavily. "
            "Accepts multiple queries separated by pipes and returns "
            "structured results with sources and content."
        ),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="web_search",
                name="Web Search",
                description="Search the web for current information on any topic",
                tags=["search", "web", "tavily", "current", "news"],
            )
        ],
    )


# ─── Server ───────────────────────────────────────────────────────────────────

def main():
    agent_card = build_agent_card()

    request_handler = DefaultRequestHandler(
        agent_executor=WebSearchAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card=agent_card))
    routes.extend(create_jsonrpc_routes(request_handler=request_handler, rpc_url="/", enable_v0_3_compat=True))

    app = Starlette(routes=routes)

    print("Starting Web Search Agent on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=1000)

if __name__ == "__main__":
    main()