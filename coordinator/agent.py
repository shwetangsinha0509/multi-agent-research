import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import asyncio
import httpx
from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from coordinator.planner import create_research_plan, create_clarification_questions
from dotenv import load_dotenv

load_dotenv()

# ─── Agent URLs ───────────────────────────────────────────────────────────────

WEB_SEARCH_AGENT_URL = os.getenv("WEB_AGENT_URL", "http://127.0.0.1:8001")
DOCUMENT_AGENT_URL   = os.getenv("DOC_AGENT_URL", "http://127.0.0.1:8002")
ANALYSIS_AGENT_URL   = os.getenv("ANALYSIS_AGENT_URL", "http://127.0.0.1:8003")


# ─── State ────────────────────────────────────────────────────────────────────

class CoordinatorState(TypedDict):
    messages:            Annotated[list, add_messages]
    topic:               str
    custom_instructions: str
    has_document:        bool
    collection_name:     str
    research_plan:       dict
    web_results:         str
    document_results:    str
    current_report:      dict | None
    report_version:      int
    phase:               str  # "clarifying" | "researching" | "done"
    status_updates:      list[str]


# ─── A2A Client ───────────────────────────────────────────────────────────────

async def call_agent(agent_url: str, message: str) -> str:
    """
    Send a task to a specialist agent via A2A JSON-RPC protocol.
    """
    import uuid
    payload = {
        "jsonrpc": "2.0",
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "messageId": uuid.uuid4().hex,
                "parts": [{"kind": "text", "text": message}]
            }
        },
        "id": uuid.uuid4().hex
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{agent_url}/",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()

    # Extract text from A2A v1 response
    try:
        # v1 response structure
        result = data.get("result", {})

        # Try direct message parts first
        if "parts" in result:
            for part in result["parts"]:
                if part.get("kind") == "text":
                    return part["text"]
                if part.get("type") == "text":
                    return part["text"]

        # Try status.message.parts
        status = result.get("status", {})
        msg = status.get("message", {})
        parts = msg.get("parts", [])
        for part in parts:
            if part.get("kind") == "text":
                return part["text"]
            if part.get("type") == "text":
                return part["text"]

        # Try artifacts
        artifacts = result.get("artifacts", [])
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    return part["text"]

    except (KeyError, TypeError) as e:
        print(f"Parse error: {e}")

    return str(data)


# ─── Nodes ────────────────────────────────────────────────────────────────────

def check_input_node(state: CoordinatorState) -> CoordinatorState:
    """
    First node. Checks if we have custom instructions.
    Sets phase to 'clarifying' if not, 'researching' if yes.
    Also detects if this is a refinement request.
    """
    messages = state.get("messages", [])
    current_report = state.get("current_report")
    last_message = messages[-1].content if messages else ""

    # If report exists and this isn't the first message → refinement
    if current_report and state.get("phase") == "done":
        return {
            **state,
            "phase": "refining",
            "custom_instructions": last_message
        }

    # If clarifying phase and user just answered → move to researching
    if state.get("phase") == "clarifying":
        return {
            **state,
            "phase": "researching",
            "custom_instructions": last_message
        }

    # First message — check if custom instructions were provided
    custom_instructions = state.get("custom_instructions", "").strip()
    if custom_instructions:
        return {**state, "phase": "researching"}
    else:
        return {**state, "phase": "clarifying"}


def clarify_node(state: CoordinatorState) -> CoordinatorState:
    """
    Asks the user clarification questions when no custom instructions provided.
    """
    topic = state.get("topic", "")
    questions = create_clarification_questions(topic)

    return {
        **state,
        "messages": [AIMessage(content=questions)],
        "phase": "clarifying"
    }


async def research_node(state: CoordinatorState) -> CoordinatorState:
    """
    Creates research plan and delegates to specialist agents in parallel.
    Collects web search results and document results.
    """
    topic = state.get("topic", "")
    custom_instructions = state.get("custom_instructions", "")
    has_document = state.get("has_document", False)
    collection_name = state.get("collection_name", "")

    status_updates = list(state.get("status_updates", []))

    # Step 1: Create research plan
    status_updates.append("plan_created")
    plan = create_research_plan(topic, custom_instructions, has_document)

    # Step 2: Web search — send all queries in one task
    status_updates.append("web_search_started")
    queries_str = " | ".join(plan.get("web_search_queries", [topic]))
    web_message = f"QUERIES: {queries_str}"

    try:
        web_results = await call_agent(WEB_SEARCH_AGENT_URL, web_message)
        status_updates.append("web_search_complete")
    except Exception as e:
        web_results = f"Web search failed: {str(e)}"
        status_updates.append("web_search_failed")

    # Step 3: Document search (only if document uploaded)
    document_results = "No document uploaded."
    if has_document and collection_name:
        status_updates.append("document_search_started")
        doc_queries = plan.get("document_queries", [topic])

        doc_tasks = []
        for query in doc_queries:
            doc_message = f"QUERY: {query} | COLLECTION: {collection_name}"
            doc_tasks.append(call_agent(DOCUMENT_AGENT_URL, doc_message))

        try:
            doc_results = await asyncio.gather(*doc_tasks)
            document_results = "\n\n---\n\n".join(doc_results)
            status_updates.append("document_search_complete")
        except Exception as e:
            document_results = f"Document search failed: {str(e)}"
            status_updates.append("document_search_failed")

    return {
        **state,
        "research_plan": plan,
        "web_results": web_results,
        "document_results": document_results,
        "status_updates": status_updates,
        "phase": "analyzing"
    }


async def analysis_node(state: CoordinatorState) -> CoordinatorState:
    status_updates = list(state.get("status_updates", []))
    status_updates.append("analysis_started")

    # Truncate research data to fit within model context limits
    # Character limits chosen to stay within 8000 TPM for gpt-oss-120b
    MAX_WEB_CHARS = 4000
    MAX_DOC_CHARS = 2000

    web_results = state.get("web_results", "")
    document_results = state.get("document_results", "")

    payload = {
        "topic": state.get("topic", ""),
        "custom_instructions": state.get("custom_instructions", ""),
        "web_results": web_results[:MAX_WEB_CHARS],
        "document_results": document_results[:MAX_DOC_CHARS],
        "report_structure": state.get("research_plan", {}).get("report_structure", []),
        "is_refinement": False,
        "current_report": None
    }

    try:
        result = await call_agent(ANALYSIS_AGENT_URL, json.dumps(payload))
        report_json = json.loads(result)
        status_updates.append("analysis_complete")
    except Exception as e:
        report_json = {
            "error": str(e),
            "title": state.get("topic", "Research Report"),
            "sections": [],
            "sources": [],
            "metadata": {"version": 1}
        }
        status_updates.append("analysis_failed")

    return {
        **state,
        "current_report": report_json,
        "report_version": 1,
        "status_updates": status_updates,
        "phase": "done",
        "messages": [AIMessage(content="report_ready")]
    }


async def refine_node(state: CoordinatorState) -> CoordinatorState:
    status_updates = list(state.get("status_updates", []))
    status_updates.append("refinement_started")

    refinement_request = state.get("messages", [])[-1].content
    current_report = state.get("current_report")
    current_version = state.get("report_version", 1)

    # Truncate research data to fit within model context limits
    MAX_WEB_CHARS = 4000  # smaller for refinement since current_report also takes tokens
    MAX_DOC_CHARS = 1000

    web_results = state.get("web_results", "")
    document_results = state.get("document_results", "")

    payload = {
        "topic": state.get("topic", ""),
        "custom_instructions": state.get("custom_instructions", ""),
        "web_results": web_results[:MAX_WEB_CHARS],
        "document_results": document_results[:MAX_DOC_CHARS],
        "report_structure": [],
        "is_refinement": True,
        "refinement_request": refinement_request,
        "current_report": {
            "title": current_report.get("title", ""),
            "sections": current_report.get("sections", []),
            "sources": current_report.get("sources", []),
            "metadata": current_report.get("metadata", {})
        }
    }

    try:
        result = await call_agent(ANALYSIS_AGENT_URL, json.dumps(payload))
        report_json = json.loads(result)

        # Force correct version number — don't trust LLM to increment it
        new_version = current_version + 1
        if "metadata" in report_json:
            report_json["metadata"]["version"] = new_version

        status_updates.append("refinement_complete")
    except Exception as e:
        print("Refine error:", e)
        report_json = current_report
        status_updates.append("refinement_failed")

    return {
        **state,
        "current_report": report_json,
        "report_version": current_version + 1,
        "status_updates": status_updates,
        "phase": "done",
        "messages": [AIMessage(content="report_ready")]
    }


# ─── Routing ──────────────────────────────────────────────────────────────────

def route_after_check(state: CoordinatorState) -> str:
    phase = state.get("phase", "clarifying")
    if phase == "clarifying":
        return "clarify"
    elif phase == "refining":
        return "refine"
    else:
        return "research"


def route_after_research(state: CoordinatorState) -> str:
    return "analyze"


# ─── Graph ────────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(CoordinatorState)

    builder.add_node("check_input", check_input_node)
    builder.add_node("clarify",     clarify_node)
    builder.add_node("research",    research_node)
    builder.add_node("analyze",     analysis_node)
    builder.add_node("refine",      refine_node)

    builder.add_edge(START, "check_input")

    builder.add_conditional_edges(
        "check_input",
        route_after_check,
        {
            "clarify":  "clarify",
            "research": "research",
            "refine":   "refine"
        }
    )

    builder.add_edge("clarify",  END)
    builder.add_edge("research", "analyze")
    builder.add_edge("analyze",  END)
    builder.add_edge("refine",   END)

    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


graph = build_graph()