import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import uvicorn
from starlette.applications import Starlette

from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.helpers import new_text_message
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0.3,
            max_tokens=12000
        )
    return _llm


# ─── Agent Executor ───────────────────────────────────────────────────────────

class AnalysisAgentExecutor(AgentExecutor):
    """
    Core logic of the Analysis Agent.
    Receives all gathered research from Coordinator via A2A.
    Uses LLM to synthesize findings into structured JSON report.
    No tools — pure LLM reasoning.
    """

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        user_message = context.get_user_input()

        # Parse the task payload
        # Expected format: JSON string with topic, instructions,
        # web_results, document_results, report_structure
        try:
            payload = json.loads(user_message)
        except json.JSONDecodeError:
            from json_repair import repair_json
            repaired = repair_json(raw)
            report_json = json.loads(repaired)
            await event_queue.enqueue_event(
                new_text_message(json.dumps({
                    "error": "Invalid payload — expected JSON"
                }))
            )
            return

        topic = payload.get("topic", "")
        custom_instructions = payload.get("custom_instructions", "")
        web_results = payload.get("web_results", "")
        document_results = payload.get("document_results", "")
        report_structure = payload.get("report_structure", [])
        is_refinement = payload.get("is_refinement", False)
        refinement_request = payload.get("refinement_request", "")
        current_report = payload.get("current_report", None)

        # Build the prompt
        if is_refinement:
            if current_report is None:
                raise ValueError("Refinement requested but current_report is missing")

            prompt = self._build_refinement_prompt(
                topic,
                current_report,
                refinement_request,
                web_results,
                document_results
            )
        else:
            prompt = self._build_analysis_prompt(
                topic,
                custom_instructions,
                web_results,
                document_results,
                report_structure
            )

        try:
            response = get_llm().invoke(prompt)
            raw = response.content.strip()

            # Remove any markdown fences
            for fence in ["```json", "```JSON", "```", "`"]:
                if raw.startswith(fence):
                    raw = raw[len(fence):]
                if raw.endswith(fence):
                    raw = raw[:-len(fence)]
            raw = raw.strip()

            # Find the first { and last } to extract just the JSON object
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                raw = raw[start:end]

            # Fix common LLM unicode issues
            raw = raw.replace("\u2011", "-")
            raw = raw.replace("\u2013", "-")
            raw = raw.replace("\u2014", "-")
            raw = raw.replace("\u202f", " ")
            raw = raw.replace("\u00a0", " ")
            raw = raw.replace("\u201c", '"').replace("\u201d", '"')
            raw = raw.replace("\u2018", "'").replace("\u2019", "'")

            # Try standard JSON first
            try:
                report_json = json.loads(raw)
            except json.JSONDecodeError:
                # Fall back to json-repair which fixes common LLM JSON issues
                from json_repair import repair_json
                repaired = repair_json(raw)
                report_json = json.loads(repaired)

            answer = json.dumps(report_json)

        except Exception as e:
            answer = json.dumps({"error": str(e)})

        await event_queue.enqueue_event(new_text_message(answer))

    def _build_analysis_prompt(
    self, topic, custom_instructions,
    web_results, document_results, report_structure
    ) -> str:
        doc_section = f"""
DOCUMENT RESEARCH:
{document_results}
""" if document_results and document_results != "No document uploaded." else ""

        sections_str = "\n".join(
            [f"- {s}" for s in report_structure]
        ) if report_structure else ""

        return f"""You are a professional research analyst. Write a comprehensive research report on the topic below.

TOPIC: {topic}

USER INSTRUCTIONS: {custom_instructions if custom_instructions else "Write a balanced, professional report."}

RESEARCH DATA:
{web_results}
{doc_section}

REQUIRED SECTIONS:
{sections_str}

Follow the user's instructions exactly — respect their requested length, depth, tone, focus area, and sections.
Use the research data to support your analysis. Add your own knowledge where the research data has gaps.

Return ONLY a valid JSON object in this exact structure:
{{
    "title": "Report title",
    "sections": [
        {{
            "id": "section_1",
            "title": "Section Title",
            "content": "Full section content.",
            "highlights": ["Key fact 1", "Key fact 2"]
        }}
    ],
    "sources": [
        {{"title": "Source title", "url": "https://url.com"}}
    ],
    "metadata": {{
        "topic": "{topic}",
        "custom_instructions": "{custom_instructions}",
        "version": 1
    }}
}}

RULES:
- Return ONLY the JSON object — no markdown fences, no text before or after
- Start with {{ and end with }}
- Use double quotes throughout
- Follow the user's length and depth instructions precisely
- Use proper Unicode currency symbols: ₹ for Indian Rupee, $ for USD, € for Euro, £ for GBP
- Never use ■ or any block character as a placeholder for currency or any other symbol
- ONLY use statistics, figures, and data points that appear in the RESEARCH DATA provided
- If a specific number or statistic is not in the research data, do NOT invent it — describe the trend qualitatively instead
- Every specific data point must be attributable to a source in the research data
- Financial projections and forecasts are only acceptable if they come from the research data — never generate your own projections
- If research data is insufficient for a section, say so explicitly in the content rather than fabricating data
- Citations in the sources array must only include sources that actually appeared in the research data"""


    def _build_refinement_prompt(
    self, topic, current_report,
    refinement_request, web_results, document_results
    ) -> str:
        return f"""You are a professional research analyst editing an existing report.

CURRENT REPORT:
{json.dumps(current_report, indent=2)}

CHANGE REQUEST:
{refinement_request}

RESEARCH DATA (use to enrich new content):
{web_results if web_results else "None"}

Apply ONLY the requested changes. Return the complete updated report as JSON.
If the user asks to add sections, add them. If they ask to modify content, modify it.
Do not remove existing content unless explicitly asked.
Increment metadata.version by 1.

RULES:
- ONLY use statistics and figures from the existing report or the available research data above
- Do not generate new data points, projections, or citations that are not in the research
- If research data is insufficient for a new section, describe trends qualitatively instead of inventing numbers
- Every specific statistic must be traceable to a source in the research data or existing report
- Return ONLY the JSON object — no markdown fences, no text before or after"""

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        raise NotImplementedError("Cancel not supported")


# ─── Agent Card ───────────────────────────────────────────────────────────────

def build_agent_card() -> AgentCard:
    return AgentCard(
        name="Analysis Agent",
        description=(
            "Synthesizes research from web search and documents into "
            "a structured JSON report. Uses LLM reasoning to analyze, "
            "organize, and write professional report content. "
            "Also handles report refinement based on user feedback."
        ),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[
            AgentSkill(
                id="analyze_and_write",
                name="Analyze and Write Report",
                description="Synthesize research into a structured JSON report",
                tags=["analysis", "writing", "report", "synthesis"],
            ),
            AgentSkill(
                id="refine_report",
                name="Refine Report",
                description="Apply user-requested changes to an existing report",
                tags=["refinement", "editing", "report"],
            )
        ],
    )


# ─── Server ───────────────────────────────────────────────────────────────────

def main():
    agent_card = build_agent_card()

    request_handler = DefaultRequestHandler(
        agent_executor=AnalysisAgentExecutor(),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card=agent_card))
    routes.extend(create_jsonrpc_routes(request_handler=request_handler, rpc_url="/", enable_v0_3_compat=True))

    app = Starlette(routes=routes)

    print("Starting Analysis Agent on port 8003...")
    uvicorn.run(app, host="127.0.0.1", port=8003)


if __name__ == "__main__":
    main()