import json
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.2
        )
    return _llm


def create_research_plan(
    topic: str,
    custom_instructions: str,
    has_document: bool
) -> dict:
    """
    Takes topic + custom instructions and creates a structured research plan.
    Returns a dict with web search queries, document queries, and report structure.
    """

    doc_note = "A reference document has been uploaded — include document search queries." \
        if has_document else "No document uploaded — focus on web research only."

    prompt = f"""You are a research planning expert.
Create a detailed research plan for the following topic and instructions.

TOPIC: {topic}
CUSTOM INSTRUCTIONS: {custom_instructions}
DOCUMENT STATUS: {doc_note}

Return ONLY a valid JSON object with this exact structure, no markdown, no explanation:
{{
    "web_search_queries": ["query 1", "query 2", "query 3"],
    "document_queries": ["doc query 1", "doc query 2"],
    "report_structure": ["Section 1 Title", "Section 2 Title", "Section 3 Title"],
    "tone": "professional",
    "estimated_pages": 3
}}

Rules:
- web_search_queries: 5-8 specific, targeted search queries. Include queries that target statistics, 
  market data, financial figures, and recent research reports. 
  Make queries specific enough to return data-rich results.
  Example: "UPI market share percentage 2024 RBI report statistics" not just "UPI market"
- document_queries: 2-3 queries to search the uploaded document (empty list if no document)
- report_structure: section titles that match the custom instructions — as many as needed
- tone: "professional", "academic", "executive", or "casual" based on instructions
- estimated_pages: based on depth requested in instructions, no fixed limit
- Return ONLY the JSON, nothing else"""

    response = get_llm().invoke(prompt)
    raw = response.content.strip()

    # Strip markdown fences if present
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback plan if LLM returns invalid JSON
        return {
            "web_search_queries": [topic, f"{topic} latest research", f"{topic} statistics"],
            "document_queries": [topic] if has_document else [],
            "report_structure": [
                "Executive Summary",
                "Key Findings",
                "Analysis",
                "Implications",
                "Conclusion"
            ],
            "tone": "professional",
            "estimated_pages": 3
        }


def create_clarification_questions(topic: str) -> str:
    """
    Generates targeted clarification questions when user provides no custom instructions.
    Returns a friendly message string.
    """

    prompt = f"""You are a research assistant helping a user define their research report.
The user wants a report on: "{topic}"
They haven't provided any specific instructions.

Generate 3-4 short, targeted clarification questions to understand:
1. The purpose of the report
2. Any specific angle, region, or industry focus
3. Preferred depth and length
4. Any specific sections they want

Return ONLY the questions as a friendly conversational message.
Start with: "Before I start researching, a few quick questions to make your report more targeted:"
Then list the questions numbered.
Keep it concise — no more than 5 lines total."""

    response = get_llm().invoke(prompt)
    return response.content.strip()