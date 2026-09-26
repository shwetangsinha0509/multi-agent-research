# Multi-Agent Research & Report System

An AI-powered research platform that generates professional PDF reports on any topic using a multi-agent architecture. Built with LangGraph, FastMCP, and the A2A protocol — each agent specializes in a task and communicates through industry-standard protocols.

---

## Demo

1. Enter a research topic
2. Answer a few targeted clarification questions (or provide custom instructions upfront)
3. Watch live progress as agents research the web and analyze findings
4. Get a structured, professional PDF report with tables, citations, and sourced data
5. Refine the report iteratively through natural language — "make it longer", "add a regulatory section", "include more tables"

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  FastAPI Backend                     │
│            /research  /refine  /chat                 │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         Coordinator Agent (LangGraph)                │
│  Clarifies → Plans → Delegates → Synthesizes        │
└────────┬───────────────┬──────────────┬─────────────┘
         │ A2A           │ A2A          │ A2A
         ▼               ▼              ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│ Web Search   │ │  Document   │ │   Analysis   │
│    Agent     │ │    Agent    │ │    Agent     │
│  (Tavily)    │ │ (MCP Client)│ │  (Pure LLM)  │
└──────────────┘ └──────┬──────┘ └──────────────┘
                        │ MCP
                        ▼
               ┌────────────────┐
               │  RAG MCP Server│  ← Custom FastMCP Server
               │  (ChromaDB +   │
               │  Embeddings)   │
               └────────────────┘
```

---

## Key Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Multi-agent orchestration | LangGraph `StateGraph` coordinates 3 specialist agents |
| Agent-to-agent communication | Google's A2A protocol (`a2a-sdk`) for task delegation |
| Custom MCP server | FastMCP exposes RAG pipeline as a reusable MCP tool |
| MCP client | Document Agent connects to RAG MCP server via MCP protocol |
| Stateful conversation | LangGraph `MemorySaver` persists state across turns |
| Clarification-first flow | Coordinator asks targeted questions before researching |
| Iterative refinement | Multi-turn report editing via natural language |
| RAG pipeline | ChromaDB + Sentence Transformers for document search |
| PDF generation | ReportLab with Unicode font support, tables, and styled sections |
| SSE streaming | Real-time progress updates to the frontend during research |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph — StateGraph, conditional edges, memory |
| Agent communication | A2A Protocol (Google) — a2a-sdk v1.1.2 |
| Custom MCP server | FastMCP — exposes RAG pipeline as MCP tools |
| LLM | Groq — openai/gpt-oss-120b |
| Web search | Tavily Search API |
| Embeddings | Sentence Transformers — all-MiniLM-L6-v2 (local, no API cost) |
| Vector store | ChromaDB — persistent local storage |
| PDF generation | ReportLab with NotoSans Unicode font |
| Backend | FastAPI with SSE (Server-Sent Events) |
| Frontend | Vanilla HTML/JS — split panel with live progress and report preview |

---

## Project Structure

```
multi_agent_research/
├── main.py                    ← FastAPI app — /research, /refine, /chat, SSE
├── coordinator/
│   ├── agent.py               ← LangGraph graph — clarify, research, analyze, refine nodes
│   └── planner.py             ← Generates research plans and clarification questions
├── agents/
│   ├── web_search_agent.py    ← A2A server — calls Tavily for web research
│   ├── document_agent.py      ← A2A server — MCP client to RAG server
│   └── analysis_agent.py      ← A2A server — synthesizes research into JSON report
├── mcp_servers/
│   └── rag_server.py          ← Custom FastMCP server — exposes RAG pipeline
├── rag/
│   ├── ingestor.py            ← PDF loading, chunking, ChromaDB storage
│   └── retriever.py           ← Semantic chunk retrieval
├── report/
│   ├── generator.py           ← ReportLab PDF generation from JSON report
│   ├── store.py               ← Report version management
│   └── fonts/                 ← NotoSans fonts for Unicode support
└── static/
    └── index.html             ← Split-panel UI with live progress and report preview
```

---

## How It Works

**Research Flow:**
1. User enters topic → Coordinator checks for custom instructions
2. If none → Coordinator generates targeted clarification questions
3. User answers → Coordinator creates a research plan with specific queries
4. Research plan dispatched to Web Search Agent (via A2A) and Document Agent (via A2A)
5. Document Agent connects to RAG MCP Server (via MCP) to search uploaded PDFs
6. All results sent to Analysis Agent (via A2A) → structured JSON report generated
7. Report saved, PDF rendered, browser preview shown

**Refinement Flow:**
1. User types change request ("add more sections", "make tables for comparisons")
2. Coordinator routes to refine node — skips research agents
3. Analysis Agent receives current report + change request → returns updated JSON
4. New PDF rendered, version history updated

---

## Run Locally

**1. Clone and set up**
```bash
git clone https://github.com/shwetangsinha0509/multi-agent-research.git
cd multi-agent-research
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

**2. Environment variables**

Create `.env`:
```
GROQ_API_KEY=your_groq_key_here
TAVILY_API_KEY=your_tavily_key_here
```

Get free API keys:
- Groq: [console.groq.com](https://console.groq.com)
- Tavily: [app.tavily.com](https://app.tavily.com)

**3. Start all services** (5 separate terminals)

```bash
# Terminal 1 — RAG MCP Server
python mcp_servers/rag_server.py

# Terminal 2 — Document Agent
python agents/document_agent.py

# Terminal 3 — Web Search Agent
python agents/web_search_agent.py

# Terminal 4 — Analysis Agent
python agents/analysis_agent.py

# Terminal 5 — Main App
uvicorn main:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## Portfolio Context

This is Project 3 in my AI/ML engineering portfolio, building on:

- **[Project 1](https://github.com/shwetangsinha0509/rag-doc-qa)** — RAG Q&A system built from raw libraries (pypdf, ChromaDB, Groq)
- **[Project 2](https://github.com/shwetangsinha0509/research-agent)** — Conversational research agent with LangGraph and tool calling

This project introduces multi-agent orchestration, custom MCP server development, and the A2A protocol — demonstrating the full agentic AI stack from protocol-level implementation to deployed product.

---

## Limitations

- **Single user** — no authentication; concurrent users share the same process
- **No streaming** — agent completes full ReAct loop before returning (SSE shows progress but not token-by-token streaming)
- **Local vector store** — ChromaDB writes to disk; not suitable for multi-instance deployment without shared storage
- **Token limits** — research payload truncated at ~6000 chars to fit within model context window