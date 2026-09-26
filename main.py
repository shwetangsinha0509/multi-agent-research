import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import uuid
import json
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from dotenv import load_dotenv
from rag.ingestor import ingest_pdf
from coordinator.agent import graph, CoordinatorState
from report.store import save_report, load_report, list_reports
from report.generator import generate_pdf
from langchain_core.messages import HumanMessage

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory SSE event queues per thread_id
# When coordinator updates status, it puts events here
# Frontend SSE connection reads from here
sse_queues: dict[str, asyncio.Queue] = {}


# ─── Request Models ───────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str
    custom_instructions: str = ""
    thread_id: str
    collection_name: str | None = None


class ChatRequest(BaseModel):
    message: str
    thread_id: str
    collection_name: str | None = None


class RefineRequest(BaseModel):
    refinement_request: str
    thread_id: str


# ─── SSE Queue helpers ────────────────────────────────────────────────────────

def get_queue(thread_id: str) -> asyncio.Queue:
    if thread_id not in sse_queues:
        sse_queues[thread_id] = asyncio.Queue()
    return sse_queues[thread_id]


async def push_status(thread_id: str, event: str, data: dict = {}):
    queue = get_queue(thread_id)
    print(f"=== PUSH_STATUS: {thread_id[:8]}... event={event} queue_size={queue.qsize()}")
    await queue.put({"event": event, "data": json.dumps(data)})


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/events/{thread_id}")
async def sse_endpoint(thread_id: str):
    """SSE stream — frontend connects here to get live status updates."""
    queue = get_queue(thread_id)

    async def event_generator():
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield item
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "{}"}

    return EventSourceResponse(event_generator())


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    """Upload and ingest a reference PDF."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported.")

    collection_name = f"doc_{uuid.uuid4().hex[:8]}"
    save_path = UPLOAD_DIR / f"{collection_name}.pdf"

    with open(save_path, "wb") as f:
        f.write(await file.read())

    chunk_count = ingest_pdf(str(save_path), collection_name)

    return {
        "message": "Document ingested successfully.",
        "collection_name": collection_name,
        "chunks": chunk_count,
        "thread_id": thread_id
    }


@app.post("/research")
async def start_research(request: ResearchRequest):
    """
    Start a new research session.
    If no custom_instructions → coordinator will ask clarification questions.
    If custom_instructions provided → goes straight to research.
    """
    thread_id = request.thread_id

    # Build initial state
    initial_state = {
        "messages": [HumanMessage(content=request.topic)],
        "topic": request.topic,
        "custom_instructions": request.custom_instructions,
        "has_document": bool(request.collection_name),
        "collection_name": request.collection_name or "",
        "research_plan": {},
        "web_results": "",
        "document_results": "",
        "current_report": None,
        "report_version": 0,
        "phase": "",
        "status_updates": []
    }

    config = {"configurable": {"thread_id": thread_id}}

    # Run graph asynchronously so SSE can stream updates
    async def run_graph():
        try:
            await push_status(thread_id, "started", {"message": "Research started"})
            result = await graph.ainvoke(initial_state, config)

            phase = result.get("phase", "")
            status_updates = result.get("status_updates", [])

            for update in status_updates:
                await push_status(thread_id, update, {})
                await asyncio.sleep(0.1)

            if phase == "clarifying":
                messages = result.get("messages", [])
                last_msg = messages[-1].content if messages else ""
                await push_status(thread_id, "clarification_needed", {
                    "message": last_msg
                })

            elif phase == "done":
                report = result.get("current_report")
                if report and "error" not in report:
                    version = result.get("report_version", 1)
                    report_id = save_report(thread_id, report, version)
                    await push_status(thread_id, "report_ready", {
                        "report_id": report_id,
                        "version": version,
                        "title": report.get("title", "Research Report")
                    })
                else:
                    await push_status(thread_id, "error", {
                        "message": "Report generation failed"
                    })

        except Exception as e:
            import traceback
            traceback.print_exc()
            await push_status(thread_id, "error", {"message": str(e)})

    asyncio.create_task(run_graph())
    return {"status": "started", "thread_id": thread_id}


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Handle clarification responses and follow-up messages.
    Used after the coordinator asks clarification questions.
    """
    thread_id = request.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    async def run_graph():
        try:
            print("=== CHAT RUN_GRAPH STARTED ===")
            await push_status(thread_id, "started", {"message": "Processing..."})

            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=request.message)]},
                config
            )

            phase = result.get("phase", "")
            status_updates = result.get("status_updates", [])

            for update in status_updates:
                await push_status(thread_id, update, {})
                await asyncio.sleep(0.1)

            if phase == "done":
                report = result.get("current_report")
                if report and "error" not in report:
                    version = result.get("report_version", 1)
                    report_id = save_report(thread_id, report, version)
                    await push_status(thread_id, "report_ready", {
                        "report_id": report_id,
                        "version": version,
                        "title": report.get("title", "")
                    })
                else:
                    await push_status(thread_id, "error", {"message": "Failed"})

        except Exception as e:
            import traceback
            traceback.print_exc()
            await push_status(thread_id, "error", {"message": str(e)})

    asyncio.create_task(run_graph())
    return {"status": "processing"}


@app.post("/refine")
async def refine_report(request: RefineRequest):
    """
    Apply user-requested changes to existing report.
    Skips research agents — only reruns Analysis Agent.
    """
    thread_id = request.thread_id
    config = {"configurable": {"thread_id": thread_id}}

    async def run_graph():
        await push_status(thread_id, "refinement_started", {})

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=request.refinement_request)]},
            config
        )

        report = result.get("current_report")
        if report and "error" not in report:
            version = result.get("report_version", 1)
            report_id = save_report(thread_id, report, version)
            await push_status(thread_id, "report_ready", {
                "report_id": report_id,
                "version": version,
                "title": report.get("title", "")
            })
        else:
            await push_status(thread_id, "error", {"message": "Refinement failed"})

    asyncio.create_task(run_graph())
    return {"status": "refining"}


@app.get("/report/{report_id}")
async def get_report(report_id: str):
    """Return report JSON for browser preview."""
    report = load_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")
    return JSONResponse(content=report)


@app.get("/report/{report_id}/pdf")
async def download_report_pdf(report_id: str):
    """Generate and stream PDF download."""
    report = load_report(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found.")

    pdf_bytes = generate_pdf(report)

    return StreamingResponse(
        iter([pdf_bytes]),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{report_id}.pdf"'
        }
    )


@app.get("/reports/{thread_id}")
async def get_report_history(thread_id: str):
    """Return list of all report versions for a thread."""
    reports = list_reports(thread_id)
    return {"reports": reports}


# ─── Frontend ─────────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")