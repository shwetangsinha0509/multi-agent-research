import json
import os
from datetime import datetime
from pathlib import Path

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def save_report(thread_id: str, report_json: dict, version: int) -> str:
    """
    Save a report JSON to disk.
    Returns the report_id used to retrieve it later.
    """
    report_id = f"{thread_id}_v{version}"
    file_path = REPORTS_DIR / f"{report_id}.json"

    # Add save timestamp to metadata
    if "metadata" in report_json:
        report_json["metadata"]["saved_at"] = datetime.now().isoformat()
        report_json["metadata"]["report_id"] = report_id

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)

    return report_id


def load_report(report_id: str) -> dict | None:
    """Load a report JSON from disk by report_id."""
    file_path = REPORTS_DIR / f"{report_id}.json"
    if not file_path.exists():
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_reports(thread_id: str) -> list[dict]:
    """
    List all report versions for a given thread_id.
    Returns list of dicts with report_id, version, title, saved_at.
    """
    reports = []
    for file_path in sorted(REPORTS_DIR.glob(f"{thread_id}_v*.json")):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            metadata = data.get("metadata", {})
            reports.append({
                "report_id": metadata.get("report_id", file_path.stem),
                "version": metadata.get("version", 1),
                "title": data.get("title", "Untitled Report"),
                "saved_at": metadata.get("saved_at", "")
            })
        except Exception:
            continue

    return sorted(reports, key=lambda x: x["version"])