import io
import re
import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, ListFlowable, ListItem, Table, TableStyle
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ─── Font Registration ────────────────────────────────────────────────────────

FONTS_DIR = Path(__file__).parent / "fonts"

def register_fonts() -> tuple[str, str]:
    """
    Register Unicode-capable NotoSans fonts.
    Returns (regular_font, bold_font) names to use in styles.
    Falls back to Helvetica if fonts are unavailable.
    """
    try:
        pdfmetrics.registerFont(
            TTFont('NotoSans', str(FONTS_DIR / 'NotoSans-Regular.ttf'))
        )
        pdfmetrics.registerFont(
            TTFont('NotoSans-Bold', str(FONTS_DIR / 'NotoSans-Bold.ttf'))
        )
        return 'NotoSans', 'NotoSans-Bold'
    except Exception as e:
        print(f"Font registration failed, falling back to Helvetica: {e}")
        return 'Helvetica', 'Helvetica-Bold'

BODY_FONT, BOLD_FONT = register_fonts()

# ─── Color Palette ────────────────────────────────────────────────────────────

PRIMARY    = HexColor("#1a1d27")
ACCENT     = HexColor("#6c63ff")
TEXT       = HexColor("#2d2d2d")
MID_GRAY   = HexColor("#888888")
DARK       = HexColor("#1a1a1a")

# ─── Markdown Utilities ───────────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    """Remove markdown formatting that ReportLab cannot render."""
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'`(.*?)`', r'\1', text)
    return text


def is_markdown_header(text: str) -> tuple[bool, str]:
    """Check if text is a markdown header. Returns (is_header, clean_text)."""
    match = re.match(r'^#{1,6}\s+(.+)$', text.strip())
    if match:
        return True, match.group(1).strip()
    return False, text


def safe_html(text: str) -> str:
    """Escape HTML special characters for ReportLab paragraphs."""
    return (text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;"))

# ─── Table Utilities ──────────────────────────────────────────────────────────

def parse_markdown_table(lines: list[str]) -> list[list[str]] | None:
    """
    Parse a list of markdown table lines into a 2D list of cell strings.
    Returns None if the lines don't form a valid table.
    """
    rows = []
    for line in lines:
        if not line.strip():
            break
        # Skip separator rows (e.g. |---|---|)
        if re.match(r'^\|?[\s\-\|:]+\|?$', line):
            continue
        if '|' not in line:
            break
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if cells:
            rows.append(cells)
    return rows if len(rows) >= 2 else None


def build_reportlab_table(rows: list[list[str]]) -> Table:
    """Convert a 2D list of strings into a styled ReportLab Table."""
    from reportlab.platypus import Paragraph
    from reportlab.lib.styles import ParagraphStyle

    max_cols = max(len(r) for r in rows)
    normalized = [r + [''] * (max_cols - len(r)) for r in rows]

    header_style = ParagraphStyle(
        'TableHeader',
        fontName=BOLD_FONT,
        fontSize=9,
        textColor=HexColor("#6c63ff"),
        leading=12,
    )
    cell_style = ParagraphStyle(
        'TableCell',
        fontName=BODY_FONT,
        fontSize=8,
        textColor=HexColor("#cccccc"),
        leading=11,
    )

    # Convert all cells to Paragraph objects for proper text wrapping
    para_rows = []
    for i, row in enumerate(normalized):
        style = header_style if i == 0 else cell_style
        para_row = [Paragraph(safe_html(strip_markdown(str(cell))), style) for cell in row]
        para_rows.append(para_row)

    available_width = 16 * cm
    col_width = available_width / max_cols

    table = Table(para_rows, colWidths=[col_width] * max_cols, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0),  HexColor("#1a1d27")),
        ('BACKGROUND',    (0, 1), (-1, -1), HexColor("#13151f")),
        ('TOPPADDING',    (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING',   (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 8),
        ('GRID',          (0, 0), (-1, -1), 0.5, HexColor("#2a2d3e")),
        ('ROWBACKGROUNDS',(0, 1), (-1, -1), [HexColor("#13151f"), HexColor("#1a1d27")]),
        ('VALIGN',        (0, 0), (-1, -1), 'TOP'),
    ]))
    return table

# ─── Styles ───────────────────────────────────────────────────────────────────

def build_styles() -> dict:
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontSize=22,
            textColor=PRIMARY,
            spaceAfter=6,
            alignment=TA_LEFT,
            fontName=BOLD_FONT,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading1"],
            fontSize=13,
            textColor=ACCENT,
            spaceBefore=20,
            spaceAfter=8,
            fontName=BOLD_FONT,
        ),
        "subheading": ParagraphStyle(
            "SubHeading",
            parent=base["Normal"],
            fontSize=10,
            textColor=DARK,
            fontName=BOLD_FONT,
            spaceBefore=10,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "BodyText",
            parent=base["Normal"],
            fontSize=10,
            textColor=TEXT,
            spaceAfter=8,
            leading=16,
            alignment=TA_JUSTIFY,
            fontName=BODY_FONT,
        ),
        "highlight": ParagraphStyle(
            "Highlight",
            parent=base["Normal"],
            fontSize=9,
            textColor=TEXT,
            spaceAfter=4,
            leading=14,
            leftIndent=12,
            fontName=BODY_FONT,
        ),
        "source": ParagraphStyle(
            "Source",
            parent=base["Normal"],
            fontSize=8,
            textColor=MID_GRAY,
            spaceAfter=4,
            leading=12,
            fontName=BODY_FONT,
        ),
    }

# ─── Content Renderer ─────────────────────────────────────────────────────────

def render_content(content: str, styles: dict, story: list) -> None:
    """
    Parse section content and append rendered flowables to story.
    Handles: markdown headers, markdown tables, and plain paragraphs.
    """
    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Markdown subheading
        is_header, header_text = is_markdown_header(line)
        if is_header:
            header_text = safe_html(strip_markdown(header_text))
            story.append(Paragraph(header_text, styles["subheading"]))
            i += 1
            continue

        # Markdown table — collect all consecutive table lines
        if '|' in line:
            table_lines = []
            while i < len(lines) and ('|' in lines[i] or
                  re.match(r'^\s*[\-\|:]+\s*$', lines[i])):
                table_lines.append(lines[i])
                i += 1
            rows = parse_markdown_table(table_lines)
            if rows:
                story.append(Spacer(1, 8))
                story.append(build_reportlab_table(rows))
                story.append(Spacer(1, 8))
            else:
                # Fallback: render as plain text
                for tl in table_lines:
                    text = safe_html(strip_markdown(tl))
                    story.append(Paragraph(text, styles["body"]))
            continue

        # Plain paragraph
        text = safe_html(strip_markdown(line))
        story.append(Paragraph(text, styles["body"]))
        i += 1

# ─── PDF Generator ────────────────────────────────────────────────────────────

def generate_pdf(report_json: dict) -> bytes:
    """
    Convert a report JSON into a PDF.
    Returns the PDF as bytes.
    """
    buffer = io.BytesIO()
    styles = build_styles()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2.5*cm,
        rightMargin=2.5*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm,
    )

    story = []

    # ── Title ─────────────────────────────────────────────────────────────────
    title = report_json.get("title", "Research Report")
    story.append(Paragraph(safe_html(title), styles["title"]))
    story.append(HRFlowable(
        width="100%", thickness=2, color=ACCENT, spaceAfter=20
    ))

    # ── Sections ──────────────────────────────────────────────────────────────
    for section in report_json.get("sections", []):
        section_title = section.get("title", "")
        content = section.get("content", "")
        highlights = section.get("highlights", [])

        story.append(Paragraph(safe_html(strip_markdown(section_title)), styles["heading"]))
        render_content(content, styles, story)

        if highlights:
            story.append(Spacer(1, 6))
            bullet_items = [
                ListItem(
                    Paragraph(safe_html(strip_markdown(h)), styles["highlight"]),
                    bulletColor=ACCENT,
                    leftIndent=20,
                )
                for h in highlights
            ]
            story.append(ListFlowable(bullet_items, bulletType="bullet", start="•"))
            story.append(Spacer(1, 8))

    # ── Sources ───────────────────────────────────────────────────────────────
    sources = report_json.get("sources", [])
    if sources:
        story.append(Spacer(1, 16))
        story.append(HRFlowable(
            width="100%", thickness=1, color=MID_GRAY, spaceAfter=12
        ))
        story.append(Paragraph("Sources", styles["heading"]))
        for i, source in enumerate(sources, 1):
            source_title = safe_html(source.get("title", "Unknown source"))
            source_url = source.get("url", "")
            if source_url:
                text = f'{i}. <a href="{source_url}" color="#6c63ff">{source_title}</a>'
            else:
                text = f"{i}. {source_title}"
            story.append(Paragraph(text, styles["source"]))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()