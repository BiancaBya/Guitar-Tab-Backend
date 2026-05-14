import math
from datetime import datetime
from io import BytesIO
from typing import Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


TabNote = Dict[str, Optional[float]]


STRING_LABELS = ["e", "B", "G", "D", "A", "E"]
ITEMS_PER_ROW = 16
ROWS_PER_PAGE = 3

PAGE_WIDTH, PAGE_HEIGHT = landscape(A4)
MARGIN_X = 18 * mm
MARGIN_TOP = 16 * mm
MARGIN_BOTTOM = 14 * mm
TITLE_HEIGHT = 24 * mm
ROW_HEIGHT = 47 * mm
STRING_GAP = 6.4 * mm
TAB_LEFT_LABEL_WIDTH = 8 * mm

BG_COLOR = colors.HexColor("#0F172A")
PANEL_COLOR = colors.HexColor("#111827")
LINE_COLOR = colors.HexColor("#64748B")
TEXT_COLOR = colors.HexColor("#E5E7EB")
MUTED_COLOR = colors.HexColor("#94A3B8")
ACCENT_COLOR = colors.HexColor("#22D3EE")
GREEN_COLOR = colors.HexColor("#22C55E")


def _safe_text(value: object, fallback: str = "Untitled") -> str:
    text = str(value or fallback).strip()
    return text if text else fallback


def _draw_wrapped_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    font_name: str,
    font_size: int,
    color=TEXT_COLOR,
):
    words = text.split()
    lines: List[str] = []
    current = ""

    for word in words:
        candidate = f"{current} {word}".strip()
        if stringWidth(candidate, font_name, font_size) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    c.setFont(font_name, font_size)
    c.setFillColor(color)

    for index, line in enumerate(lines[:2]):
        c.drawString(x, y - index * (font_size + 3), line)


def _group_notes_by_time(notes: List[TabNote]) -> List[Dict[str, object]]:
    valid_notes = [
        note for note in notes
        if note.get("string") is not None and note.get("fret") is not None
    ]

    valid_notes.sort(key=lambda item: float(item.get("time") or 0))

    groups: List[Dict[str, object]] = []

    for note in valid_notes:
        note_time = float(note.get("time") or 0)
        existing_group = None

        for group in groups:
            if abs(float(group["time"]) - note_time) < 0.01:
                existing_group = group
                break

        if existing_group is None:
            groups.append({"time": note_time, "notes": [note]})
        else:
            existing_group_notes = existing_group["notes"]
            if isinstance(existing_group_notes, list):
                existing_group_notes.append(note)

    return groups


def _draw_page_background(c: canvas.Canvas):
    c.setFillColor(BG_COLOR)
    c.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)


def _draw_header(
    c: canvas.Canvas,
    filename: str,
    variant_label: str,
    note_count: int,
    page_number: int,
    total_pages: int,
):
    _draw_page_background(c)

    c.setFillColor(TEXT_COLOR)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(MARGIN_X, PAGE_HEIGHT - MARGIN_TOP, "TabGenius Export")

    badge_text = variant_label
    badge_width = stringWidth(badge_text, "Helvetica-Bold", 10) + 12 * mm
    badge_height = 8 * mm
    badge_x = PAGE_WIDTH - MARGIN_X - badge_width
    badge_y = PAGE_HEIGHT - MARGIN_TOP - 1 * mm

    c.setFillColor(GREEN_COLOR if "Beginner" in variant_label else ACCENT_COLOR)
    c.roundRect(badge_x, badge_y - badge_height + 2, badge_width, badge_height, 4 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(badge_x + badge_width / 2, badge_y - badge_height / 2 - 1, badge_text)

    _draw_wrapped_text(
        c,
        filename,
        MARGIN_X,
        PAGE_HEIGHT - MARGIN_TOP - 11 * mm,
        PAGE_WIDTH - 2 * MARGIN_X - badge_width - 8 * mm,
        "Helvetica-Bold",
        13,
        TEXT_COLOR,
    )

    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")
    c.setFont("Helvetica", 8)
    c.setFillColor(MUTED_COLOR)
    c.drawString(MARGIN_X, PAGE_HEIGHT - MARGIN_TOP - 23 * mm, f"Notes: {note_count}  |  Generated: {generated_at}")

    c.setStrokeColor(colors.HexColor("#1E293B"))
    c.setLineWidth(1)
    c.line(MARGIN_X, PAGE_HEIGHT - MARGIN_TOP - 29 * mm, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - MARGIN_TOP - 29 * mm)

    c.setFont("Helvetica", 8)
    c.setFillColor(MUTED_COLOR)
    c.drawRightString(PAGE_WIDTH - MARGIN_X, MARGIN_BOTTOM - 4 * mm, f"Page {page_number} / {total_pages}")


def _draw_tab_row(c: canvas.Canvas, row_groups: List[Dict[str, object]], y_top: float, row_number: int):
    panel_x = MARGIN_X
    panel_y = y_top - ROW_HEIGHT + 4 * mm
    panel_width = PAGE_WIDTH - 2 * MARGIN_X
    panel_height = ROW_HEIGHT - 3 * mm

    c.setFillColor(PANEL_COLOR)
    c.roundRect(panel_x, panel_y, panel_width, panel_height, 5 * mm, stroke=0, fill=1)

    label_x = panel_x + 5 * mm
    tab_x = panel_x + TAB_LEFT_LABEL_WIDTH + 8 * mm
    tab_width = panel_width - TAB_LEFT_LABEL_WIDTH - 14 * mm
    first_line_y = y_top - 11 * mm

    c.setFont("Helvetica", 7)
    c.setFillColor(MUTED_COLOR)
    c.drawString(panel_x + 5 * mm, y_top - 4 * mm, f"Line {row_number}")

    for string_index, label in enumerate(STRING_LABELS):
        line_y = first_line_y - string_index * STRING_GAP

        c.setFont("Courier-Bold", 9)
        c.setFillColor(MUTED_COLOR)
        c.drawString(label_x, line_y - 3, label)

        c.setStrokeColor(LINE_COLOR)
        c.setLineWidth(0.45)
        c.line(tab_x, line_y, tab_x + tab_width, line_y)

        c.setStrokeColor(colors.HexColor("#334155"))
        c.setLineWidth(0.6)
        c.line(tab_x, line_y - 4, tab_x, line_y + 4)
        c.line(tab_x + tab_width, line_y - 4, tab_x + tab_width, line_y + 4)

    column_width = tab_width / ITEMS_PER_ROW

    for col_index, group in enumerate(row_groups):
        center_x = tab_x + column_width * col_index + column_width / 2
        group_notes = group.get("notes", [])

        if not isinstance(group_notes, list):
            continue

        for note in group_notes:
            try:
                string_number = int(note.get("string"))
                fret_number = int(note.get("fret"))
            except (TypeError, ValueError):
                continue

            if string_number < 1 or string_number > 6:
                continue

            line_y = first_line_y - (string_number - 1) * STRING_GAP
            fret_text = str(fret_number)
            fret_width = max(5.5 * mm, stringWidth(fret_text, "Courier-Bold", 11) + 2.5 * mm)

            c.setFillColor(PANEL_COLOR)
            c.roundRect(center_x - fret_width / 2, line_y - 4.1, fret_width, 8.2, 1.4 * mm, stroke=0, fill=1)

            c.setFillColor(ACCENT_COLOR)
            c.setFont("Courier-Bold", 11)
            c.drawCentredString(center_x, line_y - 3.6, fret_text)


def build_tablature_pdf(filename: str, notes: List[TabNote], variant: str) -> BytesIO:
    grouped = _group_notes_by_time(notes)
    chunks = [grouped[index:index + ITEMS_PER_ROW] for index in range(0, len(grouped), ITEMS_PER_ROW)]

    if not chunks:
        chunks = [[]]

    total_pages = max(1, math.ceil(len(chunks) / ROWS_PER_PAGE))
    variant_label = "Beginner Friendly" if variant == "beginner" else "Original Tab"

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=landscape(A4))
    c.setTitle(f"{_safe_text(filename)} - {variant_label}")
    c.setAuthor("TabGenius")

    for page_index in range(total_pages):
        _draw_header(c, _safe_text(filename), variant_label, len(grouped), page_index + 1, total_pages)

        page_chunks = chunks[page_index * ROWS_PER_PAGE:(page_index + 1) * ROWS_PER_PAGE]
        row_start_y = PAGE_HEIGHT - MARGIN_TOP - TITLE_HEIGHT - 10 * mm

        for local_row_index, row_groups in enumerate(page_chunks):
            y_top = row_start_y - local_row_index * ROW_HEIGHT
            global_row_number = page_index * ROWS_PER_PAGE + local_row_index + 1
            _draw_tab_row(c, row_groups, y_top, global_row_number)

        if page_index == 0 and len(grouped) == 0:
            c.setFillColor(MUTED_COLOR)
            c.setFont("Helvetica", 12)
            c.drawCentredString(PAGE_WIDTH / 2, PAGE_HEIGHT / 2, "No playable notes available for this variant.")

        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer
