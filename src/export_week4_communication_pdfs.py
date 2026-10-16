"""Export the Week 4 update and CURO proposal as polished PDF attachments."""

from __future__ import annotations

import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageTemplate,
    Paragraph,
    Spacer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "output" / "pdf"


def inline_markup(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"&lt;(https?://.+?)&gt;", r"<link href='\1' color='#2563EB'>\1</link>", text)
    return text


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DocumentTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#15324B"),
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "meta": ParagraphStyle(
            "Metadata",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceAfter=3,
        ),
        "heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#15324B"),
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "DocumentBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.2,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "DocumentBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.5,
            leftIndent=15,
            firstLineIndent=-8,
            bulletIndent=5,
            textColor=colors.HexColor("#1F2937"),
            spaceAfter=4,
        ),
        "question": ParagraphStyle(
            "ResearchQuestion",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=10,
            leading=14,
            leftIndent=14,
            rightIndent=14,
            borderColor=colors.HexColor("#CBD5E1"),
            borderWidth=0.7,
            borderPadding=8,
            backColor=colors.HexColor("#F8FAFC"),
            spaceAfter=8,
        ),
    }


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.setLineWidth(0.5)
    canvas.line(0.72 * inch, 0.52 * inch, 7.78 * inch, 0.52 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(0.72 * inch, 0.34 * inch, "FlashAttention Profiling Research")
    canvas.drawRightString(7.78 * inch, 0.34 * inch, f"Page {document.page}")
    canvas.restoreState()


def document(path: Path) -> BaseDocTemplate:
    doc = BaseDocTemplate(
        str(path),
        pagesize=LETTER,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.68 * inch,
        title=path.stem.replace("_", " "),
        author="Giang Nguyen",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="research", frames=[frame], onPage=footer))
    return doc


def proposal_story() -> list[object]:
    path = REPO_ROOT / "docs" / "proposals" / "CURO_PROPOSAL_FIRST_DRAFT.md"
    text = path.read_text()
    title = text.splitlines()[0].removeprefix("# ")
    body = text.split("## Research proposal\n\n", 1)[1].split(
        "\n\n## Notes for mentor review", 1
    )[0]
    style = styles()
    story: list[object] = [Paragraph(inline_markup(title), style["title"])]
    for label, value in (
        ("Status", "First draft for mentor review"),
        ("Target term", "Spring 2027, pending renewed confirmation"),
        ("Faculty supervisor", "________________________________"),
        ("Scope", "Dense prefill attention on NVIDIA GPUs"),
    ):
        story.append(Paragraph(f"<b>{label}:</b> {escape(value)}", style["meta"]))
    story.append(Spacer(1, 9))
    story.append(Paragraph("Research Proposal", style["heading"]))
    for paragraph in body.split("\n\n"):
        story.append(Paragraph(inline_markup(paragraph), style["body"]))
    return story


def update_story() -> list[object]:
    path = REPO_ROOT / "docs" / "updates" / "WEEK4_RESEARCH_UPDATE.md"
    lines = path.read_text().splitlines()
    style = styles()
    story: list[object] = [
        Paragraph(inline_markup(lines[0].removeprefix("# ")), style["title"]),
        Paragraph("Dense-prefill attention configuration selection", style["meta"]),
        Spacer(1, 5),
    ]
    index = 1
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline_markup(line[3:]), style["heading"]))
        elif line.startswith("- "):
            story.append(
                Paragraph(inline_markup(line[2:]), style["bullet"], bulletText="•")
            )
        elif line.startswith("Repository: "):
            story.append(Spacer(1, 5))
            story.append(Paragraph(inline_markup(line), style["meta"]))
        else:
            paragraph_lines = [line]
            while index + 1 < len(lines):
                following = lines[index + 1].strip()
                if not following or following.startswith(("## ", "- ")):
                    break
                index += 1
                paragraph_lines.append(following)
            paragraph = " ".join(paragraph_lines)
            paragraph_style = (
                style["question"]
                if paragraph.startswith("Can a lightweight")
                else style["body"]
            )
            story.append(Paragraph(inline_markup(paragraph), paragraph_style))
        index += 1
    return story


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = (
        (OUTPUT_DIR / "CURO_Proposal_First_Draft.pdf", proposal_story()),
        (OUTPUT_DIR / "Week4_Research_Update.pdf", update_story()),
    )
    for path, story in outputs:
        document(path).build(story)
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
