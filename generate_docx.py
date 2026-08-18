"""
generate_docx.py

Python port of the former generate_docx.js. Reads
output/extracted_requirements.json and output/project_config.json (both
LLM-generated) and produces a fully structured BRS Word document using
python-docx.

All document content — project name, scope, categories, NFRs, constraints,
assumptions, definitions — comes from project_config.json. Nothing is
hardcoded, so the same script works for any software product.

Each requirement bullet is an object: { text, priority } where priority is
one of "M" / "S" / "C" (MoSCoW). Plain-string bullets are tolerated for
stale inputs and default to priority "M".

Usage:
python generate_docx.py [requirements.json] [output.docx] [project_config.json]
"""

from __future__ import annotations

import datetime
import json
import os
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Twips

BLUE = "1F4E79"
LIGHT_BLUE = "D6E4F0"
MID_BLUE = "2E75B6"
GRAY = "595959"

BORDER_COLOR = "CCCCCC"

PRIORITY_COLOR = {
    "M": MID_BLUE,
    "S": GRAY,
    "C": "9B9B9B",
}

DEFAULT_INPUT = "output/extracted_requirements.json"
DEFAULT_OUTPUT = "output/BRS.docx"
DEFAULT_CONFIG = "output/project_config.json"


def _run(paragraph, text, size=11, bold=False, color=None, italic=False):
    """Append an Arial text run to a paragraph."""
    run = paragraph.add_run(str(text))
    run.font.name = "Arial"
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    return run


def _set_paragraph_border(paragraph, edge, color=MID_BLUE, sz=6, space=1):
    """Attach a single border rule to the given paragraph edge."""
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    border = OxmlElement(f"w:{edge}")
    border.set(qn("w:val"), "single")
    border.set(qn("w:sz"), str(sz))
    border.set(qn("w:space"), str(space))
    border.set(qn("w:color"), color)
    p_bdr.append(border)
    p_pr.append(p_bdr)


def _set_cell_shading(cell, fill):
    """Apply a solid background fill to a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _set_cell_borders(cell):
    """Apply hairline borders to every edge of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), BORDER_COLOR)
        tc_borders.append(border)
    tc_pr.append(tc_borders)


def _set_cell_margins(cell, top=80, bottom=80, left=120, right=120):
    """Set internal margins (dxa) of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement("w:tcMar")
    for edge, value in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        margin = OxmlElement(f"w:{edge}")
        margin.set(qn("w:w"), str(value))
        margin.set(qn("w:type"), "dxa")
        tc_mar.append(margin)
    tc_pr.append(tc_mar)


def _set_cell_valign(cell, value="top"):
    """Set vertical alignment of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    v_align = OxmlElement("w:vAlign")
    v_align.set(qn("w:val"), value)
    tc_pr.append(v_align)


def _fill_cell(cell, text, size=10, bold=False, color=None, fill=None):
    """Fill a cell with a single styled run and apply common formatting."""
    cell.text = ""
    paragraph = cell.paragraphs[0]
    _run(paragraph, text, size=size, bold=bold, color=color)
    _set_cell_borders(cell)
    _set_cell_margins(cell)
    _set_cell_valign(cell, "top")
    if fill:
        _set_cell_shading(cell, fill)


def _add_page_number_field(paragraph):
    """Insert a PAGE field into the paragraph (used in the footer)."""
    run = paragraph.add_run()
    run.font.name = "Arial"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor.from_string(GRAY)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)


def _add_toc_field(paragraph):
    """Insert a TOC field the user refreshes once in Word."""
    run = paragraph.add_run()
    run.font.name = "Arial"
    run.font.size = Pt(11)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = 'TOC \\o "1-3" \\h \\z \\u'
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    placeholder = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = "Right-click and select 'Update Field' to build the Table of Contents."
    placeholder.append(text)
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(placeholder)
    run._r.append(fld_end)


def _enable_field_updates(doc):
    """Ask Word to refresh TOC/page fields when the document is opened."""
    settings = doc.settings.element
    update_fields = OxmlElement("w:updateFields")
    update_fields.set(qn("w:val"), "true")
    settings.append(update_fields)


def _heading(doc, text, level):
    """Add a heading with the project's Arial styling."""
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    if level == 1:
        _run(paragraph, text, size=18, bold=True, color=BLUE)
        paragraph.paragraph_format.space_before = Pt(18)
        paragraph.paragraph_format.space_after = Pt(6)
        _set_paragraph_border(paragraph, "bottom", sz=6, space=4)
    elif level == 2:
        _run(paragraph, text, size=14, bold=True, color=MID_BLUE)
        paragraph.paragraph_format.space_before = Pt(12)
        paragraph.paragraph_format.space_after = Pt(4)
    else:
        _run(paragraph, text, size=12, bold=True, color=GRAY)
        paragraph.paragraph_format.space_before = Pt(8)
        paragraph.paragraph_format.space_after = Pt(3)
    return paragraph


def _para(doc, text, size=11):
    """Add a plain body paragraph."""
    paragraph = doc.add_paragraph()
    _run(paragraph, text, size=size)
    return paragraph


def _bullet(doc, text):
    """Add a bulleted paragraph."""
    paragraph = doc.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.left_indent = Inches(0.5)
    paragraph.paragraph_format.first_line_indent = Inches(-0.25)
    _run(paragraph, text, size=11)
    return paragraph


def _table(doc, widths):
    """Create a bordered table with fixed column widths (twips)."""
    table = doc.add_table(rows=0, cols=len(widths))
    table.autofit = False
    table.style = "Table Grid"
    return table


def _set_row_widths(row, widths):
    """Assign explicit column widths across all cells in a row."""
    for cell, width in zip(row.cells, widths):
        cell.width = Twips(width)


def _configure_styles(doc):
    """Set Normal and Heading fonts to the project's Arial palette."""
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal.font.size = Pt(11)
    for name, size, color in (("Heading 1", 18, BLUE), ("Heading 2", 14, MID_BLUE), ("Heading 3", 12, GRAY)):
        style = styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)


def _setup_page(doc):
    """Letter page, 1-inch margins."""
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)


def _cover_page(doc, cfg, today, summary):
    """Render the title page with project metadata table."""
    for _ in range(6):
        doc.add_paragraph()
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _run(paragraph, "BUSINESS REQUIREMENTS SPECIFICATION", size=28, bold=True, color=BLUE)
    doc.add_paragraph()
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_paragraph_border(paragraph, "bottom", sz=12, space=4)
    _run(paragraph, cfg.get("project_name") or "Software System", size=16, color=GRAY)
    for _ in range(3):
        doc.add_paragraph()
    widths = [2400, 3600]
    table = _table(doc, widths)
    rows = [
        ("Date", today),
        ("Version", cfg.get("version") or "1.0"),
        ("Status", cfg.get("status") or "Draft"),
        ("Prepared by", cfg.get("prepared_by") or "VLM Pipeline"),
        ("Source", f"Video analysis ({summary.get('total_chunks', 0)} chunks)"),
    ]
    for label, value in rows:
        row = table.add_row()
        _fill_cell(row.cells[0], label, bold=True, fill=LIGHT_BLUE)
        _fill_cell(row.cells[1], value)
        _set_row_widths(row, widths)
    doc.add_page_break()


def _table_of_contents(doc):
    """Add the Table of Contents page."""
    _heading(doc, "Table of Contents", 1)
    paragraph = doc.add_paragraph()
    _add_toc_field(paragraph)
    doc.add_page_break()


def _section1(doc, cfg):
    """Introduction: purpose, scope, definitions & acronyms."""
    _heading(doc, "1. Introduction", 1)
    _heading(doc, "1.1 Purpose", 2)
    _para(doc, cfg.get("purpose") or "")
    _heading(doc, "1.2 Scope", 2)
    _para(doc, cfg.get("scope") or "")
    _heading(doc, "1.3 Definitions & Acronyms", 2)
    definitions = cfg.get("definitions") or []
    if definitions:
        widths = [2200, 7160]
        table = _table(doc, widths)
        row = table.add_row()
        _fill_cell(row.cells[0], "Term", bold=True, color="FFFFFF", fill=BLUE)
        _fill_cell(row.cells[1], "Definition", bold=True, color="FFFFFF", fill=BLUE)
        _set_row_widths(row, widths)
        for term, definition in definitions:
            row = table.add_row()
            _fill_cell(row.cells[0], term, bold=True)
            _fill_cell(row.cells[1], definition)
            _set_row_widths(row, widths)
    else:
        _para(doc, "See project glossary.")
    doc.add_page_break()


def _section2(doc, cfg):
    """Overall description: perspective, user classes, environment."""
    _heading(doc, "2. Overall Description", 1)
    _heading(doc, "2.1 Product Perspective", 2)
    _para(doc, cfg.get("scope") or "")
    _heading(doc, "2.2 User Classes", 2)
    user_classes = cfg.get("user_classes") or [
        "End Users: interact with the system to complete primary tasks"
    ]
    for user_class in user_classes:
        _bullet(doc, user_class)
    _heading(doc, "2.3 Operating Environment", 2)
    environment = cfg.get("operating_environment") or [
        "Desktop or web-based application"
    ]
    for item in environment:
        _bullet(doc, item)
    doc.add_page_break()


def _section3(doc, cfg, all_bullets):
    """Functional requirements tables, categorized by inferred keywords."""
    _heading(doc, "3. Functional Requirements", 1)
    _para(
        doc,
        "The following functional requirements are organized by feature area. "
        "Each requirement was extracted from video content and refined through "
        "automated multi-judge analysis. Priority follows MoSCoW (Must / Should / "
        "Could Have), classified by business criticality rather than literal wording.",
    )
    categories = cfg.get("categories") or [
        {"id": "FR-01", "title": "General Requirements", "keywords": []}
    ]

    def categorize(bullet):
        lower = bullet["text"].lower()
        for category in categories:
            for keyword in category.get("keywords") or []:
                if keyword.lower() in lower:
                    return category
        return categories[-1]

    category_map = {cat["id"]: {**cat, "bullets": []} for cat in categories}
    for bullet in all_bullets:
        category_map[categorize(bullet)["id"]]["bullets"].append(bullet)

    counter = [0]

    def next_id():
        counter[0] += 1
        return f"REQ-{counter[0]:03d}"

    widths = [1200, 7560, 600]
    for category in categories:
        populated = category_map[category["id"]]
        if not populated["bullets"]:
            continue
        _heading(doc, f"{category['id']}: {category['title']}", 2)
        table = _table(doc, widths)
        row = table.add_row()
        for cell, label in zip(row.cells, ("ID", "Requirement", "P")):
            _fill_cell(cell, label, bold=True, color="FFFFFF", fill=BLUE)
        _set_row_widths(row, widths)
        for idx, bullet in enumerate(populated["bullets"]):
            shade = "FFFFFF" if idx % 2 == 0 else "F5F9FC"
            priority = bullet.get("priority") or "M"
            row = table.add_row()
            _fill_cell(row.cells[0], next_id(), fill=shade, color=GRAY)
            _fill_cell(row.cells[1], bullet["text"], fill=shade)
            _fill_cell(
                row.cells[2],
                priority,
                fill=shade,
                bold=True,
                color=PRIORITY_COLOR.get(priority, MID_BLUE),
            )
            _set_row_widths(row, widths)
        legend = doc.add_paragraph()
        _run(
            legend,
            "P = Priority: M = Must Have, S = Should Have, C = Could Have",
            size=8,
            color=GRAY,
            italic=True,
        )
    doc.add_page_break()


def _section4(doc, cfg):
    """Non-functional requirements grouped by category."""
    _heading(doc, "4. Non-Functional Requirements", 1)
    nfrs = cfg.get("non_functional_requirements") or {}
    if nfrs:
        for index, (category, items) in enumerate(nfrs.items(), start=1):
            _heading(doc, f"4.{index} {category}", 2)
            for item in items or []:
                _bullet(doc, item)
    else:
        _para(
            doc,
            "Non-functional requirements to be defined in collaboration with stakeholders.",
        )
    doc.add_page_break()


def _section5(doc, cfg):
    """Constraints & assumptions."""
    _heading(doc, "5. Constraints & Assumptions", 1)
    _heading(doc, "5.1 Constraints", 2)
    constraints = cfg.get("constraints") or [
        "Requirements were extracted from video content and may not be exhaustive."
    ]
    for constraint in constraints:
        _bullet(doc, constraint)
    _heading(doc, "5.2 Assumptions", 2)
    assumptions = cfg.get("assumptions") or [
        "Users have basic familiarity with the application domain."
    ]
    for assumption in assumptions:
        _bullet(doc, assumption)
    doc.add_page_break()


def _appendix(doc, all_bullets, summary, today):
    """Pipeline metadata table."""
    _heading(doc, "Appendix A: Analysis Metadata", 1)
    _para(
        doc,
        "The following metrics describe the automated extraction pipeline run "
        "that produced this document.",
    )
    counts = {"M": 0, "S": 0, "C": 0}
    for bullet in all_bullets:
        priority = bullet.get("priority") or "M"
        counts[priority] = counts.get(priority, 0) + 1
    rows = [
        ("Total chunks processed", str(summary.get("total_chunks", 0))),
        ("Successfully refined", str(summary.get("successfully_refined", 0))),
        ("Cross-block duplicates removed", str(summary.get("duplicate_bullets_removed_cross_block", 0))),
        ("Total requirements extracted", str(len(all_bullets))),
        ("Priority breakdown", f"M: {counts['M']}  S: {counts['S']}  C: {counts['C']}"),
        ("Inference time", f"{summary.get('inference_time_sec', 0)}s"),
        ("Judges refinement time", f"{summary.get('judges_time_sec', 0)}s"),
        ("Total pipeline time", f"{summary.get('total_time_sec', 0)}s"),
        ("Generated on", today),
    ]
    widths = [3000, 3000]
    table = _table(doc, widths)
    row = table.add_row()
    _fill_cell(row.cells[0], "Metric", bold=True, fill=LIGHT_BLUE)
    _fill_cell(row.cells[1], "Value", bold=True, fill=LIGHT_BLUE)
    _set_row_widths(row, widths)
    for label, value in rows:
        row = table.add_row()
        _fill_cell(row.cells[0], label)
        _fill_cell(row.cells[1], value, bold=True)
        _set_row_widths(row, widths)


def _header_footer(doc, cfg, today):
    """Running header (document title) and footer (date + page number)."""
    section = doc.sections[0]
    header = section.header
    header_paragraph = header.paragraphs[0]
    header_paragraph.text = ""
    _run(header_paragraph, "Business Requirements Specification", size=10, bold=True, color=BLUE)
    _run(header_paragraph, f"   |   {cfg.get('confidentiality') or 'CONFIDENTIAL'}", size=9, color=GRAY)
    _set_paragraph_border(header_paragraph, "bottom", sz=6, space=1)

    footer = section.footer
    footer_paragraph = footer.paragraphs[0]
    footer_paragraph.text = ""
    footer_paragraph.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
    _run(footer_paragraph, today, size=9, color=GRAY)
    tab = footer_paragraph.add_run("\t")
    tab.font.name = "Arial"
    tab.font.size = Pt(9)
    _run(footer_paragraph, "Page ", size=9, color=GRAY)
    _add_page_number_field(footer_paragraph)
    _set_paragraph_border(footer_paragraph, "top", sz=6, space=1)


def generate_docx(input_file, output_file, config_file):
    """
    Render the BRS Word document from the two pipeline JSON artifacts.

    Args:
        input_file:  Path to extracted_requirements.json.
        output_file: Path to write BRS.docx to.
        config_file: Path to project_config.json.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    summary = raw.get("summary") or {}
    all_bullets = []
    for block in raw.get("requirements") or []:
        for bullet in block.get("bullets") or []:
            if isinstance(bullet, str):
                all_bullets.append({"text": bullet, "priority": "M"})
            else:
                all_bullets.append(
                    {"text": bullet.get("text", ""), "priority": bullet.get("priority", "M")}
                )

    today = datetime.date.today().strftime("%B %d, %Y")

    doc = Document()
    _setup_page(doc)
    _configure_styles(doc)
    _cover_page(doc, cfg, today, summary)
    _table_of_contents(doc)
    _section1(doc, cfg)
    _section2(doc, cfg)
    _section3(doc, cfg, all_bullets)
    _section4(doc, cfg)
    _section5(doc, cfg)
    _appendix(doc, all_bullets, summary, today)
    _header_footer(doc, cfg, today)
    _enable_field_updates(doc)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    doc.save(output_file)
    print(f"BRS written to {output_file}")
    print(f"  Project: {cfg.get('project_name') or 'Unknown'}")
    print(f"  Categories: {len(cfg.get('categories') or [])}")
    print(f"  Requirements: {len(all_bullets)}")


def main() -> None:
    """CLI entry point mirroring the old Node invocation."""
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    output_file = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT
    config_file = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_CONFIG
    generate_docx(input_file, output_file, config_file)


if __name__ == "__main__":
    main()
