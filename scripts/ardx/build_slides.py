#!/usr/bin/env python3
"""Render a biweekly progress deck to .pptx from a plain Markdown content file.

The split is deliberate.  Content comes from a Markdown file that is written
from the result records and lives in git, so every number on a slide is
traceable.  Layout comes from a PowerPoint template that the presenter owns,
so visual changes never require touching this script.  The emitted deck is an
ordinary editable .pptx.

Content grammar (one slide per top-level rule):

    # Title                     -> title slide (first occurrence only)
    ## Heading                  -> new content slide
    - bullet                    -> bullet, nesting by two-space indent
    | a | b |                   -> table (consecutive rows are one table)
    ![alt](path/to/image.png)   -> picture filling the body area
    > note                      -> speaker note on the current slide

Run outside the training environment; see requirements/reporting.txt.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from pptx import Presentation
    from pptx.util import Emu, Pt
except ModuleNotFoundError:  # pragma: no cover - depends on the reporting env
    print(
        "python-pptx is missing. This script runs outside the training environment:\n"
        "  conda create -y -n ard-report python=3.11\n"
        "  conda run -n ard-report pip install -r requirements/reporting.txt",
        file=sys.stderr,
    )
    raise SystemExit(2) from None

TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")
TABLE_DIVIDER = re.compile(r"^\|[\s:|-]+\|\s*$")
IMAGE = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)\s*$")
BULLET = re.compile(r"^(\s*)[-*]\s+(.*)$")


@dataclass
class Slide:
    """One parsed slide: a heading plus ordered body blocks and notes."""

    heading: str
    bullets: list[tuple[int, str]] = field(default_factory=list)
    table: list[list[str]] = field(default_factory=list)
    image: str | None = None
    notes: list[str] = field(default_factory=list)


def parse(text: str) -> tuple[str, list[Slide]]:
    """Parse the content grammar into a deck title and its slides."""
    title = ""
    slides: list[Slide] = []
    current: Slide | None = None

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("# ") and not title:
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            current = Slide(heading=line[3:].strip())
            slides.append(current)
            continue
        if current is None:
            continue
        if line.startswith("> "):
            current.notes.append(line[2:].strip())
            continue
        image_match = IMAGE.match(line)
        if image_match:
            current.image = image_match.group(1).strip()
            continue
        if TABLE_DIVIDER.match(line):
            continue
        row_match = TABLE_ROW.match(line)
        if row_match:
            current.table.append([cell.strip() for cell in row_match.group(1).split("|")])
            continue
        bullet_match = BULLET.match(line)
        if bullet_match:
            indent = len(bullet_match.group(1)) // 2
            current.bullets.append((indent, bullet_match.group(2).strip()))
            continue
        current.bullets.append((0, line.strip()))

    return title, slides


def _pick_layout(presentation: Any, wanted: str, fallback: int) -> Any:
    """Return the named layout when the template defines it, else an index."""
    for layout in presentation.slide_layouts:
        if layout.name.strip().lower() == wanted:
            return layout
    return presentation.slide_layouts[fallback]


def _body_placeholder(slide: Any) -> Any:
    """Return the first non-title placeholder, or None when the layout lacks one."""
    for shape in slide.placeholders:
        if shape.placeholder_format.idx != 0:
            return shape
    return None


def _fill_bullets(slide: Any, bullets: list[tuple[int, str]]) -> None:
    body = _body_placeholder(slide)
    if body is None:
        return
    frame = body.text_frame
    frame.clear()
    for position, (indent, content) in enumerate(bullets):
        paragraph = frame.paragraphs[0] if position == 0 else frame.add_paragraph()
        paragraph.text = content
        paragraph.level = min(indent, 4)


def _add_table(slide: Any, rows: list[list[str]], left: Any, top: Any, width: Any, height: Any) -> None:
    columns = max(len(row) for row in rows)
    shape = slide.shapes.add_table(len(rows), columns, left, top, width, height)
    table = shape.table
    for r, row in enumerate(rows):
        for c in range(columns):
            cell = table.cell(r, c)
            cell.text = row[c] if c < len(row) else ""
            for paragraph in cell.text_frame.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(12)


def build(title: str, slides: list[Slide], template: Path | None, output: Path, root: Path) -> None:
    """Write the deck, taking every layout from the template when one is given."""
    presentation = Presentation(str(template)) if template else Presentation()

    if title:
        layout = _pick_layout(presentation, "title slide", 0)
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = title
        subtitle = _body_placeholder(slide)
        if subtitle is not None:
            subtitle.text_frame.text = "ARD progress review"

    content_layout = _pick_layout(presentation, "title and content", 1)
    blank_layout = _pick_layout(presentation, "title only", 5)

    left = Emu(685800)
    top = Emu(1828800)
    width = presentation.slide_width - 2 * left
    height = presentation.slide_height - top - Emu(457200)

    for item in slides:
        needs_blank = bool(item.table) or item.image is not None
        slide = presentation.slides.add_slide(blank_layout if needs_blank else content_layout)
        if slide.shapes.title is not None:
            slide.shapes.title.text = item.heading

        if item.bullets and not needs_blank:
            _fill_bullets(slide, item.bullets)

        if item.table:
            _add_table(slide, item.table, left, top, width, height)
        elif item.image is not None:
            path = (root / item.image).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"slide image not found: {path}")
            slide.shapes.add_picture(str(path), left, top, height=height)

        if item.notes:
            slide.notes_slide.notes_text_frame.text = "\n".join(item.notes)

    output.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(str(output))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("content", type=Path, help="Markdown content file")
    parser.add_argument("--output", type=Path, required=True, help="destination .pptx")
    parser.add_argument("--template", type=Path, default=None, help="template .pptx supplying the layouts")
    args = parser.parse_args()

    if args.template is not None and not args.template.is_file():
        raise SystemExit(f"template not found: {args.template}")

    title, slides = parse(args.content.read_text(encoding="utf-8"))
    if not slides:
        raise SystemExit(f"no '## ' slide headings found in {args.content}")

    build(title, slides, args.template, args.output, args.content.parent)
    print(f"{args.output}  ({len(slides)} content slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
