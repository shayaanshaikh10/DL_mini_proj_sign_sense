"""
Build the project report (Word .docx) from the college template.

Keeps the template's cover page, certificate, header and table-of-contents style,
removes its instruction pages, and fills in the SignSense content, figures,
tables and code images. Formatting follows the template's guidelines:
Times New Roman, 16 bold titles, 14 bold subtitles, 12 body text, justified,
1.5 line spacing, 8 pt paragraph spacing, page numbers bottom-centre from the
Introduction (Abstract = page i).

Run from the project root:
    python scripts/build_report.py                      # first pass
    python scripts/build_report.py --pages pages.json   # with TOC page numbers

App screenshots are inserted if these files exist in "mini project report/":
    app_home.png, app_prediction.png
"""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from scripts.report_text import CHAPTERS, ABSTRACT, REFERENCES, TOC_ENTRIES

REPORT_DIR = config.BASE_DIR / "mini project report"
TEMPLATE_PATH = REPORT_DIR / "DL Course Project Format 2026-27.docx"
OUTPUT_PATH = REPORT_DIR / "SignSense_Report.docx"

TITLE = "SignSense: Deep CNN-Based Traffic Sign Recognition"
# (name, roll number). The cover lists students by roll number; the certificate
# keeps the order the team chose.
COVER_STUDENTS = [("Yash Rane", 39), ("Dishant Sasane", 42), ("Shayaan Shaikh", 45)]
CERTIFICATE_STUDENTS = [("Shayaan Shaikh", 45), ("Yash Rane", 39), ("Dishant Sasane", 42)]
COURSE_LINE = ("Submitted in partial fulfillment of the requirements of the Course Project for "
               "Deep Learning in Semester VII of Final Year Artificial Intelligence & Data Science")

FONT = '<w:rFonts w:ascii="Times New Roman" w:eastAsia="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'
CODE_FONT = '<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>'
BODY_SPACING = '<w:spacing w:before="0" w:after="160" w:line="360" w:lineRule="auto"/>'
TEXT_WIDTH_INCHES = 6.2
EMU_PER_INCH = 914400
BULLET_NUM_ID = 90          # numbering id we add for bullet lists
FOOTER_REL_ID = "rIdFooterSignSense"


# ---------------------------------------------------------------
# Small XML builders
# ---------------------------------------------------------------
def run(text, size=24, bold=False, italic=False, highlight=False):
    """One text run in Times New Roman."""
    props = FONT + ("<w:b/>" if bold else "") + ("<w:i/>" if italic else "")
    props += '<w:highlight w:val="yellow"/>' if highlight else ""
    props += f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>'
    return f'<w:r><w:rPr>{props}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def rich_runs(text, size=24):
    """Turn text with **bold** parts into runs."""
    parts = re.split(r"(\*\*.+?\*\*)", text)
    return "".join(run(p[2:-2], size, bold=True) if p.startswith("**") else run(p, size)
                   for p in parts if p)


def paragraph(text, align="both", extra_ppr=""):
    """Normal body paragraph: 12 pt, justified, 1.5 spacing, 8 pt after."""
    return (f'<w:p><w:pPr>{extra_ppr}{BODY_SPACING}<w:jc w:val="{align}"/></w:pPr>'
            f'{rich_runs(text)}</w:p>')


def bullet(text):
    """Bulleted body paragraph."""
    numbering = f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{BULLET_NUM_ID}"/></w:numPr>'
    return (f'<w:p><w:pPr><w:pStyle w:val="ListParagraph"/>{numbering}'
            f'<w:spacing w:before="0" w:after="160" w:line="360" w:lineRule="auto"/>'
            f'<w:ind w:left="720" w:hanging="360"/><w:jc w:val="both"/></w:pPr>{rich_runs(text)}</w:p>')


def heading(text, level):
    """Chapter title (16 pt bold) or sub-title (14 pt bold). Chapters flow on to keep 12-15 pages."""
    size = 32 if level == 1 else 28
    before = 360 if level == 1 else 200
    spacing = f'<w:spacing w:before="{before}" w:after="160" w:line="360" w:lineRule="auto"/>'
    return (f'<w:p><w:pPr><w:keepNext/>{spacing}<w:jc w:val="left"/>'
            f'<w:outlineLvl w:val="{level - 1}"/></w:pPr>{run(text, size, bold=True)}</w:p>')


def caption(text):
    """Centred figure/table caption."""
    return (f'<w:p><w:pPr><w:spacing w:before="0" w:after="160" w:line="360" w:lineRule="auto"/>'
            f'<w:jc w:val="center"/></w:pPr>{run(text, 24)}</w:p>')


def placeholder_box(text):
    """Bordered placeholder paragraph for a missing screenshot."""
    border = '<w:pBdr>' + "".join(
        f'<w:{side} w:val="single" w:sz="8" w:space="8" w:color="808080"/>'
        for side in ["top", "left", "bottom", "right"]) + '</w:pBdr>'
    return (f'<w:p><w:pPr>{border}<w:spacing w:before="600" w:after="600"/>'
            f'<w:jc w:val="center"/></w:pPr>{run(text, 24, italic=True, highlight=True)}</w:p>')


# ---------------------------------------------------------------
# Images
# ---------------------------------------------------------------
class MediaRegistry:
    """Collects images to add to the .docx and returns relationship ids."""

    def __init__(self):
        self.images = []   # (relationship id, zip path, source file)

    def add(self, source_path):
        index = len(self.images) + 1
        rel_id = f"rIdSignSenseImg{index}"
        self.images.append((rel_id, f"word/media/signsense_{index}.png", source_path))
        return rel_id, index


def image_paragraph(media, source_path, width_inches, max_height_inches=8.0):
    """Centred inline picture, scaled to the given width (and capped in height)."""
    rel_id, index = media.add(source_path)
    with Image.open(source_path) as img:
        px_width, px_height = img.size
    width = min(width_inches, TEXT_WIDTH_INCHES)
    height = width * px_height / px_width
    if height > max_height_inches:
        height = max_height_inches
        width = height * px_width / px_height
    cx, cy = int(width * EMU_PER_INCH), int(height * EMU_PER_INCH)
    doc_pr_id = 5000 + index
    graphic = (
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{doc_pr_id}" name="figure{index}.png"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>')
    drawing = (f'<w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
               f'<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{doc_pr_id}" name="Figure {index}"/>'
               f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
               f'{graphic}</wp:inline></w:drawing>')
    return (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="120" w:after="60"/><w:jc w:val="center"/></w:pPr>'
            f'<w:r>{drawing}</w:r></w:p>')


# ---------------------------------------------------------------
# Tables
# ---------------------------------------------------------------
def table(rows, col_widths, header=True):
    """Simple bordered table, 12 pt Times New Roman, single line spacing inside cells."""
    borders = "".join(f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                      for side in ["top", "left", "bottom", "right", "insideH", "insideV"])
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in col_widths)
    xml = (f'<w:tbl><w:tblPr><w:tblW w:w="{sum(col_widths)}" w:type="dxa"/><w:jc w:val="center"/>'
           f'<w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
           f'<w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
           f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>')
    for row_index, row in enumerate(rows):
        is_header = header and row_index == 0
        xml += "<w:tr>" + ('<w:trPr><w:tblHeader/></w:trPr>' if is_header else "")
        for cell_text, width in zip(row, col_widths):
            shade = '<w:shd w:val="clear" w:color="auto" w:fill="D9D9D9"/>' if is_header else ""
            align = "center" if is_header else "left"
            xml += (f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shade}</w:tcPr>'
                    f'<w:p><w:pPr><w:spacing w:before="40" w:after="40" w:line="240" w:lineRule="auto"/>'
                    f'<w:jc w:val="{align}"/></w:pPr>{run(str(cell_text), 24, bold=is_header)}</w:p></w:tc>')
        xml += "</w:tr>"
    xml += "</w:tbl>"
    # small gap after the table
    return xml + '<w:p><w:pPr><w:spacing w:before="0" w:after="80"/></w:pPr></w:p>'


# ---------------------------------------------------------------
# Chapter content
# ---------------------------------------------------------------
def screenshot_or_placeholder(media, file_name, description):
    """Insert an app screenshot if the file exists, otherwise a placeholder box."""
    path = REPORT_DIR / file_name
    if path.exists():
        return image_paragraph(media, path, 5.6, max_height_inches=4.2)
    return placeholder_box(f"[Insert screenshot: {description}. Save it as \"{file_name}\" "
                           f"in the \"mini project report\" folder and rebuild.]")


def render_item(item, media):
    """Convert one content item (from report_text.py) into XML."""
    kind = item[0]
    if kind == "h1":
        return heading(item[1], 1)
    if kind == "h2":
        return heading(item[1], 2)
    if kind == "p":
        return paragraph(item[1])
    if kind == "bullets":
        return "".join(bullet(text) for text in item[1])
    if kind == "figure":
        _, path, width, text = item[:4]
        max_height = item[4] if len(item) > 4 else 8.0
        return image_paragraph(media, config.BASE_DIR / path, width, max_height) + caption(text)
    if kind == "screenshot":
        _, file_name, description, text = item
        return screenshot_or_placeholder(media, file_name, description) + caption(text)
    if kind == "table":
        _, text, rows, widths = item
        return caption(text) + table(rows, widths)
    raise ValueError(f"Unknown content item: {kind}")


def references_xml():
    """References chapter in the template's style."""
    xml = heading("References", 1)
    for ref in REFERENCES:
        xml += paragraph(ref)
    return xml


# ---------------------------------------------------------------
# Template editing
# ---------------------------------------------------------------
def set_paragraph_text(paragraph_xml, new_text):
    """Replace all text in a template paragraph, keeping its paragraph and first-run formatting."""
    ppr = re.search(r"<w:pPr>.*?</w:pPr>", paragraph_xml, re.S)
    rpr = re.search(r"<w:r\b[^>]*>\s*(<w:rPr>.*?</w:rPr>)", paragraph_xml, re.S)
    open_tag = re.match(r"<w:p\b[^>]*>", paragraph_xml).group(0)
    run_xml = (f'<w:r>{rpr.group(1) if rpr else ""}'
               f'<w:t xml:space="preserve">{escape(new_text)}</w:t></w:r>') if new_text else ""
    return f'{open_tag}{ppr.group(0) if ppr else ""}{run_xml}</w:p>'


def split_blocks(document_xml):
    """Return (head, list of top-level <w:p>/<w:tbl> blocks, final sectPr, tail)."""
    head, rest = document_xml.split("<w:body>")
    body, tail = rest.split("</w:body>")
    blocks = [m.group(0) for m in re.finditer(r"<w:(p|tbl)\b.*?</w:\1>", body, re.S)]
    final_sect = re.search(r"<w:sectPr\b(?:(?!<w:sectPr).)*?</w:sectPr>\s*$", body, re.S).group(0)
    blocks = [re.sub(r"<w:bookmark(Start|End)\b[^>]*/>", "", b) for b in blocks]
    blocks = [b.replace("<w:lastRenderedPageBreak/>", "") for b in blocks]
    return head + "<w:body>", blocks, final_sect, "</w:body>" + tail


def section_properties(page_numbers=None, first_page_title=False, next_page=True):
    """sectPr with the template's page size, margins and header."""
    footer = f'<w:footerReference w:type="default" r:id="{FOOTER_REL_ID}"/>' if page_numbers else ""
    number_type = ""
    if page_numbers:
        number_type = f'<w:pgNumType w:fmt="{page_numbers}" w:start="1"/>'
    return ('<w:sectPr><w:headerReference w:type="default" r:id="rId9"/>' + footer
            + ('<w:type w:val="nextPage"/>' if next_page else "")
            + '<w:pgSz w:w="11910" w:h="16840"/>'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" '
            'w:header="720" w:footer="720" w:gutter="0"/>'
            + number_type + '<w:cols w:space="720"/>'
            + ("<w:titlePg/>" if first_page_title else "")
            + '<w:docGrid w:linePitch="299"/></w:sectPr>')


def add_sectpr_to_paragraph(paragraph_xml, sectpr):
    """Put a section break at the end of an existing paragraph."""
    if "<w:pPr>" in paragraph_xml:
        return paragraph_xml.replace("</w:pPr>", sectpr + "</w:pPr>", 1)
    return re.sub(r"(<w:p\b[^>]*>)", r"\1<w:pPr>" + sectpr + "</w:pPr>", paragraph_xml, count=1)


def edit_front_matter(blocks):
    """Fill in the cover page and certificate."""
    blocks[2] = set_paragraph_text(blocks[2], TITLE)
    blocks[4] = set_paragraph_text(blocks[4], COURSE_LINE)
    for i, (name, roll) in zip([9, 10, 11], COVER_STUDENTS):
        blocks[i] = set_paragraph_text(blocks[i], f"{name} (Roll No. {roll})")
    names = ", ".join(f"{name} (Roll No. {roll})" for name, roll in CERTIFICATE_STUDENTS)
    blocks[29] = set_paragraph_text(
        blocks[29],
        f"This is to certify that the project entitled \u201c{TITLE}\u201d is a bonafide work of "
        f"{names} submitted to the University of Mumbai in partial fulfillment of the requirement "
        "of the Course Project for Deep Learning in Semester VII of Final Year Artificial "
        "Intelligence & Data Science.")
    blocks[30] = set_paragraph_text(blocks[30], "")   # stray "." in the template
    return blocks


def build_abstract_paragraphs():
    """Abstract text in the template's abstract style (12 pt, justified, 1.5 spacing)."""
    return "".join(paragraph(text) for text in ABSTRACT)


def build_toc_table(template_table, pages):
    """Rebuild the template's table of contents with our chapters and page numbers."""
    rows = re.findall(r"<w:tr\b.*?</w:tr>", template_table, re.S)
    header_row, abstract_row, chapter_row, sub_row = rows[0], rows[1], rows[2], rows[5]
    table_start = template_table[:template_table.index(rows[0])]
    new_rows = [header_row, fill_row(abstract_row, ["", "", "", "Abstract", "i"])]
    for number, title, level in TOC_ENTRIES:
        key = f"{number}. {title}" if level == 1 and number else (f"{number} {title}" if number else title)
        page = str(pages.get(key, "")) if pages else ""
        if level == 1:
            new_rows.append(fill_row(chapter_row, [number, "", "", title, page]))
        else:
            new_rows.append(fill_row(sub_row, ["", number, "", title, page]))
    return table_start + "".join(new_rows) + "</w:tbl>"


def fill_row(row_xml, texts):
    """Set the text of each of the 5 cells in a TOC row, keeping cell formatting."""
    cells = re.findall(r"<w:tc\b.*?</w:tc>", row_xml, re.S)
    new_cells = []
    for cell, text in zip(cells, texts):
        para = re.search(r"<w:p\b.*?</w:p>", cell, re.S).group(0)
        mark_rpr = re.search(r"<w:pPr>.*?(<w:rPr>.*?</w:rPr>).*?</w:pPr>", para, re.S)
        rpr = mark_rpr.group(1) if mark_rpr else "<w:rPr>" + FONT + "</w:rPr>"
        ppr = re.search(r"<w:pPr>.*?</w:pPr>", para, re.S).group(0)
        open_tag = re.match(r"<w:p\b[^>]*>", para).group(0)
        run_xml = f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>' if text else ""
        new_cells.append(cell.replace(para, f"{open_tag}{ppr}{run_xml}</w:p>"))
    row_start = row_xml[:row_xml.index(cells[0])]
    return row_start + "".join(new_cells) + "</w:tr>"


def build_body(blocks, final_sect, media, pages):
    """Assemble the full document body."""
    blocks = edit_front_matter(blocks)
    # Section 1: cover + certificate (no page numbers). Two empty cover lines (12, 18)
    # are dropped because our title needs two lines; fillers 42-49 are dropped too.
    certificate_end = add_sectpr_to_paragraph(blocks[41], section_properties(first_page_title=True))
    cover = [b for i, b in enumerate(blocks[0:41]) if i not in (12, 18)]
    front = "".join(cover) + certificate_end
    # Section 2: abstract (page i) + table of contents (page ii), roman numbers
    abstract = blocks[50] + build_abstract_paragraphs() + blocks[52]
    toc = blocks[53] + build_toc_table(blocks[54], pages)
    toc_end = f'<w:p><w:pPr>{section_properties(page_numbers="lowerRoman")}</w:pPr></w:p>'
    # Section 3: chapters + references, arabic numbers starting at 1
    chapters = "".join(render_item(item, media) for item in CHAPTERS) + references_xml()
    last_sect = section_properties(page_numbers="decimal")
    return front + abstract + toc + toc_end + chapters + last_sect


# ---------------------------------------------------------------
# Package-level edits (footer, numbering, relationships)
# ---------------------------------------------------------------
FOOTER_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
    '<w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
    f'<w:r><w:rPr>{FONT}<w:sz w:val="24"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
    f'<w:r><w:rPr>{FONT}<w:sz w:val="24"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
    f'<w:r><w:rPr>{FONT}<w:sz w:val="24"/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r>'
    f'<w:r><w:rPr>{FONT}<w:sz w:val="24"/></w:rPr><w:t>1</w:t></w:r>'
    f'<w:r><w:rPr>{FONT}<w:sz w:val="24"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r>'
    '</w:p></w:ftr>')

BULLET_ABSTRACT_NUM = (
    '<w:abstractNum w:abstractNumId="90"><w:multiLevelType w:val="hybridMultilevel"/>'
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="\u2022"/>'
    '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr>'
    f'<w:rPr>{FONT}</w:rPr></w:lvl></w:abstractNum>')


def update_package(files, media):
    """Add footer, bullet numbering, image files and relationships to the package."""
    files["word/footer_signsense.xml"] = FOOTER_XML.encode("utf-8")

    numbering = files["word/numbering.xml"].decode("utf-8")
    first_num = numbering.index("<w:num ")
    numbering = numbering[:first_num] + BULLET_ABSTRACT_NUM + numbering[first_num:]
    numbering = numbering.replace("</w:numbering>",
                                  f'<w:num w:numId="{BULLET_NUM_ID}"><w:abstractNumId w:val="90"/></w:num></w:numbering>')
    files["word/numbering.xml"] = numbering.encode("utf-8")

    rels = files["word/_rels/document.xml.rels"].decode("utf-8")
    new_rels = (f'<Relationship Id="{FOOTER_REL_ID}" Type="http://schemas.openxmlformats.org/'
                f'officeDocument/2006/relationships/footer" Target="footer_signsense.xml"/>')
    for rel_id, zip_path, source in media.images:
        files[zip_path] = Path(source).read_bytes()
        new_rels += (f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.org/'
                     f'officeDocument/2006/relationships/image" Target="{zip_path[5:]}"/>')
    files["word/_rels/document.xml.rels"] = rels.replace("</Relationships>", new_rels + "</Relationships>").encode("utf-8")

    types = files["[Content_Types].xml"].decode("utf-8")
    types = types.replace("</Types>", '<Override PartName="/word/footer_signsense.xml" ContentType='
                          '"application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
    files["[Content_Types].xml"] = types.encode("utf-8")


def main():
    """Read the template, build the report and save it."""
    parser = argparse.ArgumentParser(description="Build the SignSense report from the template")
    parser.add_argument("--pages", help="JSON file with heading -> page number (for the TOC)")
    parser.add_argument("--output", default=str(OUTPUT_PATH), help="where to save the .docx")
    args = parser.parse_args()
    pages = json.loads(Path(args.pages).read_text(encoding="utf-8-sig")) if args.pages else None

    with zipfile.ZipFile(TEMPLATE_PATH) as template:
        files = {name: template.read(name) for name in template.namelist()}

    head, blocks, final_sect, tail = split_blocks(files["word/document.xml"].decode("utf-8"))
    media = MediaRegistry()
    body = build_body(blocks, final_sect, media, pages)
    files["word/document.xml"] = (head + body + tail).encode("utf-8")
    update_package(files, media)

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as output:
        # [Content_Types].xml must be the first entry
        output.writestr("[Content_Types].xml", files.pop("[Content_Types].xml"))
        for name, data in files.items():
            output.writestr(name, data)
    print(f"Report written to {args.output} ({len(media.images)} images)")


if __name__ == "__main__":
    main()
