"""Export the Chapter 5/6 working draft to editable Word using only stdlib."""
from pathlib import Path
import re
import zipfile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'analysis/THESIS_CHAPTERS_5_6_DRAFT.md'
OUT = ROOT / 'analysis/THESIS_CHAPTERS_5_6_DRAFT.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'


def plain(text):
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', text)
    return re.sub(r'\*\*|`', '', text)


def paragraph(text, style='Normal', page_break=False):
    properties = f'<w:pStyle w:val="{style}"/>'
    if page_break:
        properties += '<w:pageBreakBefore/>'
    return f'<w:p><w:pPr>{properties}</w:pPr><w:r><w:t xml:space="preserve">{escape(plain(text))}</w:t></w:r></w:p>'


def table(lines):
    rows = [[plain(cell.strip()) for cell in line.strip().strip('|').split('|')] for line in lines]
    rows = [row for row in rows if not all(re.fullmatch(r':?-+:?', cell.strip()) for cell in row)]
    width = 9360 // len(rows[0])
    grid = ''.join(f'<w:gridCol w:w="{width}"/>' for _ in rows[0])
    result = '<w:tbl><w:tblPr><w:tblStyle w:val="ResearchTable"/><w:tblW w:w="9360" w:type="dxa"/></w:tblPr>' + f'<w:tblGrid>{grid}</w:tblGrid>'
    for index, row in enumerate(rows):
        result += '<w:tr>' + ('<w:trPr><w:tblHeader/></w:trPr>' if index == 0 else '')
        for cell in row:
            result += f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>' + paragraph(cell, 'TableHeader' if index == 0 else 'TableText') + '</w:tc>'
        result += '</w:tr>'
    return result + '</w:tbl>'


def main():
    lines = SOURCE.read_text(encoding='utf-8').splitlines()
    body = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                block.append(lines[i]); i += 1
            body.append(table(block)); continue
        heading = re.match(r'^(#{1,3})\s+(.+)', line)
        if heading:
            body.append(paragraph(heading.group(2), ['Title', 'Heading1', 'Heading2'][len(heading.group(1)) - 1], heading.group(2) == 'CHAPTER 6'))
        else:
            body.append(paragraph(line, 'Caption' if line.startswith('**Table ') else 'Normal'))
        i += 1
    body.append('<w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/></w:sectPr>')
    document = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="{W}"><w:body>{"".join(body)}</w:body></w:document>'
    styles = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W}">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="480" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:pPr><w:jc w:val="both"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:jc w:val="center"/><w:spacing w:before="240" w:after="160"/></w:pPr><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:outlineLvl w:val="0"/><w:jc w:val="left"/><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Heading 2"/><w:basedOn w:val="Heading1"/><w:pPr><w:outlineLvl w:val="1"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:jc w:val="left"/><w:spacing w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:b/><w:sz w:val="22"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableText"><w:name w:val="Table Text"/><w:basedOn w:val="Normal"/><w:pPr><w:jc w:val="left"/><w:spacing w:after="70" w:line="240" w:lineRule="auto"/></w:pPr><w:rPr><w:sz w:val="20"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="TableHeader"><w:name w:val="Table Header"/><w:basedOn w:val="TableText"/><w:rPr><w:b/></w:rPr></w:style>
<w:style w:type="table" w:styleId="ResearchTable"><w:name w:val="Research Table"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="6"/><w:bottom w:val="single" w:sz="6"/><w:insideH w:val="single" w:sz="4" w:color="BBBBBB"/></w:tblBorders><w:tblCellMar><w:top w:w="80" w:type="dxa"/><w:left w:w="60" w:type="dxa"/><w:bottom w:w="80" w:type="dxa"/><w:right w:w="60" w:type="dxa"/></w:tblCellMar></w:tblPr></w:style>
</w:styles>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
    doc_rels = '''<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'''
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in {'[Content_Types].xml': content_types, '_rels/.rels': root_rels, 'word/document.xml': document, 'word/styles.xml': styles, 'word/_rels/document.xml.rels': doc_rels}.items():
            archive.writestr(name, content.encode('utf-8'))
    print(f'Exported editable Word draft: {OUT}')


if __name__ == '__main__':
    main()
