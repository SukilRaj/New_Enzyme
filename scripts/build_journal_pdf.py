"""Render the editable journal summary using ReportLab and local fonts."""
from pathlib import Path
import re
import textwrap
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, Preformatted, KeepTogether

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'docs/journal/PROJECT_AND_JOURNAL_SUMMARY.md'
OUTPUT=ROOT/'docs/journal/ENZAIME_HELIX_PROJECT_SUMMARY.pdf'


def main():
    fonts=Path('C:/Windows/Fonts')
    for name,file in [('Body','times.ttf'),('BodyBold','timesbd.ttf'),('BodyItalic','timesi.ttf'),
                      ('Sans','arial.ttf'),('SansBold','arialbd.ttf'),('Mono','consola.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
    pdfmetrics.registerFontFamily('Sans',normal='Sans',bold='SansBold',italic='Sans',boldItalic='SansBold')
    styles=getSampleStyleSheet()
    specs={
        'BodyText':dict(fontName='Body',fontSize=10.6,leading=14.5,spaceAfter=7.5,textColor=colors.black,splitLongWords=True,allowWidows=0,allowOrphans=0),
        'Title':dict(fontName='SansBold',fontSize=23,leading=28,spaceAfter=13,textColor=colors.black),
        'Heading1':dict(fontName='SansBold',fontSize=14,leading=18,spaceBefore=15,spaceAfter=8,textColor=colors.black,keepWithNext=True),
        'Heading2':dict(fontName='SansBold',fontSize=11.4,leading=15,spaceBefore=12,spaceAfter=6,textColor=colors.black,keepWithNext=True),
        'Caption':dict(fontName='BodyItalic',fontSize=9.5,leading=12.5,spaceAfter=10),
        'Metadata':dict(fontName='Sans',fontSize=10,leading=14,spaceAfter=6),
        'Cell':dict(fontName='Sans',fontSize=8.3,leading=11,spaceAfter=0),
        'CellNumber':dict(fontName='Sans',fontSize=8.4,leading=11,alignment=TA_CENTER,spaceAfter=0),
        'CellHead':dict(fontName='SansBold',fontSize=8.4,leading=11,textColor=colors.white,spaceAfter=0),
        'CodeBlock':dict(fontName='Mono',fontSize=8.8,leading=12,spaceBefore=6,spaceAfter=12,leftIndent=9),
    }
    for name,values in specs.items():
        if name in styles:styles[name].__dict__.update(values)
        else:styles.add(ParagraphStyle(name,**values))
    def inline(text):
        s=escape(text)
        s=re.sub(r'`([^`]+)`',r'<font name="Mono" size="9">\1</font>',s)
        s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
        s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<link href="\2" color="#244f64">\1</link>',s)
        return s
    width=A4[0]-94
    flow=[];lines=SOURCE.read_text(encoding='utf-8').splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                code.extend(textwrap.wrap(lines[i],width=88,replace_whitespace=False,drop_whitespace=False) or ['']);i+=1
            flow.append(Preformatted('\n'.join(code),styles['CodeBlock']));i+=1;continue
        if line.startswith('|'):
            table=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',c) for c in cells):table.append(cells)
                i+=1
            count=len(table[0]);numeric=any('MAE' in h for h in table[0])
            if numeric:
                widths=([width*.46]+[width*.54/(count-1)]*(count-1)) if count>3 else [width/count]*count
                if 'Training and test budget' in table[0]:widths=[width*.32,width*.28,width*.1333,width*.1333,width*.1334]
            else:widths={3:[width*.22,width*.38,width*.4],4:[width*.22,width*.26,width*.26,width*.26]}.get(count,[width/count]*count)
            data=[]
            for row_index,row in enumerate(table):
                data.append([Paragraph(inline(c),styles['CellHead' if row_index==0 else 'CellNumber' if numeric and j>0 and re.match(r'^[−\-0-9]',c) and len(c)<20 else 'Cell']) for j,c in enumerate(row)])
            t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#344c5b')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f5f6')]),
                ('GRID',(0,0),(-1,-1),.45,colors.HexColor('#d9d9d9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
                ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
            t.spaceBefore=4;t.spaceAfter=12
            if numeric and len(table)<9:
                group=[t]
                if flow and getattr(flow[-1], '_bookmark_text', None):
                    group.insert(0,flow.pop())
                flow.append(KeepTogether(group))
            else:
                flow.append(t)
            continue
        if line.startswith('!['):
            match=re.match(r'!\[([^]]*)\]\(([^)]+)\)',line)
            from PIL import Image as PILImage
            path=SOURCE.parent/match.group(2)
            with PILImage.open(path) as im:w,h=im.size
            factor=min(width/w,300/h)
            img=Image(str(path),width=w*factor,height=h*factor);img.hAlign='CENTER'
            img.keepWithNext=True;flow.extend([Spacer(1,7),img,Spacer(1,4)]);i+=1;continue
        level=3 if line.startswith('### ') else 2 if line.startswith('## ') else 1 if line.startswith('# ') else 0
        if level:
            text=line[level+1:];style={1:'Title',2:'Heading1',3:'Heading2'}[level]
            paragraph=Paragraph(inline(text),styles[style]);paragraph._bookmark_text=text if level>1 else None
            flow.append(paragraph);i+=1;continue
        if line.startswith('Project summary') or line.startswith('Evidence snapshot'):
            flow.append(Paragraph(inline(line),styles['Metadata']));i+=1;continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','```','![')):
            para.append(lines[i].strip());i+=1
        text=' '.join(para);style='Caption' if text.startswith('Figure ') else 'BodyText'
        flow.append(Paragraph(inline(text),styles[style]))
    class JournalDoc(SimpleDocTemplate):
        def afterFlowable(self,element):
            heading=getattr(element,'_bookmark_text',None)
            if heading:
                key=f'heading-{self.seq.nextf("heading")}';self.canv.bookmarkPage(key)
                level=1 if element.style.name=='Heading2' else 0
                self.canv.addOutlineEntry(heading,key,level=level,closed=False)
    def decorate(canvas,doc):
        canvas.saveState();canvas.setFillColor(colors.black);canvas.setFont('Sans',8)
        if doc.page>1:canvas.drawString(47,A4[1]-31,'EnzAIme Helix  |  Project summary and manuscript basis')
        canvas.setStrokeColor(colors.HexColor('#d9d9d9'));canvas.line(47,40,A4[0]-47,40)
        canvas.setFont('Sans',7.6);canvas.drawString(47,27,'Prepared 26 October 2026  |  Exploratory computational study')
        canvas.drawRightString(A4[0]-47,27,str(doc.page));canvas.restoreState()
    doc=JournalDoc(str(OUTPUT),pagesize=A4,leftMargin=47,rightMargin=47,topMargin=47,bottomMargin=54,
                   title='EnzAIme Helix project summary and manuscript basis',author='',subject='Implemented system, reproducible results and research limitations')
    doc.build(flow,onFirstPage=decorate,onLaterPages=decorate)
    print(str(OUTPUT))


if __name__=='__main__':main()
