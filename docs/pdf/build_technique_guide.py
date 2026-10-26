"""Generate a technique-only Markdown/PDF pair; never imports inference code."""
from collections import defaultdict
import hashlib
import html
import json
from pathlib import Path
import subprocess

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate,Flowable,Frame,KeepTogether,PageBreak,PageTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.platypus.tableofcontents import TableOfContents
from pypdf import PdfReader,PdfWriter

from optimization_techniques import CHAPTERS,DEFAULTS,TECHNIQUES

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'output/pdf'
PDF=ROOT/'docs/guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf'
MD=ROOT/'docs/guides/H3_OPTIMIZATION_TECHNIQUES.md'
DATE='2026-10-26'
W,H=595.276,841.890
MARGIN=44
WIDTH=W-2*MARGIN
INK=colors.HexColor('#142B40')
TEAL=colors.HexColor('#087E8B')
MUTED=colors.HexColor('#536779')
PALE=colors.HexColor('#EDF5F5')
RULE=colors.HexColor('#D9E2E9')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def catalogue():
    prior=json.loads((ROOT/'docs/data/h3_documentation_coverage_20261020.json').read_text(encoding='utf8'))
    ids=[t['id'] for t in TECHNIQUES];assert len(set(ids))==len(ids)
    paths=sorted({p for t in TECHNIQUES for p in t['sources']})
    assert all((ROOT/p).is_file() for p in paths)
    runtime=sorted(p.relative_to(ROOT).as_posix() for p in (ROOT/'patches').glob('*.py'))
    assert not set(runtime)-set(paths),'Runtime modules missing from technique explanations'
    legacy={x['id'] for x in prior['features']}
    mapped={i for t in TECHNIQUES for i in t['legacy_ids']}
    assert legacy==mapped,(legacy-mapped,mapped-legacy)
    assert all(len(t[k].split())>=6 for t in TECHNIQUES for k in ('method','saves','contract'))
    sources=[dict(path=p,sha256=digest(ROOT/p),
        canonical_lf_sha256=hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
        techniques=[t['id'] for t in TECHNIQUES if p in t['sources']]) for p in paths]
    result=dict(date=DATE,kind='optimization techniques; mechanisms, removed work and numerical/ownership contracts',
        chapters=CHAPTERS,techniques=TECHNIQUES,defaults=DEFAULTS,sources=sources)
    coverage=dict(date=DATE,techniques=len(TECHNIQUES),chapters=len(CHAPTERS),runtime_modules=len(runtime),
        runtime_coverage={p:[t['id'] for t in TECHNIQUES if p in t['sources']] for p in runtime},
        prior_52_feature_coverage={i:[t['id'] for t in TECHNIQUES if i in t['legacy_ids']] for i in sorted(legacy)},
        attention_native_sources={p.relative_to(ROOT).as_posix():[t['id'] for t in TECHNIQUES if p.relative_to(ROOT).as_posix() in t['sources']]
            for p in (ROOT/'experiments/kitchen_attention/native').iterdir() if p.is_file()},
        sources=sources,scope='All current runtime modules, all 52 prior audited items, retained native kernels and described historical alternatives; binary/provider artifacts are provenance, not independent techniques',
        observation_charts_in_technical_guide=False)
    assert all(coverage['attention_native_sources'].values())
    return result,coverage


def markdown(data):
    lines=['# MiniMax-H3 TP2 optimization techniques','',
        'Updated '+DATE+'. This guide explains algorithms, data movement, ownership, fusion and numerical contracts. Measurements and experiment chronology remain in the separate reports.','',
        f"The catalogue contains {len(TECHNIQUES)} techniques in {len(CHAPTERS)} areas. Active, alternative, experimental, historical and retired implementations are all retained in scope.",'',
        '## Current selections','', '| Boundary | Selection | Scope |','|---|---|---|']
    lines+=['| '+' | '.join(row)+' |' for row in DEFAULTS]
    lines+=['','## Reading the technique entries','',
        'Each entry states the mechanism, the work or storage removed, the numerical/ownership contract, and exact implementation sources. A retained experimental or retired method is not a deployment default.','',
        '## Notation','',
        'M: activation rows; K/N: projection input/output width; H/Hkv: query/KV heads; D: head dimension; r: tensor-parallel owner; c: residual gauge divisor. SUM and MAX denote distributed collectives. Memory bounds include backing storage retained by views.','']
    for ch in CHAPTERS:
        lines += ['## '+ch['id']+' - '+ch['title'],'',ch['intro'],'',
                  '**Dataflow:** '+' -> '.join(ch['flow'])+'.','']
        for t in TECHNIQUES:
            if t['chapter']!=ch['id']:continue
            lines += ['### '+t['id']+' '+t['title'],'','**Selection:** '+t['status']+'.','',
                      t['method'],'','**Work removed:** '+t['saves'],'','**Contract:** '+t['contract'],'',
                      '**Implementation:** '+', '.join('['+p+'](../../'+p+')' for p in t['sources'])+'.','']
    lines+=['## Core equations','',
        '- Shared quantizer: global_max[m] = MAX_r(local_max_r[m]); scale[m] = max(global_max[m] / 127, floor). Both owners quantize against that scale.',
        '- Row projection: Y = dequantize(SUM_r(INT8_GEMM_r), shared_row_scale, weight_scale) + bias. Scaled FP16 transport explicitly approximates the integer SUM.',
        '- Scalar pooling: coarse_logit[i,b] = dot(q[i], mean_k[b]) / sqrt(D) + log(valid_count[b]); merge only omitted centroids with fine token contributions.',
        '- Row-A LoRA: hidden = SUM_r(A_r x_r) in FP32; delta = strength * B hidden, applied in adapter order.',
        '- Residual gauge: y = x / c; normalized epsilon = eps / c^2; branch outputs are divided by c before the storage cast.','',
        '## Coverage and implementation identity','',
        '[Machine-readable catalogue](../data/h3_optimization_techniques_20261026.json) and [coverage matrix](../data/h3_optimization_coverage_20261026.json) map every prior audited item and runtime module to technique IDs. The PDF embeds source hashes.','',
        'Frozen core: ComfyUI d49e888586dd8ae012c0667b33466b815fee07f7; Kitchen 0.2.37; TileLang 0.1.13. Documentation generation changes no operators, checkpoints or inference dependencies.','']
    return '\n'.join(lines)


def fonts():
    for name,file in [('Body','segoeui.ttf'),('Bold','segoeuib.ttf'),('Semi','seguisb.ttf'),('Mono','consola.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Body',boldItalic='Bold')


class FlowDiagram(Flowable):
    def __init__(self,stages):super().__init__();self.stages=stages;self.width=WIDTH;self.height=103
    def draw(self):
        c=self.canv;gap=9;bw=(WIDTH-gap*(len(self.stages)-1))/len(self.stages)
        style=ParagraphStyle('flow',fontName='Semi',fontSize=8,leading=10.5,alignment=1,textColor=INK)
        for i,text in enumerate(self.stages):
            x=i*(bw+gap);c.setFillColor(PALE);c.setStrokeColor(RULE);c.roundRect(x,23,bw,66,5,fill=1,stroke=1)
            p=Paragraph(html.escape(text),style);_,hh=p.wrap(bw-12,80);p.drawOn(c,x+6,56-hh/2)
            if i<len(self.stages)-1:
                c.setStrokeColor(TEAL);c.setLineWidth(1.1);c.line(x+bw,56,x+bw+gap-1,56)
                c.line(x+bw+gap-4,59,x+bw+gap-1,56);c.line(x+bw+gap-4,53,x+bw+gap-1,56)
        c.setFillColor(MUTED);c.setFont('Body',7.5);c.drawString(0,6,'Conceptual dataflow and ownership; independent techniques have their own contracts.')


class GuideDoc(BaseDocTemplate):
    def __init__(self,path):
        super().__init__(str(path),pagesize=(W,H),leftMargin=MARGIN,rightMargin=MARGIN,
            topMargin=60,bottomMargin=55,title='MiniMax-H3 TP2 Optimization Techniques',author='H3 implementation documentation')
        frame=Frame(MARGIN,55,WIDTH,H-115,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='guide',frames=[frame],onPage=self.decorate))
    def decorate(self,c,doc):
        c.saveState();c.setStrokeColor(RULE);c.line(MARGIN,42,W-MARGIN,42)
        c.setFont('Body',7.5);c.setFillColor(MUTED)
        c.drawString(MARGIN,28,'H3 / TP2 OPTIMIZATION TECHNIQUES  |  26 OCT 2026')
        c.drawRightString(W-MARGIN,28,str(doc.page));c.restoreState()
    def afterFlowable(self,flowable):
        key=getattr(flowable,'bookmark',None)
        if key:
            text=flowable.getPlainText();level=flowable.outline_level
            self.canv.bookmarkPage(key);self.canv.addOutlineEntry(text,key,level=level,closed=False)
            if level==0:self.notify('TOCEntry',(0,text,self.page,key))


def pdf(data,coverage):
    fonts();OUTPUT.mkdir(parents=True,exist_ok=True)
    body=ParagraphStyle('body',fontName='Body',fontSize=9.7,leading=13.6,textColor=INK,spaceAfter=7)
    small=ParagraphStyle('small',parent=body,fontSize=8,leading=10.8,textColor=MUTED)
    mono=ParagraphStyle('mono',parent=small,fontName='Mono',fontSize=7.3,leading=10,wordWrap='CJK')
    h1=ParagraphStyle('chapter',fontName='Bold',fontSize=24,leading=28,textColor=INK,spaceAfter=12)
    h2=ParagraphStyle('technique',fontName='Bold',fontSize=12,leading=15,textColor=TEAL,spaceAfter=6)
    title=ParagraphStyle('title',fontName='Bold',fontSize=35,leading=40,textColor=INK,spaceAfter=16)
    def p(text,style=body):return Paragraph(html.escape(text),style)
    def mark(text,key,level=0):
        x=p(text,h1 if level==0 else h2);x.bookmark=key;x.outline_level=level;return x
    def table(rows,widths):
        result=Table([[p(c,small) for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        result.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7F9')]),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.35,RULE)]))
        return result
    story=[Spacer(1,48),p('MINIMAX-H3 / TWO T4 GPUS',small),Spacer(1,20),
        p('Optimization\nTechniques'.replace('\n','<br/>'),title)]
    # A dedicated Paragraph carries deliberate rich-text line breaks.
    story[-1]=Paragraph('Optimization<br/>Techniques',title)
    story += [p('Implementation mechanisms, dataflow, fusion, ownership and numerical contracts.',body),Spacer(1,25),
        table([['Catalogue','Implementation coverage'],[str(len(TECHNIQUES))+' techniques / 10 areas','All 64 runtime modules and all 52 prior audited items'],
               ['Production and alternatives','Experimental, historical and retired techniques retained']], [WIDTH*.46,WIDTH*.54]),
        Spacer(1,25),p('Updated 26 October 2026. Technique explanations are separated from measurement reports. Source links and a hash manifest make the implementation scope reviewable.',body),
        Spacer(1,20),p('Frozen ComfyUI: d49e888586dd8ae012c0667b33466b815fee07f7',mono),
        p('Kitchen 0.2.37 / TileLang 0.1.13. No inference environment changes.',small),PageBreak(),mark('Contents','contents')]
    toc=TableOfContents();toc.levelStyles=[ParagraphStyle('toc',fontName='Body',fontSize=10.2,leading=17,spaceBefore=5,textColor=INK)]
    story += [toc,PageBreak(),mark('Reading the technique catalogue','reading'),
        p('Each entry gives a mechanism, the work or storage removed, and its numerical and ownership contract. Implementation paths refer to the repository. Selection labels distinguish active code from alternatives, experiments and retired methods.'),
        p('M denotes activation rows; K and N are projection input/output widths; H and Hkv are query/KV heads; D is head dimension; r is the TP owner. SUM and MAX are distributed collectives. A view retains its backing allocation.'),
        Spacer(1,10),table([['Boundary','Current selection','Scope']]+DEFAULTS,[WIDTH*.22,WIDTH*.27,WIDTH*.51]),Spacer(1,14),
        p('Numerical contracts are part of the technique. INT32 SUM, scaled FP16 SUM, Group32, Kitchen INT8 and Native SDPA must not be conflated. A returned FP32 dtype can contain values previously rounded to BF16.'),
        p('Technique coverage includes retained implementations without claiming that every candidate is selected. Benchmark tables, failure narratives and experiment chronology remain in separate reports.')]
    for ch in CHAPTERS:
        story += [PageBreak(),mark(ch['id']+' / '+ch['title'],'chapter'+ch['id']),p(ch['intro']),FlowDiagram(ch['flow']),Spacer(1,8)]
        for t in TECHNIQUES:
            if t['chapter']!=ch['id']:continue
            heading=mark(t['id']+'  '+t['title'],t['id'],1)
            cells=[heading,p('Selection: '+t['status'],small),p(t['method']),
                Paragraph('<b>Work removed.</b> '+html.escape(t['saves']),body),
                Paragraph('<b>Contract.</b> '+html.escape(t['contract']),body),
                p('Implementation: '+'; '.join(t['sources']),mono),Spacer(1,13)]
            story.append(KeepTogether(cells))
    story += [PageBreak(),mark('Core equations','equations')]
    equations=[
        ('Shared activation scale','global_max[m] = MAX_r(local_max_r[m]); scale[m] = max(global_max[m] / 127, floor). Both owners quantize against the same row scale.'),
        ('Row projection','Y = dequantize(SUM_r(INT8_GEMM_r), row_scale, weight_scale) + bias. Scaled FP16 transport deliberately approximates the integer SUM.'),
        ('Omitted-leaf pooling','coarse_logit[i,b] = dot(q[i], mean_k[b]) / sqrt(D) + log(valid_count[b]). Merge omitted centroids and selected fine tokens using a common online normalization state.'),
        ('Row-A LoRA','hidden = SUM_r(A_r x_r) in FP32; delta = strength * B hidden. Apply separate adapters in their original order.'),
        ('Residual gauge','y = x / c; normalized epsilon = eps / c^2. Fold branch output division before the storage cast; restore public unscaled outputs.')]
    for label,text in equations:story += [p(label,h2),p(text),Spacer(1,10)]
    story += [PageBreak(),mark('Implementation source catalogue','sources'),p('Every listed path is linked to one or more technique IDs and included in the embedded source-hash manifest. Historical report sources describe retained methods whose current runtime selection is retired.',small)]
    for source in data['sources']:
        story += [KeepTogether([p(source['path'],mono),p('Techniques: '+', '.join(source['techniques']),small),Spacer(1,7)])]
    GuideDoc(OUTPUT/'technique_guide_unattached.pdf').multiBuild(story)
    reader=PdfReader(OUTPUT/'technique_guide_unattached.pdf')
    texts=[page.extract_text() or '' for page in reader.pages]
    assert all(t['id'] in '\n'.join(texts) for t in TECHNIQUES)
    manifest=dict(date=DATE,kind='technique-only guide',pages=len(reader.pages),techniques=len(TECHNIQUES),
        base_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        identity='sha256 identifies raw working-source bytes; canonical_lf_sha256 permits cross-platform text comparison. Base commit is not a clean-tree claim',
        catalogue_sha256=digest(ROOT/'docs/data/h3_optimization_techniques_20261026.json'),sources=data['sources'],coverage=coverage)
    writer=PdfWriter(clone_from=reader)
    writer.add_attachment('h3_technique_source_manifest.json',json.dumps(manifest,indent=2).encode())
    writer.add_metadata({'/Title':'MiniMax-H3 TP2 Optimization Techniques','/Subject':'Mechanisms, fusion, dataflow and numerical ownership contracts; no benchmark narrative','/Author':'H3 technical documentation'})
    with PDF.open('wb') as stream:writer.write(stream)
    (OUTPUT/'h3_technique_source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    (OUTPUT/'h3_technique_text.txt').write_text('\n\n'.join(texts),encoding='utf8')
    return len(reader.pages)


def main():
    data,coverage=catalogue()
    (ROOT/'docs/data/h3_optimization_techniques_20261026.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    (ROOT/'docs/data/h3_optimization_coverage_20261026.json').write_text(json.dumps(coverage,indent=2)+'\n',encoding='utf8')
    MD.write_text(markdown(data),encoding='utf8')
    pages=pdf(data,coverage)
    print(json.dumps(dict(pdf=str(PDF),markdown=str(MD),pages=pages,techniques=len(TECHNIQUES),runtime_modules=coverage['runtime_modules'],prior_items=len(coverage['prior_52_feature_coverage'])),indent=2))


if __name__=='__main__':main()
