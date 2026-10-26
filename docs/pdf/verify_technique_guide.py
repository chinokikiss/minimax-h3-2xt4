"""Offline completeness, PDF geometry and visual-review sheet generation."""
import json
from pathlib import Path
import re

import pdfplumber
from pypdf import PdfReader
from PIL import Image,ImageDraw,ImageOps

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/pdf'
data=json.loads((ROOT/'docs/data/h3_optimization_techniques_20261026.json').read_text(encoding='utf8'))
pdf=ROOT/'docs/guides/MiniMax_H3_TP2_Optimization_Technical_Guide.pdf'
reader=PdfReader(pdf);texts=[p.extract_text() or '' for p in reader.pages]
coverage={t['id']:[i+1 for i,text in enumerate(texts) if re.search(r'^'+t['id']+r'\s+'+re.escape(t['title']),text,re.M)] for t in data['techniques']}
assert all(len(p)==1 for p in coverage.values()),{k:v for k,v in coverage.items() if len(v)!=1}
issues=[]
with pdfplumber.open(pdf) as doc:
    for i,page in enumerate(doc.pages,1):
        for ch in page.chars:
            if ch['text'].strip() and (ch['x0']<39 or ch['x1']>page.width-39 or ch['top']<35 or ch['bottom']>page.height-20):
                issues.append(dict(page=i,text=ch['text'],bbox=[ch['x0'],ch['top'],ch['x1'],ch['bottom']]))
assert not issues,issues[:12]
assert 'h3_technique_source_manifest.json' in reader.attachments
manifest=json.loads(reader.attachments['h3_technique_source_manifest.json'][0])
assert manifest['techniques']==len(coverage)
images=sorted(OUT.glob('technique_page-*.png'))
assert len(images)==len(reader.pages),(len(images),len(reader.pages))
for start in range(0,len(images),10):
    sheet=Image.new('RGB',(1590,960),'#d9e2e9');draw=ImageDraw.Draw(sheet)
    for j,path in enumerate(images[start:start+10]):
        img=Image.open(path).convert('RGB');img.thumbnail((300,424))
        x=15+(j%5)*315;y=30+(j//5)*470
        sheet.paste(img,(x,y));draw.text((x,y-20),'Page '+str(start+j+1),fill='#142b40')
    sheet.save(OUT/f'techniques_contact_{start//10+1}.png')
result=dict(pages=len(reader.pages),techniques=len(coverage),all_technique_headings_present_once=True,
    text_outside_page_safe_area=issues,embedded_manifest=True,pdf_pages_by_technique=coverage,
    visual_review='Review all five contact sheets and full-size samples before delivery')
(OUT/'technique_guide_qa.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='pdf_pages_by_technique'},indent=2))
