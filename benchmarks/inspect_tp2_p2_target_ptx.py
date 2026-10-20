"""Inventory actual Linux nvcc12.8 / TileLang0.1.15 SM75 PTX, not dynamic counters."""
import hashlib
import json
from pathlib import Path
import re

root=Path('D:/attention_eval/tp2_p2_20261018');rows=[];seen=set()
for p in root.rglob('*.ptx'):
    if 'offline' in p.parts:continue
    text=p.read_text();sha=hashlib.sha256(text.encode()).hexdigest()
    if sha in seen:continue
    seen.add(sha);source=p.with_suffix('.cu');cu=source.read_text() if source.exists() else ''
    bounds=re.search(r'__launch_bounds__\((\d+)',cu)
    sig=re.search(r'void main_kernel\((.*?)\);',cu,re.S)
    ops={}
    for line in text.splitlines():
        s=line.strip()
        if s.startswith(('mma.sync','bar.sync','cvt.','mul.f32','fma.rn.f32','ex2.','shfl.sync','ld.shared','st.shared')):
            op=re.split(r'[\s\{]',s)[0];ops[op]=ops.get(op,0)+1
    rows.append(dict(file=str(p),ptx_sha256=sha,target=re.search(r'\.target\s+([^\n]+)',text).group(1),
        threads=int(bounds[1]) if bounds else None,real_DiT_shape='6848' in cu or '6814' in cu,
        scalar_math=bool(re.search(r'Counts\[',cu)),precomputed_scores=bool(re.search(r'CS\[',cu)),
        restoration_loads=bool(re.search(r'VS\[',cu)),source_signature=sig[1] if sig else None,
        static_instruction_occurrences=ops,
        scope='Actual target JIT PTX. Counts are static textual occurrences, not dynamic instructions, stalls or achieved occupancy. Physical register/shared resources come from Systems launches.'))
(root/'target_ptx_inventory.json').write_text(json.dumps(rows,indent=2));print('TARGET_PTX_INVENTORY',len(rows),'unique PTX artifacts')
