"""Reuse the bounded T4 bootstrap without modifying the archived first pass."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parent
text=(ROOT/'build_tp2_kitchen_attention_job.py').read_text(encoding='utf8')
text=text.replace('kitchen_inspired','kitchen_secondpass')
text=text.replace("names=['benchmarks/tp2_kitchen_attention.py'", "names=['benchmarks/kitchen_v2_attention.py','patches/tilelang_grouped_tp_kitchen_v2.py','benchmarks/tp2_kitchen_attention.py'")
text=text.replace("str(SRC/'benchmarks/tp2_kitchen_attention.py')", "str(SRC/'benchmarks/kitchen_v2_attention.py')")
text=text.replace("OUT=pathlib.Path('/kaggle/working/kitchen_group32')", "OUT=pathlib.Path('/kaggle/working/kitchen_group32_v2')")
text=text.replace("'screen',min(240,max(30,remaining()-140))", "'screen',min(280,max(30,remaining()-140))")
text=text.replace('h3-tp2-kitchen-group32','h3-tp2-kitchen-group32-v2').replace('H3 TP2 Kitchen Group32','H3 TP2 Kitchen Group32 V2')
ast.parse(text)
exec(compile(text,str(ROOT/'build_tp2_kitchen_attention_job.py'),'exec'),{'__file__':str(ROOT/'build_tp2_kitchen_attention_job.py'),'__name__':'__main__'})
