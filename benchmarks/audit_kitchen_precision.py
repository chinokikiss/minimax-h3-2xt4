"""Check that scheduling candidates preserve the frozen attention math AST."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def function(path,name):
    return next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)


class RemoveScheduling(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='Kernel':
            for kw in node.keywords:
                if kw.arg=='threads':kw.value=ast.Constant(128)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='gemm':node.keywords=[kw for kw in node.keywords if kw.arg!='policy']
        return node


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    old=function(root/'patches/tilelang_grouped_tp_optimized.py','kernel')
    candidate=function(root/'patches/tilelang_grouped_tp_kitchen.py','kernel')
    assert [arg.arg for arg in candidate.args.args[-2:]]==['threads','warp_policy']
    candidate.args.args=candidate.args.args[:-2];candidate.args.defaults=candidate.args.defaults[:-2]
    candidate.body=[n for n in candidate.body if not (isinstance(n,ast.If) and 'Unknown warp configuration' in ast.unparse(n))
                    and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='gp' for t in n.targets))]
    candidate=RemoveScheduling().visit(candidate)
    before=ast.dump(old,include_attributes=False);after=ast.dump(candidate,include_attributes=False)
    assert before==after,'Unexpected arithmetic/control-flow change in attention body'
    result=dict(status='pass',scope='CPU source AST equivalence after removing explicit thread/GEMM-policy options; not GPU execution or numerical proof',
                normalized_kernel_ast_sha256=hashlib.sha256(before.encode()).hexdigest(),
                frozen_source='patches/tilelang_grouped_tp_optimized.py',candidate_source='patches/tilelang_grouped_tp_kitchen.py')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print('PASS: attention math/control flow unchanged by scheduling additions')


if __name__=='__main__':main()
