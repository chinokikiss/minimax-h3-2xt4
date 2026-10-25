"""Revalidate current sources on CPU, reusing only hash-bound pinned wheels."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import traceback
from prepare_kaggle_native_torch import inventory,install_overlay,run,core_packages


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--prior-sha256',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    receipt=dict(status='preparing',scope='CPU-only current-source preflight; no GPU setup allocation')
    try:
        matches=[p for p in Path('/kaggle/input').rglob('preflight_receipt.json')
            if hashlib.sha256(p.read_bytes()).hexdigest()==args.prior_sha256]
        assert len(matches)==1,'Reviewed pinned wheel receipt unavailable'
        prior=json.loads(matches[0].read_text());assert prior['status']=='cpu_ready'
        bundles=list(Path('/kaggle/input').rglob('local_conda.json'));assert len(bundles)==1
        spec=importlib.util.spec_from_file_location('frozen',bundles[0].parent/'frozen_kaggle_setup.py')
        frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
        frozen.require_cpu_setup();manifest=frozen.prepare_source(bundles[0].parent,args.source)
        assert core_packages(manifest['packages'])==prior['core_pins']
        base=inventory();minor=f'{sys.version_info.major}.{sys.version_info.minor}'
        wheels=prior['target_wheelsets'][minor];archive=matches[0].parent/wheels['archive']
        with archive.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==wheels['sha256']
        wheelhouse=Path('/kaggle/temp/revalidated_core_wheels');wheelhouse.mkdir(exist_ok=True)
        with tarfile.open(archive,'r:') as tar:tar.extractall(wheelhouse,filter='data')
        overlay=Path('/kaggle/temp/revalidated_core_overlay')
        install_overlay(wheelhouse,overlay,prior['core_pins'],args.output/'core_install.log',wheels['wheels_sha256'])
        assert inventory()==base,'Native platform packages changed'
        profile=dict(packages=dict(prior['core_pins']),native_base_packages=base,
            source_archive_sha256=manifest['source_archive_sha256'])
        for name in ('torch','torchvision','torchaudio','numpy'):
            if name in base:profile['packages'][name]=base[name]
        (args.output/'runtime_profile.json').write_text(json.dumps(profile,indent=2))
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(overlay),str(args.source),str(args.source/'ComfyUI'),str(args.source/'benchmarks')]),
            COMFY_ROOT=str(args.source/'ComfyUI'),H3_RUNTIME_LOCK=str(args.output/'runtime_profile.json'))
        tree=ast.parse((args.source/'benchmarks/prepare_kaggle_native_torch.py').read_text())
        probe=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='probe' for t in n.targets))
        run([sys.executable,'-c',probe],args.output/'api_probe.log',env,120)
        line=next(s for s in (args.output/'api_probe.log').read_text().splitlines() if s.startswith('H3_NATIVE_API_RECEIPT '))
        api=json.loads(line.split(' ',1)[1])
        tests="""import sys,unittest
sys.argv=['cpu-tests','--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import comfy.cli_args
sys.argv=['cpu-tests']
names=['benchmarks.test_h3_row_partial_combined','benchmarks.test_h3_reference_loading',
 'benchmarks.test_h3_loading_stages.ComponentAndIdle']
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
result=unittest.TextTestRunner(verbosity=2).run(suite)
assert result.wasSuccessful()
"""
        run([sys.executable,'-c',tests],args.output/'current_cpu_tests.log',env,120)
        run([sys.executable,str(args.source/'benchmarks/check_linux_pread.py'),'--output',str(args.output/'linux_pread.json')],
            args.output/'linux_pread.log',env,60)
        for wheelset in prior['target_wheelsets'].values():
            path=matches[0].parent/wheelset['archive']
            with path.open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==wheelset['sha256']
            shutil.copyfile(path,args.output/path.name)
        receipt.update(status='cpu_ready',api=api,core_pins=prior['core_pins'],target_wheelsets=prior['target_wheelsets'],
            base_packages=base,python=sys.version,core_imports='passed',linux_pread='passed',current_cpu_tests='passed',
            reused_cpu_receipt_sha256=args.prior_sha256,source_archive_sha256=manifest['source_archive_sha256'])
    except BaseException as error:
        receipt.update(status='blocked',error=repr(error));(args.output/'exception.txt').write_text(traceback.format_exc());raise
    finally:(args.output/'preflight_receipt.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
