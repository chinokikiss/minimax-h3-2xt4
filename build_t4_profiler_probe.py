"""Build an isolated private Kaggle hardware/profiler capability check."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
folder = root / 'kaggle_t4_profiler_probe'
folder.mkdir(exist_ok=True)
source = (root / 'benchmarks/t4_profiler_probe.py').read_text()
notebook = dict(nbformat=4, nbformat_minor=5,
    metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    cells=[dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                source=source.splitlines(keepends=True))])
(folder / 'experiment.ipynb').write_text(json.dumps(notebook), encoding='utf-8')
(folder / 'kernel-metadata.json').write_text(json.dumps(dict(
    id='chinokiki9/h3-t4-profiler-capability-probe', title='H3 T4 Profiler Capability Probe',
    code_file='experiment.ipynb', language='python', kernel_type='notebook', is_private=True,
    enable_gpu=True, enable_tpu=False, enable_internet=True, dataset_sources=[],
    competition_sources=[], kernel_sources=[], model_sources=[]), indent=2))
print(folder)
