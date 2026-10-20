"""Preserve current Kaggle log snapshots without waiting for benchmarks to finish."""
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parent / 'kaggle_retest_output' / 'partial'
RUNS = {
    'main': 'chinokiki9/minimax-h3-cuda13-retest',
    'corrected': 'chinokiki9/minimax-h3-fp32-and-pp-retest',
}

def capture(name, slug):
    api = KaggleApi()
    api.authenticate()
    deadline = time.monotonic() + 15
    try:
        with (ROOT / f'{name}_live.log').open('w', encoding='utf-8') as log:
            for event in api.kernels_logs_stream(slug):
                log.write(event.get('data', ''))
                log.flush()
                if time.monotonic() > deadline:
                    break
    except Exception as exc:
        print(name, type(exc).__name__, str(exc)[:150], flush=True)

if __name__ == '__main__':
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'capture.json').write_text(json.dumps({
        'captured_utc':datetime.now(timezone.utc).isoformat(), 'runs':RUNS,
        'evidence':'Live log snapshots; raw timing samples and unprinted correctness metrics may be unavailable.',
    }, indent=2), encoding='utf-8')
    threads = [threading.Thread(target=capture, args=(name,slug), daemon=True) for name,slug in RUNS.items()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=16)
    for name in RUNS:
        path = ROOT / f'{name}_live.log'
        print(name, path.stat().st_size if path.exists() else 0, 'log bytes', flush=True)
