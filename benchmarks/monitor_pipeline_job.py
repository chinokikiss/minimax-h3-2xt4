"""NVML bus telemetry for a separate audit process, without CUDA contexts.

Sampled PCIe throughput is an estimate, not an exact wire byte counter. Stage
markers are emitted after CUDA fences by the audit harness. Profiled work and
validation remain outside those stage ranges.
"""
import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command is required')
    import pynvml as nv
    nv.nvmlInit()
    handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())]
    result = dict(method='NVML throughput sampled every 50 ms; trapezoid integration; approximate PCIe bytes, all traffic',
                  samples=[], stages=[], unsupported=[], passed=False,
                  system_total_bytes=psutil.virtual_memory().total,
                  nvml_total_bytes=[nv.nvmlDeviceGetMemoryInfo(h).total for h in handles])
    stop = threading.Event()
    write_lock = threading.Lock()

    def snapshot():
        # A watchdog can terminate the whole process group mid-stage. Save
        # periodically as well as at stage boundaries, with atomic replacement.
        with write_lock:
            data=dict(result, samples=list(result['samples']),
                      stages=[dict(s) for s in result['stages']],
                      unsupported=list(result['unsupported']), saved_at=time.time())
            temporary=args.output.with_suffix(args.output.suffix+'.partial')
            temporary.write_text(json.dumps(data,indent=2))
            temporary.replace(args.output)

    def monitor():
        next_save=time.monotonic()+30
        while not stop.is_set():
            vm = psutil.virtual_memory()
            devices = []
            for i, handle in enumerate(handles):
                values = dict(index=i, used=nv.nvmlDeviceGetMemoryInfo(handle).used)
                try:
                    utilization=nv.nvmlDeviceGetUtilizationRates(handle)
                    values.update(gpu_util_percent=utilization.gpu,
                                  memory_util_percent=utilization.memory)
                except nv.NVMLError:
                    values.update(gpu_util_percent=None,memory_util_percent=None)
                for name, selector in [('rx', nv.NVML_PCIE_UTIL_RX_BYTES), ('tx', nv.NVML_PCIE_UTIL_TX_BYTES)]:
                    try:
                        # NVML reports KB/s over its most recent sampling period.
                        values[name+'_bytes_s'] = nv.nvmlDeviceGetPcieThroughput(handle, selector)*1024
                    except nv.NVMLError as error:
                        values[name+'_bytes_s'] = None
                        label = f'{i}/{name}/{type(error).__name__}'
                        if label not in result['unsupported']:
                            result['unsupported'].append(label)
                devices.append(values)
            result['samples'].append(dict(time=time.time(), available=vm.available,
                system_used=vm.total-vm.available, devices=devices))
            if time.monotonic()>=next_save:
                snapshot();next_save=time.monotonic()+30
            stop.wait(.05)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for line in proc.stdout:
            stamp = time.time()
            print(line.rstrip(), flush=True)
            if line.startswith('AUDIT STAGE '):
                fields = line.split()
                duration = float(fields[4])/1000
                result['stages'].append(dict(strategy=fields[2], stage=fields[3],
                    end=stamp, start=stamp-duration, seconds=duration))
                # Preserve telemetry for completed stages if the outer bounded
                # runner subsequently kills this process group.
                snapshot()
        result['returncode'] = proc.wait()
        result['passed'] = result['returncode'] == 0
    finally:
        stop.set()
        thread.join()
        for stage in result['stages']:
            rows = [r for r in result['samples'] if stage['start'] <= r['time'] <= stage['end']]
            estimates = []
            for i in range(len(handles)):
                values = {}
                for direction in ('rx','tx'):
                    field = direction+'_bytes_s'
                    pairs = list(zip(rows, rows[1:]))
                    values[direction+'_bytes_estimate'] = (
                        sum((b['time']-a['time'])*(a['devices'][i][field]+b['devices'][i][field])/2
                            for a,b in pairs) if pairs and all(r['devices'][i][field] is not None for r in rows) else None)
                estimates.append(values)
            stage['pcie'] = estimates
        snapshot()
        nv.nvmlShutdown()
    raise SystemExit(result.get('returncode', 1))


if __name__ == '__main__':
    main()
