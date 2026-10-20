"""Unchanged full requests with narrowly scoped buffered TE read observers."""
import functools
import json
import os
from pathlib import Path
import threading

os.environ['MINIMAX_H3_TE_ABLATION']='baseline'
import te_pipeline_ablation as te
import cpu_causal_trace as cpu
import te_io_telemetry as telemetry


class IORecorder(te.TERecorder):
    def start(self,tag,kineto=False):
        self.io_observer=telemetry.ReadObserver()
        self.io_stream=(self.output/f'io_reads_r{self.rank}.jsonl').open('w',buffering=1024*1024)
        self.io_count=0;self.io_dropped=0;self.io_lock=threading.Lock();self.io_accept=True
        (self.output/f'io_capabilities_r{self.rank}.json').write_text(json.dumps(telemetry.capabilities(),indent=2))
        super().start(tag,False)  # No Kineto/GIL sampler: reuse existing GPU evidence.

    def stop(self):
        result=super().stop()
        with self.io_lock:
            self.io_accept=False;self.io_stream.close()
        self.io_observer.close()
        (self.output/f'io_counts_r{self.rank}.json').write_text(json.dumps(dict(reads=self.io_count,dropped=self.io_dropped)))
        return result

    def save_read(self,row,path):
        row.update(rank=self.rank,pid=os.getpid(),source=str(path),**self.fields())
        # A two-request diagnostic normally has a few thousand reads. Never
        # silently truncate evidence; the analyzer rejects any dropped rows.
        with self.io_lock:
            # Background next-request reads can finish after capture closes.
            # They do not reopen/export traces or change the underlying read.
            if not self.io_accept:return
            if self.io_count>=20000:self.io_dropped+=1;return
            self.io_stream.write(json.dumps(row,separators=(',',':'))+'\n')
            self.io_count+=1
        self.emit('te_buffered_read',row['start_ns'],row['end_ns'],source=str(path),
            bytes=row.get('returned_bytes'),io_row=self.io_count-1)


def install(recorder):
    cpu.install(recorder)
    def open_file(original):
        @functools.wraps(original)
        def wrapped(path,*a,**k):
            file=original(path,*a,**k)
            mode=a[0] if a else k.get('mode','r')
            if (recorder.enabled and mode=='rb' and path.name.startswith('te.block')
                and path.suffix=='.safetensors'):
                return telemetry.ObservedFile(file,path,recorder)
            return file
        return wrapped
    cpu.base.patch(Path,'open',open_file)


te.pipeline.tracing.Recorder=IORecorder
te.pipeline.tracing.install=install
if __name__=='__main__':te.pipeline.main()
