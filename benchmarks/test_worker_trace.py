"""Reject incomplete CUDA coverage; exercise CPU probe without CUDA."""
from pathlib import Path
import hashlib
import json
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from benchmarks.worker_trace_coverage import check
from benchmarks.worker_cpu_probe import WorkerCPUProbe,validate_profile


class Coverage(unittest.TestCase):
    def fixture(self,path,missing_rank=False,missing_block=False,blocks=False):
        db=sqlite3.connect(path)
        db.executescript('''create table StringIds(id integer,value text);
create table NVTX_EVENTS(start integer,end integer,text text,textId integer,globalTid integer);
create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer,nameId integer);
create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,correlationId integer,deviceId integer,shortName integer);''')
        for rank in (0,1):
            pid=(100+rank)<<24;tid=pid+3
            db.execute('insert into NVTX_EVENTS values (?,?,?,?,?)',(0,10000,
                f'denoise/nfe1/rank{rank}' if blocks else f'H3_CAPTURE/rank={rank}',None,tid))
            for i in range(50 if blocks else 4):
                start=10+i*100
                if blocks:db.execute('insert into NVTX_EVENTS values (?,?,?,?,?)',
                    (start,start+50,f"H3_DIT/rank={rank}/block=('dit', {i}, {rank})/block_compute",None,tid))
                db.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?)',(start+1,start+2,tid,i,1))
                if (not missing_rank or rank!=1) and (not missing_block or i!=49):
                    db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?)',(start+3,start+10,pid,i,rank,1))
        db.commit();db.close()

    def test_two_correlated_worker_devices_are_required(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'valid.sqlite';self.fixture(p);self.assertTrue(check(p)['passed'])
            p=Path(t)/'missing.sqlite';self.fixture(p,missing_rank=True);self.assertFalse(check(p)['passed'])

    def test_every_trained_block_needs_correlated_kernels(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'valid.sqlite';self.fixture(p,blocks=True)
            self.assertTrue(check(p,prefix='denoise/nfe',minimum=50,require_blocks=True)['passed'])
            p=Path(t)/'missing.sqlite';self.fixture(p,blocks=True,missing_block=True)
            self.assertFalse(check(p,prefix='denoise/nfe',minimum=1,require_blocks=True)['passed'])

    def test_each_resumed_chunk_needs_kernels(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'chunks.sqlite';self.fixture(p)
            db=sqlite3.connect(p)
            try:
                for rank in (0,1):
                    pid=(100+rank)<<24;tid=pid+3
                    db.execute('insert into NVTX_EVENTS values (?,?,?,?,?)',(8000,9000,f'H3_CAPTURE/rank={rank}/chunk1',None,tid))
                db.commit()
                self.assertFalse(check(p,minimum=4,minimum_per_label=4)['passed'])
                for rank in (0,1):
                    pid=(100+rank)<<24;tid=pid+3
                    for i in range(4):
                        start=8010+i*100
                        db.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?)',(start,start+1,tid,50+i,1))
                        db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?)',(start+2,start+3,pid,50+i,rank,1))
                db.commit()
                self.assertTrue(check(p,minimum=8,minimum_per_label=4)['passed'])
            finally:db.close()

    def test_driver_api_launches_are_correlated_without_runtime_table(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'driver.sqlite';self.fixture(p,blocks=True)
            db=sqlite3.connect(p)
            try:
                db.execute('alter table CUPTI_ACTIVITY_KIND_RUNTIME rename to CUPTI_ACTIVITY_KIND_DRIVER');db.commit()
            finally:db.close()
            self.assertTrue(check(p,prefix='denoise/nfe',minimum=50,require_blocks=True)['passed'])

    def test_cpu_probe_calls_original_once_and_retains_call_stack_evidence(self):
        with tempfile.TemporaryDirectory() as t:
            probe=WorkerCPUProbe(0,t,interval=.002);probe.start();calls=[]
            def original(segments):calls.append(1);threading.Event().wait(.02);return 7
            self.assertEqual(probe.invoke('norm/modulate',original,([(0,8,0)],),{}),7)
            value=probe.finish()
            self.assertEqual(calls,[1])
            self.assertFalse(value['calls'][0]['cprofile_enabled']);self.assertTrue(value['stack_samples'])
            self.assertEqual(value['segments'],[dict(start=0,end=8,row=0)])

    def test_incremental_cpu_evidence_survives_without_finish(self):
        with tempfile.TemporaryDirectory() as t:
            probe=WorkerCPUProbe(0,t,interval=.002);probe.start()
            try:
                probe.mark_progress('weight_acquire',('dit',28,0))
                probe.invoke('norm/modulate',lambda segments:7,([(0,8,0)],),{})
                probe.checkpoint()
                value=json.loads((Path(t)/'rank0_cpu.json').read_text())
                self.assertFalse(value['final']);self.assertEqual(value['calls'][0]['call'],1)
                self.assertIn('28',value['progress']['block'])
            finally:probe.finish()

    def test_resident_profile_clock_remains_valid_with_concurrent_disk_thread(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'data';path.write_bytes(b'X'*65536);stop=threading.Event()
            def background():
                while not stop.is_set():hashlib.sha256(path.read_bytes()).digest()
            thread=threading.Thread(target=background,name='concurrent-disk-fixture');thread.start()
            probe=WorkerCPUProbe(0,t,interval=.002,deterministic=True);probe.start()
            try:
                def original(segments):threading.Event().wait(.02);return sum(range(1000))
                for _ in range(2):probe.invoke('norm/modulate',original,([(0,8,0)],),{})
                value=probe.finish()
                self.assertTrue(all(row['profile_validation']['passed'] for row in value['calls']))
                self.assertTrue(all(row['profile_clock']=='monotonic_wall' for row in value['calls']))
                self.assertTrue(all(row['profile_engine']=='stdlib_profile_thread_local' for row in value['calls']))
                self.assertTrue(all(fn['function'] not in ('background','read_bytes') for row in value['calls'] for fn in row['functions']))
            finally:
                stop.set();thread.join(2)
                if probe.enabled:probe.finish()
            self.assertFalse(thread.is_alive())

    def test_corrupt_profile_times_are_rejected(self):
        stats=SimpleNamespace(total_tt=.004,stats={('x',1,'bad'):(1,1,42.,42.,{})})
        self.assertFalse(validate_profile(stats)['passed'])
        stats.stats={('x',1,'bad'):(1,1,-1.,-1.,{})}
        self.assertFalse(validate_profile(stats)['passed'])

    def test_backend_case_cannot_execute_twice(self):
        from benchmarks.launch_dit_profile import claim_profile
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);config=root/'config.json';config.write_text('{}');inputs=root/'inputs.pt';inputs.write_bytes(b'private')
            claim_profile(root,config,inputs);before=(root/'profile.claimed').read_bytes()
            with self.assertRaises(FileExistsError):claim_profile(root,config,inputs)
            self.assertEqual((root/'profile.claimed').read_bytes(),before)

    def test_explicit_checkpoints_restart_and_persist_final_stop(self):
        from benchmarks.worker_cuda_checkpoint import CUDAFlushCheckpoint
        events=[]
        api=SimpleNamespace(synchronize=lambda device:events.append(('sync',device)),
            profiler=SimpleNamespace(stop=lambda:events.append('stop'),start=lambda:events.append('start')),
            nvtx=SimpleNamespace(range_push=lambda label:None,range_pop=lambda:None))
        with tempfile.TemporaryDirectory() as t:
            checkpoint=CUDAFlushCheckpoint(0,t,backend=api)
            checkpoint.flush('dit_block_4');checkpoint.flush('worker_close',resume=False)
            self.assertEqual(events,[('sync',0),'stop','start',('sync',0),'stop'])
            value=json.loads((Path(t)/'rank0_cuda_checkpoints.json').read_text())
            self.assertTrue(value['records'][0]['restarted']);self.assertFalse(value['records'][1]['restarted'])
            self.assertTrue(all(row['status']=='complete' and row['stopped'] for row in value['records']))

    def test_checkpoint_failure_is_durable_and_background_cuda_is_rejected(self):
        from benchmarks.worker_cuda_checkpoint import CUDAFlushCheckpoint
        events=[]
        def fail():raise RuntimeError('stop failed')
        api=SimpleNamespace(synchronize=lambda device:events.append('sync'),
            profiler=SimpleNamespace(stop=fail,start=lambda:events.append('start')),
            nvtx=SimpleNamespace(range_push=lambda label:None,range_pop=lambda:None))
        with tempfile.TemporaryDirectory() as t:
            checkpoint=CUDAFlushCheckpoint(1,t,backend=api)
            with self.assertRaises(RuntimeError):checkpoint.flush('error')
            value=json.loads((Path(t)/'rank1_cuda_checkpoints.json').read_text())
            self.assertEqual(value['records'][0]['status'],'failed');self.assertFalse(value['records'][0]['stopped'])
            failures=[]
            def background():
                try:checkpoint.flush('background')
                except RuntimeError:failures.append(1)
            thread=threading.Thread(target=background);thread.start();thread.join(2)
            self.assertEqual(failures,[1]);self.assertEqual(events,['sync'])

    def test_background_cpu_preparation_never_creates_cuda_observer_events(self):
        from benchmarks.dit_performance_observer import Phases
        phases=Phases(0);phases.enabled=True;seen=[]
        def background():
            try:
                with phases.span('CPU/quantize'):seen.append('executed')
            except BaseException as error:seen.append(error)
        with patch('torch.cuda.Event',side_effect=AssertionError('CPU preparation touched CUDA')):
            thread=threading.Thread(target=background);thread.start();thread.join(2)
        self.assertFalse(thread.is_alive());self.assertEqual(seen,['executed']);self.assertFalse(phases.cpu)


if __name__=='__main__':unittest.main()
