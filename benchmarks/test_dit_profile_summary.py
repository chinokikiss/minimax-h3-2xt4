"""Process-qualified Nsight attribution; synthetic trace rows, no GPU/API calls."""
from pathlib import Path
import sqlite3
import tempfile
import unittest
from benchmarks.summarize_dit_single_profile import attribute,union_ns


class Attribution(unittest.TestCase):
    def test_union_does_not_double_count_overlapping_kernels(self):
        self.assertEqual(union_ns([(0,10),(5,20),(25,30)]),25)

    def test_correlation_ids_are_qualified_by_process_and_warmup_is_excluded(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'trace.sqlite';db=sqlite3.connect(path)
            db.executescript('''create table StringIds(id integer,value text);
create table NVTX_EVENTS(start integer,end integer,text text,textId integer,globalTid integer);
create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer,nameId integer);
create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,correlationId integer,deviceId integer,shortName integer);''')
            db.executemany('insert into StringIds values (?,?)',[(1,'kernelA'),(2,'kernelB'),(3,'cudaLaunchKernel')])
            for rank in (0,1):
                pid=(100+rank)<<24;tid=pid+3
                db.executemany('insert into NVTX_EVENTS values (?,?,?,?,?)',[
                    (10,100,f'denoise/nfe1/rank{rank}',None,tid),
                    (15,90,f'H3_DIT/rank={rank}/block=0/'+('QKV/column' if rank==0 else 'FC1/column'),None,tid)])
                db.executemany('insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?)',[(20,21,tid,7,3),(1,2,tid,8,3)])
                db.executemany('insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?)',[(40,60,pid,7,rank,rank+1),(1,4,pid,8,rank,rank+1)])
            db.commit();db.close();result=attribute(path)
            self.assertEqual(result['matched_denoise_kernels'],2)
            self.assertEqual({(p['device'],p['phase']) for p in result['phases']},{(0,'QKV/column'),(1,'FC1/column')})
            self.assertTrue(all(p['kernel_count']==1 for p in result['phases']))


if __name__=='__main__':unittest.main()
