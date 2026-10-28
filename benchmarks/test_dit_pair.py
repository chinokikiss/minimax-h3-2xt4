"""CPU contracts for paired calls, allocator semantics and CUDA attribution."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import torch
from benchmarks.analyze_dit_pair import attribute,merge,duration,RangeIndex
from benchmarks.dit_pair_contract import fixture,identity,NOISE_SHA,TOKENS,SEGMENTS,digest


class Contract(unittest.TestCase):
    def test_seeded_pair_and_packet_keep_native_layout_without_alias(self):
        from patches.comfy_tp_runtime import packet
        data=fixture();proof=identity(data)
        self.assertEqual(proof['noise_sha256'],identity(fixture())['noise_sha256'])
        self.assertEqual(proof['prior_noise_sha256'],list(NOISE_SHA))
        self.assertEqual(data['minimax_payload']['layout'].seq_len,TOKENS)
        copied=packet(data,'cpu')
        self.assertEqual(copied['minimax_payload']['layout'].segments,SEGMENTS)
        self.assertEqual(digest(copied['x'][0]),digest(data['x'][0]))
        self.assertNotEqual(copied['x'][0].data_ptr(),data['x'][0].data_ptr())
        self.assertIn('synthetic',proof['distribution'])
        tags=data['minimax_payload']['text_token_tags']
        self.assertEqual(int((tags==0).sum()),1504)
        self.assertTrue(torch.equal(tags[:6],torch.ones(6,dtype=torch.long)))
        self.assertTrue(torch.equal(tags[6:767],torch.zeros(761,dtype=torch.long)))
        self.assertTrue(torch.equal(tags[773:1516],torch.zeros(743,dtype=torch.long)))

    def test_interval_unions_and_half_open_nested_scopes(self):
        self.assertEqual(merge([(1,4),(2,8),(8,9),(15,20)]),[(1,9),(15,20)])
        self.assertEqual(duration([(1,4),(2,8)]),7)
        index=RangeIndex([(1,9,'outer'),(2,5,'inner'),(5,7,'next')])
        self.assertEqual([r[2] for r in index.at(4)],['outer','inner'])
        self.assertEqual([r[2] for r in index.at(5)],['outer','next'])
        self.assertEqual(index.at(9),())


def database(path,omit_last=False):
    db=sqlite3.connect(path)
    db.executescript('''create table StringIds(id integer,value text);
      create table NVTX_EVENTS(start integer,end integer,globalTid integer,text text,textId integer);
      create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer,nameId integer);
      create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,correlationId integer,deviceId integer,shortName integer,demangledName integer);''')
    db.executemany('insert into StringIds values (?,?)',[(1,'cudaLaunchKernel'),(2,'compute_kernel'),(3,'nccl_sum')])
    for rank in (0,1):
        pid=(rank+1)<<24;tid=pid+1
        for run,start in [(1,1000000),(2,4000000)]:
            db.execute('insert into NVTX_EVENTS values (?,?,?,?,?)',(start,start+1000000,tid,f'H3_PAIR_CALL/run={run}/rank={rank}',None))
            for block in range(50):
                a=start+block*10000;correlation=run*100+block
                db.execute('insert into NVTX_EVENTS values (?,?,?,?,?)',(a,a+8000,tid,f"H3_DIT/rank={rank}/block=('dit', {block}, {rank})/block_compute",None))
                db.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?,?,?)',(a+100,a+200,tid,correlation,1))
                if not (omit_last and rank==1 and run==2 and block==49):
                    db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?)',(a+300,a+7000,pid,correlation,rank,2,2))
            # A background NCCL launch is not correlated to the main thread.
            db.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?)',(start+600000,start+700000,pid,9999,rank,3,3))
    db.commit();db.close()


class Attribution(unittest.TestCase):
    def test_both_runs_both_ranks_and_background_kernels_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.sqlite';database(path)
            result=attribute(path)
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['calls']),4)
        for call in result['calls']:
            self.assertEqual(call['trained_blocks_with_kernels'],50)
            self.assertEqual(call['kernel_count'],51)
            self.assertEqual(call['uncorrelated_kernel_count'],1)
            self.assertAlmostEqual(call['all_gpu_activity_union_ms']+call['idle_ms'],call['span_ms'])
            self.assertGreater(call['nccl_union_ms'],0)

    def test_missing_one_block_in_second_rank_second_call_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trace.sqlite';database(path,omit_last=True)
            result=attribute(path)
        self.assertFalse(result['complete'])
        self.assertEqual(result['calls'][-1]['trained_blocks_with_kernels'],49)


if __name__=='__main__':unittest.main()
