"""Small deterministic CPU gates for the real filesystem benchmark."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from benchmarks.kaggle_storage_probe import partitions, select_files, measure, copy_corpus, verify_corpus, mount_for


class StorageProbe(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]/'output/storage_probe_fixture'
        root.mkdir(parents=True, exist_ok=True)
        self.directory = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.directory.name).resolve()
        self.assertTrue(self.root.is_relative_to(root.resolve()))
        self.addCleanup(self.directory.cleanup)
        for i in range(3):
            (self.root/f'{i}.bin').write_bytes(bytes(range(256))*8192)
        self.entries, self.selection = select_files(self.root, total=5*2**20, per_file=2*2**20)

    def test_partition_visits_every_byte_once_even_inside_a_file(self):
        groups = partitions(self.entries, 2)
        self.assertEqual([sum(e['length'] for e in g) for g in groups], [5*2**19]*2)
        for original in self.entries:
            intervals = sorted((e['offset'],e['offset']+e['length']) for group in groups for e in group if e['path']==original['path'])
            self.assertEqual(intervals[0][0], 0)
            self.assertEqual(intervals[-1][1], original['length'])
            self.assertTrue(all(a[1]==b[0] for a,b in zip(intervals, intervals[1:])))

    def test_sequential_and_range_counts_and_deadline_reporting(self):
        for mode in ('sequential','random_4k','strided_12800'):
            for workers in (1,2):
                r = measure(dict(entries=self.entries,mode=mode,workers=workers,requests=53,seconds=10))
                expected = self.selection['selected_bytes'] if mode=='sequential' else 53*(4096 if mode=='random_4k' else 12800)
                self.assertEqual(r['bytes'], expected)
                self.assertTrue(r['complete'])
                self.assertGreater(r['mib_per_second'],0)
        r=measure(dict(entries=self.entries,mode='sequential',seconds=-1))
        self.assertFalse(r['complete']);self.assertEqual(r['bytes'],0)

    def test_copies_are_byte_checked_and_capacity_is_capped(self):
        copied,_=copy_corpus(self.entries,self.root/'copied')
        verify_corpus(copied)
        self.assertEqual(sum(e['length'] for e in copied),5*2**20)
        self.assertEqual(copied[0]['sha256'],hashlib.sha256(Path(copied[0]['path']).read_bytes()).hexdigest())
        Path(copied[0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):verify_corpus(copied)

    def test_mount_uses_longest_actual_mount(self):
        mounts='1 0 0:1 / / rw - overlay overlay rw\n2 1 8:1 / /kaggle/input ro - ext4 /dev/sda ro\n'
        self.assertEqual(mount_for('/kaggle/input/data/file',mounts)['source'],'/dev/sda')
        self.assertEqual(mount_for('/tmp/file',mounts)['filesystem'],'overlay')


if __name__=='__main__':unittest.main()
