"""Linux correctness smoke for mincore, counter brackets and unchanged reads."""
import hashlib
import os
from pathlib import Path
import tempfile
import unittest

import te_io_telemetry as telemetry


@unittest.skipUnless(os.name=='posix','Linux telemetry qualification')
class FileTelemetryTests(unittest.TestCase):
    def test_read_forwarding_and_residency(self):
        payload=bytes(range(256))*8192
        with tempfile.TemporaryDirectory(prefix='te_io_test_') as directory:
            path=Path(directory)/'probe.bin';path.write_bytes(payload)
            observer=telemetry.ReadObserver(period=.01)
            try:
                with path.open('rb',buffering=0) as file:
                    os.fdatasync(file.fileno())
                    os.posix_fadvise(file.fileno(),0,0,os.POSIX_FADV_DONTNEED)
                    initial=telemetry.residency(file.fileno())
                    self.assertIn('resident_fraction',initial)
                    buffer=bytearray(len(payload))
                    count,row=observer.measure(file,len(buffer),lambda:file.readinto(buffer))
                    self.assertEqual(count,len(payload))
                    self.assertEqual(hashlib.sha256(buffer).digest(),hashlib.sha256(payload).digest())
                    self.assertGreaterEqual(row['wall_ns'],row['cpu_ns'])
                    self.assertGreaterEqual(row['counters']['io'].get('rchar',0),len(payload))
                    self.assertEqual(row['resident_after']['resident_fraction'],1.0)
                    file.seek(0)
                    count,hot=observer.measure(file,len(buffer),lambda:file.readinto(buffer))
                    self.assertEqual(hot['resident_before']['resident_fraction'],1.0)
                    self.assertEqual(count,len(payload))
            finally:observer.close()


if __name__=='__main__':unittest.main()
