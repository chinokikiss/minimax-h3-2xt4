import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.private_reference_delivery import private_reference_prefix, mirror_reference_prefix


class PrivateReferenceDelivery(unittest.TestCase):
    def test_binary_roundtrip_preserves_original_bytes_and_reference_order(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            output = source / 'delivered'
            rows = []
            for index, raw in enumerate((bytes(range(256)) * 17, b'__REFERENCE_ROWS__\x00\xff' * 31)):
                name = f'input_{index}.bin'
                (source / name).write_bytes(raw)
                rows.append(dict(file=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
            namespace = {}
            exec(private_reference_prefix(rows, source, output), namespace)
            self.assertEqual(namespace['REFERENCE_ROWS'], rows)
            for row in rows:
                self.assertEqual((source / row['file']).read_bytes(), (output / row['file']).read_bytes())

    def test_changed_input_rejected_before_notebook_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'image').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                private_reference_prefix([dict(file='image', bytes=7, sha256='0' * 64)], root, root / 'out')

    def test_immutable_mirror_is_byte_exact_and_corruption_blocks_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = b'original reference bytes\x00\xff'
            row = dict(file='ref.png', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            urls={'ref.png':'https://raw.githubusercontent.com/owner/repo/'+'1'*40+'/ref.png'}
            prefix = mirror_reference_prefix([row], urls, root / 'output')
            with patch('urllib.request.urlopen',return_value=io.BytesIO(raw)):exec(prefix, {})
            self.assertEqual((root / 'output' / row['file']).read_bytes(), raw)
            with patch('urllib.request.urlopen',return_value=io.BytesIO(b'corrupted')):
                with self.assertRaises(ValueError):exec(prefix, {})
            with self.assertRaises(ValueError):
                mirror_reference_prefix([row],{'ref.png':urls['ref.png'].replace('1'*40,'main')},root)
