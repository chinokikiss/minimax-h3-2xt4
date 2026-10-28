"""Regression for the observed pre-denoiser HTTP 503, with no network requests."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import requests
from benchmarks.tp2_public_dit_assets import recover, download


def http_error(code):
    response = requests.Response()
    response.status_code = code
    return requests.HTTPError(response=response)


class DownloadRecovery(unittest.TestCase):
    def test_transient_503_recovers_and_is_bounded(self):
        calls, waits = [], []
        def operation(attempt):
            calls.append(attempt)
            if attempt < 2:
                raise http_error(503)
            return 'verified-by-caller'
        self.assertEqual(recover(operation, label='shard', sleep=waits.append), 'verified-by-caller')
        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(waits, [2, 5])

    def test_exhausted_retry_and_permanent_failures(self):
        calls = []
        def fail(attempt):
            calls.append(attempt)
            raise http_error(503)
        with self.assertRaises(requests.HTTPError):
            recover(fail, label='shard', sleep=lambda seconds: None)
        self.assertEqual(calls, [0, 1, 2])
        for error in (http_error(403), ValueError('bad identity')):
            with self.assertRaises(type(error)):
                recover(lambda attempt: (_ for _ in ()).throw(error), label='shard',
                        sleep=lambda seconds: self.fail('permanent error retried'))

    def test_corrupt_download_is_rejected_without_redownload(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'shard'
            target.write_bytes(b'bad')
            row = dict(repository='public/model', file='shard', revision='a'*40,
                       bytes=3, sha256='0'*64)
            with patch('huggingface_hub.hf_hub_download', return_value=str(target)) as fetch:
                with self.assertRaisesRegex(ValueError, 'Immutable checkpoint mismatch'):
                    download(row, directory)
                self.assertEqual(fetch.call_count, 1)


if __name__ == '__main__':
    unittest.main()
