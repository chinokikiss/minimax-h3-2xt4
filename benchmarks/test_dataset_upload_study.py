"""Regression tests for early probe setup failures; no credentials/network."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks import kaggle_dataset_upload_study as probe


class UploadStudy(unittest.TestCase):
    def test_cli_uses_native_distribution_console_entry_not_missing_main(self):
        from unittest.mock import Mock
        invoked=Mock(return_value=None)
        entry=SimpleNamespace(group='console_scripts',name='kaggle',load=Mock(return_value=invoked))
        with patch.object(probe.md,'distribution',return_value=SimpleNamespace(entry_points=[entry])), \
                patch.object(probe.os,'execv',side_effect=AssertionError('No python -m entry point')), \
                patch.object(probe.sys,'argv',[]):
            probe.operation('cli',Path('payload'),'owner/private')
            self.assertEqual(probe.sys.argv,['kaggle','datasets','create','-p','payload','-q','-t'])
        invoked.assert_called_once_with()

    def test_missing_scratch_parent_created_and_payload_streamed(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'not-created'/'nested'/'probe'
            block=b'x'*2**20
            with patch.object(probe.os,'urandom',return_value=block) as random:
                folder,digest=probe.prepare_payload(base,blocks=2)
            self.assertEqual((folder/'synthetic.dat').stat().st_size,2*2**20)
            self.assertEqual(digest,hashlib.sha256(block+block).hexdigest())
            self.assertEqual(random.call_args_list,[( (2**20,),{}),((2**20,),{})])
            with self.assertRaises(FileExistsError):probe.prepare_payload(base,blocks=1)

    def test_early_failure_keeps_phase_and_error_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'result'
            def fail(args):
                args.output.mkdir()
                probe.write(args.output/'result.json',dict(status='running',phase='payload_preparation',methods=[]))
                raise FileNotFoundError('synthetic setup failure')
            with patch.object(probe,'study',side_effect=fail),patch('sys.argv',['probe','--output',str(output)]):
                with self.assertRaises(FileNotFoundError):probe.main()
            result=json.loads((output/'result.json').read_text())
            self.assertEqual(result['status'],'failed');self.assertEqual(result['phase'],'payload_preparation')
            self.assertEqual(result['error_type'],'FileNotFoundError');self.assertEqual(result['methods'],[])
            self.assertTrue((output/'exception.txt').is_file())


if __name__=='__main__':unittest.main(verbosity=2)
