"""Invalidate stale idle ownership before trained weights can be consumed."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from patches.comfy_tp_runtime import RankRuntime


class Identity(unittest.TestCase):
    def test_native_replacement_and_offline_manifest_change_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);native=root/'native.safetensors';native.write_bytes(b'native')
            shard=root/'te.block00.rank0.safetensors';shard.write_bytes(b'shard')
            manifest=shard.with_suffix('.json');manifest.write_text('{}')
            s=native.stat();t=shard.stat();m=manifest.stat()
            reader=SimpleNamespace(path=native,signature=(s.st_size,s.st_mtime_ns),
                rank_offline_verified={str(shard):(t.st_size,t.st_mtime_ns,m.st_mtime_ns,m.st_size)})
            owner=SimpleNamespace(readers={'te':reader})
            RankRuntime._check_idle_identity(owner)
            manifest.write_text('{"changed":true}')
            with self.assertRaisesRegex(RuntimeError,'Rank-local checkpoint changed'):
                RankRuntime._check_idle_identity(owner)
            native.write_bytes(b'changed native checkpoint')
            with self.assertRaisesRegex(RuntimeError,'TE checkpoint changed'):
                RankRuntime._check_idle_identity(owner)


if __name__=='__main__':unittest.main()
