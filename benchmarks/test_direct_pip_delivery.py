"""Guard exact pins, native-platform preservation and reuse without GPU work."""
import unittest
from benchmarks.prepare_kaggle_direct_pip import requirements


class DirectPins(unittest.TestCase):
    def fixture(self):
        pins={'comfy-kitchen':'0.2.37','tilelang':'0.1.13','psutil':'7.2.2'}
        return dict(core_pins=pins,requirements={'3.11':{n:f'{n}=={v} --hash=sha256:'+64*'a' for n,v in pins.items()}})

    def test_reuse_native_exact_packages_and_install_only_differences(self):
        changed,text=requirements(self.fixture(),{'psutil':'7.2.2','tilelang':'0.1.15'},'3.11')
        self.assertEqual(changed,{'comfy-kitchen':'0.2.37','tilelang':'0.1.13'})
        self.assertIn('tilelang==0.1.13 --hash=sha256:',text)
        self.assertNotIn('psutil',text)

    def test_native_torch_or_unqualified_abi_is_rejected(self):
        lock=self.fixture();lock['core_pins']['torch']='2.12.1'
        with self.assertRaises(ValueError):requirements(lock,{},'3.11')
        with self.assertRaises(RuntimeError):requirements(self.fixture(),{},'3.14')


if __name__=='__main__':unittest.main()
