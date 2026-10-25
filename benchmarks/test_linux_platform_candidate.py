"""Guard the Linux-only supplement against changing the frozen packages."""
import unittest
from unittest.mock import patch
from benchmarks.prepare_frozen_linux_candidate import validate_platform_resolution


def row(name,version,requires=()):
    return dict(metadata=dict(name=name,version=version,requires_dist=list(requires)),
                download_info=dict(url='https://example.invalid/'+name+'.whl',
                                   archive_info=dict(hashes=dict(sha256='a'*64))))


class PlatformCandidate(unittest.TestCase):
    def check(self,rows):
        from packaging.markers import default_environment
        env=default_environment();env.update(platform_system='Linux',sys_platform='linux',
            os_name='posix',platform_machine='x86_64',python_version='3.11',python_full_version='3.11.15')
        with patch('packaging.markers.default_environment',return_value=env):
            return validate_platform_resolution(dict(install=rows),dict(packages={'torch':'2.12.1+cu132'}))

    def test_exact_base_and_reachable_extra_dependency(self):
        rows=[row('torch','2.12.1+cu132',['cuda-toolkit[cublas]==13.2.1; platform_system == "Linux"']),
              row('cuda-toolkit','13.2.1',['nvidia-cublas==13.4.0.1; extra == "cublas"']),
              row('nvidia-cublas','13.4.0.1')]
        self.assertEqual(len(self.check(rows)),3)

    def test_substitution_orphan_missing_and_unhashed_rejected(self):
        for rows,message in (
            ([row('torch','2.11.0')],'substitution'),
            ([row('torch','2.12.1+cu132'),row('unrequested','1')],'not reachable'),
            ([row('torch','2.12.1+cu132',['cuda-toolkit==13.2.1'])],'Incomplete'),
            ([row('torch','2.12.1+cu132',['cuda-toolkit==13.2.1']),row('cuda-toolkit','13.1.0')],'Unsatisfied')):
            with self.subTest(message=message),self.assertRaisesRegex(RuntimeError,message):self.check(rows)
        candidate=row('torch','2.12.1+cu132');candidate['download_info']['archive_info']={}
        with self.assertRaisesRegex(RuntimeError,'hashed'):self.check([candidate])


if __name__=='__main__':unittest.main()
