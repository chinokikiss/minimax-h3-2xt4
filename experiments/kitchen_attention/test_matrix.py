"""CPU checks of benchmark coverage and comparable output boundaries."""
import unittest
import torch
from .matrix import shapes,bnd


class MatrixContracts(unittest.TestCase):
    def test_both_physical_partitions_cover_short_and_long(self):
        for rank in (0,1):
            cases=shapes(rank)
            self.assertEqual([s.n for s in cases[:4]],[353,512,6814,51220])
            for shape in cases:
                self.assertEqual(shape.h%shape.hk,0)
                self.assertLessEqual(shape.prefix,shape.n)
                if shape.sparse:self.assertEqual((shape.n,shape.h),(shape.nk,shape.hk))
        self.assertEqual(shapes(0)[-1].n+shapes(1)[-1].n,1797)

    def test_common_bnd_conversion_preserves_head_token_order(self):
        tensor=torch.arange(2*3*5*7).reshape(2,3,5,7)
        output=bnd(tensor)
        self.assertEqual(tuple(output.shape),(2,5,21))
        for batch in range(2):
            for token in range(5):
                self.assertTrue(torch.equal(output[batch,token],tensor[batch,:,token].flatten()))

    def test_short_split_and_long_inline_workspace_regimes(self):
        cases={s.name:s for s in shapes(0)}
        workspace=lambda s:s.h*((s.n+63)//64*64)*((s.nk+63)//64)*4
        self.assertLess(workspace(cases['dit_short_512']),96*2**20)
        self.assertGreater(workspace(cases['dit_long_51220']),96*2**20)


if __name__=='__main__':unittest.main()
