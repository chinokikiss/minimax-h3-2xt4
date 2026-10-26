import unittest
import torch
from .graph import signature


class GraphContractTests(unittest.TestCase):
    def test_inference_geometry_and_strided_value(self):
        with torch.inference_mode():
            q=torch.empty(1,4,17,64);k=torch.empty(1,2,19,64)
            v=torch.empty(1,19,2,64).transpose(1,2)
            self.assertEqual(signature(q,k,v)[2][0],tuple(k.shape))
            with self.assertRaises(ValueError):signature(q,k,v.double())
            with self.assertRaises(ValueError):signature(q,k,v[:,:,:18])
            with self.assertRaises(ValueError):signature(q[:, :3],k,v)

    def test_training_rejected(self):
        x=torch.empty(1,2,3,64)
        with self.assertRaises(ValueError):signature(x,x,x)
        with torch.no_grad():
            with self.assertRaises(ValueError):signature(x.requires_grad_(),x,x)


if __name__=='__main__':unittest.main()
