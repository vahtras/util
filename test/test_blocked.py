import unittest
import numpy
from ..blocked import *

class TestBlocked(unittest.TestCase):

    def setUp(self):
        self.bdm = BlockDiagonalMatrix((2, 1), (2, 1))

    def tearDown(self):
        pass

    def assert_allclose(self, this, ref):
        for t, r in zip(this, ref):
            numpy.testing.assert_allclose(t, r)

    def test_rowdim(self):
        self.assertTupleEqual(self.bdm.nrow, (2, 1))

    def test_coldim(self):
        self.assertTupleEqual(self.bdm.ncol, (2, 1))

    def test_row_offset(self):
        self.assertTupleEqual(self.bdm.irow, (0, 2))

    def test_col_offset(self):
        self.assertTupleEqual(self.bdm.icol, (0, 2))

    def test_incompatible_dim_raises(self):
        self.assertRaises(AssertionError, BlockDiagonalMatrix, (1, 2), (3,))

    def test_str(self):
        self.assertEqual(str(self.bdm), """
Block 1

 (2, 2) 
              Column   1    Column   2

Block 2

 (1, 1) 
              Column   1
""")


    def test_unblock(self):
        blocked = BlockDiagonalMatrix((2, 1), (2, 1))
        blocked[0][:, :] = ((1, 2), (3, 4))
        blocked[1][:, :] = ((5,),)
        unblocked = [[1, 2, 0], [3, 4, 0], [0, 0, 5]]
        numpy.testing.assert_allclose(blocked.unblock(), unblocked)

    def test_pack(self):
        blocked = BlockDiagonalMatrix((2, 1), (2, 1))
        blocked[0][:, :] = ((1, 2), (2, 1))
        blocked[1][:, :] = ((5,),)
        packed = [[1, 2, 1], [5]]
        self.assert_allclose(blocked.pack().subblock, packed)
        

    def test_unit(self):
        ref = [[[1, 0], [0, 1]], [[1]]]
        blocked_unit = unit((2, 1))
        self.assert_allclose(blocked_unit, ref)

    def test_mul(self):
        self.bdm[0][:, :] = [[0, 1], [2, 3]]
        self.bdm[1][:, :] = [[2]]
        self.assert_allclose(self.bdm*self.bdm, [[[2, 3], [6, 11]], [[4]]])

    def test_rmul(self):
        self.bdm[0][:, :] = [[0, 1], [2, 3]]
        self.bdm[1][:, :] = [[2]]
        self.assert_allclose(2*self.bdm, [[[0, 2], [4, 6]], [[4]]])

    def test_add(self):
        self.bdm[0][:, :] = [[0, 1], [2, 3]]
        self.bdm[1][:, :] = [[2]]
        self.assert_allclose(self.bdm + self.bdm, [[[0, 2], [4, 6]], [[4]]])

    def test_sub(self):
        self.bdm[0][:, :] = [[0, 1], [2, 3]]
        self.bdm[1][:, :] = [[2]]
        self.assert_allclose(self.bdm - self.bdm, [[[0, 0], [0, 0]], [[0]]])

    def test_neg(self):
        self.bdm[0][:, :] = [[0, 1], [2, 3]]
        self.bdm[1][:, :] = [[2]]
        self.assert_allclose(-self.bdm, [[[0, -1], [-2, -3]], [[-2]]])

    def test_div(self):
        self.bdm[0][:, :] = [[0, 1], [2, 3]]
        self.bdm[1][:, :] = [[2]]
        self.assert_allclose(self.bdm/self.bdm, [[[1, 0], [0, 1]], [[1]]])

    def test_scalar_div(self):
        self.bdm[0][:, :] = [[0, 1], [2, 3]]
        self.bdm[1][:, :] = [[2]]
        self.assert_allclose(self.bdm/2, [[[0, .5], [1, 1.5]], [[1]]])

    def test_rdiv(self):
        self.bdm[0][:, :] = [[0, 1], [2, 3]]
        self.bdm[1][:, :] = [[2]]
        self.assert_allclose(1/self.bdm, [[[-1.5, 0.5], [1, 0]], [[0.5]]])

    def test_sqrt(self):
        self.bdm[0][:, :] = [[4, 0], [0, 4]]
        self.bdm[1][:, :] = [[9]]
        self.assert_allclose(self.bdm.sqrt(), [[[2, 0], [0, 2]], [[3]]])

    def test_isqrt(self):
        self.bdm[0][:, :] = [[4, 0], [0, 4]]
        self.bdm[1][:, :] = [[9]]
        self.assert_allclose(self.bdm.invsqrt(), [[[.5, 0], [0, .5]], [[1./3]]])

    def test_eigvec(self):
        self.bdm[0][:, :] = [[4, 0], [0, 4]]
        self.bdm[1][:, :] = [[9]]
        u, v = self.bdm.eigvec()
        self.assert_allclose(u, [4, 9])
        self.assert_allclose(v, [[[1, 0], [0, 1]], [[1]]])

    def test_transpose(self):
        self.bdm[0][:, :] = [[1, 2], [3, 4]]
        bdm_T = self.bdm.T()
        self.assert_allclose(bdm_T[0], [[1, 3], [2, 4]])

    def test_get_columns(self):
        self.bdm[0][:, :] = [[1, 2], [3, 4]]
        subset = self.bdm.get_columns((1, 0))
        self.assert_allclose(subset[0], [1, 3])

    def test_func(self):
        self.bdm[0][:, :] = [[1, 0], [0, 16]]
        self.bdm[1][:, :] = [[25]]
        self.assert_allclose(self.bdm.sqrt(), [[[1, 0], [0, 4]], [[5]]])
        

class BlockedTriangularTest(unittest.TestCase):

    def setUp(self):
        pass

    def teardown(self):
        pass

    def test_str(self):
        bt = triangular((2, 1))
        print bt
        self.assertEqual(str(bt), """
Block 1

    0.00000000
    0.00000000    0.00000000

Block 2

    0.00000000
"""
        ) 

    def test_init(self):
        bt = triangular.init([[1., 2., 3.], [4.]])
        numpy.testing.assert_almost_equal(bt.subblock[0], [1., 2., 3.])
        numpy.testing.assert_almost_equal(bt.subblock[1], [4.])

        
        

