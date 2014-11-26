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

