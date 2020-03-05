import unittest
import numpy as np
import numpy.testing as npt

from util.full import Triangular


class TestTriangular(unittest.TestCase):
    def setUp(self):
        self.twodim = Triangular.init([1.0, 2.0, 3.0])
        np.random.seed(0)
        self.antisym = Triangular((2, 2), anti=True)
        self.antisym[1, 0] = 2.0

    def test_init_square_shape(self):
        self.assertEqual(self.twodim.sshape, (2, 2))

    def test_init_values(self):
        npt.assert_equal(np.array(self.twodim), [1.0, 2.0, 3.0])

    def test_str(self):
        self.assertEqual(
            str(self.twodim),
            """
    1.00000000
    2.00000000    3.00000000
""",
        )

    def test_getitem_by_tuple(self):
        self.assertEqual(self.twodim[0, 0], 1.0)
        self.assertEqual(self.twodim[1, 0], 2.0)
        self.assertEqual(self.twodim[0, 1], 2.0)
        self.assertEqual(self.twodim[1, 1], 3.0)

    def test_getitem_by_tuple_as(self):
        self.assertEqual(self.antisym[0, 0], 0.0)
        self.assertEqual(self.antisym[1, 0], 2.0)
        self.assertEqual(self.antisym[0, 1], -2.0)
        self.assertEqual(self.antisym[1, 1], 0.0)

    def test_getitem_by_scalar(self):
        self.assertEqual(self.twodim[0], 1.0)
        self.assertEqual(self.twodim[1], 2.0)
        self.assertEqual(self.twodim[2], 3.0)

    def test_setitem(self):
        self.twodim[0, 1] = 4.0
        npt.assert_equal(np.array(self.twodim), [1.0, 4.0, 3.0])

    def test_setitem_anti1(self):
        self.antisym[0, 1] = -2.0
        npt.assert_equal(np.array(self.antisym), [0.0, 2.0, 0.0])

    def test_setitem_anti2(self):
        self.antisym[1, 0] = 2.0
        npt.assert_equal(np.array(self.antisym), [0.0, 2.0, 0.0])

    def test_unpack(self):
        npt.assert_equal(self.twodim.unpack(), [[1.0, 2.0], [2.0, 3.0]])

    def test_unpack_as(self):
        As = Triangular((2, 2), anti=True)
        As.random()
        npt.assert_allclose(
            As.unpack(), [[0.0000000, -0.71518937], [0.71518937, 0.00000000]]
        )

    def test_mul(self):
        npt.assert_equal(
            self.twodim * self.twodim, [[5.0, 8.0], [8.0, 13.0]]
        )

    def test_mul_scalar(self):
        npt.assert_equal(np.array(self.twodim * 2), [2.0, 4.0, 6])

    def test_random(self):
        self.twodim.random()
        npt.assert_allclose(
            np.array(self.twodim), [0.5488135, 0.71518937, 0.60276338]
        )

    def test_random_antisymmetric(self):
        As = Triangular((2, 2), anti=True)
        As.random()
        npt.assert_allclose(
            np.array(As), [0.0000000, 0.71518937, 0.00000000]
        )
