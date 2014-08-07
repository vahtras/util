import unittest
import numpy as np
from ..full import matrix, init

class TestMatrix(unittest.TestCase):

    def test_diag(self):
        ref = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        this = matrix.diag([1,1,1])
        np.testing.assert_equal(this, ref)

    def test_str(self):
        M = matrix((2,2))
        M[0,0] = M[1,1] = 1
        this = str(M)
        ref = """
 (2, 2) 
              Column   1    Column   2
       1      1.00000000    0.00000000
       2      0.00000000    1.00000000
"""
        self.assertEqual(this, ref)

    def test_str2(self):
        M = matrix((6,6))
        for i in range(6):
            M[i, i] = float(i)
        this = str(M)
        ref = """
 (6, 6) 
              Column   1    Column   2    Column   3    Column   4    Column   5
       2      0.00000000    1.00000000    0.00000000    0.00000000    0.00000000
       3      0.00000000    0.00000000    2.00000000    0.00000000    0.00000000
       4      0.00000000    0.00000000    0.00000000    3.00000000    0.00000000
       5      0.00000000    0.00000000    0.00000000    0.00000000    4.00000000

              Column   6
       6      5.00000000
"""
        self.assertEqual(this, ref)

    def test_str3(self):
        M = matrix((2,2,2))
        M[0,0,0] = M[1,1,1] = 1
        this = str(M)
        ref = """
 (2, 2, 2) 
[0]
 (2, 2) 
              Column   1    Column   2
       1      1.00000000    0.00000000
[1]
 (2, 2) 
              Column   1    Column   2
       2      0.00000000    1.00000000
"""
        self.assertEqual(this, ref)

    def test_mul(self):
        M = matrix((2, 2))
        M[0, 0] = M[1, 1] = 2.0
        M[0, 1] = M[1, 0] = 1.0
        np.testing.assert_equal(M*M, [[5, 4], [4, 5]])

    def test_mul_scalar_matrix(self):
        M = matrix((2, 2))
        M[0, 0] = M[1, 1] = 2.0
        M[0, 1] = M[1, 0] = 1.0
        np.testing.assert_equal(2*M, [[4, 2], [2, 4]])

    def test_mul_matrix_scalar(self):
        M = matrix((2, 2))
        M[0, 0] = M[1, 1] = 2.0
        M[0, 1] = M[1, 0] = 1.0
        np.testing.assert_equal(M*2, [[4, 2], [2, 4]])

    def test_outer(self):
        V1 = matrix((2, )); V1[0]=1.0
        V2 = matrix((2, )); V2[1]=1.0
        V12 = V1.x(V2)
        np.testing.assert_equal(V12, [[0, 1], [0, 0]])

    def test_div_scalar(self):
        M = matrix((2, 2))
        M[0, 0] = 1; M[0, 1] = 2; M[1, 0] = 3; M[1, 1] = 4
        np.testing.assert_equal(M/2, [[0.5, 1.0], [1.5, 2.0]])

    def test_div_self(self):
        M = matrix((2, 2))
        M[0, 0] = 1; M[0, 1] = 2; M[1, 0] = 3; M[1, 1] = 4
        np.testing.assert_equal(M/M, [[1.0, 0.0], [0.0, 1.0]])

    def test_neg(self):
        M = matrix((2, 2))
        M[0, 0] = 1; M[0, 1] = 2; M[1, 0] = 3; M[1, 1] = 4
        np.testing.assert_equal(-M, [[-1.0, -2.0], [-3.0, -4.0]])

    def test_and(self):
        M = matrix((2, 2))
        M[0, 0] = 1; M[0, 1] = 2; M[1, 0] = 3; M[1, 1] = 4
        np.testing.assert_equal(M&M, 30)

    def test_scatter(self):
        M = matrix((2, 2))
        M1 = matrix((3, 3))
        M[0, 0] = 1; M[0, 1] = 2; M[1, 0] = 3; M[1, 1] = 4
        M.scatter(M1, [0, 2], [0, 2]) 
        np.testing.assert_equal(M1, [[1, 0, 2], [0, 0, 0], [3, 0, 4]])

    def test_scatter_add(self):
        M = matrix((2, 2))
        M1 = matrix((3, 3)); M1[1, 1] = 5
        M[0, 0] = 1; M[0, 1] = 2; M[1, 0] = 3; M[1, 1] = 4
        M.scatteradd(M1, [0, 2], [0, 2]) 
        np.testing.assert_equal(M1, [[1, 0, 2], [0, 5, 0], [3, 0, 4]])

    def test_pack(self):
        M = matrix((2, 2))
        M1 = matrix((3, 3))
        M1[0, 0] = 1; M1[0, 2] = 2; M1[2, 0] = 3; M1[2, 2] = 4
        M1.packto(M, [0, 2], [0, 2]) 
        np.testing.assert_equal(M, [[1, 2], [3, 4]])

    def test_pack_rows(self):
        M = matrix((2, 3))
        M1 = matrix((3, 3))
        M1[0, 0] = 1; M1[0, 2] = 2; M1[2, 0] = 3; M1[2, 2] = 4
        M1.packto(M, rows=[0, 2])
        np.testing.assert_equal(M, [[1, 0, 2], [3, 0, 4]])

    def test_pack_columns(self):
        M = matrix((3, 2))
        M1 = matrix((3, 3))
        M1[0, 0] = 1; M1[0, 2] = 2; M1[2, 0] = 3; M1[2, 2] = 4
        M1.packto(M, columns=[0, 2])
        np.testing.assert_equal(M, [[1, 2], [0, 0], [3, 4]])

    def test_commute(self):
        z = init([[1, 0], [0, -1]])
        x = init([[0, 1], [1, 0]])
        np.testing.assert_equal(z^x, [[0, 2], [-2, 0]])

    def test_inv_init_none(self):
        M = init([[2, 1], [1, 2]])
        self.assertEqual(M._I, None)

    def test_inv(self):
        M = init([[3, 1], [1, -3]])
        M.inv()
        np.testing.assert_almost_equal(M._I, [[0.3, 0.1], [0.1, -0.3]])

    def test_rdiv(self):
        M = init([[3, 1], [1, -3]])
        np.testing.assert_almost_equal(1/M, [[0.3, 0.1], [0.1, -0.3]])

    def test_det(self):
        M = init([[3, 1], [1, -3]])
        self.assertAlmostEqual(M.det(), -10)

    def test_minor(self):
        M1 = matrix((3, 3))
        M1[0, 0] = 1; M1[0, 2] = 2; M1[2, 0] = 3; M1[2, 2] = 4
        np.testing.assert_equal(M1.minor(1, 1), [[1, 2], [3, 4]])

    def test_cofactor(self):
        M = init([[4, 1], [1, 3]])
        np.testing.assert_almost_equal(M.cofactor(), [[3., -1.], [-1., 4.]])
    
        
        


