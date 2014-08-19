import unittest
from ..unformatted import FortranBinary

"""Fortran examples
    1.
      integer, parameter :: n = 3
      double precision x(n)
      x = (/ 1.0D0, 2.0D0, 3.0D0 /)
      open(1, file='fort.1', status='new', form='unformatted')
      write(1) n
      write(1) x
      close(1)
      end
    2.
      character*5 lab
      integer n
      lab = 'LABEL'
      open(1, file='fort.2', status='new', form='unformatted')
      write(1) 0
      write(1) lab
      close(1)
      end
    3.
      integer, parameter :: n = 3
      double precision x(n), y(n)
      x = (/ 1.0D0, 2.0D0, 3.0D0 /)
      y = (/ 5.0D0, 6.0D0, 7.0D0 /)
      open(3, file='fort.3', status='new', form='unformatted')
      write(3) x
      write(3) y
      close(3)
      end
"""

class TestUnformatted(unittest.TestCase):

    def setUp(self):
        import os
        test_dir, _ = os.path.split(__file__)
        self.fort1 = os.path.join(test_dir, 'fort.1')
        self.fort2 = os.path.join(test_dir, 'fort.2')
        self.fort3 = os.path.join(test_dir, 'fort.3')
        pass

    def test_1_lengths(self):
        record_lengths = [rec.reclen for rec in FortranBinary(self.fort1)]
        self.assertListEqual(record_lengths, [4, 24])

    def test_1_int(self):
        fort1 = FortranBinary(self.fort1)
        first = fort1.next().read(1, 'i')
        self.assertTupleEqual(first, (3,))

    def test_1_floats(self):
        fort1 = FortranBinary(self.fort1)
        fort1.next()
        second = fort1.next().read(3, 'd')
        self.assertTupleEqual(second, (1., 2., 3.))

    def test_2_lengths(self):
        record_lengths = [rec.reclen for rec in FortranBinary(self.fort2)]
        self.assertListEqual(record_lengths, [4, 5])

    def test_2_int(self):
        fort2 = FortranBinary(self.fort2)
        first = fort2.next().read(1, 'i')
        self.assertTupleEqual(first, (0,))

    def test_2_char(self):
        fort2 = FortranBinary(self.fort2)
        fort2.next()
        second = "".join(fort2.next().read(5, 'c'))
        self.assertEqual(second, 'LABEL')

    def test_3_lengths(self):
        record_lengths = [rec.reclen for rec in FortranBinary(self.fort3)]
        self.assertListEqual(record_lengths, [24, 24])

    def test_3_floats(self):
        fort3 = FortranBinary(self.fort3)
        first = fort3.next().read(3, 'd')
        second = fort3.next().read(3, 'd')
        self.assertTupleEqual(second, (5., 6., 7.))

    def test_3_stops(self):
        def read_past_eof(filename):
            fort = FortranBinary(filename)
            fort.next() #x
            fort.next() #y
            fort.next() #eof
        self.assertRaises(StopIteration, read_past_eof, self.fort3)

if __name__ == "__main__":
    unittest.main()
