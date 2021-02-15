import numpy


class BlockDiagonalMatrix:
    """
    Blocked matrix class based on lists of full matrices
    """

    def __init__(self, nrow, ncol):
        """ Constructur of the class."""

        assert len(nrow) == len(ncol)
        from . import full

        self.nblocks = len(nrow)
        self.nrow = nrow
        self.ncol = ncol
        self.subblock = []
        self.irow = []
        self.icol = []
        for i in range(self.nblocks):
            self.subblock.append(full.matrix((nrow[i], ncol[i])))
            self.irow.append(sum(self.nrow[:i]))
            self.icol.append(sum(self.ncol[:i]))
        self.irow = tuple(self.irow)
        self.icol = tuple(self.icol)

    def __str__(self):
        """ Formatted output based on full matrix class """

        retstr = ""
        for i in range(self.nblocks):
            if self.nrow[i] * self.ncol[i]:
                retstr += "\nBlock %d\n" % (i + 1) + str(self.subblock[i])
        return retstr

    def __repr__(self):
        return f'BlockDiagonalMatrix({self.nrow}, {self.ncol})'

    def __getitem__(self, n):
        """ Index argument returns subblock
        Example
        >>> M = BlockDiagonalMatrix([2], [2])
        >>> print M[0]
        <BLANKLINE>
         (2, 2)
                      Column   1    Column   2
        <BLANKLINE>
        """
        return self.subblock[n]

    @staticmethod
    def init(*arr):
        from .full import init

        matrices = tuple(init(a) for a in arr)
        for m in matrices:
            assert len(m.shape) == 2, "blocked only for two dimensions"
        rdim = tuple(m.shape[0] for m in matrices)
        cdim = tuple(m.shape[1] for m in matrices)
        new = BlockDiagonalMatrix(rdim, cdim)
        for i, a in enumerate(arr):
            new.subblock[i][:, :] = matrices[i]
        return new

    @staticmethod
    def init_from_array(arr, rdim, cdim):
        from . import full

        assert numpy.dot(rdim, cdim) == len(arr)
        new = BlockDiagonalMatrix(rdim, cdim)
        start = 0
        end = 0
        for i, block in enumerate(new.subblock):
            end += rdim[i] * cdim[i]
            block[:, :] = full.init(
                arr[start:end]
            ).reshape(block.shape, order="F")
            start += rdim[i] * cdim[i]
        return new

    @property
    def size(self):
        return sum(b.size for b in self.subblock)

    def ravel(self, **kwargs):
        from . import full

        linear = full.matrix(self.size)
        start = 0
        end = 0
        for s in self.subblock:
            end += s.size
            linear[start:end] = s.ravel(**kwargs)
            start += s.size
        return linear

    def __matmul__(self, other):
        """Multiplication blockwise """

        new = BlockDiagonalMatrix(self.nrow, other.ncol)
        for i in range(self.nblocks):
            if self.nrow[i]:
                new.subblock[i] = self.subblock[i] @ other.subblock[i]
        return new

    def __rmul__(self, other):
        """Scalar multiplication"""

        new = BlockDiagonalMatrix(self.nrow, self.ncol)
        for i in range(self.nblocks):
            if self.nrow[i]:
                new.subblock[i] = other * self.subblock[i]
        return new

    def __add__(self, other):
        """Addition blockwise"""
        bdm = BlockDiagonalMatrix(self.nrow, self.ncol)
        bdm.subblock = [s + o for s, o in zip(self.subblock, other.subblock)]
        return bdm

    def __sub__(self, other):
        """Subtraction blockwize"""
        bdm = BlockDiagonalMatrix(self.nrow, self.ncol)
        bdm.subblock = [s - o for s, o in zip(self.subblock, other.subblock)]
        return bdm

    def __neg__(self):
        """Negation blockwise"""

        bdm = BlockDiagonalMatrix(self.nrow, self.ncol)
        bdm.subblock = [-b for b in self.subblock]
        return bdm

    def __truediv__(self, other):
        "Scalar division"

        bdm = BlockDiagonalMatrix(self.nrow, self.ncol)
        bdm.subblock = [b * (1/other) for b in self.subblock]
        return bdm

    def __rtruediv__(self, other):
        "Reverse Scalar division"

        bdm = BlockDiagonalMatrix(self.nrow, self.ncol)
        bdm.subblock = [other / b for b in self.subblock]
        return bdm

    def solve(self, other):
        "Solve linear equations blockwise" ""

        bdm = BlockDiagonalMatrix(self.nrow, self.ncol)
        bdm.subblock = [s.solve(o) for s, o in zip(self.subblock, other.subblock)]
        return bdm

    def pack(self):
        for i in range(self.nblocks):
            assert self.nrow[i] == self.ncol[i]
        new = BlockedTriangular(self.nrow)
        for i in range(self.nblocks):
            new.subblock[i] = self.subblock[i].pack()
        return new

    def unblock(self):
        from . import full

        nrows = sum(self.nrow)
        ncols = sum(self.ncol)
        new = full.matrix((nrows, ncols))
        for i in range(self.nblocks):
            new[
                self.irow[i]: self.irow[i] + self.nrow[i],
                self.icol[i]: self.icol[i] + self.ncol[i],
            ] = self.subblock[i]
        return new

    def T(self):
        """
        Transpose
        Example:
        >>> M = BlockDiagonalMatrix([2],[2])
        >>> M.subblock[0][0, 1] = 1
        >>> M.subblock[0][1, 0] = 2
        >>> print M, M.T()
        <BLANKLINE>
        Block 1
        <BLANKLINE>
         (2, 2)
                      Column   1    Column   2
               1      0.00000000    1.00000000
               2      2.00000000    0.00000000
        <BLANKLINE>
        Block 1
        <BLANKLINE>
         (2, 2)
                      Column   1    Column   2
               1      0.00000000    2.00000000
               2      1.00000000    0.00000000
        <BLANKLINE>
        """
        new = BlockDiagonalMatrix(self.ncol, self.nrow)
        for i in range(self.nblocks):
            new.subblock[i] = self.subblock[i].T
        return new

    def sqrt(self):
        new = BlockDiagonalMatrix(self.ncol, self.nrow)
        for i in range(self.nblocks):
            new.subblock[i] = self.subblock[i].sqrt()
        return new

    def invsqrt(self):
        new = BlockDiagonalMatrix(self.ncol, self.nrow)
        for i in range(self.nblocks):
            new.subblock[i] = self.subblock[i].invsqrt()
        return new

    def func(self, f):
        """ Blockwise function of matrix"""
        new = BlockDiagonalMatrix(self.ncol, self.nrow)
        for i in range(self.nblocks):
            new.subblock[i] = self.subblock[i].func(f)
        return new

    def tr(self):
        """Sum blockwise traces
        """

        sum = 0
        for i in range(self.nblocks):
            if self.nrow[i]:
                sum += self.subblock[i].tr()
        return sum

    def eigvec(self):
        u = BlockDiagonalMatrix(self.nrow, self.nblocks * [1])
        v = BlockDiagonalMatrix(self.nrow, self.ncol)
        for i in range(self.nblocks):
            u.subblock[i], v.subblock[i] = self.subblock[i].eigvec()
        return u, v

    def qr(self):
        q = BlockDiagonalMatrix(self.nrow, self.nrow)
        r = BlockDiagonalMatrix(self.nrow, self.ncol)
        for i in range(self.nblocks):
            q.subblock[i], r.subblock[i] = self.subblock[i].qr()
        return q, r

    def GS(self, S):
        new = BlockDiagonalMatrix(self.nrow, self.ncol)
        for i in range(self.nblocks):
            new.subblock[i] = self.subblock[i].GS(S.subblock[i])
        return new

    def get_columns(self, columns_per_symmetry):
        new = BlockDiagonalMatrix(self.nrow, columns_per_symmetry)
        for oldblock, newblock, cols in zip(self, new, columns_per_symmetry):
            if cols:
                newblock[:, :] = oldblock[:, :cols]
        return new


def unit(nbl, factor=1):
    from .full import unit

    new = BlockDiagonalMatrix(nbl, nbl)
    for i in range(len(nbl)):
        if nbl[i]:
            new.subblock[i] = unit(nbl[i], factor)
    return new


class BlockedTriangular(object):
    def __init__(self, dim):
        from . import full

        self.nblocks = len(dim)
        self.dim = dim
        self.subblock = []
        for i in range(self.nblocks):
            self.subblock.append(full.Triangular((dim[i], dim[i])))

    def __str__(self):
        retstr = ""
        for i in range(self.nblocks):
            if self.dim[i]:
                retstr += "\nBlock %d\n" % (i + 1) + str(self.subblock[i])
        return retstr

    def __add__(self, other):
        bt = BlockedTriangular(self.dim)
        bt.subblock = [s + o for s, o in zip(self.subblock, other.subblock)]
        return bt

    def __sub__(self, other):
        new = BlockedTriangular(self.dim)
        for i in range(self.nblocks):
            new.subblock[i] = self.subblock[i] - other.subblock[i]
        return new

    def unpack(self):
        new = BlockDiagonalMatrix(self.dim, self.dim)
        for i in range(self.nblocks):
            new.subblock[i] = self.subblock[i].unpack()
        return new

    def unblock(self):
        return self.unpack().unblock().pack()

    @staticmethod
    def init(blocks):
        from . import full

        BlockedTriangular_matrices = [full.Triangular.init(block) for block in blocks]
        dimensions = [tmat.dim for tmat in BlockedTriangular_matrices]
        new = BlockedTriangular(dimensions)
        new.subblock = BlockedTriangular_matrices
        return new


# Keep non-standard names for back compatibility
triangular = BlockedTriangular
