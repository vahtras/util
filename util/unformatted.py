"""
    This module defines a class FortranBinary for reading binary 
    files generated by FORTRAN unformatted I/O
"""
import struct

class FortranBinary(object):
    """Class for binary files compatible with Fortran Unformatted I/O"""

    pad = 4

    def __init__(self, name, mode="rb"):
        self.name = name
        self.file = open(name, mode)
        self.data = None
        self.rec = None
        self.loc = 0
        self.reclen = 0

    def __iter__(self):
        return self

    def __next__(self): #pragma: no cover
        return self.next()

    def next(self):
        """Read a Fortran record"""
        head = self.file.read(self.pad)
        if head:
            size = struct.unpack('i', head)[0]
            self.data = self.file.read(size)
            self.reclen = size
            tail = self.file.read(self.pad)
            assert head == tail
            self.loc = 0
            self.rec = Rec(self.data)
            return self.rec
        else:
            raise StopIteration

    def readbuf(self, num, fmt):
        """Read data from current record"""
        vec = self.rec.read(num, fmt)
        return vec

    def find(self, label):
        """Find string label in file"""
        if isinstance(label, str):
            try:
                blabel = bytes(label, 'utf-8')
            except TypeError:
                blabel = label
        elif isinstance(label, bytes):
            blabel = label
        else:
            raise ValueError

        for rec in self:
            if blabel in rec:
                return rec

    def close(self):
        """Close file"""
        self.file.close()

    def record_byte_lengths(self):
        """Return record byte lengths in file as tuple"""

        reclengths = [record.reclen for record in self]
        return tuple(reclengths)

class Rec(object):
    """Representation of a single Fortran record"""

    def __init__(self, data):
        self.data = data
        self.loc = 0
        self.reclen = len(data)

    def __contains__(self, obj):
        return obj in self.data

    def read(self, num, fmt):
        """Read data from current record"""
        start, stop = self.loc, self.loc+struct.calcsize(fmt*num)
        vec = struct.unpack(fmt*num, self.data[start:stop])
        self.loc = stop
        return vec

if __name__ == "__main__": #pragma: no cover
    pass
