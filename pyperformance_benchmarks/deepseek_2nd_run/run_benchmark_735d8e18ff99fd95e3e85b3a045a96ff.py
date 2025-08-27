import hashlib
import os
import struct
import pyperf
from typing import Optional, List, Tuple, Dict, Any, Union, BinaryIO, cast

int2byte = struct.Struct('>B').pack

class BitfieldBase(object):

    def __init__(self, x: Union[BinaryIO, 'BitfieldBase']) -> None:
        if isinstance(x, BitfieldBase):
            self.f = x.f
            self.bits = x.bits
            self.bitfield = x.bitfield
            self.count = x.bitfield
        else:
            self.f = x
            self.bits = 0
            self.bitfield = 0
            self.count = 0

    def _read(self, n: int) -> bytes:
        s = self.f.read(n)
        if (not s):
            raise Exception('Length Error')
        self.count += len(s)
        return s

    def needbits(self, n: int) -> None:
        while (self.bits < n):
            self._more()

    def _mask(self, n: int) -> int:
        return ((1 << n) - 1)

    def toskip(self) -> int:
        return (self.bits & 7)

    def align(self) -> None:
        self.readbits(self.toskip())

    def dropbits(self, n: int = 8) -> None:
        while ((n >= self.bits) and (n > 7)):
            n -= self.bits
            self.bits = 0
            n -= (len(self.f._read((n >> 3))) << 3)
        if n:
            self.readbits(n)

    def dropbytes(self, n: int = 1) -> None:
        self.dropbits((n << 3))

    def tell(self) -> Tuple[int, int]:
        return ((self.count - ((self.bits + 7) >> 3)), (7 - ((self.bits - 1) & 7)))

    def tellbits(self) -> int:
        (bytes, bits) = self.tell()
        return ((bytes << 3) + bits)

class Bitfield(BitfieldBase):

    def _more(self) -> None:
        c = self._read(1)
        self.bitfield += (ord(c) << self.bits)
        self.bits += 8

    def snoopbits(self, n: int = 8) -> int:
        if (n > self.bits):
            self.needbits(n)
        return (self.bitfield & self._mask(n))

    def readbits(self, n: int = 8) -> int:
        if (n > self.bits):
            self.needbits(n)
        r = (self.bitfield & self._mask(n))
        self.bits -= n
        self.bitfield >>= n
        return r

class RBitfield(BitfieldBase):

    def _more(self) -> None:
        c = self._read(1)
        self.bitfield <<= 8
        self.bitfield += ord(c)
        self.bits += 8

    def snoopbits(self, n: int = 8) -> int:
        if (n > self.bits):
            self.needbits(n)
        return ((self.bitfield >> (self.bits - n)) & self._mask(n))

    def readbits(self, n: int = 8) -> int:
        if (n > self.bits):
            self.needbits(n)
        r = ((self.bitfield >> (self.bits - n)) & self._mask(n))
        self.bits -= n
        self.bitfield &= (~ (self._mask(n) << self.bits))
        return r

def printbits(v: int, n: int) -> str:
    o = ''
    for i in range(n):
        if (v & 1):
            o = ('1' + o)
        else:
            o = ('0' + o)
        v >>= 1
    return o

class HuffmanLength(object):

    def __init__(self, code: int, bits: int = 0) -> None:
        self.code = code
        self.bits = bits
        self.symbol: Optional[int] = None
        self.reverse_symbol: Optional[int] = None

    def __repr__(self) -> str:
        return repr((self.code, self.bits, self.symbol, self.reverse_symbol))

    @staticmethod
    def _sort_func(obj: 'HuffmanLength') -> Tuple[int, int]:
        return (obj.bits, obj.code)

def reverse_bits(v: int, n: int) -> int:
    a = (1 << 0)
    b = (1 << (n - 1))
    z = 0
    for i in range((n - 1), (- 1), (- 2)):
        z |= ((v >> i) & a)
        z |= ((v << i) & b)
        a <<= 1
        b >>= 1
    return z

def reverse_bytes(v: int, n: int) -> int:
    a = (255 << 0)
    b = (255 << (n - 8))
    z = 0
    for i in range((n - 8), (- 8), (- 16)):
        z |= ((v >> i) & a)
        z |= ((v << i) & b)
        a <<= 8
        b >>= 8
    return z

class HuffmanTable(object):

    def __init__(self, bootstrap: List[Tuple[int, int]]) -> None:
        l: List[HuffmanLength] = []
        (start, bits) = bootstrap[0]
        for (finish, endbits) in bootstrap[1:]:
            if bits:
                for code in range(start, finish):
                    l.append(HuffmanLength(code, bits))
            (start, bits) = (finish, endbits)
            if (endbits == (- 1)):
                break
        l.sort(key=HuffmanLength._sort_func)
        self.table = l

    def populate_huffman_symbols(self) -> None:
        (bits, symbol) = ((- 1), (- 1))
        for x in self.table:
            symbol += 1
            if (x.bits != bits):
                symbol <<= (x.bits - bits)
                bits = x.bits
            x.symbol = symbol
            x.reverse_symbol = reverse_bits(symbol, bits)

    def tables_by_bits(self) -> Dict[int, List[HuffmanLength]]:
        d: Dict[int, List[HuffmanLength]] = {}
        for x in self.table:
            try:
                d[x.bits].append(x)
            except:
                d[x.bits] = [x]
        return d

    def min_max_bits(self) -> None:
        (self.min_bits, self.max_bits) = (16, (- 1))
        for x in self.table:
            if (x.bits < self.min_bits):
                self.min_bits = x.bits
            if (x.bits > self.max_bits):
                self.max_bits = x.bits

    def _find_symbol(self, bits: int, symbol: int, table: List[HuffmanLength]) -> int:
        for h in table:
            if ((h.bits == bits) and (h.reverse_symbol == symbol)):
                return h.code
        return (- 1)

    def find_next_symbol(self, field: BitfieldBase, reversed: bool = True) -> int:
        cached_length = (- 1)
        cached = 0
        for x in self.table:
            if (cached_length != x.bits):
                cached = field.snoopbits(x.bits)
                cached_length = x.bits
            if ((reversed and (x.reverse_symbol == cached)) or ((not reversed) and (x.symbol == cached))):
                field.readbits(x.bits)
                return x.code
        raise Exception(('unfound symbol, even after end of table @%r' % field.tell()))
        for bits in range(self.min_bits, (self.max_bits + 1)):
            r = self._find_symbol(bits, field.snoopbits(bits), self.table)
            if (0 <= r):
                field.readbits(bits)
                return r
            elif (bits == self.max_bits):
                raise Exception('unfound symbol, even after max_bits')
        return -1

class OrderedHuffmanTable(HuffmanTable):

    def __init__(self, lengths: List[int]) -> None:
        l = len(lengths)
        z = (list(zip(range(l), lengths)) + [(l, (- 1))])
        HuffmanTable.__init__(self, z)

def code_length_orders(i: int) -> int:
    return (16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15)[i]

def distance_base(i: int) -> int:
    return (1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577)[i]

def length_base(i: int) -> int:
    return (3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258)[(i - 257)]

def extra_distance_bits(n: int) -> int:
    if (0 <= n <= 1):
        return 0
    elif (2 <= n <= 29):
        return ((n >> 1) - 1)
    else:
        raise Exception('illegal distance code')

def extra_length_bits(n: int) -> int:
    if ((257 <= n <= 260) or (n == 285)):
        return 0
    elif (261 <= n <= 284):
        return (((n - 257) >> 2) - 1)
    else:
        raise Exception('illegal length code')

def move_to_front(l: List[Any], c: int) -> None:
    l[:] = ((l[c:(c + 1)] + l[0:c]) + l[(c + 1):])

def bwt_transform(L: bytes) -> List[int]:
    F = bytes(sorted(L))
    base: List[int] = []
    for i in range(256):
        base.append(F.find(int2byte(i)))
    pointers = ([(- 1)] * len(L))
    for (i, symbol) in enumerate(L):
        pointers[base[symbol]] = i
        base[symbol] += 1
    return pointers

def bwt_reverse(L: bytes, end: int) -> bytes:
    out: List[int] = []
    if len(L):
        T = bwt_transform(L)
        for i in range(len(L)):
            end = T[end]
            out.append(L[end])
    return bytes(out)

def compute_used(b: RBitfield) -> List[bool]:
    huffman_used_map = b.readbits(16)
    map_mask = (1 << 15)
    used: List[bool] = []
    while (map_mask > 0):
        if (huffman_used_map & map_mask):
            huffman_used_bitmap = b.readbits(16)
            bit_mask = (1 << 15)
            while (bit_mask > 0):
                if (huffman_used_bitmap & bit_mask):
                    pass
                used += [bool((huffman_used_bitmap & bit_mask))]
                bit_mask >>= 1
        else:
            used += ([False] * 16)
        map_mask >>= 1
    return used

def compute_selectors_list(b: RBitfield, huffman_groups: int) -> List[int]:
    selectors_used = b.readbits(15)
    mtf = list(range(huffman_groups))
    selectors_list: List[int] = []
    for i in range(selectors_used):
        c = 0
        while b.readbits(1):
            c += 1
            if (c >= huffman_groups):
                raise Exception('Bzip2 chosen selector greater than number of groups (max 6)')
        if (c >= 0):
            move_to_front(mtf, c)
        selectors_list.append(mtf[0])
    return selectors_list

def compute_tables(b: RBitfield, huffman_groups: int, symbols_in_use: int) -> List[OrderedHuffmanTable]:
    groups_lengths: List[List[int]] = []
    for j in range(huffman_groups):
        length = b.readbits(5)
        lengths: List[int] = []
        for i in range(symbols_in_use):
            if (not (0 <= length <= 20)):
                raise Exception('Bzip2 Huffman length code outside range 0..20')
            while b.readbits(1):
                length -= ((b.readbits(1) * 2) - 1)
            lengths += [length]
        groups_lengths += [lengths]
    tables: List[OrderedHuffmanTable] = []
    for g in groups_lengths:
        codes = OrderedHuffmanTable(g)
        codes.populate_huffman_symbols()
        codes.min_max_bits()
        tables.append(codes)
    return tables

def decode_huffman_block(b: RBitfield, out: List[bytes]) -> None:
    randomised = b.readbits(1)
    if randomised:
        raise Exception('Bzip2 randomised support not implemented')
    pointer = b.readbits(24)
    used = compute_used(b)
    huffman_groups = b.readbits(3)
    if (not (2 <= huffman_groups <= 6)):
        raise Exception('Bzip2: Number of Huffman groups not in range 2..6')
    selectors_list = compute_selectors_list(b, huffman_groups)
    symbols_in_use = (sum(used) + 2)
    tables = compute_tables(b, huffman_groups, symbols_in_use)
    favourites = [int2byte(i) for (i, x) in enumerate(used) if x]
    selector_pointer = 0
    decoded = 0
    repeat = 0
    repeat_power = 0
    buffer: List[bytes] = []
    t: Optional[OrderedHuffmanTable] = None
    while True:
        decoded -= 1
        if (decoded <= 0):
            decoded = 50
            if (selector_pointer <= len(selectors_list)):
                t = tables[selectors_list[selector_pointer]]
                selector_pointer += 1
        r = t.find_next_symbol(b, False) if t else -1
        if (0 <= r <= 1):
            if (repeat == 0):
                repeat_power = 1
            repeat += (repeat_power << r)
            repeat_power <<= 1
            continue
        elif (repeat > 0):
            buffer.append((favourites[0] * repeat))
            repeat = 0
        if (r == (symbols_in_use - 1)):
            break
        else:
            o = favourites[(r - 1)]
            move_to_front(favourites, (r - 1))
            buffer.append(o)
            pass
    nt = nearly_there = bwt_reverse(b''.join(buffer), pointer)
    i = 0
    while (i < len(nearly_there)):
        if ((i < (len(nearly_there) - 4)) and (nt[i] == nt[(i + 1)] == nt[(i + 2)] == nt[(i + 3)])):
            out.append((nearly_there[i:(i + 1)] * (ord(nearly_there[(i + 4):(i + 5)]) + 4)))
            i += 5
        else:
            out.append(nearly_there[i:(i + 1)])
            i += 1

def bzip2_main(input: BinaryIO) -> bytes:
    b = RBitfield(input)
    method = b.readbits(8)
    if (method != ord('h')):
        raise Exception("Unknown (not type 'h'uffman Bzip2) compression method")
    blocksize = b.readbits(8)
    if (ord('1') <= blocksize <= ord('9')):
        blocksize = (blocksize - ord('0'))
    else:
        raise Exception("Unknown (not size '0'-'9') Bzip2 blocksize")
    out: List[bytes] = []
    while True:
        blocktype = b.readbits(48)
        b.readbits(32)
        if (blocktype == 54156738319193):
            decode_huffman_block(b, out)
        elif (blocktype == 25779555029136):
            b.align()
            break
        else:
            raise Exception('Illegal Bzip2 blocktype')
    return b''.join(out)

def gzip_main(field: Bitfield) -> str:
    b = Bitfield(field)
    method = b.readbits(8)
    if (method != 8):
        raise Exception('Unknown (not type eight DEFLATE) compression method')
    flags = b.readbits(8)
    b.readbits(32)
    b.readbits(8)
    b.readbits(8)
    if (flags & 4):
        xlen = b.readbits(16)
        b.dropbytes(xlen)
    while (flags & 8):
        if (not b.readbits(8)):
            break
    while (flags & 16):
        if (not b.readbits(8)):
            break
    if (flags & 2):
        b.readbits(16)
    out: List[str] = []
    while True:
        lastbit = b.readbits(1)
        blocktype = b.readbits(2)
        if (blocktype == 0):
            b.align()
            length = b.readbits(16)
            if (length