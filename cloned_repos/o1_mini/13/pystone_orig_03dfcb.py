"""
"PYSTONE" Benchmark Program

Version:        Python/1.1 (corresponds to C/1.1 plus 2 Pystone fixes)

Author:         Reinhold P. Weicker,  CACM Vol 27, No 10, 10/84 pg. 1013.

                Translated from ADA to C by Rick Richardson.
                Every method to preserve ADA-likeness has been used,
                at the expense of C-ness.

                Translated from C to Python by Guido van Rossum.

Version History:

                Version 1.1 corrects two bugs in version 1.0:

                First, it leaked memory: in Proc1(), NextRecord ends
                up having a pointer to itself.  I have corrected this
                by zapping NextRecord.PtrComp at the end of Proc1().

                Second, Proc3() used the operator != to compare a
                record to None.  This is rather inefficient and not
                true to the intention of the original benchmark (where
                a pointer comparison to None is intended; the !=
                operator attempts to find a method __cmp__ to do value
                comparison of the record).  Version 1.1 runs 5-10
                percent faster than version 1.0, so benchmark figures
                of different versions can't be compared directly.
    
"""
from __future__ import annotations
from typing import Optional, Tuple, List
from time import clock
import sys

LOOPS: int = 50000
__version__: str = '1.1'
Ident1, Ident2, Ident3, Ident4, Ident5 = range(1, 6)

TRUE: int = 1
FALSE: int = 0

class Record:
    def __init__(
        self, 
        PtrComp: Optional[Record] = None, 
        Discr: int = 0, 
        EnumComp: int = 0, 
        IntComp: int = 0, 
        StringComp: int = 0
    ) -> None:
        self.PtrComp: Optional[Record] = PtrComp
        self.Discr: int = Discr
        self.EnumComp: int = EnumComp
        self.IntComp: int = IntComp
        self.StringComp: int = StringComp

    def copy(self) -> Record:
        return Record(
            self.PtrComp, 
            self.Discr, 
            self.EnumComp, 
            self.IntComp, 
            self.StringComp
        )

def main(loops: int = LOOPS) -> None:
    benchtime, stones = pystones(loops)
    print('Pystone(%s) time for %d passes = %g' % (__version__, loops, benchtime))
    print('This machine benchmarks at %g pystones/second' % stones)

def pystones(loops: int = LOOPS) -> Tuple[float, float]:
    return Proc0(loops)

IntGlob: int = 0
BoolGlob: int = FALSE
Char1Glob: str = '\x00'
Char2Glob: str = '\x00'
Array1Glob: List[int] = [0] * 51
Array2Glob: List[List[int]] = [x[:] for x in [Array1Glob] * 51]
PtrGlb: Optional[Record] = None
PtrGlbNext: Optional[Record] = None

def Proc0(loops: int = LOOPS) -> Tuple[float, float]:
    global IntGlob
    global BoolGlob
    global Char1Glob
    global Char2Glob
    global Array1Glob
    global Array2Glob
    global PtrGlb
    global PtrGlbNext
    starttime: float = clock()
    for _ in range(loops):
        pass
    nulltime: float = clock() - starttime
    PtrGlbNext = Record()
    PtrGlb = Record()
    PtrGlb.PtrComp = PtrGlbNext
    PtrGlb.Discr = Ident1
    PtrGlb.EnumComp = Ident3
    PtrGlb.IntComp = 40
    PtrGlb.StringComp = 'DHRYSTONE PROGRAM, SOME STRING'
    String1Loc: str = "DHRYSTONE PROGRAM, 1'ST STRING"
    Array2Glob[8][7] = 10
    starttime = clock()
    for _ in range(loops):
        Proc5()
        Proc4()
        IntLoc1: int = 2
        IntLoc2: int = 3
        String2Loc: str = "DHRYSTONE PROGRAM, 2'ND STRING"
        EnumLoc: int = Ident2
        BoolGlob = not Func2(String1Loc, String2Loc)
        while IntLoc1 < IntLoc2:
            IntLoc3: int = 5 * IntLoc1 - IntLoc2
            IntLoc3 = Proc7(IntLoc1, IntLoc2)
            IntLoc1 = IntLoc1 + 1
        Proc8(Array1Glob, Array2Glob, IntLoc1, IntLoc3)
        PtrGlb = Proc1(PtrGlb)
        CharIndex: str = 'A'
        while CharIndex <= Char2Glob:
            if EnumLoc == Func1(CharIndex, 'C'):
                EnumLoc = Proc6(Ident1)
            CharIndex = chr(ord(CharIndex) + 1)
        IntLoc3 = IntLoc2 * IntLoc1
        IntLoc2 = IntLoc3 / IntLoc1
        IntLoc2 = 7 * (IntLoc3 - IntLoc2) - IntLoc1
        IntLoc1 = Proc2(IntLoc1)
    benchtime: float = clock() - starttime - nulltime
    if benchtime == 0.0:
        loopsPerBenchtime: float = 0.0
    else:
        loopsPerBenchtime = loops / benchtime
    return (benchtime, loopsPerBenchtime)

def Proc1(PtrParIn: Optional[Record]) -> Optional[Record]:
    global PtrGlb
    PtrParIn.PtrComp = NextRecord = PtrGlb.copy()
    PtrParIn.IntComp = 5
    NextRecord.IntComp = PtrParIn.IntComp
    NextRecord.PtrComp = PtrParIn.PtrComp
    NextRecord.PtrComp = Proc3(NextRecord.PtrComp)
    if NextRecord.Discr == Ident1:
        NextRecord.IntComp = 6
        NextRecord.EnumComp = Proc6(PtrParIn.EnumComp)
        NextRecord.PtrComp = PtrGlb.PtrComp
        NextRecord.IntComp = Proc7(NextRecord.IntComp, 10)
    else:
        PtrParIn = NextRecord.copy()
    NextRecord.PtrComp = None
    return PtrParIn

def Proc2(IntParIO: int) -> int:
    IntLoc: int = IntParIO + 10
    while True:
        if Char1Glob == 'A':
            IntLoc = IntLoc - 1
            IntParIO = IntLoc - IntGlob
            EnumLoc: int = Ident1
        if 'EnumLoc' in locals() and EnumLoc == Ident1:
            break
    return IntParIO

def Proc3(PtrParOut: Optional[Record]) -> Optional[Record]:
    global IntGlob
    if PtrGlb is not None:
        PtrParOut = PtrGlb.PtrComp
    else:
        IntGlob = 100
    if PtrGlb is not None:
        PtrGlb.IntComp = Proc7(10, IntGlob)
    return PtrParOut

def Proc4() -> None:
    global Char2Glob
    BoolLoc: bool = Char1Glob == 'A'
    BoolLoc = BoolLoc or bool(BoolGlob)
    Char2Glob = 'B'

def Proc5() -> None:
    global Char1Glob
    global BoolGlob
    Char1Glob = 'A'
    BoolGlob = FALSE

def Proc6(EnumParIn: int) -> int:
    EnumParOut: int = EnumParIn
    if not Func3(EnumParIn):
        EnumParOut = Ident4
    if EnumParIn == Ident1:
        EnumParOut = Ident1
    elif EnumParIn == Ident2:
        if IntGlob > 100:
            EnumParOut = Ident1
        else:
            EnumParOut = Ident4
    elif EnumParIn == Ident3:
        EnumParOut = Ident2
    elif EnumParIn == Ident4:
        pass
    elif EnumParIn == Ident5:
        EnumParOut = Ident3
    return EnumParOut

def Proc7(IntParI1: int, IntParI2: int) -> int:
    IntLoc: int = IntParI1 + 2
    IntParOut: int = IntParI2 + IntLoc
    return IntParOut

def Proc8(
    Array1Par: List[int], 
    Array2Par: List[List[int]], 
    IntParI1: int, 
    IntParI2: int
) -> None:
    global IntGlob
    IntLoc: int = IntParI1 + 5
    Array1Par[IntLoc] = IntParI2
    Array1Par[IntLoc + 1] = Array1Par[IntLoc]
    Array1Par[IntLoc + 30] = IntLoc
    for IntIndex in range(IntLoc, IntLoc + 2):
        Array2Par[IntLoc][IntIndex] = IntLoc
    Array2Par[IntLoc][IntLoc - 1] = Array2Par[IntLoc][IntLoc - 1] + 1
    Array2Par[IntLoc + 20][IntLoc] = Array1Par[IntLoc]
    IntGlob = 5

def Func1(CharPar1: str, CharPar2: str) -> int:
    CharLoc1: str = CharPar1
    CharLoc2: str = CharLoc1
    if CharLoc2 != CharPar2:
        return Ident1
    else:
        return Ident2

def Func2(StrParI1: str, StrParI2: str) -> int:
    IntLoc: int = 1
    while IntLoc <= 1:
        if Func1(StrParI1[IntLoc], StrParI2[IntLoc + 1]) == Ident1:
            CharLoc: str = 'A'
            IntLoc = IntLoc + 1
    if 'W' <= CharLoc <= 'Z':
        IntLoc = 7
    if CharLoc == 'X':
        return TRUE
    elif StrParI1 > StrParI2:
        IntLoc = IntLoc + 7
        return TRUE
    else:
        return FALSE

def Func3(EnumParIn: int) -> int:
    EnumLoc: int = EnumParIn
    if EnumLoc == Ident3:
        return TRUE
    return FALSE

if __name__ == '__main__':
    def error(msg: str) -> None:
        print(msg, end=' ', file=sys.stderr)
        print('usage: %s [number_of_loops]' % sys.argv[0], file=sys.stderr)
        sys.exit(100)
    nargs: int = len(sys.argv) - 1
    if nargs > 1:
        error('%d arguments are too many;' % nargs)
    elif nargs == 1:
        try:
            loops: int = int(sys.argv[1])
        except ValueError:
            error('Invalid argument %r;' % sys.argv[1])
    else:
        loops = LOOPS
    main(loops)
