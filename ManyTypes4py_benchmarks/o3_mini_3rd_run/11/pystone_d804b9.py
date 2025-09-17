#!/usr/bin/env python3
"""
"PYSTONE" Benchmark Program

Version:        Python/1.1 (corresponds to C/1.1 plus 2 Pystone fixes)

Author:         Reinhold P. Weicker,  CACM Vol 27, No 10, 10/84 pg. 1013.

                Translated from ADA to C by Rick Richardson.
                Every method to preserve ADA-likeness has been used,
                at the expense of C-ness.

                Translated from C to Python by Guido van Rossum.
                
                Adapted to both run on Transcrypt and Python by Jacques de Hooge.

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
                
                For Transcrypt, some not yet supported contructs were avoided
"""

from typing import Optional, List, Tuple
import time

LOOPS: int = 50000

def clock() -> float:
    return time.time()

__version__ = '1.1'
[Ident1, Ident2, Ident3, Ident4, Ident5] = range(1, 6)

class Record:
    def __init__(self, 
                 PtrComp: Optional["Record"] = None, 
                 Discr: int = 0, 
                 EnumComp: int = 0, 
                 IntComp: int = 0, 
                 StringComp: str = "") -> None:
        self.PtrComp: Optional["Record"] = PtrComp
        self.Discr: int = Discr
        self.EnumComp: int = EnumComp
        self.IntComp: int = IntComp
        self.StringComp: str = StringComp

    def copy(self) -> "Record":
        return Record(self.PtrComp, self.Discr, self.EnumComp, self.IntComp, self.StringComp)

TRUE: int = 1
FALSE: int = 0

# Global Variables
IntGlob: int = 0
BoolGlob: int = FALSE
Char1Glob: str = '\x00'
Char2Glob: str = '\x00'
Array1Glob: List[int] = [0 for i in range(51)]
Array2Glob: List[List[int]] = [[0 for i in range(51)] for j in range(51)]
PtrGlb: Optional[Record] = None
PtrGlbNext: Optional[Record] = None

def main(loops: int = LOOPS) -> None:
    benchtime, stones = pystones(loops)
    print(f'Pystone({__version__}) time for {loops} passes = {benchtime}')
    print(f'This machine benchmarks at {stones} pystones/second')

def pystones(loops: int = LOOPS) -> Tuple[float, float]:
    return Proc0(loops)

def Proc0(loops: int = LOOPS) -> Tuple[float, float]:
    global IntGlob, BoolGlob, Char1Glob, Char2Glob, Array1Glob, Array2Glob, PtrGlb, PtrGlbNext
    starttime: float = clock()
    for i in range(loops):
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
    for i in range(loops):
        Proc5()
        Proc4()
        IntLoc1: int = 2
        IntLoc2: int = 3
        String2Loc: str = "DHRYSTONE PROGRAM, 2'ND STRING"
        EnumLoc: int = Ident2
        global BoolGlob
        BoolGlob = not Func2(String1Loc, String2Loc)
        IntLoc3: int = 0
        while IntLoc1 < IntLoc2:
            IntLoc3 = 5 * IntLoc1 - IntLoc2
            IntLoc3 = Proc7(IntLoc1, IntLoc2)
            IntLoc1 = IntLoc1 + 1
        Proc8(Array1Glob, Array2Glob, IntLoc1, IntLoc3)
        # It is assumed PtrGlb is not None here.
        PtrGlb = Proc1(PtrGlb)  # type: ignore
        CharIndex: str = 'A'
        while CharIndex <= Char2Glob:
            if EnumLoc == Func1(CharIndex, 'C'):
                EnumLoc = Proc6(Ident1)
            CharIndex = chr(ord(CharIndex) + 1)
        IntLoc3 = IntLoc2 * IntLoc1
        # In Python 3, / produces float but in original code it's probably integer division.
        IntLoc2 = int(IntLoc3 / IntLoc1)
        IntLoc2 = 7 * (IntLoc3 - IntLoc2) - IntLoc1
        IntLoc1 = Proc2(IntLoc1)
    benchtime: float = clock() - starttime - nulltime
    loopsPerBenchtime: float = loops / benchtime if benchtime != 0.0 else 0.0
    return (benchtime, loopsPerBenchtime)

def Proc1(PtrParIn: Record) -> Record:
    global PtrGlb
    NextRecord: Record = PtrGlb.copy()  # type: ignore
    PtrParIn.PtrComp = NextRecord
    PtrParIn.IntComp = 5
    NextRecord.IntComp = PtrParIn.IntComp
    NextRecord.PtrComp = PtrParIn.PtrComp
    NextRecord.PtrComp = Proc3(NextRecord.PtrComp)  # type: ignore
    if NextRecord.Discr == Ident1:
        NextRecord.IntComp = 6
        NextRecord.EnumComp = Proc6(PtrParIn.EnumComp)
        NextRecord.PtrComp = PtrGlb  # type: ignore
        NextRecord.IntComp = Proc7(NextRecord.IntComp, 10)
    else:
        PtrParIn = NextRecord.copy()
    NextRecord.PtrComp = None
    return PtrParIn

def Proc2(IntParIO: int) -> int:
    global Char1Glob, IntGlob
    IntLoc: int = IntParIO + 10
    while True:
        if Char1Glob == 'A':
            IntLoc = IntLoc - 1
            IntParIO = IntLoc - IntGlob
            EnumLoc: int = Ident1
        else:
            EnumLoc = Ident2  # ensure EnumLoc is defined in any branch
        if EnumLoc == Ident1:
            break
    return IntParIO

def Proc3(PtrParOut: Optional[Record]) -> Optional[Record]:
    global IntGlob, PtrGlb
    if PtrGlb is not None:
        PtrParOut = PtrGlb.PtrComp
    else:
        IntGlob = 100
    PtrGlb.IntComp = Proc7(10, IntGlob)  # type: ignore
    return PtrParOut

def Proc4() -> None:
    global Char2Glob, Char1Glob, BoolGlob
    BoolLoc: bool = (Char1Glob == 'A')
    BoolLoc = BoolLoc or bool(BoolGlob)
    Char2Glob = 'B'

def Proc5() -> None:
    global Char1Glob, BoolGlob
    Char1Glob = 'A'
    BoolGlob = FALSE

def Proc6(EnumParIn: int) -> int:
    EnumParOut: int = EnumParIn
    if not Func3(EnumParIn):
        EnumParOut = Ident4
    if EnumParIn == Ident1:
        EnumParOut = Ident1
    elif EnumParIn == Ident2:
        global IntGlob
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

def Proc8(Array1Par: List[int], Array2Par: List[List[int]], IntParI1: int, IntParI2: int) -> None:
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
    CharLoc: str = ''
    while IntLoc <= 1:
        if Func1(StrParI1[IntLoc], StrParI2[IntLoc + 1]) == Ident1:
            CharLoc = 'A'
            IntLoc = IntLoc + 1
    if CharLoc >= 'W' and CharLoc <= 'Z':
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
    main(LOOPS)