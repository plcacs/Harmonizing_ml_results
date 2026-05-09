from org.transcrypt.stubs.browser import __pragma__

class Matrix:
    def __init__(self, nRows: int, nCols: int, elements: list = []):
        self.nRows: int = nRows
        self.nCols: int = nCols
        if len(elements):
            self._: list = elements
        else:
            self._: list = [[0 for col in range(nCols)] for row in range(nRows)]

    def __matmul__(self, other: 'Matrix') -> 'Matrix':
        result: Matrix = Matrix(self.nRows, other.nCols)
        for iTargetRow in range(result.nRows):
            for iTargetCol in range(result.nCols):
                for iTerm in range(self.nCols):
                    result._[iTargetRow][iTargetCol] += self._[iTargetRow][iTerm] * other._[iTerm][iTargetCol]
        return result

    def __imatmul__(self, other: 'Matrix') -> 'Matrix':
        return self.__matmul__(other)

    def __mul__(self, other: 'Matrix') -> 'Matrix':
        if type(other) == Matrix:
            result: Matrix = Matrix(self.nRows, self.nCols)
            for iRow in range(self.nRows):
                for iCol in range(self.nCols):
                    result._[iRow][iCol] = self._[iRow][iCol] * other._[iRow][iCol]
            return result
        else:
            return self.__rmul__(other)

    def __rmul__(self, scalar: float) -> 'Matrix':
        result: Matrix = Matrix(self.nRows, self.nCols)
        for iRow in range(self.nRows):
            for iCol in range(self.nCols):
                result._[iRow][iCol] = scalar * self._[iRow][iCol]
        return result

    def __imul__(self, other: 'Matrix') -> 'Matrix':
        return self.__mul__(other)

    def __add__(self, other: 'Matrix') -> 'Matrix':
        result: Matrix = Matrix(self.nRows, self.nCols)
        for iRow in range(self.nRows):
            for iCol in range(self.nCols):
                result._[iRow][iCol] = self._[iRow][iCol] + other._[iRow][iCol]
        return result

    def __getitem__(self, index: int) -> list:
        return self._[index]

    def __setitem__(self, index: int, value: list) -> None:
        self._[index] = value

    def __repr__(self) -> str:
        return repr(self._)

    def __floordiv__(self, other: 'Matrix') -> str:
        return 'Overloading __floordiv__ has no meaning for matrices'

    def __truediv__(self, other: 'Matrix') -> str:
        return 'Overloading __truediv__ has no meaning for matrices'

class Functor:
    def __init__(self, factor: float):
        self.factor: float = factor
    __pragma__('kwargs')

    def __call__(self, x: float, y: float = -1, *args: list, m: float = -2, n: float, **kwargs: dict) -> tuple:
        return (self.factor * x, self.factor * y, [self.factor * arg for arg in args], self.factor * m, self.factor * n)
    __pragma__('nokwargs')
f = Functor(10)
__pragma__('kwargs')

class Bitwise:
    def __lshift__(self, other: object) -> None:
        autoTester.check('lshift')

    def __rlshift__(self, other: object) -> None:
        autoTester.check('rlshift')

    def __rshift__(self, other: object) -> None:
        autoTester.check('rshift')

    def __rrshift__(self, other: object) -> None:
        autoTester.check('rrshift')

    def __or__(self, other: object) -> None:
        autoTester.check('or')

    def __ror__(self, other: object) -> None:
        autoTester.check('ror')

    def __xor__(self, other: object) -> None:
        autoTester.check('xor')

    def __rxor__(self, other: object) -> None:
        autoTester.check('rxor')

    def __and__(self, other: object) -> None:
        autoTester.check('and')

    def __rand__(self, other: object) -> None:
        autoTester.check('rand')
bitwise = Bitwise()
__pragma__('opov')
bitwise << []
[] << bitwise
autoTester.check(32 << 2)
bitwise >> []
[] >> bitwise
autoTester.check(32 >> 2)
bitwise | []
[] | bitwise
autoTester.check(1 | 4)
bitwise ^ []
[] ^ bitwise
autoTester.check(11 ^ 13)
bitwise & []
[] & bitwise
autoTester.check(12 & 20)
a = 32
a <<= 2
autoTester.check(a)
__pragma__('noopov')
autoTester.check(32 << 2)
autoTester.check(32 >> 2)
autoTester.check(1 | 4)
autoTester.check(11 ^ 13)
autoTester.check(12 & 20)
a = 32
a <<= 2
autoTester.check(a)

class A:
    def __init__(self):
        self.b: dict = {}
a = A()
a.b['c'] = 'd'
__pragma__('opov')
a.b['c'] += 'e'
__pragma__('noopov')
autoTester.check(a.b['c'])
