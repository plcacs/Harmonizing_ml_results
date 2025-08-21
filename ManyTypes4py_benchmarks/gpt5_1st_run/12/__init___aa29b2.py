from typing import Any, Dict, List, Tuple, Union
from org.transcrypt.stubs.browser import __pragma__

Element = Union[int, float]
Scalar = Union[int, float]


class Matrix:
    nRows: int
    nCols: int
    _: List[List[Element]]

    def __init__(self, nRows: int, nCols: int, elements: List[List[Element]] = []) -> None:
        self.nRows = nRows
        self.nCols = nCols
        if len(elements):
            self._ = elements
        else:
            self._ = [[0 for col in range(nCols)] for row in range(nRows)]

    def __matmul__(self, other: "Matrix") -> "Matrix":
        result = Matrix(self.nRows, other.nCols)
        for iTargetRow in range(result.nRows):
            for iTargetCol in range(result.nCols):
                for iTerm in range(self.nCols):
                    result._[iTargetRow][iTargetCol] += self._[iTargetRow][iTerm] * other._[iTerm][iTargetCol]
        return result

    def __imatmul__(self, other: "Matrix") -> "Matrix":
        return self.__matmul__(other)

    def __mul__(self, other: Union["Matrix", Scalar]) -> "Matrix":
        if type(other) == Matrix:
            result = Matrix(self.nRows, self.nCols)
            for iRow in range(self.nRows):
                for iCol in range(self.nCols):
                    result._[iRow][iCol] = self._[iRow][iCol] * other._[iRow][iCol]
            return result
        else:
            return self.__rmul__(other)  # type: ignore[arg-type]

    def __rmul__(self, scalar: Scalar) -> "Matrix":
        result = Matrix(self.nRows, self.nCols)
        for iRow in range(self.nRows):
            for iCol in range(self.nCols):
                result._[iRow][iCol] = scalar * self._[iRow][iCol]
        return result

    def __imul__(self, other: Union["Matrix", Scalar]) -> "Matrix":
        return self.__mul__(other)

    def __add__(self, other: "Matrix") -> "Matrix":
        result = Matrix(self.nRows, self.nCols)
        for iRow in range(self.nRows):
            for iCol in range(self.nCols):
                result._[iRow][iCol] = self._[iRow][iCol] + other._[iRow][iCol]
        return result

    def __getitem__(self, index: int) -> List[Element]:
        return self._[index]

    def __setitem__(self, index: int, value: List[Element]) -> None:
        self._[index] = value

    def __repr__(self) -> str:
        return repr(self._)

    def __floordiv__(self, other: Any) -> str:
        return 'Overloading __floordiv__ has no meaning for matrices'

    def __truediv__(self, other: Any) -> str:
        return 'Overloading __truediv__ has no meaning for matrices'


class Functor:
    factor: int

    def __init__(self, factor: int) -> None:
        self.factor = factor

    __pragma__('kwargs')
    def __call__(
        self,
        x: int,
        y: int = -1,
        *args: int,
        m: int = -2,
        n: int,
        **kwargs: Any
    ) -> Tuple[int, int, List[int], int, int]:
        return (self.factor * x, self.factor * y, [self.factor * arg for arg in args], self.factor * m, self.factor * n)
    __pragma__('nokwargs')


f: Functor = Functor(10)

__pragma__('kwargs')
def g(
    x: int,
    y: int = -1,
    *args: int,
    m: int = -2,
    n: int,
    **kwargs: Any
) -> Tuple[int, int, Tuple[int, ...], int, int]:
    return (x, y, args, m, n)
__pragma__('nokwargs')


def run(autoTester: Any) -> None:
    m0 = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 8, 10]])
    m1 = Matrix(3, 3, [[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    m4 = Matrix(3, 3, [[1, 1, 2], [2, 2, 3], [3, 3, -5]])
    m5 = Matrix(3, 3, [[1, 1, 2], [2, 2, 3], [3, 3, -5]])
    x = 3
    y = x * 4 * x
    fast = 2 * 3
    fast += 1
    __pragma__('opov')
    m1[1][2] = m0[1][2]
    slow = 2 + 3
    m2 = m0 * m1 + m1 * (m0 + m1)
    m3 = 2 * (2 * m0 * 3 * m1 + m2 * 4) * 2
    autoTester.check(m0[1][1], m0[1][2], m1[1][1], m1[1][2])
    m1 += m0
    m2 *= m1
    m5 @= m4
    m6 = m0 @ m1
    autoTester.check(m0 / m1)
    autoTester.check(m0 // m1)
    __pragma__('noopov')
    fast2 = 16 * y + 1
    fast *= 2
    autoTester.check(m0, m1)
    autoTester.check(x, y)
    autoTester.check(m2)
    autoTester.check(m3)
    autoTester.check(m5)
    autoTester.check(m6)
    autoTester.check(fast, slow, fast2)
    x = 'marker'
    __pragma__('opov')
    autoTester.check(f(3, 4, 30, 40, m=300, n=400, p=3000, q=4000))
    autoTester.check(g(3, 4, 30, 40, m=300, n=400, p=3000, q=4000))
    autoTester.check(set((1, 2, 3)) == set((3, 2, 1)))
    autoTester.check(set((1, 2, 3)) != set((3, 2, 1)))
    autoTester.check(set((1, 3)) == set((3, 2, 1)))
    autoTester.check(set((1, 3)) != set((3, 2, 1)))
    autoTester.check(set((1, 2)) < set((3, 2, 1)))
    autoTester.check(set((1, 2, 3)) <= set((3, 2, 1)))
    autoTester.check(set((1, 2, 3)) > set((2, 1)))
    autoTester.check(set((1, 2, 3)) >= set((3, 2, 1)))
    autoTester.check((1, 2, 3) == (1, 2, 3))
    autoTester.check([1, 2, 3] == [1, 2, 3])
    autoTester.check((1, 2, 3) != (1, 2, 3))
    autoTester.check([1, 2, 3] != [1, 2, 3])
    autoTester.check((2, 1, 3) == (1, 2, 3))
    autoTester.check([2, 1, 3] == [1, 2, 3])
    autoTester.check((2, 1, 3) != (1, 2, 3))
    autoTester.check([2, 1, 3] != [1, 2, 3])
    __pragma__('noopov')

    class Bitwise:
        def __lshift__(self, other: Any) -> None:
            autoTester.check('lshift')

        def __rlshift__(self, other: Any) -> None:
            autoTester.check('rlshift')

        def __rshift__(self, other: Any) -> None:
            autoTester.check('rshift')

        def __rrshift__(self, other: Any) -> None:
            autoTester.check('rrshift')

        def __or__(self, other: Any) -> None:
            autoTester.check('or')

        def __ror__(self, other: Any) -> None:
            autoTester.check('ror')

        def __xor__(self, other: Any) -> None:
            autoTester.check('xor')

        def __rxor__(self, other: Any) -> None:
            autoTester.check('rxor')

        def __and__(self, other: Any) -> None:
            autoTester.check('and')

        def __rand__(self, other: Any) -> None:
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
        b: Dict[str, str]

        def __init__(self) -> None:
            self.b = {}

    a = A()
    a.b['c'] = 'd'
    __pragma__('opov')
    a.b['c'] += 'e'
    __pragma__('noopov')
    autoTester.check(a.b['c'])