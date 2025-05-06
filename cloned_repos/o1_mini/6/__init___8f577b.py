from org.transcrypt.stubs.browser import __pragma__
__pragma__('kwargs')

from typing import Any, Tuple, Dict

class A:

    def __init__(self, x: int = 123, y: int = 456, *args: Any, m: Any, n: int = 456, **kwargs: Any) -> None:
        self.x: int = x
        self.y: int = y
        self.args: Tuple[Any, ...] = args
        self.m: Any = m
        self.n: int = n
        self.kwargs: Dict[str, Any] = kwargs
        self.extra: str = 'hello'

    def f(self, autoTester: Any) -> None:
        autoTester.check(self.x, self.y, self.args, self.m, self.n, self.kwargs, self.extra)

class B(A):

    def __init__(self, x: Any, y: int = -1, *args: Any, m: int = -2, n: Any, **kwargs: Any) -> None:
        A.__init__(self, y, x, *args, m=n, n=m, **kwargs)

class C:
    __pragma__('nokwargs')

    def tricky(self, *args: Any) -> Tuple[Any, ...]:
        return args
    __pragma__('kwargs')

def run(autoTester: Any) -> None:

    def f(x: Any, y: int = -1, *args: Any, m: int = -2, n: Any, **kwargs: Any) -> None:
        autoTester.check('#203', kwargs.__class__.__name__)
        autoTester.check('#203', sorted(kwargs.keys()))

        def f2(x: Any, y: int = -3, *args: Any, m: int = -4, n: Any, **kwargs: Any) -> None:
            autoTester.check(x, y, args, m, n, kwargs)
        f2(11, 22, 1010, 2020, m=100100, n=200200, p=10001000, q=20002000)
        autoTester.check(x, y, args, m, n, kwargs)
        
    f(1, 2, 10, 20, m=100, n=200, p=1000, q=2000)
    b: B = B(3, 4, 30, 40, m=300, n=400, p=3000, q=4000)
    b.f(autoTester)

    def g(*args: Any, **kwargs: Any) -> None:
        autoTester.check(args, kwargs)
    g(*(1, 2, 3), **{'p': 'aP', 'q': 'aQ', 'r': 'anR'})
    (lambda x, y: Any, ... )(1, 2, 8, 16, m=128, n=256.3, p=1024.3, q=2048.3)
    autoTester.check(C().tricky(*range(4)))
    autoTester.check('{}-{}'.format(1, 3, 5, 7, 9))
    autoTester.check('{}-{}'.format(*range(4)))
