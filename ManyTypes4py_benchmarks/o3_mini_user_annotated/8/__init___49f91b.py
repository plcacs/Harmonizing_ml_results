from typing import Any

def run(autoTester: Any) -> None:
    def show(*args: Any) -> None:
        autoTester.check(*args)
        # print(*args)  # Leave in for debugging purposes

    class R:
        def __init__(self, a: int, b: int) -> None:
            self.a: int = a
            self.b: int = b

    class A(R):
        def __init__(self, a: int, b: int, c: int) -> None:
            super().__init__(a, b)
            self.c: int = c

        def f(self, x: int, y: int) -> None:
            show('A.f:', x, y, self.a, self.b, self.c)
            
        def g(self, x: int, y: int) -> None:
            show('A.g:', x, y)
            
    class B(R):
        def __init__(self, a: int, b: int, d: int) -> None:
            super().__init__(a, b)
            self.d: int = d

        def f(self, x: int, y: int) -> None:
            show('B.f:', x, y, self.a, self.b, self.d)
            
        def h(self, x: int, y: int) -> None:
            show('A.h:', x, y, self.a, self.b, self.d)

    class C(A):
        def __init__(self, a: int, b: int, c: int) -> None:
            super().__init__(a, b, c)
        
        def f(self, x: int, y: int) -> None:
            super().f(x, y)
            show('C.f:', x, y, self.a, self.b, self.c)
            
    class D(B):
        def __init__(self, a: int, b: int, d: int) -> None:
            super().__init__(a, b, d)
        
        def f(self, x: int, y: int) -> None:
            super().f(x, y)
            show('D.f:', x, y, self.a, self.b, self.d)
            
    class E(C, D):        
        def __init__(self, a: int, b: int, c: int, d: int) -> None:
            R.__init__(self, a, b)  # Inherited via multiple legs of a diamond, so call explicitly
            self.c: int = c         # Could also have used C.__init__, but symmetry preferred
            self.d: int = d         # Don't use both C.__init__ and D.__init__, since R.__init__ will be called by both

        def f(self, x: int, y: int) -> None:
            C.f(self, x, y)         # Ambiguous part of diamond, don't use super()
            D.f(self, x, y)         # Ambiguous part of diamond, don't use super()
            show('E.f:', x, y, self.a, self.b, self.c, self.d)
            
        def g(self, x: int, y: int) -> None:
            super().g(x, y)         # Unique, using super() is OK
            show('E.g:', x, y, self.a, self.b, self.c, self.d)
            
        def h(self, x: int, y: int) -> None:
            super().h(x, y)         # Unique, using super() is OK
            show('E.h:', x, y, self.a, self.b, self.c, self.d)
           
    rr: R = R(100, 200)

    show('--1--')

    a: A = A(101, 201, 301)
    a.f(711, 811)
    a.g(721, 821)

    show('--2--')

    b: B = B(102, 202, 302)
    b.f(712, 812)
    b.h(732, 832)

    show('--3--')

    c: C = C(103, 203, 303)
    c.f(713, 813)
    c.g(723, 823)

    show('--4--')

    d: D = D(104, 204, 304)
    d.f(714, 814)
    d.h(734, 834)

    show('--5--')

    e: E = E(105, 205, 305, 405)
    e.f(715, 815)
    e.g(725, 825)
    e.h(735, 835)