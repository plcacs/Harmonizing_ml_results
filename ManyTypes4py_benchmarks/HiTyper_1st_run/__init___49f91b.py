def run(autoTester: bool) -> None:

    def show(*args) -> None:
        autoTester.check(*args)

    class R:

        def __init__(self, a: Any, b: Any) -> None:
            self.a = a
            self.b = b

    class A(R):

        def __init__(self, a: Any, b: Any, c) -> None:
            super().__init__(a, b)
            self.c = c

        def f(self, x: Any, y: Any) -> None:
            show('A.f:', x, y, self.a, self.b, self.c)

        def g(self, x: Any, y: Any) -> None:
            show('A.g:', x, y)

    class B(R):

        def __init__(self, a: Any, b: Any, d) -> None:
            super().__init__(a, b)
            self.d = d

        def f(self, x: Any, y: Any) -> None:
            show('B.f:', x, y, self.a, self.b, self.d)

        def h(self, x: Any, y: Any) -> None:
            show('A.h:', x, y, self.a, self.b, self.d)

    class C(A):

        def __init__(self, a: Any, b: Any, c) -> None:
            super().__init__(a, b, c)

        def f(self, x: Any, y: Any) -> None:
            super().f(x, y)
            show('C.f:', x, y, self.a, self.b, self.c)

    class D(B):

        def __init__(self, a: Any, b: Any, d) -> None:
            super().__init__(a, b, d)

        def f(self, x: Any, y: Any) -> None:
            super().f(x, y)
            show('D.f:', x, y, self.a, self.b, self.d)

    class E(C, D):

        def __init__(self, a: Any, b: Any, c, d) -> None:
            R.__init__(self, a, b)
            self.c = c
            self.d = d

        def f(self, x: Any, y: Any) -> None:
            C.f(self, x, y)
            D.f(self, x, y)
            show('E.f:', x, y, self.a, self.b, self.c, self.d)

        def g(self, x: Any, y: Any) -> None:
            super().g(x, y)
            show('E.g:', x, y, self.a, self.b, self.c, self.d)

        def h(self, x: Any, y: Any) -> None:
            super().h(x, y)
            show('E.h:', x, y, self.a, self.b, self.c, self.d)
    rr = R(100, 200)
    show('--1--')
    a = A(101, 201, 301)
    a.f(711, 811)
    a.g(721, 821)
    show('--2--')
    b = B(102, 202, 302)
    b.f(712, 812)
    b.h(732, 832)
    show('--3--')
    c = C(103, 203, 303)
    c.f(713, 813)
    c.g(723, 823)
    show('--4--')
    d = D(104, 204, 304)
    d.f(714, 814)
    d.h(734, 834)
    show('--5--')
    e = E(105, 205, 305, 405)
    e.f(715, 815)
    e.g(725, 825)
    e.h(735, 835)