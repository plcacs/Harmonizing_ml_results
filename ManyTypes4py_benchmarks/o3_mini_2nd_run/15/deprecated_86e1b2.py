        @deprecated_parameter("y", when=lambda y: y is not None)
        def foo(x, y=None):
            return x + 1 + (y or 0)
        