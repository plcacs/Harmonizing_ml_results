from dataclasses import dataclass
from typing import Any, Dict

SOME_GLOBAL_VAR: str = "Ahhhh I'm a global var!!"
'\nThis is a global var.\n'

def func_with_no_args() -> None:
    """
    This function has no args.
    """
    return None

def func_with_args(a: int, b: int, c: int = 3) -> int:
    """
    This function has some args.

    # Parameters

    a : `int`
        A number.
    b : `int`
        Another number.
    c : `int`, optional (default = `3`)
        Yet another number.

    Notes
    -----

    These are some notes.

    # Returns

    `int`
        The result of `a + b * c`.
    """
    return a + b * c

class SomeClass:
    """
    I'm a class!

    # Parameters

    x : `float`
        This attribute is called `x`.
    """
    some_class_level_variable: int = 1
    '\n    This is how you document a class-level variable.\n    '
    some_class_level_var_with_type: int = 1

    def __init__(self) -> None:
        self.x: float = 1.0

    def _private_method(self) -> None:
        """
        Private methods should not be included in documentation.
        """
        pass

    def some_method(self) -> None:
        """
        I'm a method!

        But I don't do anything.

        # Returns

        `None`
        """
        return None

    def method_with_alternative_return_section(self) -> int:
        """
        Another method.

        # Returns

        A completely arbitrary number.
        """
        return 3

    def method_with_alternative_return_section3(self) -> int:
        """
        Another method.

        # Returns

        number : `int`
            A completely arbitrary number.
        """
        return 3

class AnotherClassWithReallyLongConstructor:
    def __init__(
        self,
        a_really_long_argument_name: int = 0,
        another_long_name: int = 2,
        these_variable_names_are_terrible: str = 'yea I know',
        **kwargs: Any
    ) -> None:
        self.a: int = a_really_long_argument_name
        self.b: int = another_long_name
        self.c: str = these_variable_names_are_terrible
        self.other: Dict[str, Any] = kwargs

@dataclass
class ClassWithDecorator:
    pass

class _PrivateClass:
    def public_method_on_private_class(self) -> None:
        """
        This should not be documented since the class is private.
        """
        pass