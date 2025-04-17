from dataclasses import dataclass
from typing import Any, Dict

SOME_GLOBAL_VAR: str = "Ahhhh I'm a global var!!"

def func_with_no_args() -> None:
    return None

def func_with_args(a: int, b: int, c: int = 3) -> int:
    return a + b * c

class SomeClass:
    some_class_level_variable: int = 1
    some_class_level_var_with_type: int = 1

    def __init__(self) -> None:
        self.x: float = 1.0

    def _private_method(self) -> None:
        pass

    def some_method(self) -> None:
        return None

    def method_with_alternative_return_section(self) -> int:
        return 3

    def method_with_alternative_return_section3(self) -> int:
        return 3

class AnotherClassWithReallyLongConstructor:
    def __init__(
        self,
        a_really_long_argument_name: int = 0,
        another_long_name: float = 2,
        these_variable_names_are_terrible: str = "yea I know",
        **kwargs: Dict[str, Any],
    ) -> None:
        self.a: int = a_really_long_argument_name
        self.b: float = another_long_name
        self.c: str = these_variable_names_are_terrible
        self.other: Dict[str, Any] = kwargs

@dataclass
class ClassWithDecorator:
    x: int

class _PrivateClass:
    def public_method_on_private_class(self) -> None:
        pass
