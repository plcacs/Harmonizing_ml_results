from dataclasses import dataclass

SOME_GLOBAL_VAR: str = "Ahhhh I'm a global var!!"

def func_with_no_args() -> None:
    return None

def func_with_args(a: int, b: int, c: int = 3) -> int:
    return a + b * c

class SomeClass:
    some_class_level_variable: int = 1
    some_class_level_var_with_type: int = 1

    def __init__(self):
        self.x: float = 1.0

    def _private_method(self):
        pass

    def some_method() -> None:
        return None

    def method_with_alternative_return_section() -> int:
        return 3

    def method_with_alternative_return_section3() -> int:
        return 3

class AnotherClassWithReallyLongConstructor:
    def __init__(self, a_really_long_argument_name: int = 0, another_long_name: int = 2, these_variable_names_are_terrible: str = 'yea I know', **kwargs):
        self.a = a_really_long_argument_name
        self.b = another_long_name
        self.c = these_variable_names_are_terrible
        self.other = kwargs

@dataclass
class ClassWithDecorator:
    pass

class _PrivateClass:
    def public_method_on_private_class(self):
        pass
