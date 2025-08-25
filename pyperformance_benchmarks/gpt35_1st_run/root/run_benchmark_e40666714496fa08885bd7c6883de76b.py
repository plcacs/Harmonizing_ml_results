from typing import Protocol, runtime_checkable

@runtime_checkable
class HasX(Protocol):
    x: None

@runtime_checkable
class HasManyAttributes(Protocol):
    a: None
    b: None
    c: None
    d: None
    e: None

@runtime_checkable
class SupportsInt(Protocol):

    def __int__(self) -> None: ...

@runtime_checkable
class SupportsManyMethods(Protocol):

    def one(self) -> None: ...
    def two(self) -> None: ...
    def three(self) -> None: ...
    def four(self) -> None: ...
    def five(self) -> None: ...

@runtime_checkable
class SupportsIntAndX(Protocol):
    x: None

    def __int__(self) -> None: ...

class Empty():
    pass

class PropertyX():
    @property
    def x(self) -> int: ...

class HasIntMethod():
    def __int__(self) -> int: ...

class HasManyMethods():
    def one(self) -> int: ...
    def two(self) -> str: ...
    def three(self) -> bytes: ...
    def four(self) -> memoryview: ...
    def five(self) -> bytearray: ...

class PropertyXWithInt():
    @property
    def x(self) -> int: ...

    def __int__(self) -> int: ...

class ClassVarX():
    x: int = 42

class ClassVarXWithInt():
    x: int = 42

    def __int__(self) -> int: ...

class InstanceVarX():
    def __init__(self):
        self.x: int = 42

class ManyInstanceVars():
    def __init__(self):
        a: int
        b: int
        c: int
        d: int
        e: int

class InstanceVarXWithInt():
    def __init__(self):
        self.x: int = 42

    def __int__(self) -> int: ...

class NominalX(HasX):
    def __init__(self):
        self.x: int = 42

class NominalSupportsInt(SupportsInt):
    def __int__(self) -> int: ...

class NominalXWithInt(SupportsIntAndX):
    def __init__(self):
        self.x: int = 42
