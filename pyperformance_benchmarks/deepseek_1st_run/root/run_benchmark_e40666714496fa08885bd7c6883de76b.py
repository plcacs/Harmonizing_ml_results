"""
Test the performance of isinstance() checks against runtime-checkable protocols.

For programmes that make extensive use of this feature,
these calls can easily become a bottleneck.
See https://github.com/python/cpython/issues/74690

The following situations all exercise different code paths
in typing._ProtocolMeta.__instancecheck__,
so each is tested in this benchmark:

  (1) Comparing an instance of a class that directly inherits
      from a protocol to that protocol.
  (2) Comparing an instance of a class that fulfils the interface
      of a protocol using instance attributes.
  (3) Comparing an instance of a class that fulfils the interface
      of a protocol using class attributes.
  (4) Comparing an instance of a class that fulfils the interface
      of a protocol using properties.

Protocols with callable and non-callable members also
exercise different code paths in _ProtocolMeta.__instancecheck__,
so are also tested separately.
"""
import time
from typing import Protocol, runtime_checkable, Type, List, Any
import pyperf

@runtime_checkable
class HasX(Protocol):
    """A runtime-checkable protocol with a single non-callable member"""
    x: Any

@runtime_checkable
class HasManyAttributes(Protocol):
    """A runtime-checkable protocol with many non-callable members"""
    a: Any
    b: Any
    c: Any
    d: Any
    e: Any

@runtime_checkable
class SupportsInt(Protocol):
    """A runtime-checkable protocol with a single callable member"""

    def __int__(self) -> int:
        ...

@runtime_checkable
class SupportsManyMethods(Protocol):
    """A runtime-checkable protocol with many callable members"""

    def one(self) -> int:
        ...

    def two(self) -> str:
        ...

    def three(self) -> bytes:
        ...

    def four(self) -> memoryview:
        ...

    def five(self) -> bytearray:
        ...

@runtime_checkable
class SupportsIntAndX(Protocol):
    """A runtime-checkable protocol with a mix of callable and non-callable members"""
    x: Any

    def __int__(self) -> int:
        ...

class Empty:
    """Empty class with no attributes"""
    pass

class PropertyX:
    """Class with a property x"""

    @property
    def x(self) -> int:
        return 42

class HasIntMethod:
    """Class with an __int__ method"""

    def __int__(self) -> int:
        return 42

class HasManyMethods:
    """Class with many methods"""

    def one(self) -> int:
        return 42

    def two(self) -> str:
        return '42'

    def three(self) -> bytes:
        return b'42'

    def four(self) -> memoryview:
        return memoryview(b'42')

    def five(self) -> bytearray:
        return bytearray(b'42')

class PropertyXWithInt:
    """Class with a property x and an __int__ method"""

    @property
    def x(self) -> int:
        return 42

    def __int__(self) -> int:
        return 42

class ClassVarX:
    """Class with a ClassVar x"""
    x: int = 42

class ClassVarXWithInt:
    """Class with a ClassVar x and an __int__ method"""
    x: int = 42

    def __int__(self) -> int:
        return 42

class InstanceVarX:
    """Class with an instance var x"""

    def __init__(self) -> None:
        self.x: int = 42

class ManyInstanceVars:
    """Class with many instance vars"""

    def __init__(self) -> None:
        for attr in 'abcde':
            setattr(self, attr, 42)

class InstanceVarXWithInt:
    """Class with an instance var x and an __int__ method"""

    def __init__(self) -> None:
        self.x: int = 42

    def __int__(self) -> int:
        return 42

class NominalX(HasX):
    """Class that explicitly subclasses HasX"""

    def __init__(self) -> None:
        self.x: int = 42

class NominalSupportsInt(SupportsInt):
    """Class that explicitly subclasses SupportsInt"""

    def __int__(self) -> int:
        return 42

class NominalXWithInt(SupportsIntAndX):
    """Class that explicitly subclasses NominalXWithInt"""

    def __init__(self) -> None:
        self.x: int = 42

def bench_protocols(loops: int) -> float:
    protocols: List[Type[Protocol]] = [HasX, HasManyAttributes, SupportsInt, SupportsManyMethods, SupportsIntAndX]
    instances: List[Any] = [cls() for cls in (Empty, PropertyX, HasIntMethod, HasManyMethods, PropertyXWithInt, ClassVarX, ClassVarXWithInt, InstanceVarX, ManyInstanceVars, InstanceVarXWithInt, NominalX, NominalSupportsInt, NominalXWithInt)]
    t0: float = time.perf_counter()
    for _ in range(loops):
        for protocol in protocols:
            for instance in instances:
                isinstance(instance, protocol)
    return time.perf_counter() - t0

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Test the performance of isinstance() checks against runtime-checkable protocols'
    runner.bench_time_func('typing_runtime_protocols', bench_protocols)
