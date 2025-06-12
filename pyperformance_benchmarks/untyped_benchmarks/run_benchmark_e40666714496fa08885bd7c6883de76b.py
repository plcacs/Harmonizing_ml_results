
'\nTest the performance of isinstance() checks against runtime-checkable protocols.\n\nFor programmes that make extensive use of this feature,\nthese calls can easily become a bottleneck.\nSee https://github.com/python/cpython/issues/74690\n\nThe following situations all exercise different code paths\nin typing._ProtocolMeta.__instancecheck__,\nso each is tested in this benchmark:\n\n  (1) Comparing an instance of a class that directly inherits\n      from a protocol to that protocol.\n  (2) Comparing an instance of a class that fulfils the interface\n      of a protocol using instance attributes.\n  (3) Comparing an instance of a class that fulfils the interface\n      of a protocol using class attributes.\n  (4) Comparing an instance of a class that fulfils the interface\n      of a protocol using properties.\n\nProtocols with callable and non-callable members also\nexercise different code paths in _ProtocolMeta.__instancecheck__,\nso are also tested separately.\n'
import time
from typing import Protocol, runtime_checkable
import pyperf

@runtime_checkable
class HasX(Protocol):
    'A runtime-checkable protocol with a single non-callable member'
    x = None

@runtime_checkable
class HasManyAttributes(Protocol):
    'A runtime-checkable protocol with many non-callable members'
    a = None
    b = None
    c = None
    d = None
    e = None

@runtime_checkable
class SupportsInt(Protocol):
    'A runtime-checkable protocol with a single callable member'

    def __int__(self):
        ...

@runtime_checkable
class SupportsManyMethods(Protocol):
    'A runtime-checkable protocol with many callable members'

    def one(self):
        ...

    def two(self):
        ...

    def three(self):
        ...

    def four(self):
        ...

    def five(self):
        ...

@runtime_checkable
class SupportsIntAndX(Protocol):
    'A runtime-checkable protocol with a mix of callable and non-callable members'
    x = None

    def __int__(self):
        ...

class Empty():
    'Empty class with no attributes'

class PropertyX():
    'Class with a property x'

    @property
    def x(self):
        return 42

class HasIntMethod():
    'Class with an __int__ method'

    def __int__(self):
        return 42

class HasManyMethods():
    'Class with many methods'

    def one(self):
        return 42

    def two(self):
        return '42'

    def three(self):
        return b'42'

    def four(self):
        return memoryview(b'42')

    def five(self):
        return bytearray(b'42')

class PropertyXWithInt():
    'Class with a property x and an __int__ method'

    @property
    def x(self):
        return 42

    def __int__(self):
        return 42

class ClassVarX():
    'Class with a ClassVar x'
    x = 42

class ClassVarXWithInt():
    'Class with a ClassVar x and an __int__ method'
    x = 42

    def __int__(self):
        return 42

class InstanceVarX():
    'Class with an instance var x'

    def __init__(self):
        self.x = 42

class ManyInstanceVars():
    'Class with many instance vars'

    def __init__(self):
        for attr in 'abcde':
            setattr(self, attr, 42)

class InstanceVarXWithInt():
    'Class with an instance var x and an __int__ method'

    def __init__(self):
        self.x = 42

    def __int__(self):
        return 42

class NominalX(HasX):
    'Class that explicitly subclasses HasX'

    def __init__(self):
        self.x = 42

class NominalSupportsInt(SupportsInt):
    'Class that explicitly subclasses SupportsInt'

    def __int__(self):
        return 42

class NominalXWithInt(SupportsIntAndX):
    'Class that explicitly subclasses NominalXWithInt'

    def __init__(self):
        self.x = 42

def bench_protocols(loops):
    protocols = [HasX, HasManyAttributes, SupportsInt, SupportsManyMethods, SupportsIntAndX]
    instances = [cls() for cls in (Empty, PropertyX, HasIntMethod, HasManyMethods, PropertyXWithInt, ClassVarX, ClassVarXWithInt, InstanceVarX, ManyInstanceVars, InstanceVarXWithInt, NominalX, NominalSupportsInt, NominalXWithInt)]
    t0 = time.perf_counter()
    for _ in range(loops):
        for protocol in protocols:
            for instance in instances:
                isinstance(instance, protocol)
    return (time.perf_counter() - t0)
if (__name__ == '__main__'):
    runner = pyperf.Runner()
    runner.metadata['description'] = 'Test the performance of isinstance() checks against runtime-checkable protocols'
    runner.bench_time_func('typing_runtime_protocols', bench_protocols)
