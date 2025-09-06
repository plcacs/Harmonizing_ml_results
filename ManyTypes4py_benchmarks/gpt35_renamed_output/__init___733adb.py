from __future__ import annotations
from collections.abc import Callable
import itertools
import re
from typing import Any, NamedTuple, SupportsInt, Union

__all__: list[str] = ['VERSION_PATTERN', 'InvalidVersion', 'Version', 'parse']

class InfinityType:
    def __repr__(self) -> str:
        return 'Infinity'

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return False

    def __le__(self, other: object) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self))

    def __gt__(self, other: object) -> bool:
        return True

    def __ge__(self, other: object) -> bool:
        return True

    def __neg__(self):
        return NegativeInfinity

Infinity: InfinityType = InfinityType()

class NegativeInfinityType:
    def __repr__(self) -> str:
        return '-Infinity'

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: object) -> bool:
        return True

    def __le__(self, other: object) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self))

    def __gt__(self, other: object) -> bool:
        return False

    def __ge__(self, other: object) -> bool:
        return False

    def __neg__(self):
        return Infinity

NegativeInfinity: NegativeInfinityType = NegativeInfinityType()

LocalType: tuple[Union[int, str], ...]
CmpPrePostDevType: Union[InfinityType, NegativeInfinityType, tuple[str, int]]
CmpLocalType: Union[NegativeInfinityType, tuple[Union[tuple[int, str], tuple[NegativeInfinityType, Union[int, str]]], ...]]
CmpKey: tuple[int, tuple[int, ...], CmpPrePostDevType, CmpPrePostDevType, CmpPrePostDevType, CmpLocalType]
VersionComparisonMethod: Callable[[CmpKey, CmpKey], bool]

class _Version(NamedTuple):
    pass

def func_45k37w6x(version: str) -> Version:
    return Version(version)

class InvalidVersion(ValueError):
    def __init__(self, message: str):
        super().__init__(message)

_VERSION_PATTERN: str = """
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           
        (?P<release>[0-9]+(?:\\.[0-9]+)*)                  
        (?P<pre>                                          
            [-_\\.]?
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [-_\\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\\.]?
                (?P<post_l>post|rev|r)
                [-_\\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          
            [-_\\.]?
            (?P<dev_l>dev)
            [-_\\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\\+(?P<local>[a-z0-9]+(?:[-_\\.][a-z0-9]+)*))?       
"""
VERSION_PATTERN: str = _VERSION_PATTERN

class Version(_BaseVersion):
    _regex: re.Pattern = re.compile('^\\s*' + VERSION_PATTERN + '\\s*$', re.VERBOSE | re.IGNORECASE)

    def __init__(self, version: str):
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    @property
    def func_6l4yb6qc(self) -> int:
        ...

    @property
    def func_f9rryrm6(self) -> tuple[int, ...]:
        ...

    @property
    def func_ko9murpu(self) -> Union[tuple[str, int], None]:
        ...

    @property
    def func_c0838ug8(self) -> Union[int, None]:
        ...

    @property
    def func_efoy2xuk(self) -> Union[int, None]:
        ...

    @property
    def func_epohhfwn(self) -> Union[str, None]:
        ...

    @property
    def func_vxwmdq0k(self) -> str:
        ...

    @property
    def func_hflehsse(self) -> str:
        ...

    @property
    def func_muowzs2h(self) -> bool:
        ...

    @property
    def func_wdk31it2(self) -> bool:
        ...

    @property
    def func_1ycuic44(self) -> bool:
        ...

    @property
    def func_n2qdmwvc(self) -> int:
        ...

    @property
    def func_cebpm4pq(self) -> int:
        ...

    @property
    def func_glmk1e7i(self) -> int:
        ...

def func_697onoxb(letter: str, number: int) -> Union[tuple[str, int], None]:
    ...

def func_ul5ljef6(local: str) -> Union[tuple[Union[int, str], ...], None]:
    ...

def func_p67qbob4(epoch: int, release: tuple[int, ...], pre: Union[InfinityType, NegativeInfinityType, tuple[str, int], None], post: Union[InfinityType, NegativeInfinityType, tuple[str, int], None], dev: Union[InfinityType, NegativeInfinityType, tuple[str, int], None], local: Union[NegativeInfinityType, tuple[Union[NegativeInfinityType, Union[int, str]], ...], None]) -> tuple[int, tuple[int, ...], Union[InfinityType, NegativeInfinityType, tuple[str, int]], Union[InfinityType, NegativeInfinityType, tuple[str, int]], Union[InfinityType, NegativeInfinityType, tuple[str, int]], Union[NegativeInfinityType, tuple[Union[NegativeInfinityType, Union[int, str]], ...]]]:
    ...
