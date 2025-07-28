from __future__ import annotations
from collections.abc import Callable
import itertools
import re
from typing import Any, NamedTuple, Optional, Tuple, Union, Match

__all__ = ['VERSION_PATTERN', 'InvalidVersion', 'Version', 'parse']

class InfinityType:
    def __repr__(self) -> str:
        return 'Infinity'

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: Any) -> bool:
        return False

    def __le__(self, other: Any) -> bool:
        return False

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self))

    def __gt__(self, other: Any) -> bool:
        return True

    def __ge__(self, other: Any) -> bool:
        return True

    def __neg__(self) -> NegativeInfinityType:
        return NegativeInfinity

Infinity = InfinityType()

class NegativeInfinityType:
    def __repr__(self) -> str:
        return '-Infinity'

    def __hash__(self) -> int:
        return hash(repr(self))

    def __lt__(self, other: Any) -> bool:
        return True

    def __le__(self, other: Any) -> bool:
        return True

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self))

    def __gt__(self, other: Any) -> bool:
        return False

    def __ge__(self, other: Any) -> bool:
        return False

    def __neg__(self) -> InfinityType:
        return Infinity

NegativeInfinity = NegativeInfinityType()
LocalType = Tuple[Union[int, str], ...]
CmpPrePostDevType = Union[InfinityType, NegativeInfinityType, Tuple[str, int]]
CmpLocalType = Union[NegativeInfinityType, Tuple[Union[Tuple[int, str], Tuple[NegativeInfinityType, Union[int, str]]], ...]]
CmpKey = Tuple[int, Tuple[int, ...], CmpPrePostDevType, CmpPrePostDevType, CmpPrePostDevType, CmpLocalType]
VersionComparisonMethod = Callable[[CmpKey, CmpKey], bool]

class _Version(NamedTuple):
    epoch: int
    release: Tuple[int, ...]
    pre: Optional[Tuple[str, int]]
    post: Optional[Tuple[str, int]]
    dev: Optional[Tuple[str, int]]
    local: Optional[Tuple[Union[int, str], ...]]

def parse(version: str) -> Version:
    return Version(version)

class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.

    The ``InvalidVersion`` exception is raised when a version string is
    improperly formatted. Pandas uses this exception to ensure that all
    version strings are PEP 440 compliant.

    See Also
    --------
    util.version.Version : Class for handling and parsing version strings.

    Examples
    --------
    >>> pd.util.version.Version("1.")
    Traceback (most recent call last):
    InvalidVersion: Invalid version: '1.'
    """

class _BaseVersion:
    _key: CmpKey

    def __hash__(self) -> int:
        return hash(self._key)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented  # type: ignore
        return self._key < other._key

    def __le__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented  # type: ignore
        return self._key <= other._key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented  # type: ignore
        return self._key == other._key

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented  # type: ignore
        return self._key >= other._key

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented  # type: ignore
        return self._key > other._key

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, _BaseVersion):
            return NotImplemented  # type: ignore
        return self._key != other._key

_VERSION_PATTERN = '\n    v?\n    (?:\n        (?:(?P<epoch>[0-9]+)!)?                           # epoch\n        (?P<release>[0-9]+(?:\\.[0-9]+)*)                  # release segment\n        (?P<pre>                                          # pre-release\n            [-_\\.]?\n            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)\n            [-_\\.]?\n            (?P<pre_n>[0-9]+)?\n        )?\n        (?P<post>                                         # post release\n            (?:-(?P<post_n1>[0-9]+))\n            |\n            (?:\n                [-_\\.]?\n                (?P<post_l>post|rev|r)\n                [-_\\.]?\n                (?P<post_n2>[0-9]+)?\n            )\n        )?\n        (?P<dev>                                          # dev release\n            [-_\\.]?\n            (?P<dev_l>dev)\n            [-_\\.]?\n            (?P<dev_n>[0-9]+)?\n        )?\n    )\n    (?:\\+(?P<local>[a-z0-9]+(?:[-_\\.][a-z0-9]+)*))?       # local version\n'
VERSION_PATTERN = _VERSION_PATTERN

class Version(_BaseVersion):
    _regex: re.Pattern[str] = re.compile('^\\s*' + VERSION_PATTERN + '\\s*$', re.VERBOSE | re.IGNORECASE)

    def __init__(self, version: str) -> None:
        match: Optional[Match[str]] = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")
        self._version: _Version = _Version(
            epoch=int(match.group('epoch')) if match.group('epoch') else 0,
            release=tuple((int(i) for i in match.group('release').split('.'))),
            pre=_parse_letter_version(match.group('pre_l'), match.group('pre_n')),
            post=_parse_letter_version(match.group('post_l'), match.group('post_n1') or match.group('post_n2')),
            dev=_parse_letter_version(match.group('dev_l'), match.group('dev_n')),
            local=_parse_local_version(match.group('local'))
        )
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local
        )

    def __repr__(self) -> str:
        return f"<Version('{self}')>"

    def __str__(self) -> str:
        parts: list[str] = []
        if self.epoch != 0:
            parts.append(f'{self.epoch}!')
        parts.append('.'.join(str(x) for x in self.release))
        if self.pre is not None:
            parts.append(''.join(str(x) for x in self.pre))
        if self.post is not None:
            parts.append(f'.post{self.post}')
        if self.dev is not None:
            parts.append(f'.dev{self.dev}')
        if self.local is not None:
            parts.append(f'+{self.local}')
        return ''.join(parts)

    @property
    def epoch(self) -> int:
        return self._version.epoch

    @property
    def release(self) -> Tuple[int, ...]:
        return self._version.release

    @property
    def pre(self) -> Optional[Tuple[str, int]]:
        return self._version.pre

    @property
    def post(self) -> Optional[int]:
        return self._version.post[1] if self._version.post else None

    @property
    def dev(self) -> Optional[int]:
        return self._version.dev[1] if self._version.dev else None

    @property
    def local(self) -> Optional[str]:
        if self._version.local:
            return '.'.join(str(x) for x in self._version.local)
        else:
            return None

    @property
    def public(self) -> str:
        return str(self).split('+', 1)[0]

    @property
    def base_version(self) -> str:
        parts: list[str] = []
        if self.epoch != 0:
            parts.append(f'{self.epoch}!')
        parts.append('.'.join(str(x) for x in self.release))
        return ''.join(parts)

    @property
    def is_prerelease(self) -> bool:
        return self.dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        return self.post is not None

    @property
    def is_devrelease(self) -> bool:
        return self.dev is not None

    @property
    def major(self) -> int:
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def minor(self) -> int:
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        return self.release[2] if len(self.release) >= 3 else 0

def _parse_letter_version(letter: Optional[str], number: Optional[str]) -> Optional[Tuple[str, int]]:
    if letter:
        if number is None:
            number = "0"
        letter = letter.lower()
        if letter == 'alpha':
            letter = 'a'
        elif letter == 'beta':
            letter = 'b'
        elif letter in ['c', 'pre', 'preview']:
            letter = 'rc'
        elif letter in ['rev', 'r']:
            letter = 'post'
        return (letter, int(number))
    if not letter and number:
        letter = 'post'
        return (letter, int(number))
    return None

_local_version_separators: re.Pattern[str] = re.compile('[\\._-]')

def _parse_local_version(local: Optional[str]) -> Optional[Tuple[Union[int, str], ...]]:
    if local is not None:
        return tuple(
            (part.lower() if not part.isdigit() else int(part))
            for part in _local_version_separators.split(local)
        )
    return None

def _cmpkey(
    epoch: int,
    release: Tuple[int, ...],
    pre: Optional[Tuple[str, int]],
    post: Optional[Tuple[str, int]],
    dev: Optional[Tuple[str, int]],
    local: Optional[Tuple[Union[int, str], ...]]
) -> CmpKey:
    _release: Tuple[int, ...] = tuple(
        reversed(list(itertools.dropwhile(lambda x: x == 0, reversed(release))))
    )
    if pre is None and post is None and (dev is not None):
        _pre: CmpPrePostDevType = NegativeInfinity
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre
    if post is None:
        _post: CmpPrePostDevType = NegativeInfinity
    else:
        _post = post
    if dev is None:
        _dev: CmpPrePostDevType = Infinity
    else:
        _dev = dev
    if local is None:
        _local: CmpLocalType = NegativeInfinity
    else:
        _local = tuple(
            ((i, '') if isinstance(i, int) else (NegativeInfinity, i)) for i in local
        )
    return (epoch, _release, _pre, _post, _dev, _local)