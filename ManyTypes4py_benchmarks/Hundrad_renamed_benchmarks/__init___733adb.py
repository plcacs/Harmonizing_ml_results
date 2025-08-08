from __future__ import annotations
from collections.abc import Callable
import itertools
import re
from typing import Any, NamedTuple, SupportsInt, Union
__all__ = ['VERSION_PATTERN', 'InvalidVersion', 'Version', 'parse']


class InfinityType:

    def __repr__(self):
        return 'Infinity'

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __gt__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __neg__(self):
        return NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:

    def __repr__(self):
        return '-Infinity'

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, type(self))

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __neg__(self):
        return Infinity


NegativeInfinity = NegativeInfinityType()
LocalType = tuple[Union[int, str], ...]
CmpPrePostDevType = Union[InfinityType, NegativeInfinityType, tuple[str, int]]
CmpLocalType = Union[NegativeInfinityType, tuple[Union[tuple[int, str],
    tuple[NegativeInfinityType, Union[int, str]]], ...]]
CmpKey = tuple[int, tuple[int, ...], CmpPrePostDevType, CmpPrePostDevType,
    CmpPrePostDevType, CmpLocalType]
VersionComparisonMethod = Callable[[CmpKey, CmpKey], bool]


class _Version(NamedTuple):
    pass


def func_45k37w6x(version):
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

    def __hash__(self):
        return hash(self._key)

    def __lt__(self, other):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key < other._key

    def __le__(self, other):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key <= other._key

    def __eq__(self, other):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key == other._key

    def __ge__(self, other):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key >= other._key

    def __gt__(self, other):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key > other._key

    def __ne__(self, other):
        if not isinstance(other, _BaseVersion):
            return NotImplemented
        return self._key != other._key


_VERSION_PATTERN = """
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\\.]?
            (?P<pre_l>alpha|a|beta|b|preview|pre|c|rc)
            [-_\\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\\.]?
                (?P<post_l>post|rev|r)
                [-_\\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\\.]?
            (?P<dev_l>dev)
            [-_\\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\\+(?P<local>[a-z0-9]+(?:[-_\\.][a-z0-9]+)*))?       # local version
"""
VERSION_PATTERN = _VERSION_PATTERN


class Version(_BaseVersion):
    _regex = re.compile('^\\s*' + VERSION_PATTERN + '\\s*$', re.VERBOSE |
        re.IGNORECASE)

    def __init__(self, version):
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")
        self._version = _Version(epoch=int(match.group('epoch')) if match.
            group('epoch') else 0, release=tuple(int(i) for i in match.
            group('release').split('.')), pre=_parse_letter_version(match.
            group('pre_l'), match.group('pre_n')), post=
            _parse_letter_version(match.group('post_l'), match.group(
            'post_n1') or match.group('post_n2')), dev=
            _parse_letter_version(match.group('dev_l'), match.group('dev_n'
            )), local=_parse_local_version(match.group('local')))
        self._key = _cmpkey(self._version.epoch, self._version.release,
            self._version.pre, self._version.post, self._version.dev, self.
            _version.local)

    def __repr__(self):
        return f"<Version('{self}')>"

    def __str__(self):
        parts = []
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
    def func_6l4yb6qc(self):
        return self._version.epoch

    @property
    def func_f9rryrm6(self):
        return self._version.release

    @property
    def func_ko9murpu(self):
        return self._version.pre

    @property
    def func_c0838ug8(self):
        return self._version.post[1] if self._version.post else None

    @property
    def func_efoy2xuk(self):
        return self._version.dev[1] if self._version.dev else None

    @property
    def func_epohhfwn(self):
        if self._version.local:
            return '.'.join(str(x) for x in self._version.local)
        else:
            return None

    @property
    def func_vxwmdq0k(self):
        return str(self).split('+', 1)[0]

    @property
    def func_hflehsse(self):
        parts = []
        if self.epoch != 0:
            parts.append(f'{self.epoch}!')
        parts.append('.'.join(str(x) for x in self.release))
        return ''.join(parts)

    @property
    def func_muowzs2h(self):
        return self.dev is not None or self.pre is not None

    @property
    def func_wdk31it2(self):
        return self.post is not None

    @property
    def func_1ycuic44(self):
        return self.dev is not None

    @property
    def func_n2qdmwvc(self):
        return self.release[0] if len(self.release) >= 1 else 0

    @property
    def func_cebpm4pq(self):
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def func_glmk1e7i(self):
        return self.release[2] if len(self.release) >= 3 else 0


def func_697onoxb(letter, number):
    if letter:
        if number is None:
            number = 0
        letter = letter.lower()
        if letter == 'alpha':
            letter = 'a'
        elif letter == 'beta':
            letter = 'b'
        elif letter in ['c', 'pre', 'preview']:
            letter = 'rc'
        elif letter in ['rev', 'r']:
            letter = 'post'
        return letter, int(number)
    if not letter and number:
        letter = 'post'
        return letter, int(number)
    return None


_local_version_separators = re.compile('[\\._-]')


def func_ul5ljef6(local):
    if local is not None:
        return tuple(part.lower() if not part.isdigit() else int(part) for
            part in _local_version_separators.split(local))
    return None


def func_p67qbob4(epoch, release, pre, post, dev, local):
    _release = tuple(reversed(list(itertools.dropwhile(lambda x: x == 0,
        reversed(release)))))
    if pre is None and post is None and dev is not None:
        _pre = NegativeInfinity
    elif pre is None:
        _pre = Infinity
    else:
        _pre = pre
    if post is None:
        _post = NegativeInfinity
    else:
        _post = post
    if dev is None:
        _dev = Infinity
    else:
        _dev = dev
    if local is None:
        _local = NegativeInfinity
    else:
        _local = tuple((i, '') if isinstance(i, int) else (NegativeInfinity,
            i) for i in local)
    return epoch, _release, _pre, _post, _dev, _local
