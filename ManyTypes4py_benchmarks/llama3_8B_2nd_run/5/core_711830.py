from __future__ import annotations
import abc
import copy
import logging
import pprint
import re
import sys
import warnings
from collections import namedtuple
from datetime import datetime, timezone
from functools import partial, wraps
from glob import iglob
from inspect import getcallargs
from operator import attrgetter
from pathlib import Path, PurePath, PurePosixPath
from typing import TYPE_CHECKING, Any, Callable, Generic, Protocol, TypeVar, runtime_checkable
from urllib.parse import urlsplit
from cachetools import Cache, cachedmethod
from cachetools.keys import hashkey
from typing_extensions import Self

class DatasetError(Exception):
    """``DatasetError`` raised by ``AbstractDataset`` implementations
    in case of failure of input/output methods.

    ``AbstractDataset`` implementations should provide instructive
    information in case of failure.
    """
    pass

class DatasetNotFoundError(DatasetError):
    """``DatasetNotFoundError`` raised by 