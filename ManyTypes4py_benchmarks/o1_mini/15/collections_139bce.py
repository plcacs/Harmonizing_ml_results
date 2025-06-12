"""
Utilities for extensions of and operations on Python collections.
"""
import io
import itertools
import types
import warnings
from collections import OrderedDict
from collections.abc import (
    Callable,
    Collection,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    Sequence,
    Set,
)
from dataclasses import fields, is_dataclass, replace
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from unittest.mock import Mock

import pydantic
from typing_extensions import TypeAlias

from prefect.utilities.annotations import BaseAnnotation as BaseAnnotation
from prefect.utilities.annotations import Quote as Quote
from prefect.utilities.annotations import quote as quote

if TYPE_CHECKING:
    pass

class AutoEnum(str, Enum):
    """
    An enum class that automatically generates value from variable names.

    This guards against common errors where variable names are updated but values are
    not.

    In addition, because AutoEnums inherit from `str`, they are automatically
    JSON-serializable.

    See https://docs.python.org/3/library/enum.html#using-automatic-values

    Example:
        