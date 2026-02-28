"""This module provides a set of classes which underpin the data loading and
saving functionality provided by ``kedro.io``.
"""
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
from kedro.utils import load_obj
if TYPE_CHECKING:
    import os
    from kedro.io.catalog_config_resolver import CatalogConfigResolver, Patterns

VERSION_FORMAT = '%Y-%m-%dT%H.%M.%S.%fZ'
VERSIONED_FLAG_KEY = 'versioned'
VERSION_KEY = 'version'
HTTP_PROTOCOLS = ('http', 'https')
PROTOCOL_DELIMITER = '://'
CLOUD_PROTOCOLS = ('abfs', 'abfss', 'adl', 'gcs', 'gdrive', 'gs', 'oci', 'oss', 's3', 's3a', 's3n')
TYPE_KEY = 'type'

class DatasetError(Exception):
    """``DatasetError`` raised by ``AbstractDataset`` implementations
    in case of failure of input/output methods.

    ``AbstractDataset`` implementations should provide instructive
    information in case of failure.
    """
    pass

class DatasetNotFoundError(DatasetError):
    """``DatasetNotFoundError`` raised by 