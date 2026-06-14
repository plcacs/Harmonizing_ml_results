import inspect
from collections.abc import Hashable
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
from snorkel.types import DataPoint, FieldMap, HashingFunction

MapFunction = Callable[[DataPoint], Optional[DataPoint]]

def get_parameters(
    f: Callable[..., Any],
    allow_args: bool = False,
    allow_kwargs: bool = False,
) -> list[str]: ...

def is_hashable(obj: Any) -> bool: ...

def get_hashable(obj: Any) -> Hashable: ...

class BaseMapper:
    name: str
    memoize: bool
    _pre: list[BaseMapper]
    _memoize_key: HashingFunction
    _cache: dict[Hashable, Optional[DataPoint]]

    def __init__(
        self,
        name: str,
        pre: list[BaseMapper],
        memoize: bool,
        memoize_key: Optional[HashingFunction] = None,
    ) -> None: ...
    def reset_cache(self) -> None: ...
    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]: ...
    def __call__(self, x: DataPoint) -> Optional[DataPoint]: ...
    def __repr__(self) -> str: ...

class Mapper(BaseMapper):
    field_names: Dict[str, str]
    mapped_field_names: Optional[Dict[str, str]]

    def __init__(
        self,
        name: str,
        field_names: Optional[Dict[str, str]] = None,
        mapped_field_names: Optional[Dict[str, str]] = None,
        pre: Optional[list[BaseMapper]] = None,
        memoize: bool = False,
        memoize_key: Optional[HashingFunction] = None,
    ) -> None: ...
    def run(self, **kwargs: Any) -> Optional[FieldMap]: ...
    def _update_fields(self, x: DataPoint, mapped_fields: FieldMap) -> DataPoint: ...
    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]: ...

class LambdaMapper(BaseMapper):
    _f: MapFunction

    def __init__(
        self,
        name: str,
        f: MapFunction,
        pre: Optional[list[BaseMapper]] = None,
        memoize: bool = False,
        memoize_key: Optional[HashingFunction] = None,
    ) -> None: ...
    def _generate_mapped_data_point(self, x: DataPoint) -> Optional[DataPoint]: ...

class lambda_mapper:
    name: Optional[str]
    pre: Optional[list[BaseMapper]]
    memoize: bool
    memoize_key: Optional[HashingFunction]

    def __init__(
        self,
        name: Optional[str] = None,
        pre: Optional[list[BaseMapper]] = None,
        memoize: bool = False,
        memoize_key: Optional[HashingFunction] = None,
    ) -> None: ...
    def __call__(self, f: MapFunction) -> LambdaMapper: ...