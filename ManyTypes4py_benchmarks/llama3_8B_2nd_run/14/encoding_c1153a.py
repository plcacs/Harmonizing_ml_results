from __future__ import annotations
from collections import defaultdict
from collections.abc import Hashable, Iterable
import itertools
from typing import TYPE_CHECKING, NpDtype

def get_dummies(data: DataFrame | Series | array_like, 
               prefix: str | list[str] | dict[str, str] = None, 
               prefix_sep: str | list[str] | dict[str, str] = '_', 
               dummy_na: bool = False, 
               columns: list[str] | None = None, 
               sparse: bool = False, 
               drop_first: bool = False, 
               dtype: NpDtype | None = None) -> DataFrame:
    ...

def _get_dummies_1d(data: array_like, 
                    prefix: str | None, 
                    prefix_sep: str | None, 
                    dummy_na: bool, 
                    sparse: bool, 
                    drop_first: bool, 
                    dtype: NpDtype | None) -> DataFrame:
    ...

def from_dummies(data: DataFrame, 
                sep: str | None = None, 
                default_category: Hashable | dict[Hashable, Hashable] | None = None) -> DataFrame:
    ...
