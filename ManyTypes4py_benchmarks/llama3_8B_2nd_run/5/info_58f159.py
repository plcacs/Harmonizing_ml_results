from __future__ import annotations
from abc import ABC, abstractmethod
import sys
from textwrap import dedent
from typing import TYPE_CHECKING, Union, Sequence, Iterator, Mapping
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
from pandas._typing import Dtype, WriteBuffer
from pandas import DataFrame, Index, Series

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pandas import DataFrame, Index, Series

class _BaseInfo(ABC):
    ...

class DataFrameInfo(_BaseInfo):
    ...

class SeriesInfo(_BaseInfo):
    ...

class _InfoPrinterAbstract:
    ...

class _DataFrameInfoPrinter(_InfoPrinterAbstract):
    ...

class _SeriesInfoPrinter(_InfoPrinterAbstract):
    ...

class _TableBuilderAbstract(ABC):
    ...

class _DataFrameTableBuilder(_TableBuilderAbstract):
    ...

class _TableBuilderVerboseMixin(_TableBuilderAbstract):
    ...

class _DataFrameTableBuilderVerbose(_DataFrameTableBuilder, _TableBuilderVerboseMixin):
    ...

class _SeriesTableBuilder(_TableBuilderAbstract):
    ...

class _SeriesTableBuilderNonVerbose(_SeriesTableBuilder):
    ...

class _SeriesTableBuilderVerbose(_SeriesTableBuilder, _TableBuilderVerboseMixin):
    ...

def _get_dataframe_dtype_counts(df: DataFrame) -> Mapping[Dtype, int]:
    ...
