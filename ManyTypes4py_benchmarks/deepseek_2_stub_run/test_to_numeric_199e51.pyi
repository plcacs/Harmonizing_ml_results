```python
import decimal
from typing import Any, Literal, overload
from typing_extensions import TypeAlias

import numpy as np
import pandas as pd
from pandas import ArrowDtype, DataFrame, Index, Series
from pandas._typing import DtypeBackend, Scalar

_errors_literal: TypeAlias = Literal["raise", "coerce"]
_downcast_literal: TypeAlias = Literal["integer", "signed", "unsigned", "float"]

@overload
def to_numeric(
    arg: Series,
    errors: _errors_literal = ...,
    downcast: _downcast_literal | None = ...,
    dtype_backend: DtypeBackend = ...,
) -> Series: ...

@overload
def to_numeric(
    arg: Index,
    errors: _errors_literal = ...,
    downcast: _downcast_literal | None = ...,
    dtype_backend: DtypeBackend = ...,
) -> Index: ...

@overload
def to_numeric(
    arg: list[Any],
    errors: _errors_literal = ...,
    downcast: _downcast_literal | None = ...,
    dtype_backend: DtypeBackend = ...,
) -> np.ndarray: ...

@overload
def to_numeric(
    arg: tuple[Any, ...],
    errors: _errors_literal = ...,
    downcast: _downcast_literal | None = ...,
    dtype_backend: DtypeBackend = ...,
) -> np.ndarray: ...

@overload
def to_numeric(
    arg: pd.Timestamp | pd.Timedelta,
    errors: _errors_literal = ...,
    downcast: _downcast_literal | None = ...,
    dtype_backend: DtypeBackend = ...,
) -> float: ...

@overload
def to_numeric(
    arg: Scalar,
    errors: _errors_literal = ...,
    downcast: _downcast_literal | None = ...,
    dtype_backend: DtypeBackend = ...,
) -> Any: ...

@overload
def to_numeric(
    arg: Any,
    errors: _errors_literal = ...,
    downcast: _downcast_literal | None = ...,
    dtype_backend: DtypeBackend = ...,
) -> Any: ...

def to_numeric(
    arg: Any,
    errors: _errors_literal = ...,
    downcast: _downcast_literal | None = ...,
    dtype_backend: DtypeBackend = ...,
) -> Any: ...

__all__: list[str] = ...
```