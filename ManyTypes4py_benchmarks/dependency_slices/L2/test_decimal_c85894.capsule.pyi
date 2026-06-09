from typing import Any

# === Third-party dependency: numpy ===
# Used symbols: abs, add, array, asarray, exp, int64, negative, positive, random, sum

# === Internal dependency: pandas ===
# re-export: from pandas.core.api import MultiIndex
# re-export: from pandas.core.api import array
# re-export: from pandas.core.api import Series
# re-export: from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
# re-export: from pandas._testing._warnings import assert_produces_warning
# re-export: from pandas._testing.asserters import assert_almost_equal
# re-export: from pandas._testing.asserters import assert_equal
# re-export: from pandas._testing.asserters import assert_extension_array_equal
# re-export: from pandas._testing.asserters import assert_frame_equal
# re-export: from pandas._testing.asserters import assert_numpy_array_equal
# re-export: from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.tests.extension.base ===
class ExtensionTests(BaseAccumulateTests, BaseCastingTests, BaseConstructorsTests, BaseDtypeTests, BaseGetitemTests, BaseGroupbyTests, BaseIndexTests, BaseInterfaceTests, BaseParsingTests, BaseMethodsTests, BaseMissingTests, BaseArithmeticOpsTests, BaseComparisonOpsTests, BaseUnaryOpsTests, BasePrintingTests, BaseReduceTests, BaseReshapingTests, BaseSetitemTests, Dim2CompatTests):
    ...

# === Internal dependency: pandas.tests.extension.decimal.array ===
class DecimalDtype(ExtensionDtype):
    def __init__(self, context = ...) -> None: ...
class DecimalArray(OpsMixin, ExtensionScalarOpsMixin, ExtensionArray):
    def __init__(self, values, dtype = ..., copy = ..., context = ...) -> None: ...
    def dtype(self) -> Any: ...
    def take(self, indexer, allow_fill = ..., fill_value = ...) -> Any: ...
    def nbytes(self) -> int: ...
    def _na_value(self) -> Any: ...
def to_decimal(values, context = ...) -> Any: ...
def make_data() -> Any: ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises