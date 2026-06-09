# === Third-party dependency: numpy ===
# Used symbols: abs, add, array, asarray, exp, int64, negative, positive, random, sum

# === Internal dependency: pandas ===
from pandas.core.api import MultiIndex
from pandas.core.api import array
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_almost_equal
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_extension_array_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.tests.extension.base ===
class ExtensionTests(BaseAccumulateTests, BaseCastingTests, BaseConstructorsTests, BaseDtypeTests, BaseGetitemTests, BaseGroupbyTests, BaseIndexTests, BaseInterfaceTests, BaseParsingTests, BaseMethodsTests, BaseMissingTests, BaseArithmeticOpsTests, BaseComparisonOpsTests, BaseUnaryOpsTests, BasePrintingTests, BaseReduceTests, BaseReshapingTests, BaseSetitemTests, Dim2CompatTests):
    ...

# === Internal dependency: pandas.tests.extension.decimal.array ===
class DecimalDtype(ExtensionDtype):
    def __init__(self, context=...): ...
class DecimalArray(OpsMixin, ExtensionScalarOpsMixin, ExtensionArray):
    def __init__(self, values, dtype=..., copy=..., context=...): ...
    def dtype(self): ...
    def take(self, indexer, allow_fill=..., fill_value=...): ...
    def nbytes(self): ...
    def _na_value(self): ...
def to_decimal(values, context=...): ...
def make_data(): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises