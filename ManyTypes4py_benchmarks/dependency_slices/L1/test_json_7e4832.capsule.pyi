# === Third-party dependency: numpy ===
# Used symbols: array, ndarray

# === Internal dependency: pandas ===
from pandas.core.api import NA
from pandas.core.api import array
from pandas.core.api import Series

# === Internal dependency: pandas._testing ===
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.tests.extension.base ===
class ExtensionTests(BaseAccumulateTests, BaseCastingTests, BaseConstructorsTests, BaseDtypeTests, BaseGetitemTests, BaseGroupbyTests, BaseIndexTests, BaseInterfaceTests, BaseParsingTests, BaseMethodsTests, BaseMissingTests, BaseArithmeticOpsTests, BaseComparisonOpsTests, BaseUnaryOpsTests, BasePrintingTests, BaseReduceTests, BaseReshapingTests, BaseSetitemTests, Dim2CompatTests):
    ...

# === Internal dependency: pandas.tests.extension.json.array ===
class JSONDtype(ExtensionDtype): ...
class JSONArray(ExtensionArray):
    def __init__(self, values, dtype=..., copy=...): ...
    def take(self, indexer, allow_fill=..., fill_value=...): ...
def make_data(): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, param, raises