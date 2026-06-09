# === Third-party dependency: numpy ===
# Used symbols: all, arange, array, asarray, int64, integer, issubdtype, linspace, max, maximum, mean, nan, random, repeat, sum

# === Internal dependency: pandas ===
from pandas.core.api import CategoricalDtype
from pandas.core.api import isna
from pandas.core.api import Index
from pandas.core.api import CategoricalIndex
from pandas.core.api import MultiIndex
from pandas.core.api import NaT
from pandas.core.api import date_range
from pandas.core.api import Categorical
from pandas.core.api import Series
from pandas.core.api import DataFrame
from pandas.core.reshape.api import cut
from pandas.core.reshape.api import qcut

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning
from pandas._testing.asserters import assert_categorical_equal
from pandas._testing.asserters import assert_dict_equal
from pandas._testing.asserters import assert_equal
from pandas._testing.asserters import assert_frame_equal
from pandas._testing.asserters import assert_index_equal
from pandas._testing.asserters import assert_numpy_array_equal
from pandas._testing.asserters import assert_series_equal

# === Internal dependency: pandas.api.typing ===
from pandas.core.groupby import SeriesGroupBy

# === Internal dependency: pandas.tests.groupby ===
def get_groupby_method_args(name, obj): ...

# === Third-party dependency: pytest ===
# Used symbols: fail, fixture, mark, raises, skip