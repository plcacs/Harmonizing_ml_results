# === Third-party dependency: numpy ===
# Used symbols: array, mean, sum

# === Internal dependency: pandas ===
from pandas.core.api import Grouper
from pandas.core.api import Categorical
from pandas.core.api import Series
from pandas.core.api import DataFrame

# === Internal dependency: pandas._config ===
def using_string_dtype(): ...

# === Internal dependency: pandas._testing ===
from pandas._testing._warnings import assert_produces_warning

# === Internal dependency: pandas.tests.groupby ===
def get_groupby_method_args(name, obj): ...

# === Third-party dependency: pytest ===
# Used symbols: fixture, mark, raises