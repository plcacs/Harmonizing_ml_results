import functools
import shutil
import tempfile
import unittest
import warnings
from contextlib import contextmanager
from distutils.version import LooseVersion
import pandas as pd
from pandas.api.types import is_list_like
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
from databricks import koalas as ks
from databricks.koalas.frame import DataFrame
from databricks.koalas.indexes import Index
from databricks.koalas.series import Series
from databricks.koalas.utils import default_session, sql_conf as sqlc, SPARK_CONF_ARROW_ENABLED

class SQLTestUtils(object):
    @contextmanager
    def sql_conf(self, pairs: dict):
        ...

    @contextmanager
    def database(self, *databases: str):
        ...

    @contextmanager
    def table(self, *tables: str):
        ...

    @contextmanager
    def tempView(self, *views: str):
        ...

    @contextmanager
    def function(self, *functions: str):
        ...

class ReusedSQLTestCase(unittest.TestCase, SQLTestUtils):
    @classmethod
    def setUpClass(cls):
        ...

    @classmethod
    def tearDownClass(cls):
        ...

    def assertPandasEqual(self, left, right, check_exact: bool = True):
        ...

    def assertPandasAlmostEqual(self, left, right):
        ...

    def assert_eq(self, left, right, check_exact: bool = True, almost: bool = False):
        ...

    @staticmethod
    def _to_pandas(obj):
        ...

class TestUtils(object):
    @contextmanager
    def temp_dir(self):
        ...

    @contextmanager
    def temp_file(self):
        ...

class ComparisonTestBase(ReusedSQLTestCase):
    @property
    def kdf(self) -> DataFrame:
        ...

    @property
    def pdf(self) -> pd.DataFrame:
        ...

def compare_both(f=None, almost: bool = True):
    ...

@contextmanager
def assert_produces_warning(expected_warning=Warning, filter_level='always', check_stacklevel: bool = True, raise_on_extra_warnings: bool = True):
    ...
