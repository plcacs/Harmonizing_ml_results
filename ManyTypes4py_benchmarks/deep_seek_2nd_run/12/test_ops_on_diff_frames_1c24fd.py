from distutils.version import LooseVersion
from itertools import product
import unittest
import pandas as pd
import numpy as np
import pyspark
from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.frame import DataFrame
from databricks.koalas.testing.utils import ReusedSQLTestCase, SQLTestUtils
from databricks.koalas.typedef.typehints import extension_dtypes, extension_dtypes_available, extension_float_dtypes_available, extension_object_dtypes_available
from typing import Any, Dict, List, Optional, Tuple, Union

class OpsOnDiffFramesEnabledTest(ReusedSQLTestCase, SQLTestUtils):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        set_option('compute.ops_on_diff_frames', True)

    @classmethod
    def tearDownClass(cls) -> None:
        reset_option('compute.ops_on_diff_frames')
        super().tearDownClass()

    @property
    def pdf1(self) -> pd.DataFrame:
        return pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [4, 5, 6, 3, 2, 1, 0, 0, 0]}, index=[0, 1, 3, 5, 6, 8, 9, 10, 11])

    @property
    def pdf2(self) -> pd.DataFrame:
        return pd.DataFrame({'a': [9, 8, 7, 6, 5, 4, 3, 2, 1], 'b': [0, 0, 0, 4, 5, 6, 1, 2, 3]}, index=list(range(9)))

    @property
    def pdf3(self) -> pd.DataFrame:
        return pd.DataFrame({'b': [1, 1, 1, 1, 1, 1, 1, 1, 1], 'c': [1, 1, 1, 1, 1, 1, 1, 1, 1]}, index=list(range(9)))

    @property
    def pdf4(self) -> pd.DataFrame:
        return pd.DataFrame({'e': [2, 2, 2, 2, 2, 2, 2, 2, 2], 'f': [2, 2, 2, 2, 2, 2, 2, 2, 2]}, index=list(range(9)))

    @property
    def pdf5(self) -> pd.DataFrame:
        return pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9], 'b': [4, 5, 6, 3, 2, 1, 0, 0, 0], 'c': [4, 5, 6, 3, 2, 1, 0, 0, 0]}, index=[0, 1, 3, 5, 6, 8, 9, 10, 11]).set_index(['a', 'b'])

    @property
    def pdf6(self) -> pd.DataFrame:
        return pd.DataFrame({'a': [9, 8, 7, 6, 5, 4, 3, 2, 1], 'b': [0, 0, 0, 4, 5, 6, 1, 2, 3], 'c': [9, 8, 7, 6, 5, 4, 3, 2, 1], 'e': [4, 5, 6, 3, 2, 1, 0, 0, 0]}, index=list(range(9))).set_index(['a', 'b'])

    @property
    def pser1(self) -> pd.Series:
        midx = pd.MultiIndex([['lama', 'cow', 'falcon', 'koala'], ['speed', 'weight', 'length', 'power']], [[0, 3, 1, 1, 1, 2, 2, 2], [0, 2, 0, 3, 2, 0, 1, 3]])
        return pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1], index=midx)

    @property
    def pser2(self) -> pd.Series:
        midx = pd.MultiIndex([['lama', 'cow', 'falcon'], ['speed', 'weight', 'length']], [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        return pd.Series([-45, 200, -1.2, 30, -250, 1.5, 320, 1, -0.3], index=midx)

    @property
    def pser3(self) -> pd.Series:
        midx = pd.MultiIndex([['koalas', 'cow', 'falcon'], ['speed', 'weight', 'length']], [[0, 0, 0, 1, 1, 1, 2, 2, 2], [1, 1, 2, 0, 0, 2, 2, 2, 1]])
        return pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1, 0.3], index=midx)

    @property
    def kdf1(self) -> DataFrame:
        return ks.from_pandas(self.pdf1)

    @property
    def kdf2(self) -> DataFrame:
        return ks.from_pandas(self.pdf2)

    @property
    def kdf3(self) -> DataFrame:
        return ks.from_pandas(self.pdf3)

    @property
    def kdf4(self) -> DataFrame:
        return ks.from_pandas(self.pdf4)

    @property
    def kdf5(self) -> DataFrame:
        return ks.from_pandas(self.pdf5)

    @property
    def kdf6(self) -> DataFrame:
        return ks.from_pandas(self.pdf6)

    @property
    def kser1(self) -> ks.Series:
        return ks.from_pandas(self.pser1)

    @property
    def kser2(self) -> ks.Series:
        return ks.from_pandas(self.pser2)

    @property
    def kser3(self) -> ks.Series:
        return ks.from_pandas(self.pser3)

    def test_ranges(self) -> None:
        self.assert_eq((ks.range(10) + ks.range(10)).sort_index(), (ks.DataFrame({'id': list(range(10))}) + ks.DataFrame({'id': list(range(10))})).sort_index())

    def test_no_matched_index(self) -> None:
        with self.assertRaisesRegex(ValueError, 'Index names must be exactly matched'):
            ks.DataFrame({'a': [1, 2, 3]}).set_index('a') + ks.DataFrame({'b': [1, 2, 3]}).set_index('b')

    def test_arithmetic(self) -> None:
        self._test_arithmetic_frame(self.pdf1, self.pdf2, check_extension=False)
        self._test_arithmetic_series(self.pser1, self.pser2, check_extension=False)

    @unittest.skipIf(not extension_dtypes_available, 'pandas extension dtypes are not available')
    def test_arithmetic_extension_dtypes(self) -> None:
        self._test_arithmetic_frame(self.pdf1.astype('Int64'), self.pdf2.astype('Int64'), check_extension=True)
        self._test_arithmetic_series(self.pser1.astype(int).astype('Int64'), self.pser2.astype(int).astype('Int64'), check_extension=True)

    @unittest.skipIf(not extension_float_dtypes_available, 'pandas extension float dtypes are not available')
    def test_arithmetic_extension_float_dtypes(self) -> None:
        self._test_arithmetic_frame(self.pdf1.astype('Float64'), self.pdf2.astype('Float64'), check_extension=True)
        self._test_arithmetic_series(self.pser1.astype('Float64'), self.pser2.astype('Float64'), check_extension=True)

    def _test_arithmetic_frame(self, pdf1: pd.DataFrame, pdf2: pd.DataFrame, *, check_extension: bool) -> None:
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)

        def assert_eq(actual: Any, expected: Any) -> None:
            if LooseVersion('1.1') <= LooseVersion(pd.__version__) < LooseVersion('1.2.2'):
                self.assert_eq(actual, expected, check_exact=not check_extension)
                if check_extension:
                    if isinstance(actual, DataFrame):
                        for dtype in actual.dtypes:
                            self.assertTrue(isinstance(dtype, extension_dtypes))
                    else:
                        self.assertTrue(isinstance(actual.dtype, extension_dtypes))
            else:
                self.assert_eq(actual, expected)
        assert_eq((kdf1.a - kdf2.b).sort_index(), (pdf1.a - pdf2.b).sort_index())
        assert_eq((kdf1.a * kdf2.a).sort_index(), (pdf1.a * pdf2.a).sort_index())
        if check_extension and (not extension_float_dtypes_available):
            self.assert_eq((kdf1['a'] / kdf2['a']).sort_index(), (pdf1['a'] / pdf2['a']).sort_index())
        else:
            assert_eq((kdf1['a'] / kdf2['a']).sort_index(), (pdf1['a'] / pdf2['a']).sort_index())
        assert_eq((kdf1 + kdf2).sort_index(), (pdf1 + pdf2).sort_index())
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b')])
        kdf1.columns = columns
        kdf2.columns = columns
        pdf1.columns = columns
        pdf2.columns = columns
        assert_eq((kdf1['x', 'a'] - kdf2['x', 'b']).sort_index(), (pdf1['x', 'a'] - pdf2['x', 'b']).sort_index())
        assert_eq((kdf1['x', 'a'] - kdf2['x']['b']).sort_index(), (pdf1['x', 'a'] - pdf2['x']['b']).sort_index())
        assert_eq((kdf1['x']['a'] - kdf2['x', 'b']).sort_index(), (pdf1['x']['a'] - pdf2['x', 'b']).sort_index())
        assert_eq((kdf1 + kdf2).sort_index(), (pdf1 + pdf2).sort_index())

    def _test_arithmetic_series(self, pser1: pd.Series, pser2: pd.Series, *, check_extension: bool) -> None:
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)

        def assert_eq(actual: Any, expected: Any) -> None:
            if LooseVersion('1.1') <= LooseVersion(pd.__version__) < LooseVersion('1.2.2'):
                self.assert_eq(actual, expected, check_exact=not check_extension)
                if check_extension:
                    self.assertTrue(isinstance(actual.dtype, extension_dtypes))
            else:
                self.assert_eq(actual, expected)
        assert_eq((kser1 + kser2).sort_index(), (pser1 + pser2).sort_index())
        assert_eq((kser1 - kser2).sort_index(), (pser1 - pser2).sort_index())
        assert_eq((kser1 * kser2).sort_index(), (pser1 * pser2).sort_index())
        if check_extension and (not extension_float_dtypes_available):
            self.assert_eq((kser1 / kser2).sort_index(), (pser1 / pser2).sort_index())
        else:
            assert_eq((kser1 / kser2).sort_index(), (pser1 / pser2).sort_index())

    def test_arithmetic_chain(self) -> None:
        self._test_arithmetic_chain_frame(self.pdf1, self.pdf2, self.pdf3, check_extension=False)
        self._test_arithmetic_chain_series(self.pser1, self.pser2, self.pser3, check_extension=False)

    @unittest.skipIf(not extension_dtypes_available, 'pandas extension dtypes are not available')
    def test_arithmetic_chain_extension_dtypes(self) -> None:
        self._test_arithmetic_chain_frame(self.pdf1.astype('Int64'), self.pdf2.astype('Int64'), self.pdf3.astype('Int64'), check_extension=True)
        self._test_arithmetic_chain_series(self.pser1.astype(int).astype('Int64'), self.pser2.astype(int).astype('Int64'), self.pser3.astype(int).astype('Int64'), check_extension=True)

    @unittest.skipIf(not extension_float_dtypes_available, 'pandas extension float dtypes are not available')
    def test_arithmetic_chain_extension_float_dtypes(self) -> None:
        self._test_arithmetic_chain_frame(self.pdf1.astype('Float64'), self.pdf2.astype('Float64'), self.pdf3.astype('Float64'), check_extension=True)
        self._test_arithmetic_chain_series(self.pser1.astype('Float64'), self.pser2.astype('Float64'), self.pser3.astype('Float64'), check_extension=True)

    def _test_arithmetic_chain_frame(self, pdf1: pd.DataFrame, pdf2: pd.DataFrame, pdf3: pd.DataFrame, *, check_extension: bool) -> None:
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)
        kdf3 = ks.from_pandas(pdf3)
        common_columns = set(kdf1.columns).intersection(kdf2.columns).intersection(kdf3.columns)

        def assert_eq(actual: Any, expected: Any) -> None:
            if LooseVersion('1.1') <= LooseVersion(pd.__version__) < LooseVersion('1.2.2'):
                self.assert_eq(actual, expected, check_exact=not check_extension)
                if check_extension:
                    if isinstance(actual, DataFrame):
                        for column, dtype in zip(actual.columns, actual.dtypes):
                            if column in common_columns:
                                self.assertTrue(isinstance(dtype, extension_dtypes))
                            else:
                                self.assertFalse(isinstance(dtype, extension_dtypes))
                    else:
                        self.assertTrue(isinstance(actual.dtype, extension_dtypes))
            else:
                self.assert_eq(actual, expected)
        assert_eq((kdf1.a - kdf2.b - kdf3.c).sort_index(), (pdf1.a - pdf2.b - pdf3.c).sort_index())
        assert_eq((kdf1.a * (kdf2.a * kdf3.c)).sort_index(), (pdf1.a * (pdf2.a * pdf3.c)).sort_index())
        if check_extension and (not extension_float_dtypes_available):
            self.assert_eq((kdf1['a'] / kdf2['a'] / kdf3['c']).sort_index(), (pdf1['a'] / pdf2['a'] / pdf3['c']).sort_index())
        else:
            assert_eq((kdf1['a'] / kdf2['a'] / kdf3['c']).sort_index(), (pdf1['a'] / pdf2['a'] / pdf3['c']).sort_index())
        if check_extension and LooseVersion('1.0') <= LooseVersion(pd.__version__) < LooseVersion('1.1'):
            self.assert_eq((kdf1 + kdf2 - kdf3).sort_index(), (pdf1 + pdf2 - pdf3).sort_index(), almost=True)
        else:
            assert_eq((kdf1 + kdf2 - kdf3).sort_index(), (pdf1 + pdf2 - pdf3).sort_index())
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b')])
        kdf1.columns = columns
        kdf2.columns = columns
        pdf1.columns = columns
        pdf2.columns = columns
        columns = pd.MultiIndex.from_tuples([('x', 'b'), ('y', 'c')])
        kdf3.columns = columns
        pdf3.columns = columns
        common_columns = set(kdf1.columns).intersection(kdf2.columns).intersection(kdf3.columns)
        assert_eq((kdf1['x', 'a'] - kdf2['x', 'b'] - kdf3['y', 'c']).sort_index(), (pdf1['x', 'a'] - pdf2['x', 'b'] - pdf3['y', 'c']).sort_index())
        assert_eq((kdf1['x', 'a'] * (kdf2['x', 'b'] * kdf3['y', 'c'])).sort_index(), (pdf1['x', 'a'] * (pdf2['x', 'b'] * pdf3['y', 'c'])).sort_index())
        if check_extension and LooseVersion('1.0') <= LooseVersion(pd.__version__) < Loose