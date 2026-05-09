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
    def kdf1(self) -> ks.DataFrame:
        return ks.from_pandas(self.pdf1)

    @property
    def kdf2(self) -> ks.DataFrame:
        return ks.from_pandas(self.pdf2)

    @property
    def kdf3(self) -> ks.DataFrame:
        return ks.from_pandas(self.pdf3)

    @property
    def kdf4(self) -> ks.DataFrame:
        return ks.from_pandas(self.pdf4)

    @property
    def kdf5(self) -> ks.DataFrame:
        return ks.from_pandas(self.pdf5)

    @property
    def kdf6(self) -> ks.DataFrame:
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

        def assert_eq(actual: ks.DataFrame, expected: pd.DataFrame) -> None:
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

        def assert_eq(actual: ks.Series, expected: pd.Series) -> None:
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

        def assert_eq(actual: ks.DataFrame, expected: pd.DataFrame) -> None:
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
        if check_extension and LooseVersion('1.0') <= LooseVersion(pd.__version__) < LooseVersion('1.1'):
            self.assert_eq((kdf1 + kdf2 - kdf3).sort_index(), (pdf1 + pdf2 - pdf3).sort_index(), almost=True)
        else:
            assert_eq((kdf1 + kdf2 - kdf3).sort_index(), (pdf1 + pdf2 - pdf3).sort_index())

    def _test_arithmetic_chain_series(self, pser1: pd.Series, pser2: pd.Series, pser3: pd.Series, *, check_extension: bool) -> None:
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)
        kser3 = ks.from_pandas(pser3)

        def assert_eq(actual: ks.Series, expected: pd.Series) -> None:
            if LooseVersion('1.1') <= LooseVersion(pd.__version__) < LooseVersion('1.2.2'):
                self.assert_eq(actual, expected, check_exact=not check_extension)
                if check_extension:
                    self.assertTrue(isinstance(actual.dtype, extension_dtypes))
            else:
                self.assert_eq(actual, expected)
        assert_eq((kser1 + kser2 - kser3).sort_index(), (pser1 + pser2 - pser3).sort_index())
        assert_eq((kser1 * kser2 * kser3).sort_index(), (pser1 * pser2 * pser3).sort_index())
        if check_extension and (not extension_float_dtypes_available):
            if LooseVersion(pd.__version__) >= LooseVersion('1.0'):
                self.assert_eq((kser1 - kser2 / kser3).sort_index(), (pser1 - pser2 / pser3).sort_index())
            else:
                expected = pd.Series([249.0, np.nan, 0.0, 0.88, np.nan, np.nan, np.nan, np.nan, np.nan, -np.inf] + [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan], index=pd.MultiIndex([['cow', 'falcon', 'koala', 'koalas', 'lama'], ['length', 'power', 'speed', 'weight']], [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4], [0, 1, 2, 2, 3, 0, 0, 1, 2, 3, 0, 0, 3, 3, 0, 2, 3]]))
                self.assert_eq((kser1 - kser2 / kser3).sort_index(), expected)
        else:
            assert_eq((kser1 - kser2 / kser3).sort_index(), (pser1 - pser2 / pser3).sort_index())
        assert_eq((kser1 + kser2 * kser3).sort_index(), (pser1 + pser2 * pser3).sort_index())

    def test_mod(self) -> None:
        pser = pd.Series([100, None, -300, None, 500, -700])
        pser_other = pd.Series([-150] * 6)
        kser = ks.from_pandas(pser)
        kser_other = ks.from_pandas(pser_other)
        self.assert_eq(kser.mod(kser_other).sort_index(), pser.mod(pser_other))
        self.assert_eq(kser.mod(kser_other).sort_index(), pser.mod(pser_other))
        self.assert_eq(kser.mod(kser_other).sort_index(), pser.mod(pser_other))

    def test_rmod(self) -> None:
        pser = pd.Series([100, None, -300, None, 500, -700])
        pser_other = pd.Series([-150] * 6)
        kser = ks.from_pandas(pser)
        kser_other = ks.from_pandas(pser_other)
        self.assert_eq(kser.rmod(kser_other).sort_index(), pser.rmod(pser_other))
        self.assert_eq(kser.rmod(kser_other).sort_index(), pser.rmod(pser_other))
        self.assert_eq(kser.rmod(kser_other).sort_index(), pser.rmod(pser_other))

    def test_getitem_boolean_series(self) -> None:
        pdf1 = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [100, 200, 300, 400, 500]}, index=[20, 10, 30, 0, 50])
        pdf2 = pd.DataFrame({'A': [0, -1, -2, -3, -4], 'B': [-100, -200, -300, -400, -500]}, index=[0, 30, 10, 20, 50])
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)
        self.assert_eq(pdf1[pdf2.A > -3].sort_index(), kdf1[kdf2.A > -3].sort_index())
        self.assert_eq(pdf1.A[pdf2.A > -3].sort_index(), kdf1.A[kdf2.A > -3].sort_index())
        self.assert_eq((pdf1.A + 1)[pdf2.A > -3].sort_index(), (kdf1.A + 1)[kdf2.A > -3].sort_index())

    def test_loc_getitem_boolean_series(self) -> None:
        pdf1 = pd.DataFrame({'A': [0, 1, 2, 3, 4], 'B': [100, 200, 300, 400, 500]}, index=[20, 10, 30, 0, 50])
        pdf2 = pd.DataFrame({'A': [0, -1, -2, -3, -4], 'B': [-100, -200, -300, -400, -500]}, index=[20, 10, 30, 0, 50])
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)
        self.assert_eq(pdf1.loc[pdf2.A > -3].sort_index(), kdf1.loc[kdf2.A > -3].sort_index())
        self.assert_eq(pdf1.A.loc[pdf2.A > -3].sort_index(), kdf1.A.loc[kdf2.A > -3].sort_index())
        self.assert_eq((pdf1.A + 1).loc[pdf2.A > -3].sort_index(), (kdf1.A + 1).loc[kdf2.A > -3].sort_index())

    def test_bitwise(self) -> None:
        pser1 = pd.Series([True, False, True, False, np.nan, np.nan, True, False, np.nan])
        pser2 = pd.Series([True, False, False, True, True, False, np.nan, np.nan, np.nan])
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)
        self.assert_eq(pser1 | pser2, (kser1 | kser2).sort_index())
        self.assert_eq(pser1 & pser2, (kser1 & kser2).sort_index())
        pser1 = pd.Series([True, False, np.nan], index=list('ABC'))
        pser2 = pd.Series([False, True, np.nan], index=list('DEF'))
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)
        self.assert_eq(pser1 | pser2, (kser1 | kser2).sort_index())
        self.assert_eq(pser1 & pser2, (kser1 & kser2).sort_index())

    @unittest.skipIf(not extension_object_dtypes_available, 'pandas extension object dtypes are not available')
    def test_bitwise_extension_dtype(self) -> None:

        def assert_eq(actual: ks.Series, expected: pd.Series) -> None:
            if LooseVersion('1.1') <= LooseVersion(pd.__version__) < LooseVersion('1.2.2'):
                self.assert_eq(actual, expected, check_exact=False)
                self.assertTrue(isinstance(actual.dtype, extension_dtypes))
            else:
                self.assert_eq(actual, expected)
        pser1 = pd.Series([True, False, True, False, np.nan, np.nan, True, False, np.nan], dtype='boolean')
        pser2 = pd.Series([True, False, False, True, True, False, np.nan, np.nan, np.nan], dtype='boolean')
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)
        assert_eq((kser1 | kser2).sort_index(), pser1 | pser2)
        assert_eq((kser1 & kser2).sort_index(), pser1 & pser2)
        pser1 = pd.Series([True, False, np.nan], index=list('ABC'), dtype='boolean')
        pser2 = pd.Series([False, True, np.nan], index=list('DEF'), dtype='boolean')
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)
        assert_eq((kser1 | kser2).sort_index(), pd.Series([True, None, None, None, True, None], index=list('ABCDEF'), dtype='boolean'))
        assert_eq((kser1 & kser2).sort_index(), pd.Series([None, False, None, False, None, None], index=list('ABCDEF'), dtype='boolean'))

    def test_concat_column_axis(self) -> None:
        pdf1 = pd.DataFrame({'A': [0, 2, 4], 'B': [1, 3, 5]}, index=[1, 2, 3])
        pdf1.columns.names = ['AB']
        pdf2 = pd.DataFrame({'C': [1, 2, 3], 'D': [4, 5, 6]}, index=[1, 3, 5])
        pdf2.columns.names = ['CD']
        kdf1 = ks.from_pandas(pdf1)
        kdf2 = ks.from_pandas(pdf2)
        kdf3 = kdf1.copy()
        kdf4 = kdf2.copy()
        pdf3 = pdf1.copy()
        pdf4 = pdf2.copy()
        columns = pd.MultiIndex.from_tuples([('X', 'A'), ('X', 'B')], names=['X', 'AB'])
        pdf3.columns = columns
        kdf3.columns = columns
        columns = pd.MultiIndex.from_tuples([('X', 'C'), ('X', 'D')], names=['Y', 'CD'])
        pdf4.columns = columns
        kdf4.columns = columns
        pdf5 = pd.DataFrame({'A': [0, 2, 4], 'B': [1, 3, 5]}, index=[1, 2, 3])
        pdf6 = pd.DataFrame({'C': [1, 2, 3]}, index=[1, 3, 5])
        kdf5 = ks.from_pandas(pdf5)
        kdf6 = ks.from_pandas(pdf6)
        ignore_indexes = [True, False]
        joins = ['inner', 'outer']
        objs = [([kdf1.A, kdf2.C], [pdf1.A, pdf2.C]), ([kdf1.A, kdf2], [pdf1.A, pdf2]), ([kdf1.A, kdf2.C], [pdf1.A, pdf2.C]), ([kdf3['X', 'A'], kdf4['X', 'C']], [pdf3['X', 'A'], pdf4['X', 'C']]), ([kdf3, kdf4['X', 'C']], [pdf3, pdf4['X', 'C']]), ([kdf3['X', 'A'], kdf4], [pdf3['X', 'A'], pdf4]), ([kdf3, kdf4], [pdf3, pdf4]), ([kdf5, kdf6], [pdf5, pdf6]), ([kdf6, kdf5], [pdf6, pdf5])]
        for ignore_index, join in product(ignore_indexes, joins):
            for i, (kdfs, pdfs) in enumerate(objs):
                with self.subTest(ignore_index=ignore_index, join=join, pdfs=pdfs, pair=i):
                    actual = ks.concat(kdfs, axis=1, ignore_index=ignore_index, join=join)
                    expected = pd.concat(pdfs, axis=1, ignore_index=ignore_index, join=join)
                    self.assert_eq(repr(actual.sort_values(list(actual.columns)).reset_index(drop=True)), repr(expected.sort_values(list(expected.columns)).reset_index(drop=True)))

    def test_combine_first(self) -> None:
        pser1 = pd.Series({'falcon': 330.0, 'eagle': 160.0})
        pser2 = pd.Series({'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0})
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)
        self.assert_eq(kser1.combine_first(kser2).sort_index(), pser1.combine_first(pser2).sort_index())
        with self.assertRaisesRegex(ValueError, '`combine_first` only allows `Series` for parameter `other`'):
            kser1.combine_first(50)
        kser1.name = ('X', 'A')
        kser2.name = ('Y', 'B')
        pser1.name = ('X', 'A')
        pser2.name = ('Y', 'B')
        self.assert_eq(kser1.combine_first(kser2).sort_index(), pser1.combine_first(pser2).sort_index())
        midx1 = pd.MultiIndex([['lama', 'cow', 'falcon', 'koala'], ['speed', 'weight', 'length', 'power']], [[0, 3, 1, 1, 1, 2, 2, 2], [0, 2, 0, 3, 2, 0, 1, 3]])
        midx2 = pd.MultiIndex([['lama', 'cow', 'falcon'], ['speed', 'weight', 'length']], [[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]])
        pser1 = pd.Series([45, 200, 1.2, 30, 250, 1.5, 320, 1], index=midx1)
        pser2 = pd.Series([-45, 200, -1.2, 30, -250, 1.5, 320, 1, -0.3], index=midx2)
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)
        self.assert_eq(kser1.combine_first(kser2).sort_index(), pser1.combine_first(pser2).sort_index())
        pdf = pd.DataFrame({'A': {'falcon': 330.0, 'eagle': 160.0}, 'B': {'falcon': 345.0, 'eagle': 200.0, 'duck': 30.0}})
        pser1 = pdf.A
        pser2 = pdf.B
        kser1 = ks.from_pandas(pser1)
        kser2 = ks.from_pandas(pser2)
        self.assert_eq(kser1.combine_first(kser2).sort_index(), pser1.combine_first(pser2).sort_index())
        kser1.name = ('X', 'A')
        kser2.name = ('Y', 'B')
        pser1.name = ('X', 'A')
        pser2.name = ('Y', 'B')
        self.assert_eq(kser1.combine_first(kser2).sort_index(), pser1.combine_first(pser2).sort_index())

    def test_insert(self) -> None:
        pdf = pd.DataFrame([1, 2, 3])
        kdf = ks.from_pandas(pdf)
        pser = pd.Series([4, 5, 6])
        kser = ks.from_pandas(pser)
        kdf.insert(1, 'y', kser)
        pdf.insert(1, 'y', pser)
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        pdf = pd.DataFrame([1, 2, 3], index=[10, 20, 30])
        kdf = ks.from_pandas(pdf)
        pser = pd.Series([4, 5, 6])
        kser = ks.from_pandas(pser)
        kdf.insert(1, 'y', kser)
        pdf.insert(1, 'y', pser)
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        pdf = pd.DataFrame({('x', 'a'): [1, 2, 3]})
        kdf = ks.from_pandas(pdf)
        pser = pd.Series([4, 5, 6])
        kser = ks.from_pandas(pser)
        pdf = pd.DataFrame({('x', 'a', 'b'): [1, 2, 3]})
        kdf = ks.from_pandas(pdf)
        kdf.insert(0, 'a', kser)
        pdf.insert(0, 'a', pser)
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        kdf.insert(0, ('b', 'c', ''), kser)
        pdf.insert(0, ('b', 'c', ''), pser)
        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_compare(self) -> None:
        if LooseVersion(pd.__version__) >= LooseVersion('1.1'):
            pser1 = pd.Series(['b', 'c', np.nan, 'g', np.nan])
            pser2 = pd.Series(['a', 'c', np.nan, np.nan, 'h'])
            kser1 = ks.from_pandas(pser1)
            kser2 = ks.from_pandas(pser2)
            self.assert_eq(pser1.compare(pser2).sort_index(), kser1.compare(kser2).sort_index())
            self.assert_eq(pser1.compare(pser2, keep_shape=True).sort_index(), kser1.compare(kser2, keep_shape=True).sort_index())
            self.assert_eq(pser1.compare(pser2, keep_equal=True).sort_index(), kser1.compare(kser2, keep_equal=True).sort_index())
            self.assert_eq(pser1.compare(pser2, keep_shape=True, keep_equal=True).sort_index(), kser1.compare(kser2, keep_shape=True, keep_equal=True).sort_index())
            pser1.index = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z'), ('x', 'k'), ('q', 'l')])
            pser2.index = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z'), ('x', 'k'), ('q', 'l')])
            kser1 = ks.from_pandas(pser1)
            kser2 = ks.from_pandas(pser2)
            self.assert_eq(pser1.compare(pser2).sort_index(), kser1.compare(kser2).sort_index())
            self.assert_eq(pser1.compare(pser2, keep_shape=True).sort_index(), kser1.compare(kser2, keep_shape=True).sort_index())
            self.assert_eq(pser1.compare(pser2, keep_equal=True).sort_index(), kser1.compare(kser2, keep_equal=True).sort_index())
            self.assert_eq(pser1.compare(pser2, keep_shape=True, keep_equal=True).sort_index(), kser1.compare(kser2, keep_shape=True, keep_equal=True).sort_index())
        else:
            kser1 = ks.Series(['b', 'c', np.nan, 'g', np.nan])
            kser2 = ks.Series(['a', 'c', np.nan, np.nan, 'h'])
            expected = ks.DataFrame([['b', 'a'], ['g', None], [None, 'h']], index=[0, 3, 4], columns=['self', 'other'])
            self.assert_eq(expected, kser1.compare(kser2).sort_index())
            expected = ks.DataFrame([['b', 'a'], [None, None], [None, None], ['g', None], [None, 'h']], index=[0, 1, 2, 3, 4], columns=['self', 'other'])
            self.assert_eq(expected, kser1.compare(kser2, keep_shape=True).sort_index())
            expected = ks.DataFrame([['b', 'a'], ['g', None], [None, 'h']], index=[0, 3, 4], columns=['self', 'other'])
            self.assert_eq(expected, kser1.compare(kser2, keep_equal=True).sort_index())
            expected = ks.DataFrame([['b', 'a'], ['c', 'c'], [None, None], ['g', None], [None, 'h']], index=[0, 1, 2, 3, 4], columns=['self', 'other'])
            self.assert_eq(expected, kser1.compare(kser2, keep_shape=True, keep_equal=True).sort_index())
            kser1 = ks.Series(['b', 'c', np.nan, 'g', np.nan], index=pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z'), ('x', 'k'), ('q', 'l')]))
            kser2 = ks.Series(['a', 'c', np.nan, np.nan, 'h'], index=pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z'), ('x', 'k'), ('q', 'l')]))
            expected = ks.DataFrame([['b', 'a'], [None, 'h'], ['g', None]], index=pd.MultiIndex.from_tuples([('a', 'x'), ('q', 'l'), ('x', 'k')]), columns=['self', 'other'])
            self.assert_eq(expected, kser1.compare(kser2).sort_index())
            expected = ks.DataFrame([['b', 'a'], [None, None], [None, None], [None, 'h'], ['g', None]], index=pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z'), ('q', 'l'), ('x', 'k')]), columns=['self', 'other'])
            self.assert_eq(expected, kser1.compare(kser2, keep_shape=True).sort_index())
            expected = ks.DataFrame([['b', 'a'], [None, 'h'], ['g', None]], index=pd.MultiIndex.from_tuples([('a', 'x'), ('q', 'l'), ('x', 'k')]), columns=['self', 'other'])
            self.assert_eq(expected, kser1.compare(kser2, keep_equal=True).sort_index())
            expected = ks.DataFrame([['b', 'a'], ['c', 'c'], [None, None], [None, 'h'], ['g', None]], index=pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z'), ('q', 'l'), ('x', 'k')]), columns=['self', 'other'])
            self.assert_eq(expected, kser1.compare(kser2, keep_shape=True, keep_equal=True).sort_index())
        with self.assertRaisesRegex(ValueError, 'Can only compare identically-labeled Series objects'):
            kser1 = ks.Series([1, 2, 3, 4, 5], index=pd.Index([1, 2, 3, 4, 5]))
            kser2 = ks.Series([2, 2, 3, 4, 1], index=pd.Index([5, 4, 3, 2, 1]))
            kser1.compare(kser2)
        with self.assertRaisesRegex(ValueError, 'Can only compare identically-labeled Series objects'):
            kser1 = ks.Series([1, 2, 3, 4, 5], index=pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'z'), ('x', 'k'), ('q', 'l')]))
            kser2 = ks.Series([2, 2, 3, 4, 1], index=pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y'), ('c', 'a'), ('x', 'k'), ('q', 'l')]))
            kser1.compare(kser2)

    def test_different_columns(self) -> None:
        kdf1 = self.kdf1
        kdf4 = self.kdf4
        pdf1 = self.pdf1
        pdf4 = self.pdf4
        self.assert_eq((kdf1 + kdf4).sort_index(), (pdf1 + pdf4).sort_index(), almost=True)
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b')])
        kdf1.columns = columns
        pdf1.columns = columns
        columns = pd.MultiIndex.from_tuples([('z', 'e'), ('z', 'f')])
        kdf4.columns = columns
        pdf4.columns = columns
        self.assert_eq((kdf1 + kdf4).sort_index(), (pdf1 + pdf4).sort_index(), almost=True)

    def test_assignment_series(self) -> None:
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kser = kdf.a
        pser = pdf.a
        kdf['a'] = self.kdf2.a
        pdf['a'] = self.pdf2.a
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        self.assert_eq(kser, pser)
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kser = kdf.a
        pser = pdf.a
        kdf['a'] = self.kdf2.b
        pdf['a'] = self.pdf2.b
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        self.assert_eq(kser, pser)
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf['c'] = self.kdf2.a
        pdf['c'] = self.pdf2.a
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b')])
        kdf.columns = columns
        pdf.columns = columns
        kdf['y', 'c'] = self.kdf2.a
        pdf['y', 'c'] = self.pdf2.a
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        pdf = pd.DataFrame({'a': [1, 2, 3], 'Koalas': [0, 1, 2]}).set_index('Koalas', drop=False)
        kdf = ks.from_pandas(pdf)
        kdf.index.name = None
        kdf['NEW'] = ks.Series([100, 200, 300])
        pdf.index.name = None
        pdf['NEW'] = pd.Series([100, 200, 300])
        self.assert_eq(kdf.sort_index(), pdf.sort_index())

    def test_assignment_frame(self) -> None:
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kser = kdf.a
        pser = pdf.a
        kdf[['a', 'b']] = self.kdf1
        pdf[['a', 'b']] = self.pdf1
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        self.assert_eq(kser, pser)
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kser = kdf.a
        pser = pdf.a
        kdf[['b', 'c']] = self.kdf1
        pdf[['b', 'c']] = self.pdf1
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        self.assert_eq(kser, pser)
        kdf = ks.from_pandas(self.pdf1)
        pdf = self.pdf1
        kdf[['c', 'd']] = self.kdf1
        pdf[['c', 'd']] = self.pdf1
        self.assert_eq(kdf.sort_index(), pdf.sort_index())
        columns = pd.MultiIndex.from_tuples([('x