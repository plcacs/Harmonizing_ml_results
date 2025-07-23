import base64
from distutils.version import LooseVersion
from io import BytesIO
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils
from typing import Optional, Tuple, List, Any
matplotlib.use('agg')

class DataFramePlotMatplotlibTest(ReusedSQLTestCase, TestUtils):
    sample_ratio_default: Optional[float] = None

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if LooseVersion(pd.__version__) >= LooseVersion('0.25'):
            pd.set_option('plotting.backend', 'matplotlib')
        set_option('plotting.backend', 'matplotlib')
        set_option('plotting.max_rows', 2000)
        set_option('plotting.sample_ratio', None)

    @classmethod
    def tearDownClass(cls) -> None:
        if LooseVersion(pd.__version__) >= LooseVersion('0.25'):
            pd.reset_option('plotting.backend')
        reset_option('plotting.backend')
        reset_option('plotting.max_rows')
        reset_option('plotting.sample_ratio')
        super().tearDownClass()

    @property
    def pdf1(self) -> pd.DataFrame:
        return pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50], 'b': [2, 3, 4, 5, 7, 9, 10, 15, 34, 45, 49]}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10])

    @property
    def kdf1(self) -> ks.DataFrame:
        return ks.from_pandas(self.pdf1)

    @staticmethod
    def plot_to_base64(ax: plt.Axes) -> bytes:
        bytes_data = BytesIO()
        ax.figure.savefig(bytes_data, format='png')
        bytes_data.seek(0)
        b64_data = base64.b64encode(bytes_data.read())
        plt.close(ax.figure)
        return b64_data

    def test_line_plot(self) -> None:

        def check_line_plot(pdf: pd.DataFrame, kdf: ks.DataFrame) -> None:
            ax1 = pdf.plot(kind='line', colormap='Paired')
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot(kind='line', colormap='Paired')
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            ax3 = pdf.plot.line(colormap='Paired')
            bin3 = self.plot_to_base64(ax3)
            ax4 = kdf.plot.line(colormap='Paired')
            bin4 = self.plot_to_base64(ax4)
            self.assertEqual(bin3, bin4)
        pdf1 = self.pdf1
        kdf1 = self.kdf1
        check_line_plot(pdf1, kdf1)
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('y', 'b')])
        pdf1.columns = columns
        kdf1.columns = columns
        check_line_plot(pdf1, kdf1)

    def test_area_plot(self) -> None:

        def check_area_plot(pdf: pd.DataFrame, kdf: ks.DataFrame) -> None:
            ax1 = pdf.plot(kind='area', colormap='Paired')
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot(kind='area', colormap='Paired')
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            ax3 = pdf.plot.area(colormap='Paired')
            bin3 = self.plot_to_base64(ax3)
            ax4 = kdf.plot.area(colormap='Paired')
            bin4 = self.plot_to_base64(ax4)
            self.assertEqual(bin3, bin4)
        pdf = self.pdf1
        kdf = self.kdf1
        check_area_plot(pdf, kdf)
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('y', 'b')])
        pdf.columns = columns
        kdf.columns = columns
        check_area_plot(pdf, kdf)

    def test_area_plot_stacked_false(self) -> None:

        def check_area_plot_stacked_false(pdf: pd.DataFrame, kdf: ks.DataFrame) -> None:
            ax1 = pdf.plot.area(stacked=False)
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot.area(stacked=False)
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
        pdf = pd.DataFrame({'sales': [3, 2, 3, 9, 10, 6], 'signups': [5, 5, 6, 12, 14, 13], 'visits': [20, 42, 28, 62, 81, 50]}, index=pd.date_range(start='2018/01/01', end='2018/07/01', freq='M'))
        kdf = ks.from_pandas(pdf)
        check_area_plot_stacked_false(pdf, kdf)
        columns = pd.MultiIndex.from_tuples([('x', 'sales'), ('x', 'signups'), ('y', 'visits')])
        pdf.columns = columns
        kdf.columns = columns
        check_area_plot_stacked_false(pdf, kdf)

    def test_area_plot_y(self) -> None:

        def check_area_plot_y(pdf: pd.DataFrame, kdf: ks.DataFrame, y: str) -> None:
            ax1 = pdf.plot.area(y=y)
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot.area(y=y)
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
        pdf = pd.DataFrame({'sales': [3, 2, 3, 9, 10, 6], 'signups': [5, 5, 6, 12, 14, 13], 'visits': [20, 42, 28, 62, 81, 50]}, index=pd.date_range(start='2018/01/01', end='2018/07/01', freq='M'))
        kdf = ks.from_pandas(pdf)
        check_area_plot_y(pdf, kdf, y='sales')
        columns = pd.MultiIndex.from_tuples([('x', 'sales'), ('x', 'signups'), ('y', 'visits')])
        pdf.columns = columns
        kdf.columns = columns
        check_area_plot_y(pdf, kdf, y=('x', 'sales'))

    def test_barh_plot_with_x_y(self) -> None:

        def check_barh_plot_with_x_y(pdf: pd.DataFrame, kdf: ks.DataFrame, x: str, y: str) -> None:
            ax1 = pdf.plot(kind='barh', x=x, y=y, colormap='Paired')
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot(kind='barh', x=x, y=y, colormap='Paired')
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            ax3 = pdf.plot.barh(x=x, y=y, colormap='Paired')
            bin3 = self.plot_to_base64(ax3)
            ax4 = kdf.plot.barh(x=x, y=y, colormap='Paired')
            bin4 = self.plot_to_base64(ax4)
            self.assertEqual(bin3, bin4)
        pdf1 = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
        kdf1 = ks.from_pandas(pdf1)
        check_barh_plot_with_x_y(pdf1, kdf1, x='lab', y='val')
        columns = pd.MultiIndex.from_tuples([('x', 'lab'), ('y', 'val')])
        pdf1.columns = columns
        kdf1.columns = columns
        check_barh_plot_with_x_y(pdf1, kdf1, x=('x', 'lab'), y=('y', 'val'))

    def test_barh_plot(self) -> None:

        def check_barh_plot(pdf: pd.DataFrame, kdf: ks.DataFrame) -> None:
            ax1 = pdf.plot(kind='barh', colormap='Paired')
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot(kind='barh', colormap='Paired')
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            ax3 = pdf.plot.barh(colormap='Paired')
            bin3 = self.plot_to_base64(ax3)
            ax4 = kdf.plot.barh(colormap='Paired')
            bin4 = self.plot_to_base64(ax4)
            self.assertEqual(bin3, bin4)
        pdf1 = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
        kdf1 = ks.from_pandas(pdf1)
        check_barh_plot(pdf1, kdf1)
        columns = pd.MultiIndex.from_tuples([('x', 'lab'), ('y', 'val')])
        pdf1.columns = columns
        kdf1.columns = columns
        check_barh_plot(pdf1, kdf1)

    def test_bar_plot(self) -> None:

        def check_bar_plot(pdf: pd.DataFrame, kdf: ks.DataFrame) -> None:
            ax1 = pdf.plot(kind='bar', colormap='Paired')
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot(kind='bar', colormap='Paired')
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            ax3 = pdf.plot.bar(colormap='Paired')
            bin3 = self.plot_to_base64(ax3)
            ax4 = kdf.plot.bar(colormap='Paired')
            bin4 = self.plot_to_base64(ax4)
            self.assertEqual(bin3, bin4)
        pdf1 = self.pdf1
        kdf1 = self.kdf1
        check_bar_plot(pdf1, kdf1)
        columns = pd.MultiIndex.from_tuples([('x', 'lab'), ('y', 'val')])
        pdf1.columns = columns
        kdf1.columns = columns
        check_bar_plot(pdf1, kdf1)

    def test_bar_with_x_y(self) -> None:
        pdf = pd.DataFrame({'lab': ['A', 'B', 'C'], 'val': [10, 30, 20]})
        kdf = ks.from_pandas(pdf)
        ax1 = pdf.plot(kind='bar', x='lab', y='val', colormap='Paired')
        bin1 = self.plot_to_base64(ax1)
        ax2 = kdf.plot(kind='bar', x='lab', y='val', colormap='Paired')
        bin2 = self.plot_to_base64(ax2)
        self.assertEqual(bin1, bin2)
        ax3 = pdf.plot.bar(x='lab', y='val', colormap='Paired')
        bin3 = self.plot_to_base64(ax3)
        ax4 = kdf.plot.bar(x='lab', y='val', colormap='Paired')
        bin4 = self.plot_to_base64(ax4)
        self.assertEqual(bin3, bin4)
        columns = pd.MultiIndex.from_tuples([('x', 'lab'), ('y', 'val')])
        pdf.columns = columns
        kdf.columns = columns
        ax5 = pdf.plot(kind='bar', x=('x', 'lab'), y=('y', 'val'), colormap='Paired')
        bin5 = self.plot_to_base64(ax5)
        ax6 = kdf.plot(kind='bar', x=('x', 'lab'), y=('y', 'val'), colormap='Paired')
        bin6 = self.plot_to_base64(ax6)
        self.assertEqual(bin5, bin6)
        ax7 = pdf.plot.bar(x=('x', 'lab'), y=('y', 'val'), colormap='Paired')
        bin7 = self.plot_to_base64(ax7)
        ax8 = kdf.plot.bar(x=('x', 'lab'), y=('y', 'val'), colormap='Paired')
        bin8 = self.plot_to_base64(ax8)
        self.assertEqual(bin7, bin8)

    def test_pie_plot(self) -> None:

        def check_pie_plot(pdf: pd.DataFrame, kdf: ks.DataFrame, y: str) -> None:
            ax1 = pdf.plot.pie(y=y, figsize=(5, 5), colormap='Paired')
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot.pie(y=y, figsize=(5, 5), colormap='Paired')
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            ax1 = pdf.plot(kind='pie', y=y, figsize=(5, 5), colormap='Paired')
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot(kind='pie', y=y, figsize=(5, 5), colormap='Paired')
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            ax11, ax12 = pdf.plot.pie(figsize=(5, 5), subplots=True, colormap='Paired')
            bin11 = self.plot_to_base64(ax11)
            bin12 = self.plot_to_base64(ax12)
            self.assertEqual(bin11, bin12)
            ax21, ax22 = kdf.plot.pie(figsize=(5, 5), subplots=True, colormap='Paired')
            bin21 = self.plot_to_base64(ax21)
            bin22 = self.plot_to_base64(ax22)
            self.assertEqual(bin21, bin22)
            ax11, ax12 = pdf.plot(kind='pie', figsize=(5, 5), subplots=True, colormap='Paired')
            bin11 = self.plot_to_base64(ax11)
            bin12 = self.plot_to_base64(ax12)
            self.assertEqual(bin11, bin12)
            ax21, ax22 = kdf.plot(kind='pie', figsize=(5, 5), subplots=True, colormap='Paired')
            bin21 = self.plot_to_base64(ax21)
            bin22 = self.plot_to_base64(ax22)
            self.assertEqual(bin21, bin22)
        pdf1 = pd.DataFrame({'mass': [0.33, 4.87, 5.97], 'radius': [2439.7, 6051.8, 6378.1]}, index=['Mercury', 'Venus', 'Earth'])
        kdf1 = ks.from_pandas(pdf1)
        check_pie_plot(pdf1, kdf1, y='mass')
        columns = pd.MultiIndex.from_tuples([('x', 'mass'), ('y', 'radius')])
        pdf1.columns = columns
        kdf1.columns = columns
        check_pie_plot(pdf1, kdf1, y=('x', 'mass'))

    def test_pie_plot_error_message(self) -> None:
        pdf = pd.DataFrame({'mass': [0.33, 4.87, 5.97], 'radius': [2439.7, 6051.8, 6378.1]}, index=['Mercury', 'Venus', 'Earth'])
        kdf = ks.from_pandas(pdf)
        with self.assertRaises(ValueError) as context:
            kdf.plot.pie(figsize=(5, 5), colormap='Paired')
        error_message = "pie requires either y column or 'subplots=True'"
        self.assertTrue(error_message in str(context.exception))

    def test_scatter_plot(self) -> None:

        def check_scatter_plot(pdf: pd.DataFrame, kdf: ks.DataFrame, x: str, y: str, c: str) -> None:
            ax1 = pdf.plot.scatter(x=x, y=y)
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot.scatter(x=x, y=y)
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            ax1 = pdf.plot(kind='scatter', x=x, y=y)
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot(kind='scatter', x=x, y=y)
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            ax1 = pdf.plot.scatter(x=x, y=y, c=c, s=50)
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot.scatter(x=x, y=y, c=c, s=50)
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
        pdf1 = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
        kdf1 = ks.from_pandas(pdf1)
        check_scatter_plot(pdf1, kdf1, x='a', y='b', c='c')
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('x', 'b'), ('y', 'c'), ('z', 'd')])
        pdf1.columns = columns
        kdf1.columns = columns
        check_scatter_plot(pdf1, kdf1, x=('x', 'a'), y=('x', 'b'), c=('y', 'c'))

    def test_hist_plot(self) -> None:

        def check_hist_plot(pdf: pd.DataFrame, kdf: ks.DataFrame) -> None:
            _, ax1 = plt.subplots(1, 1)
            ax1 = pdf.plot.hist()
            bin1 = self.plot_to_base64(ax1)
            _, ax2 = plt.subplots(1, 1)
            ax2 = kdf.plot.hist()
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            ax1 = pdf.plot.hist(bins=15)
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot.hist(bins=15)
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            ax1 = pdf.plot(kind='hist', bins=15)
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot(kind='hist', bins=15)
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            ax1 = pdf.plot.hist(bins=3, bottom=[2, 1, 3])
            bin1 = self.plot_to_base64(ax1)
            ax2 = kdf.plot.hist(bins=3, bottom=[2, 1, 3])
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
            non_numeric_pdf = self.pdf1.copy()
            non_numeric_pdf['c'] = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
            non_numeric_kdf = ks.from_pandas(non_numeric_pdf)
            ax1 = non_numeric_pdf.plot.hist(x=non_numeric_pdf.columns[0], y=non_numeric_pdf.columns[1], bins=3)
            bin1 = self.plot_to_base64(ax1)
            ax2 = non_numeric_kdf.plot.hist(x=non_numeric_pdf.columns[0], y=non_numeric_pdf.columns[1], bins=3)
            bin2 = self.plot_to_base64(ax2)
            self.assertEqual(bin1, bin2)
        pdf1 = self.pdf1
        kdf1 = self.kdf1
        check_hist_plot(pdf1, kdf1)
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('y', 'b')])
        pdf1.columns = columns
        kdf1.columns = columns
        check_hist_plot(pdf1, kdf1)

    def test_kde_plot(self) -> None:

        def moving_average(a: np.ndarray, n: int = 10) -> np.ndarray:
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        def check_kde_plot(pdf: pd.DataFrame, kdf: ks.DataFrame, *args: Any, **kwargs: Any) -> None:
            _, ax1 = plt.subplots(1, 1)
            ax1 = pdf.plot.kde(*args, **kwargs)
            _, ax2 = plt.subplots(1, 1)
            ax2 = kdf.plot.kde(*args, **kwargs)
            try:
                for i, (line1, line2) in enumerate(zip(ax1.get_lines(), ax2.get_lines())):
                    expected = line1.get_xydata().ravel()
                    actual = line2.get_xydata().ravel()
                    self.assertTrue(np.allclose(moving_average(actual), moving_average(expected), rtol=3.0))
            finally:
                ax1.cla()
                ax2.cla()
        pdf1 = self.pdf1
        kdf1 = self.kdf1
        check_kde_plot(pdf1, kdf1, bw_method=0.3)
        check_kde_plot(pdf1, kdf1, ind=[1, 2, 3], bw_method=3.0)
        columns = pd.MultiIndex.from_tuples([('x', 'a'), ('y', 'b')])
        pdf1.columns = columns
        pdf1.columns = columns
        check_kde_plot(pdf1, kdf1, bw_method=0.3)
        check_kde_plot(pdf1, kdf1, ind=[1, 2, 3], bw_method=3.0)
