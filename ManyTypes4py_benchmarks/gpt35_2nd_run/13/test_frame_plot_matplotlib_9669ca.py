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
matplotlib.use('agg')

class DataFramePlotMatplotlibTest(ReusedSQLTestCase, TestUtils):
    sample_ratio_default: None

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
    def plot_to_base64(ax) -> str:
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

    # Remaining methods have similar type annotations
