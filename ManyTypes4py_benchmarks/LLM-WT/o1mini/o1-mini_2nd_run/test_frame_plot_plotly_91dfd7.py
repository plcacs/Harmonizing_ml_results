import unittest
from distutils.version import LooseVersion
import pprint
import pandas as pd
import numpy as np
from plotly import express
import plotly.graph_objs as go
from databricks import koalas as ks
from databricks.koalas.config import set_option, reset_option
from databricks.koalas.testing.utils import ReusedSQLTestCase, TestUtils
from databricks.koalas.utils import name_like_string
from typing import List

@unittest.skipIf(LooseVersion(pd.__version__) < '1.0.0', "pandas<1.0 does not support latest plotly and/or 'plotting.backend' option.")
class DataFramePlotPlotlyTest(ReusedSQLTestCase, TestUtils):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        pd.set_option('plotting.backend', 'plotly')
        set_option('plotting.backend', 'plotly')
        set_option('plotting.max_rows', 2000)
        set_option('plotting.sample_ratio', None)

    @classmethod
    def tearDownClass(cls) -> None:
        pd.reset_option('plotting.backend')
        reset_option('plotting.backend')
        reset_option('plotting.max_rows')
        reset_option('plotting.sample_ratio')
        super().tearDownClass()

    @property
    def pdf1(self) -> pd.DataFrame:
        return pd.DataFrame({
            'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 50],
            'b': [2, 3, 4, 5, 7, 9, 10, 15, 34, 45, 49]
        }, index=[0, 1, 3, 5, 6, 8, 9, 9, 9, 10, 10])

    @property
    def kdf1(self) -> ks.DataFrame:
        return ks.from_pandas(self.pdf1)

    def test_line_plot(self) -> None:

        def check_line_plot(pdf: pd.DataFrame, kdf: ks.DataFrame) -> None:
            self.assertEqual(pdf.plot(kind='line'), kdf.plot(kind='line'))
            self.assertEqual(pdf.plot.line(), kdf.plot.line())
        
        pdf1: pd.DataFrame = self.pdf1
        kdf1: ks.DataFrame = self.kdf1
        check_line_plot(pdf1, kdf1)

    def test_area_plot(self) -> None:

        def check_area_plot(pdf: pd.DataFrame, kdf: ks.DataFrame) -> None:
            self.assertEqual(pdf.plot(kind='area'), kdf.plot(kind='area'))
            self.assertEqual(pdf.plot.area(), kdf.plot.area())
        
        pdf: pd.DataFrame = self.pdf1
        kdf: ks.DataFrame = self.kdf1
        check_area_plot(pdf, kdf)

    def test_area_plot_y(self) -> None:

        def check_area_plot_y(pdf: pd.DataFrame, kdf: ks.DataFrame, y: str) -> None:
            self.assertEqual(pdf.plot.area(y=y), kdf.plot.area(y=y))
        
        pdf: pd.DataFrame = pd.DataFrame({
            'sales': [3, 2, 3, 9, 10, 6],
            'signups': [5, 5, 6, 12, 14, 13],
            'visits': [20, 42, 28, 62, 81, 50]
        }, index=pd.date_range(start='2018/01/01', end='2018/07/01', freq='M'))
        kdf: ks.DataFrame = ks.from_pandas(pdf)
        check_area_plot_y(pdf, kdf, y='sales')

    def test_barh_plot_with_x_y(self) -> None:

        def check_barh_plot_with_x_y(pdf: pd.DataFrame, kdf: ks.DataFrame, x: str, y: str) -> None:
            self.assertEqual(pdf.plot(kind='barh', x=x, y=y), kdf.plot(kind='barh', x=x, y=y))
            self.assertEqual(pdf.plot.barh(x=x, y=y), kdf.plot.barh(x=x, y=y))
        
        pdf1: pd.DataFrame = pd.DataFrame({
            'lab': ['A', 'B', 'C'],
            'val': [10, 30, 20]
        })
        kdf1: ks.DataFrame = ks.from_pandas(pdf1)
        check_barh_plot_with_x_y(pdf1, kdf1, x='lab', y='val')

    def test_barh_plot(self) -> None:

        def check_barh_plot(pdf: pd.DataFrame, kdf: ks.DataFrame) -> None:
            self.assertEqual(pdf.plot(kind='barh'), kdf.plot(kind='barh'))
            self.assertEqual(pdf.plot.barh(), kdf.plot.barh())
        
        pdf1: pd.DataFrame = pd.DataFrame({
            'lab': [20.1, 40.5, 60.6],
            'val': [10, 30, 20]
        })
        kdf1: ks.DataFrame = ks.from_pandas(pdf1)
        check_barh_plot(pdf1, kdf1)

    def test_bar_plot(self) -> None:

        def check_bar_plot(pdf: pd.DataFrame, kdf: ks.DataFrame) -> None:
            self.assertEqual(pdf.plot(kind='bar'), kdf.plot(kind='bar'))
            self.assertEqual(pdf.plot.bar(), kdf.plot.bar())
        
        pdf1: pd.DataFrame = self.pdf1
        kdf1: ks.DataFrame = self.kdf1
        check_bar_plot(pdf1, kdf1)

    def test_bar_with_x_y(self) -> None:
        pdf: pd.DataFrame = pd.DataFrame({
            'lab': ['A', 'B', 'C'],
            'val': [10, 30, 20]
        })
        kdf: ks.DataFrame = ks.from_pandas(pdf)
        self.assertEqual(pdf.plot(kind='bar', x='lab', y='val'), kdf.plot(kind='bar', x='lab', y='val'))
        self.assertEqual(pdf.plot.bar(x='lab', y='val'), kdf.plot.bar(x='lab', y='val'))

    def test_scatter_plot(self) -> None:

        def check_scatter_plot(pdf: pd.DataFrame, kdf: ks.DataFrame, x: str, y: str, c: str) -> None:
            self.assertEqual(pdf.plot.scatter(x=x, y=y), kdf.plot.scatter(x=x, y=y))
            self.assertEqual(pdf.plot(kind='scatter', x=x, y=y), kdf.plot(kind='scatter', x=x, y=y))
            self.assertEqual(pdf.plot.scatter(x=x, y=y, c=c, s=50), kdf.plot.scatter(x=x, y=y, c=c, s=50))
        
        pdf1: pd.DataFrame = pd.DataFrame(np.random.rand(50, 4), columns=['a', 'b', 'c', 'd'])
        kdf1: ks.DataFrame = ks.from_pandas(pdf1)
        check_scatter_plot(pdf1, kdf1, x='a', y='b', c='c')

    def test_pie_plot(self) -> None:

        def check_pie_plot(kdf: ks.DataFrame) -> None:
            pdf: pd.DataFrame = kdf.to_pandas()
            self.assertEqual(
                kdf.plot(kind='pie', y=kdf.columns[0]),
                express.pie(pdf, values='a', names=pdf.index)
            )
            self.assertEqual(
                kdf.plot(kind='pie', values='a'),
                express.pie(pdf, values='a')
            )
        
        kdf1: ks.DataFrame = self.kdf1
        check_pie_plot(kdf1)

    def test_hist_plot(self) -> None:

        def check_hist_plot(kdf: ks.DataFrame) -> None:
            bins: np.ndarray = np.array([1.0, 5.9, 10.8, 15.7, 20.6, 25.5, 30.4, 35.3, 40.2, 45.1, 50.0])
            data: List[np.ndarray] = [
                np.array([5.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                np.array([4.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0])
            ]
            prev: float = bins[0]
            text_bins: List[str] = []
            for b in bins[1:]:
                text_bins.append('[%s, %s)' % (prev, b))
                prev = b
            text_bins[-1] = text_bins[-1][:-1] + ']'
            bins_center: np.ndarray = 0.5 * (bins[:-1] + bins[1:])
            name_a: str = name_like_string(kdf.columns[0])
            name_b: str = name_like_string(kdf.columns[1])
            bars: List[go.Bar] = [
                go.Bar(
                    x=bins_center,
                    y=data[0],
                    name=name_a,
                    text=text_bins,
                    hovertemplate='variable=' + name_a + '<br>value=%{text}<br>count=%{y}'
                ),
                go.Bar(
                    x=bins_center,
                    y=data[1],
                    name=name_b,
                    text=text_bins,
                    hovertemplate='variable=' + name_b + '<br>value=%{text}<br>count=%{y}'
                )
            ]
            fig: go.Figure = go.Figure(data=bars, layout=go.Layout(barmode='stack'))
            fig.layout.xaxis.title = 'value'
            fig.layout.yaxis.title = 'count'
            self.assertEqual(
                pprint.pformat(kdf.plot(kind='hist').to_dict()),
                pprint.pformat(fig.to_dict())
            )
        
        kdf1: ks.DataFrame = self.kdf1
        check_hist_plot(kdf1)
        columns: pd.MultiIndex = pd.MultiIndex.from_tuples([('x', 'y'), ('y', 'z')])
        kdf1.columns = columns
        check_hist_plot(kdf1)

    def test_kde_plot(self) -> None:
        kdf: ks.DataFrame = ks.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [1, 3, 5, 7, 9],
            'c': [2, 4, 6, 8, 10]
        })
        pdf: pd.DataFrame = pd.DataFrame({
            'Density': [0.03515491, 0.06834979, 0.00663503, 0.02372059, 0.06834979, 0.01806934, 0.01806934, 0.06834979, 0.02372059],
            'names': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],
            'index': [-3.5, 5.5, 14.5, -3.5, 5.5, 14.5, -3.5, 5.5, 14.5]
        })
        actual: go.Figure = kdf.plot.kde(bw_method=5, ind=3)
        expected: go.Figure = express.line(pdf, x='index', y='Density', color='names')
        expected.layout.xaxis.title = None
        self.assertEqual(
            pprint.pformat(actual.to_dict()),
            pprint.pformat(expected.to_dict())
        )
