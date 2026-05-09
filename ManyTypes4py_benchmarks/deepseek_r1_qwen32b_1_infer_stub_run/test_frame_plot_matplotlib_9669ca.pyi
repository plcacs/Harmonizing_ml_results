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
from typing import Optional, Union, Tuple, List

matplotlib.use('agg')

class DataFramePlotMatplotlibTest(ReusedSQLTestCase, TestUtils):
    sample_ratio_default: Optional[float] = ...

    def __init__(self, *args: Tuple, **kwargs: dict) -> None:
        ...

    @classmethod
    def setUpClass(cls) -> None:
        ...

    @classmethod
    def tearDownClass(cls) -> None:
        ...

    @property
    def pdf1(self) -> pd.DataFrame:
        ...

    @property
    def kdf1(self) -> ks.DataFrame:
        ...

    @staticmethod
    def plot_to_base64(ax: matplotlib.axes.Axes) -> bytes:
        ...

    def test_line_plot(self) -> None:
        ...

    def test_area_plot(self) -> None:
        ...

    def test_area_plot_stacked_false(self) -> None:
        ...

    def test_area_plot_y(self) -> None:
        ...

    def test_barh_plot_with_x_y(self) -> None:
        ...

    def test_barh_plot(self) -> None:
        ...

    def test_bar_plot(self) -> None:
        ...

    def test_bar_with_x_y(self) -> None:
        ...

    def test_pie_plot(self) -> None:
        ...

    def test_pie_plot_error_message(self) -> None:
        ...

    def test_scatter_plot(self) -> None:
        ...

    def test_hist_plot(self) -> None:
        ...

    def test_kde_plot(self) -> None:
        ...