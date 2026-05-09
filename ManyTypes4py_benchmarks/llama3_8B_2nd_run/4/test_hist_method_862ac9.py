import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import _matplotlib
from pandas._testing import assert_produces_warning, assert_numpy_array_equal
from pandas.plotting.common import _check_ax_scales, _check_axes_shape, _check_colors, _check_legend_labels, _check_patches_all_filled, _check_plot_works, _check_text_labels, _check_ticks_props, get_x_axis, get_y_axis
import pytest

@pytest.fixture
def ts():
    return pd.Series(np.arange(30, dtype=np.float64), index=pd.date_range('2020-01-01', periods=30, freq='B'), name='ts')

class TestSeriesPlots:
    @pytest.mark.parametrize('kwargs',