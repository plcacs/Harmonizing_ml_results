from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from flask_babel import gettext as _
from pandas import DataFrame, Series, to_numeric
from superset.exceptions import InvalidPostProcessingError
from superset.utils.core import PostProcessingBoxplotWhiskerType
from superset.utils.pandas_postprocessing.aggregate import aggregate

def boxplot(
    df: DataFrame,
    groupby: List[str],
    metrics: List[str],
    whisker_type: PostProcessingBoxplotWhiskerType,
    percentiles: Optional[Union[List[float], Tuple[float, float]]] = None,
) -> DataFrame:
    """
    Calculate boxplot statistics. For each metric, the operation creates eight
    new columns with the column name suffixed with the following values:

    - `__mean`: the mean
    - `__median`: the median
    - `__max`: the maximum value excluding outliers (see whisker type)
    - `__min`: the minimum value excluding outliers (see whisker type)
    - `__q1`: the median
    - `__q1`: the first quartile (25th percentile)
    - `__q3`: the third quartile (75th percentile)
    - `__count`: count of observations
    - `__outliers`: the values that fall outside the minimum/maximum value
                    (see whisker type)

    :param df: DataFrame containing all-numeric data (temporal column ignored)
    :param groupby: The categories to group by (x-axis)
    :param metrics: The metrics for which to calculate the distribution
    :param whisker_type: The confidence level type
    :return: DataFrame with boxplot statistics per groupby
    """

    def quartile1(series: Series) -> float:
        return np.nanpercentile(series, 25, method='midpoint')

    def quartile3(series: Series) -> float:
        return np.nanpercentile(series, 75, method='midpoint')

    if whisker_type == PostProcessingBoxplotWhiskerType.TUKEY:

        def whisker_high(series: Series) -> float:
            upper_outer_lim = quartile3(series) + 1.5 * (quartile3(series) - quartile1(series))
            return series[series <= upper_outer_lim].max()

        def whisker_low(series: Series) -> float:
            lower_outer_lim = quartile1(series) - 1.5 * (quartile3(series) - quartile1(series))
            return series[series >= lower_outer_lim].min()

    elif whisker_type == PostProcessingBoxplotWhiskerType.PERCENTILE:
        if not isinstance(percentiles, (list, tuple)) or len(percentiles) != 2 or (not isinstance(percentiles[0], (int, float))) or (not isinstance(percentiles[1], (int, float))) or (percentiles[0] >= percentiles[1]):
            raise InvalidPostProcessingError(_('percentiles must be a list or tuple with two numeric values, of which the first is lower than the second value'))
        low, high = (percentiles[0], percentiles[1])

        def whisker_high(series: Series) -> float:
            return np.nanpercentile(series, high)

        def whisker_low(series: Series) -> float:
            return np.nanpercentile(series, low)

    else:
        whisker_high = np.max
        whisker_low = np.min

    def outliers(series: Series) -> List[float]:
        above = series[series > whisker_high(series)]
        below = series[series < whisker_low(series)]
        return above.tolist() + below.tolist()

    operators: Dict[str, Callable[[Series], Any]] = {
        'mean': np.mean,
        'median': np.median,
        'max': whisker_high,
        'min': whisker_low,
        'q1': quartile1,
        'q3': quartile3,
        'count': np.ma.count,
        'outliers': outliers,
    }
    aggregates = {
        f'{metric}__{operator_name}': {'column': metric, 'operator': operator}
        for operator_name, operator in operators.items()
        for metric in metrics
    }
    for column in metrics:
        if df.dtypes[column] == np.object_:
            df[column] = to_numeric(df[column], errors='coerce')
    return aggregate(df, groupby=groupby, aggregates=aggregates)
