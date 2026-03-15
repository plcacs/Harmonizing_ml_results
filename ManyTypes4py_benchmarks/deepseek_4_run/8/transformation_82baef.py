from typing import Any, Callable, Dict, List, Union, Optional, Tuple
import numpy as np
import pandas as pd
from numpy import nan
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions import empirical_distribution as ed
from toolz import curry, merge, compose, mapcat
from fklearn.common_docstrings import learner_return_docstring, learner_pred_fn_docstring
from fklearn.training.utils import log_learner_time
from fklearn.types import LearnerReturnType, LearnerLogType
from fklearn.preprocessing.schema import column_duplicatable

@curry
@log_learner_time(learner_name='selector')
def selector(df: pd.DataFrame, training_columns: List[str], predict_columns: Optional[List[str]] = None) -> LearnerReturnType:
    """
    Filters a DataFrames by selecting only the desired columns.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame that must contain `columns`

    training_columns : list of str
        A list of column names that will remain in the dataframe during training time (fit)

    predict_columns: list of str
        A list of column names that will remain in the dataframe during prediction time (transform)
        If None, it defaults to `training_columns`.
    """
    if predict_columns is None:
        predict_columns = training_columns

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        return new_data_set[predict_columns]
    p.__doc__ = learner_pred_fn_docstring('selector')
    log = {'selector': {'training_columns': training_columns, 'predict_columns': predict_columns, 'transformed_column': list(set(training_columns).union(predict_columns))}}
    return (p, df[training_columns], log)
selector.__doc__ += learner_return_docstring('Selector')

@column_duplicatable('columns_to_cap')
@curry
@log_learner_time(learner_name='capper')
def capper(df: pd.DataFrame, columns_to_cap: List[str], precomputed_caps: Optional[Dict[str, Any]] = None) -> LearnerReturnType:
    """
    Learns the maximum value for each of the `columns_to_cap`
    and used that as the cap for those columns. If precomputed caps
    are passed, the function uses that as the cap value instead of
    computing the maximum.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame that must contain `columns_to_cap` columns.

    columns_to_cap : list of str
        A list os column names that should be caped.

    precomputed_caps : dict
        A dictionary on the format {"column_name" : cap_value}.
        That maps column names to pre computed cap values
    """
    if not precomputed_caps:
        precomputed_caps = {}
    caps = {col: precomputed_caps.get(col, df[col].max()) for col in columns_to_cap}

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        capped_cols = {col: new_data_set[col].clip(upper=caps[col]) for col in caps.keys()}
        return new_data_set.assign(**capped_cols)
    p.__doc__ = learner_pred_fn_docstring('capper')
    log = {'capper': {'caps': caps, 'transformed_column': columns_to_cap, 'precomputed_caps': precomputed_caps}}
    return (p, p(df), log)
capper.__doc__ += learner_return_docstring('Capper')

@column_duplicatable('columns_to_floor')
@curry
@log_learner_time(learner_name='floorer')
def floorer(df: pd.DataFrame, columns_to_floor: List[str], precomputed_floors: Optional[Dict[str, Any]] = None) -> LearnerReturnType:
    """
    Learns the minimum value for each of the `columns_to_floor`
    and used that as the floot for those columns. If precomputed floors
    are passed, the function uses that as the cap value instead of
    computing the minimun.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame that must contain `columns_to_floor` columns.

    columns_to_floor : list of str
        A list os column names that should be floored.

    precomputed_floors : dict
        A dictionary on the format {"column_name" : floor_value}
        that maps column names to pre computed floor values
    """
    if not precomputed_floors:
        precomputed_floors = {}
    floors = {col: precomputed_floors.get(col, df[col].min()) for col in columns_to_floor}

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        capped_cols = {col: new_data_set[col].clip(lower=floors[col]) for col in floors.keys()}
        return new_data_set.assign(**capped_cols)
    p.__doc__ = learner_pred_fn_docstring('floorer')
    log = {'floorer': {'floors': floors, 'transformed_column': columns_to_floor, 'precomputed_floors': precomputed_floors}}
    return (p, p(df), log)
floorer.__doc__ += learner_return_docstring('Floorer')

@curry
@log_learner_time(learner_name='ecdfer')
def ecdfer(df: pd.DataFrame, ascending: bool = True, prediction_column: str = 'prediction', ecdf_column: str = 'prediction_ecdf', max_range: int = 1000) -> LearnerReturnType:
    """
    Learns an Empirical Cumulative Distribution Function from the specified column
    in the input DataFrame. It is usually used in the prediction column to convert
    a predicted probability into a score from 0 to 1000.

    Parameters
    ----------
    df : Pandas' pandas.DataFrame
        A Pandas' DataFrame that must contain a `prediction_column` columns.

    ascending : bool
        Whether to compute an ascending ECDF or a descending one.

    prediction_column : str
        The name of the column in `df` to learn the ECDF from.

    ecdf_column : str
        The name of the new ECDF column added by this function

    max_range : int
        The maximum value for the ECDF. It will go will go
         from 0 to max_range.
    """
    if ascending:
        base = 0
        sign = 1
    else:
        base = max_range
        sign = -1
    values = df[prediction_column]
    ecdf = ed.ECDF(values)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(**{ecdf_column: base + sign * max_range * ecdf(new_df[prediction_column])})
    p.__doc__ = learner_pred_fn_docstring('ecdefer')
    log = {'ecdfer': {'nobs': len(values), 'prediction_column': prediction_column, 'ascending': ascending, 'transformed_column': [ecdf_column]}}
    return (p, p(df), log)
ecdfer.__doc__ += learner_return_docstring('ECDFer')

@curry
@log_learner_time(learner_name='discrete_ecdfer')
def discrete_ecdfer(df: pd.DataFrame, ascending: bool = True, prediction_column: str = 'prediction', ecdf_column: str = 'prediction_ecdf', max_range: int = 1000, round_method: Callable = int) -> LearnerReturnType:
    """
    Learns an Empirical Cumulative Distribution Function from the specified column
    in the input DataFrame. It is usually used in the prediction column to convert
    a predicted probability into a score from 0 to 1000.

    Parameters
    ----------
    df : Pandas' pandas.DataFrame
        A Pandas' DataFrame that must contain a `prediction_column` columns.

    ascending : bool
        Whether to compute an ascending ECDF or a descending one.

    prediction_column : str
        The name of the column in `df` to learn the ECDF from.

    ecdf_column : str
        The name of the new ECDF column added by this function.

    max_range : int
        The maximum value for the ECDF. It will go will go
         from 0 to max_range.

    round_method: Callable
        A function perform the round of transformed values for ex: (int, ceil, floor, round)
    """
    if ascending:
        base = 0
        sign = 1
    else:
        base = max_range
        sign = -1
    values = df[prediction_column]
    ecdf = ed.ECDF(values)
    df_ecdf = pd.DataFrame()
    df_ecdf['x'] = ecdf.x
    df_ecdf['y'] = pd.Series(base + sign * max_range * ecdf.y).apply(round_method)
    boundaries = df_ecdf.groupby('y').agg((min, max))['x']['min'].reset_index()
    y = boundaries['y']
    x = boundaries['min']
    side = ecdf.side
    log = {'discrete_ecdfer': {'map': dict(zip(x, y)), 'round_method': round_method, 'nobs': len(values), 'prediction_column': prediction_column, 'ascending': ascending, 'transformed_column': [ecdf_column]}}
    del ecdf
    del values
    del df_ecdf

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        if not ascending:
            tind = np.searchsorted(-x, -new_df[prediction_column])
        else:
            tind = np.searchsorted(x, new_df[prediction_column], side) - 1
        return new_df.assign(**{ecdf_column: y[tind].values})
    return (p, p(df), log)
discrete_ecdfer.__doc__ += learner_return_docstring('Discrete ECDFer')

@curry
def prediction_ranger(df: pd.DataFrame, prediction_min: float, prediction_max: float, prediction_column: str = 'prediction') -> LearnerReturnType:
    """
    Caps and floors the specified prediction column to a set range.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame that must contain a `prediction_column` columns.

    prediction_min : float
        The floor for the prediction.

    prediction_max : float
        The cap for the prediction.

    prediction_column : str
        The name of the column in `df` to cap and floor
    """

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(**{prediction_column: new_df[prediction_column].clip(lower=prediction_min, upper=prediction_max)})
    p.__doc__ = learner_pred_fn_docstring('prediction_ranger')
    log = {'prediction_ranger': {'prediction_min': prediction_min, 'prediction_max': prediction_max, 'transformed_column': [prediction_column]}}
    return (p, p(df), log)
prediction_ranger.__doc__ += learner_return_docstring('Prediction Ranger')

def apply_replacements(df: pd.DataFrame, columns: List[str], vec: Dict[str, Dict[Any, Any]], replace_unseen: Any) -> pd.DataFrame:
    """
    Base function to apply the replacements values found on the
    "vec" vectors into the df DataFrame.

    Parameters
    -----------

    df: pandas.DataFrame
        A Pandas DataFrame containing the data to be replaced.

    columns : list of str
        The df columns names to perform the replacements.

    vec: dict
        A dict mapping a col to dict mapping a value to its replacement. For example:
        vec = {"feature1": {1: 2, 3: 5, 6: 8}}

    replace_unseen: Any
        Default value to replace when original value is not present in the `vec` dict for the feature

    """
    column_categorizer = lambda col: df[col].apply(lambda x: np.nan if isinstance(x, float) and np.isnan(x) else vec[col].get(x, replace_unseen))
    categ_columns = {col: column_categorizer(col) for col in columns}
    return df.assign(**categ_columns)

@column_duplicatable('value_maps')
@curry
@log_learner_time(learner_name='value_mapper')
def value_mapper(df: pd.DataFrame, value_maps: Dict[str, Dict[Any, Any]], ignore_unseen: bool = True, replace_unseen_to: Any = np.nan) -> LearnerReturnType:
    """
    Map values in selected columns in the DataFrame according to dictionaries of replacements.
    Learner wrapper for apply_replacements

    Parameters
    -----------

    df: pandas.DataFrame
        A Pandas DataFrame containing the data to be replaced.

    value_maps: dict of dicts
        A dict mapping a col to dict mapping a value to its replacement. For example:
        value_maps = {"feature1": {1: 2, 3: 5, 6: 8}}

    ignore_unseen: bool
        If True, values not explicitly declared in value_maps will be left as is.
        If False, these will be replaced by replace_unseen_to.

    replace_unseen_to: Any
        Default value to replace when original value is not present in the `vec` dict for the feature.
    """

    def new_col_value_map(old_col_value_map: Dict[Any, Any], new_keys: List[Any]) -> Dict[Any, Any]:
        old_keys = old_col_value_map.keys()
        return {key: old_col_value_map[key] if key in old_keys else key for key in new_keys}
    columns = list(value_maps.keys())
    if ignore_unseen:
        value_maps = {col: new_col_value_map(value_maps[col], list(df[col].unique())) for col in columns}

    def p(df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(df, columns, value_maps, replace_unseen=replace_unseen_to)
    return (p, p(df), {'value_maps': value_maps})

@column_duplicatable('columns_to_truncate')
@curry
@log_learner_time(learner_name='truncate_categorical')
def truncate_categorical(df: pd.DataFrame, columns_to_truncate: List[str], percentile: float, replacement: Any = -9999, replace_unseen: Any = -9999, store_mapping: bool = False) -> LearnerReturnType:
    """
    Truncate infrequent categories and replace them by a single one.
    You can think of it like "others" category.

    The default behaviour is to replace the original values. To store
    the original values in a new column, specify `prefix` or `suffix`
    in the parameters, or specify a dictionary with the desired column
    mapping using the `columns_mapping` parameter.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame that must contain a `prediction_column` columns.

    columns_to_truncate : list of str
        The df columns names to perform the truncation.

    percentile : float
        Categories less frequent than the percentile will be replaced by the
        same one.

    replacement: int, str, float or nan
        The value to use when a category is less frequent that the percentile
        variable.

    replace_unseen : int, str, float, or nan
        The value to impute unseen categories.

    store_mapping : bool (default: False)
        Whether to store the feature value -> integer dictionary in the log.
    """
    get_categs = lambda col: (df[col].value_counts() / len(df)).to_dict()
    update = lambda d: map(lambda kv: (kv[0], replacement) if kv[1] <= percentile else (kv[0], kv[0]), d.items())
    categs_to_dict = lambda categ_dict: dict(categ_dict)
    vec = {column: compose(categs_to_dict, update, get_categs)(column) for column in columns_to_truncate}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_truncate, vec, replace_unseen)
    p.__doc__ = learner_pred_fn_docstring('truncate_categorical')
    log = {'truncate_categorical': {'transformed_column': columns_to_truncate, 'replace_unseen': replace_unseen}}
    if store_mapping:
        log['truncate_categorical']['mapping'] = vec
    return (p, p(df), log)
truncate_categorical.__doc__ += learner_return_docstring('Truncate Categorical')

@column_duplicatable('columns_to_rank')
@curry
@log_learner_time(learner_name='rank_categorical')
def rank_categorical(df: pd.DataFrame, columns_to_rank: List[str], replace_unseen: Any = nan, store_mapping: bool = False) -> LearnerReturnType:
    """
    Rank categorical features by their frequency in the train set.

    The default behaviour is to replace the original values. To store
    the original values in a new column, specify `prefix` or `suffix`
    in the parameters, or specify a dictionary with the desired column
    mapping using the `columns_mapping` parameter.

    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame that must contain a `prediction_column` columns.

    columns_to_rank : list of str
        The df columns names to perform the rank.

    replace_unseen : int, str, float, or nan
        The value to impute unseen categories.

    store_mapping : bool (default: False)
        Whether to store the feature value -> integer dictionary in the log
    """

    def col_categ_getter(col: str) -> Dict[Any, Any]:
        return df[col].value_counts().reset_index().sort_values([col, 'count'], ascending=[True, False]).set_index(col)['count'].rank(method='first', ascending=False).to_dict()
    vec = {column: col_categ_getter(column) for column in columns_to_rank}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_rank, vec, replace_unseen)
    p.__doc__ = learner_pred_fn_docstring('rank_categorical')
    log = {'rank_categorical': {'transformed_column': columns_to_rank,