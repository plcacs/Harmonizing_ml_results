from typing import Any, Callable, Dict, List, Union, Optional, Tuple, Set
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
def selector(
    df: pd.DataFrame,
    training_columns: List[str],
    predict_columns: Optional[List[str]] = None
) -> LearnerReturnType:
    if predict_columns is None:
        predict_columns = training_columns

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        return new_data_set[predict_columns]
    p.__doc__ = learner_pred_fn_docstring('selector')
    log: LearnerLogType = {'selector': {'training_columns': training_columns, 'predict_columns': predict_columns, 'transformed_column': list(set(training_columns).union(predict_columns))}}
    return (p, df[training_columns], log)
selector.__doc__ += learner_return_docstring('Selector')

@column_duplicatable('columns_to_cap')
@curry
@log_learner_time(learner_name='capper')
def capper(
    df: pd.DataFrame,
    columns_to_cap: List[str],
    precomputed_caps: Optional[Dict[str, float]] = None
) -> LearnerReturnType:
    if not precomputed_caps:
        precomputed_caps = {}
    caps: Dict[str, float] = {col: precomputed_caps.get(col, df[col].max()) for col in columns_to_cap}

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        capped_cols: Dict[str, pd.Series] = {col: new_data_set[col].clip(upper=caps[col]) for col in caps.keys()}
        return new_data_set.assign(**capped_cols)
    p.__doc__ = learner_pred_fn_docstring('capper')
    log: LearnerLogType = {'capper': {'caps': caps, 'transformed_column': columns_to_cap, 'precomputed_caps': precomputed_caps}}
    return (p, p(df), log)
capper.__doc__ += learner_return_docstring('Capper')

@column_duplicatable('columns_to_floor')
@curry
@log_learner_time(learner_name='floorer')
def floorer(
    df: pd.DataFrame,
    columns_to_floor: List[str],
    precomputed_floors: Optional[Dict[str, float]] = None
) -> LearnerReturnType:
    if not precomputed_floors:
        precomputed_floors = {}
    floors: Dict[str, float] = {col: precomputed_floors.get(col, df[col].min()) for col in columns_to_floor}

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        capped_cols: Dict[str, pd.Series] = {col: new_data_set[col].clip(lower=floors[col]) for col in floors.keys()}
        return new_data_set.assign(**capped_cols)
    p.__doc__ = learner_pred_fn_docstring('floorer')
    log: LearnerLogType = {'floorer': {'floors': floors, 'transformed_column': columns_to_floor, 'precomputed_floors': precomputed_floors}}
    return (p, p(df), log)
floorer.__doc__ += learner_return_docstring('Floorer')

@curry
@log_learner_time(learner_name='ecdfer')
def ecdfer(
    df: pd.DataFrame,
    ascending: bool = True,
    prediction_column: str = 'prediction',
    ecdf_column: str = 'prediction_ecdf',
    max_range: int = 1000
) -> LearnerReturnType:
    if ascending:
        base = 0
        sign = 1
    else:
        base = max_range
        sign = -1
    values: pd.Series = df[prediction_column]
    ecdf: Callable[[np.ndarray], np.ndarray] = ed.ECDF(values)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(**{ecdf_column: base + sign * max_range * ecdf(new_df[prediction_column])})
    p.__doc__ = learner_pred_fn_docstring('ecdefer')
    log: LearnerLogType = {'ecdfer': {'nobs': len(values), 'prediction_column': prediction_column, 'ascending': ascending, 'transformed_column': [ecdf_column]}}
    return (p, p(df), log)
ecdfer.__doc__ += learner_return_docstring('ECDFer')

@curry
@log_learner_time(learner_name='discrete_ecdfer')
def discrete_ecdfer(
    df: pd.DataFrame,
    ascending: bool = True,
    prediction_column: str = 'prediction',
    ecdf_column: str = 'prediction_ecdf',
    max_range: int = 1000,
    round_method: Callable = int
) -> LearnerReturnType:
    if ascending:
        base = 0
        sign = 1
    else:
        base = max_range
        sign = -1
    values: pd.Series = df[prediction_column]
    ecdf: Callable[[np.ndarray], np.ndarray] = ed.ECDF(values)
    df_ecdf: pd.DataFrame = pd.DataFrame()
    df_ecdf['x'] = ecdf.x
    df_ecdf['y'] = pd.Series(base + sign * max_range * ecdf.y).apply(round_method)
    boundaries: pd.DataFrame = df_ecdf.groupby('y').agg((min, max))['x']['min'].reset_index()
    y: pd.Series = boundaries['y']
    x: pd.Series = boundaries['min']
    side: str = ecdf.side
    log: LearnerLogType = {'discrete_ecdfer': {'map': dict(zip(x, y)), 'round_method': round_method, 'nobs': len(values), 'prediction_column': prediction_column, 'ascending': ascending, 'transformed_column': [ecdf_column]}}
    del ecdf
    del values
    del df_ecdf

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        if not ascending:
            tind: np.ndarray = np.searchsorted(-x, -new_df[prediction_column])
        else:
            tind: np.ndarray = np.searchsorted(x, new_df[prediction_column], side) - 1
        return new_df.assign(**{ecdf_column: y[tind].values})
    return (p, p(df), log)
discrete_ecdfer.__doc__ += learner_return_docstring('Discrete ECDFer')

@curry
def prediction_ranger(
    df: pd.DataFrame,
    prediction_min: float,
    prediction_max: float,
    prediction_column: str = 'prediction'
) -> LearnerReturnType:
    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(**{prediction_column: new_df[prediction_column].clip(lower=prediction_min, upper=prediction_max)})
    p.__doc__ = learner_pred_fn_docstring('prediction_ranger')
    log: LearnerLogType = {'prediction_ranger': {'prediction_min': prediction_min, 'prediction_max': prediction_max, 'transformed_column': [prediction_column]}}
    return (p, p(df), log)
prediction_ranger.__doc__ += learner_return_docstring('Prediction Ranger')

def apply_replacements(
    df: pd.DataFrame,
    columns: List[str],
    vec: Dict[str, Dict[Any, Any]],
    replace_unseen: Any
) -> pd.DataFrame:
    column_categorizer: Callable[[str], pd.Series] = lambda col: df[col].apply(lambda x: np.nan if isinstance(x, float) and np.isnan(x) else vec[col].get(x, replace_unseen))
    categ_columns: Dict[str, pd.Series] = {col: column_categorizer(col) for col in columns}
    return df.assign(**categ_columns)

@column_duplicatable('value_maps')
@curry
@log_learner_time(learner_name='value_mapper')
def value_mapper(
    df: pd.DataFrame,
    value_maps: Dict[str, Dict[Any, Any]],
    ignore_unseen: bool = True,
    replace_unseen_to: Any = np.nan
) -> LearnerReturnType:
    def new_col_value_map(old_col_value_map: Dict[Any, Any], new_keys: List[Any]) -> Dict[Any, Any]:
        old_keys: List[Any] = old_col_value_map.keys()
        return {key: old_col_value_map[key] if key in old_keys else key for key in new_keys}
    columns: List[str] = list(value_maps.keys())
    if ignore_unseen:
        value_maps = {col: new_col_value_map(value_maps[col], list(df[col].unique())) for col in columns}

    def p(df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(df, columns, value_maps, replace_unseen=replace_unseen_to)
    return (p, p(df), {'value_maps': value_maps})

@column_duplicatable('columns_to_truncate')
@curry
@log_learner_time(learner_name='truncate_categorical')
def truncate_categorical(
    df: pd.DataFrame,
    columns_to_truncate: List[str],
    percentile: float,
    replacement: Union[int, str, float] = -9999,
    replace_unseen: Union[int, str, float] = -9999,
    store_mapping: bool = False
) -> LearnerReturnType:
    get_categs: Callable[[str], Dict[Any, float]] = lambda col: (df[col].value_counts() / len(df)).to_dict()
    update: Callable[[Dict[Any, float]], List[Tuple[Any, Any]]] = lambda d: map(lambda kv: (kv[0], replacement) if kv[1] <= percentile else (kv[0], kv[0]), d.items())
    categs_to_dict: Callable[[List[Tuple[Any, Any]]], Dict[Any, Any]] = lambda categ_dict: dict(categ_dict)
    vec: Dict[str, Dict[Any, Any]] = {column: compose(categs_to_dict, update, get_categs)(column) for column in columns_to_truncate}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_truncate, vec, replace_unseen)
    p.__doc__ = learner_pred_fn_docstring('truncate_categorical')
    log: LearnerLogType = {'truncate_categorical': {'transformed_column': columns_to_truncate, 'replace_unseen': replace_unseen}}
    if store_mapping:
        log['truncate_categorical']['mapping'] = vec
    return (p, p(df), log)
truncate_categorical.__doc__ += learner_return_docstring('Truncate Categorical')

@column_duplicatable('columns_to_rank')
@curry
@log_learner_time(learner_name='rank_categorical')
def rank_categorical(
    df: pd.DataFrame,
    columns_to_rank: List[str],
    replace_unseen: Any = nan,
    store_mapping: bool = False
) -> LearnerReturnType:
    def col_categ_getter(col: str) -> Dict[Any, float]:
        return df[col].value_counts().reset_index().sort_values([col, 'count'], ascending=[True, False]).set_index(col)['count'].rank(method='first', ascending=False).to_dict()
    vec: Dict[str, Dict[Any, float]] = {column: col_categ_getter(column) for column in columns_to_rank}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_rank, vec, replace_unseen)
    p.__doc__ = learner_pred_fn_docstring('rank_categorical')
    log: LearnerLogType = {'rank_categorical': {'transformed_column': columns_to_rank, 'replace_unseen': replace_unseen}}
    if store_mapping:
        log['rank_categorical']['mapping'] = vec
    return (p, p(df), log)
rank_categorical.__doc__ += learner_return_docstring('Rank Categorical')

@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='count_categorizer')
def count_categorizer(
    df: pd.DataFrame,
    columns_to_categorize: List[str],
    replace_unseen: int = -1,
    store_mapping: bool = False
) -> LearnerReturnType:
    categ_getter: Callable[[str], Dict[Any, int]] = lambda col: df[col].value_counts().to_dict()
    vec: Dict[str, Dict[Any, int]] = {column: categ_getter(column) for column in columns_to_categorize}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_categorize, vec, replace_unseen)
    p.__doc__ = learner_pred_fn_docstring('count_categorizer')
    log: LearnerLogType = {'count_categorizer': {'transformed_column': columns_to_categorize, 'replace_unseen': replace_unseen}}
    if store_mapping:
        log['count_categorizer']['mapping'] = vec
    return (p, p(df), log)
count_categorizer.__doc__ += learner_return_docstring('Count Categorizer')

@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='label_categorizer')
def label_categorizer(
    df: pd.DataFrame,
    columns_to_categorize: List[str],
    replace_unseen: Any = nan,
    store_mapping: bool = False
) -> LearnerReturnType:
    def categ_dict(series: pd.Series) -> Dict[Any, int]:
        categs: np.ndarray = series.dropna().unique()
        return dict(map(reversed, enumerate(categs)))
    vec: Dict[str, Dict[Any, int]] = {column: categ_dict(df[column]) for column in columns_to_categorize}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_categorize, vec, replace_unseen)
    p.__doc__ = learner_pred_fn_docstring('label_categorizer')
    log: LearnerLogType = {'label_categorizer': {'transformed_column': columns_to_categorize, 'replace_unseen': replace_unseen}}
    if store_mapping:
        log['label_categorizer']['mapping'] = vec
    return (p, p(df), log)
label_categorizer.__doc__ += learner_return_docstring('Label Categorizer')

@column_duplicatable('columns_to_bin')
@curry
@log_learner_time(learner_name='quantile_biner')
def quantile_biner(
    df: pd.DataFrame,
    columns_to_bin: List[str],
    q: Union[int, List[float]] = 4,
    right: bool = False
) -> LearnerReturnType:
    bin_getter: Callable[[str], np.ndarray] = lambda col: pd.qcut(df[col], q, retbins=True)[1]
    bins: Dict[str, np.ndarray] = {column: bin_getter(column) for column in columns_to_bin}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        col_biner: Callable[[str], np.ndarray] = lambda col: np.where(new_df[col].isnull(), nan, np.digitize(new_df[col], bins[col], right=right))
        bined_columns: Dict[str, np.ndarray] = {col: col_biner(col) for col in columns_to_bin}
        return new_df.assign(**bined_columns)
    p.__doc__ = learner_pred_fn_docstring('quantile_biner')
    log: LearnerLogType = {'quantile_biner': {'transformed_column': columns_to_bin, 'q': q}}
    return (p, p(df), log)
quantile_biner.__doc__ += learner_return_docstring('Quantile Biner')

@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='onehot_categorizer')
def onehot_categorizer(
    df: pd.DataFrame,
    columns_to_categorize: List[str],
    hardcode_nans: bool = False,
    drop_first_column: bool = False,
    store_mapping: bool = False
) -> LearnerReturnType:
    categ_getter: Callable[[str], List[Any]] = lambda col: list(np.sort(df[col].dropna(axis=0, how='any').unique()))
    vec: Dict[str, List[Any]] = {column: categ_getter(column) for column in sorted(columns_to_categorize)}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        make_dummies: Callable[[str], Dict[str, pd.Series]] = lambda col: dict(map(lambda categ: ('fklearn_feat__' + col + '==' + str(categ), (new_df[col] == categ).astype(int)), vec[col][int(drop_first_column):]))
        oh_cols: Dict[str, pd.Series] = dict(mapcat(lambda col: merge(make_dummies(col), {'fklearn_feat__' + col + '==' + 'nan': (~new_df[col].isin(vec[col])).astype(int)} if hardcode_nans else {}).items(), columns_to_categorize))
        return new_df.assign(**oh_cols).drop(columns_to_categorize, axis=1)
    p.__doc__ = learner_pred_fn_docstring('onehot_categorizer')
    log: LearnerLogType