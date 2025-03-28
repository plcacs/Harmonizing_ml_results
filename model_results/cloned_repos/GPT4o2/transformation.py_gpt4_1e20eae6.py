```python
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
def selector(df: pd.DataFrame,
             training_columns: List[str],
             predict_columns: Optional[List[str]] = None) -> LearnerReturnType:
    if predict_columns is None:
        predict_columns = training_columns

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        return new_data_set[predict_columns]

    p.__doc__ = learner_pred_fn_docstring("selector")

    log: LearnerLogType = {'selector': {
        'training_columns': training_columns,
        'predict_columns': predict_columns,
        'transformed_column': list(set(training_columns).union(predict_columns))}}

    return p, df[training_columns], log


selector.__doc__ += learner_return_docstring("Selector")


@column_duplicatable('columns_to_cap')
@curry
@log_learner_time(learner_name='capper')
def capper(df: pd.DataFrame,
           columns_to_cap: List[str],
           precomputed_caps: Optional[Dict[str, float]] = None) -> LearnerReturnType:
    if not precomputed_caps:
        precomputed_caps = {}

    caps: Dict[str, float] = {col: precomputed_caps.get(col, df[col].max()) for col in columns_to_cap}

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        capped_cols = {col: new_data_set[col].clip(upper=caps[col]) for col in caps.keys()}
        return new_data_set.assign(**capped_cols)

    p.__doc__ = learner_pred_fn_docstring("capper")

    log: LearnerLogType = {'capper': {
        'caps': caps,
        'transformed_column': columns_to_cap,
        'precomputed_caps': precomputed_caps}}

    return p, p(df), log


capper.__doc__ += learner_return_docstring("Capper")


@column_duplicatable('columns_to_floor')
@curry
@log_learner_time(learner_name='floorer')
def floorer(df: pd.DataFrame,
            columns_to_floor: List[str],
            precomputed_floors: Optional[Dict[str, float]] = None) -> LearnerReturnType:
    if not precomputed_floors:
        precomputed_floors = {}

    floors: Dict[str, float] = {col: precomputed_floors.get(col, df[col].min()) for col in columns_to_floor}

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        capped_cols = {col: new_data_set[col].clip(lower=floors[col]) for col in floors.keys()}
        return new_data_set.assign(**capped_cols)

    p.__doc__ = learner_pred_fn_docstring("floorer")

    log: LearnerLogType = {'floorer': {
        'floors': floors,
        'transformed_column': columns_to_floor,
        'precomputed_floors': precomputed_floors}}

    return p, p(df), log


floorer.__doc__ += learner_return_docstring("Floorer")


@curry
@log_learner_time(learner_name='ecdfer')
def ecdfer(df: pd.DataFrame,
           ascending: bool = True,
           prediction_column: str = "prediction",
           ecdf_column: str = "prediction_ecdf",
           max_range: int = 1000) -> LearnerReturnType:
    if ascending:
        base = 0
        sign = 1
    else:
        base = max_range
        sign = -1

    values = df[prediction_column]

    ecdf = ed.ECDF(values)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(**{ecdf_column: (base + sign * max_range * ecdf(new_df[prediction_column]))})

    p.__doc__ = learner_pred_fn_docstring("ecdefer")

    log: LearnerLogType = {'ecdfer': {
        'nobs': len(values),
        'prediction_column': prediction_column,
        'ascending': ascending,
        'transformed_column': [ecdf_column]}}

    return p, p(df), log


ecdfer.__doc__ += learner_return_docstring("ECDFer")


@curry
@log_learner_time(learner_name='discrete_ecdfer')
def discrete_ecdfer(df: pd.DataFrame,
                    ascending: bool = True,
                    prediction_column: str = "prediction",
                    ecdf_column: str = "prediction_ecdf",
                    max_range: int = 1000,
                    round_method: Callable[[float], int] = int) -> LearnerReturnType:
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

    boundaries = df_ecdf.groupby("y").agg((min, max))["x"]["min"].reset_index()

    y = boundaries["y"]
    x = boundaries["min"]
    side = ecdf.side

    log: LearnerLogType = {'discrete_ecdfer': {
        'map': dict(zip(x, y)),
        'round_method': round_method,
        'nobs': len(values),
        'prediction_column': prediction_column,
        'ascending': ascending,
        'transformed_column': [ecdf_column]}}

    del ecdf
    del values
    del df_ecdf

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        if not ascending:
            tind = np.searchsorted(-x, -new_df[prediction_column])
        else:
            tind = np.searchsorted(x, new_df[prediction_column], side) - 1

        return new_df.assign(**{ecdf_column: y[tind].values})

    return p, p(df), log


discrete_ecdfer.__doc__ += learner_return_docstring("Discrete ECDFer")


@curry
def prediction_ranger(df: pd.DataFrame,
                      prediction_min: float,
                      prediction_max: float,
                      prediction_column: str = "prediction") -> LearnerReturnType:
    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(
            **{prediction_column: new_df[prediction_column].clip(lower=prediction_min, upper=prediction_max)}
        )

    p.__doc__ = learner_pred_fn_docstring("prediction_ranger")

    log: LearnerLogType = {'prediction_ranger': {
        'prediction_min': prediction_min,
        'prediction_max': prediction_max,
        'transformed_column': [prediction_column]}}

    return p, p(df), log


prediction_ranger.__doc__ += learner_return_docstring("Prediction Ranger")


def apply_replacements(df: pd.DataFrame,
                       columns: List[str],
                       vec: Dict[str, Dict],
                       replace_unseen: Any) -> pd.DataFrame:
    column_categorizer = lambda col: df[col].apply(lambda x: (np.nan
                                                              if isinstance(x, float) and np.isnan(x)
                                                              else vec[col].get(x, replace_unseen)))
    categ_columns = {col: column_categorizer(col) for col in columns}
    return df.assign(**categ_columns)


@column_duplicatable('value_maps')
@curry
@log_learner_time(learner_name="value_mapper")
def value_mapper(df: pd.DataFrame,
                 value_maps: Dict[str, Dict],
                 ignore_unseen: bool = True,
                 replace_unseen_to: Any = np.nan) -> Tuple[Callable[[pd.DataFrame], pd.DataFrame], pd.DataFrame, Dict]:
    def new_col_value_map(old_col_value_map: Dict[Any, Any],
                          new_keys: List[Any]) -> Dict[Any, Dict]:
        old_keys = old_col_value_map.keys()
        return {key: old_col_value_map[key] if key in old_keys else key for key in new_keys}

    columns = list(value_maps.keys())
    if ignore_unseen:
        value_maps = {col: new_col_value_map(value_maps[col], list(df[col].unique())) for col in columns}

    def p(df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(df, columns, value_maps, replace_unseen=replace_unseen_to)

    return p, p(df), {"value_maps": value_maps}


@column_duplicatable('columns_to_truncate')
@curry
@log_learner_time(learner_name="truncate_categorical")
def truncate_categorical(df: pd.DataFrame,
                         columns_to_truncate: List[str],
                         percentile: float,
                         replacement: Union[str, float] = -9999,
                         replace_unseen: Union[str, float] = -9999,
                         store_mapping: bool = False) -> LearnerReturnType:
    get_categs = lambda col: (df[col].value_counts() / len(df)).to_dict()
    update = lambda d: map(lambda kv: (kv[0], replacement) if kv[1] <= percentile else (kv[0], kv[0]), d.items())
    categs_to_dict = lambda categ_dict: dict(categ_dict)

    vec: Dict[str, Dict] = {column: compose(categs_to_dict, update, get_categs)(column) for column in columns_to_truncate}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_truncate, vec, replace_unseen)

    p.__doc__ = learner_pred_fn_docstring("truncate_categorical")

    log: LearnerLogType = {'truncate_categorical': {
        'transformed_column': columns_to_truncate,
        'replace_unseen': replace_unseen}
    }

    if store_mapping:
        log["truncate_categorical"]["mapping"] = vec

    return p, p(df), log


truncate_categorical.__doc__ += learner_return_docstring("Truncate Categorical")


@column_duplicatable('columns_to_rank')
@curry
@log_learner_time(learner_name="rank_categorical")
def rank_categorical(df: pd.DataFrame,
                     columns_to_rank: List[str],
                     replace_unseen: Union[str, float] = nan,
                     store_mapping: bool = False) -> LearnerReturnType:
    def col_categ_getter(col: str) -> Dict:
        return (df[col]
                .value_counts()
                .reset_index()
                .sort_values([col, "count"], ascending=[True, False])
                .set_index(col)["count"]
                .rank(method="first", ascending=False)
                .to_dict())

    vec: Dict[str, Dict] = {column: col_categ_getter(column) for column in columns_to_rank}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_rank, vec, replace_unseen)

    p.__doc__ = learner_pred_fn_docstring("rank_categorical")

    log: LearnerLogType = {'rank_categorical': {
        'transformed_column': columns_to_rank,
        'replace_unseen': replace_unseen}
    }

    if store_mapping:
        log['rank_categorical']['mapping'] = vec

    return p, p(df), log


rank_categorical.__doc__ += learner_return_docstring("Rank Categorical")


@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='count_categorizer')
def count_categorizer(df: pd.DataFrame,
                      columns_to_categorize: List[str],
                      replace_unseen: int = -1,
                      store_mapping: bool = False) -> LearnerReturnType:
    categ_getter = lambda col: df[col].value_counts().to_dict()
    vec: Dict[str, Dict] = {column: categ_getter(column) for column in columns_to_categorize}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_categorize, vec, replace_unseen)

    p.__doc__ = learner_pred_fn_docstring("count_categorizer")

    log: LearnerLogType = {'count_categorizer': {
        'transformed_column': columns_to_categorize,
        'replace_unseen': replace_unseen}
    }

    if store_mapping:
        log['count_categorizer']['mapping'] = vec

    return p, p(df), log


count_categorizer.__doc__ += learner_return_docstring("Count Categorizer")


@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='label_categorizer')
def label_categorizer(df: pd.DataFrame,
                      columns_to_categorize: List[str],
                      replace_unseen: Union[str, float] = nan,
                      store_mapping: bool = False) -> LearnerReturnType:
    def categ_dict(series: pd.Series) -> Dict:
        categs = series.dropna().unique()
        return dict(map(reversed, enumerate(categs)))  # type: ignore

    vec: Dict[str, Dict] = {column: categ_dict(df[column]) for column in columns_to_categorize}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_categorize, vec, replace_unseen)

    p.__doc__ = learner_pred_fn_docstring("label_categorizer")

    log: LearnerLogType = {'label_categorizer': {
        'transformed_column': columns_to_categorize,
        'replace_unseen': replace_unseen}
    }

    if store_mapping:
        log['label_categorizer']['mapping'] = vec

    return p, p(df), log


label_categorizer.__doc__ += learner_return_docstring("Label Categorizer")


@column_duplicatable('columns_to_bin')
@curry
@log_learner_time(learner_name='quantile_biner')
def quantile_biner(df: pd.DataFrame,
                   columns_to_bin: List[str],
                   q: int = 4,
                   right: bool = False) -> LearnerReturnType:
    bin_getter = lambda col: pd.qcut(df[col], q, retbins=True)[1]
    bins: Dict[str, np.ndarray] = {column: bin_getter(column) for column in columns_to_bin}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        col_biner = lambda col: np.where(new_df[col].isnull(), nan, np.digitize(new_df[col], bins[col], right=right))
        bined_columns = {col: col_biner(col) for col in columns_to_bin}
        return new_df.assign(**bined_columns)

    p.__doc__ = learner_pred_fn_docstring("quantile_biner")

    log: LearnerLogType = {'quantile_biner': {
        'transformed_column': columns_to_bin,
        'q': q}}

    return p, p(df), log


quantile_biner.__doc__ += learner_return_docstring("Quantile Biner")


@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='onehot_categorizer')
def onehot_categorizer(df: pd.DataFrame,
                       columns_to_categorize: List[str],
                       hardcode_nans: bool = False,
                       drop_first_column: bool = False,
                       store_mapping: bool = False) -> LearnerReturnType:
    categ_getter = lambda col: list(np.sort(df[col].dropna(axis=0, how='any').unique()))
    vec: Dict[str, List[Any]] = {column: categ_getter(column) for column in sorted(columns_to_categorize)}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        make_dummies = lambda col: dict(map(lambda categ: ("fklearn_feat__" + col + "==" + str(categ),
                                                           (new_df[col] == categ).astype(int)),
                                            vec[col][int(drop_first_column):]))

        oh_cols = dict(mapcat(lambda col: merge(make_dummies(col),
                                                {"fklearn_feat__" + col + "==" + "nan":
                                                    (~new_df[col].isin(vec[col])).astype(int)} if hardcode_nans
                                                else {}).items(),
                              columns_to_categorize))

        return new_df.assign(**oh_cols).drop(columns_to_categorize, axis=1)

    p.__doc__ = learner_pred_fn_docstring("onehot_categorizer")

    log: LearnerLogType = {'onehot_categorizer': {
        'transformed_column': columns_to_categorize,
        'hardcode_nans': hardcode_nans,
        'drop_first_column': drop_first_column}}

    if store_mapping:
        log['onehot_categorizer']['mapping'] = vec

    return p, p(df), log


onehot_categorizer.__doc__ += learner_return_docstring("Onehot Categorizer")


@column_duplicatable('columns_to_categorize')
@curry
@log_learner_time(learner_name='target_categorizer')
def target_categorizer(df: pd.DataFrame,
                       columns_to_categorize: List[str],
                       target_column: str,
                       smoothing: float = 1.0,
                       ignore_unseen: bool = True,
                       store_mapping: bool = False) -> LearnerReturnType:
    target_mean = df[target_column].mean()
    replace_unseen = nan if ignore_unseen else target_mean

    def categ_target_dict(column: str) -> Dict:
        column_agg = df.groupby(column)[target_column].agg(['count', 'mean'])
        column_target_mean = column_agg['mean']
        column_target_count = column_agg['count']

        smoothed_target_mean = (column_target_count * column_target_mean + smoothing * target_mean) / \
                               (column_target_count + smoothing)

        return smoothed_target_mean.to_dict()

    vec: Dict[str, Dict] = {column: categ_target_dict(column) for column in columns_to_categorize}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_categorize, vec, replace_unseen)

    p.__doc__ = learner_pred_fn_docstring("target_categorizer")

    log: LearnerLogType = {'target_categorizer': {
        'transformed_columns': columns_to_categorize,
        'target_column': target_column,
        'smoothing': smoothing,
        'ignore_unseen': ignore_unseen}
    }

    if store_mapping:
        log['target_categorizer']['mapping'] = vec

    return p, p(df), log


target_categorizer.__doc__ += learner_return_docstring("Target Categorizer")


@column_duplicatable('columns_to_scale')
@curry
@log_learner_time(learner_name='standard_scaler')
def standard_scaler(df: pd.DataFrame,
                    columns_to_scale: List[str]) -> LearnerReturnType:
    scaler = StandardScaler()

    scaler.fit(df[columns_to_scale].values)

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        new_data = scaler.transform(new_data_set[columns_to_scale].values)
        new_cols = pd.DataFrame(data=new_data, columns=columns_to_scale).to_dict('list')
        return new_data_set.assign(**new_cols)

    p.__doc__ = learner_pred_fn_docstring("standard_scaler")

    log: LearnerLogType = {'standard_scaler': {
        'standard_scaler': scaler.get_params(),
        'transformed_column': columns_to_scale}}

    return p, p(df), log


standard_scaler.__doc__ += learner_return_docstring("Standard Scaler")


@column_duplicatable('columns_to_transform')
@curry
@log_learner_time(learner_name='custom_transformer')
def custom_transformer(df: pd.DataFrame,
                       columns_to_transform: List[str],
                       transformation_function: Callable[[pd.Series], pd.Series],
                       is_vectorized: bool = False) -> LearnerReturnType:
    import swifter  # NOQA

    def p(df: pd.DataFrame) -> pd.DataFrame:
        if is_vectorized:
            return df.assign(**{col: transformation_function(df[col]) for col in columns_to_transform})

        return df.assign(**{col: df[col].swifter.apply(transformation_function) for col in columns_to_transform})

    p.__doc__ = learner_pred_fn_docstring("custom_transformer")

    log: LearnerLogType = {'custom_transformer': {
        'transformed_column': columns_to_transform,
        'transformation_function': transformation_function.__name__}
    }

    return p, p(df), log


custom_transformer.__doc__ += learner_return_docstring("Custom Transformer")


@curry
@log_learner_time(learner_name='null_injector')
def null_injector(df: pd.DataFrame,
                  proportion: float,
                  columns_to_inject: Optional[List[str]] = None,
                  groups: Optional[List[List[str]]] = None,
                  seed: int = 1) -> LearnerReturnType:
    if proportion < 0 or proportion > 1:
        raise ValueError('proportions must be between 0 and 1.')
    if not ((columns_to_inject is None) ^ (groups is None)):
        raise ValueError('Either columns_to_inject or groups must be None.')

    n_rows = df.shape[0]

    groups = [[f] for f in columns_to_inject] if columns_to_inject is not None else groups

    null_cols: Dict[str, pd.Series] = {}
    for seed_i, group in enumerate(groups):  # type: ignore
        np.random.seed(seed + seed_i)
        replace_mask = np.random.binomial(1, 1 - proportion, n_rows).astype(bool)
        null_cols = merge(null_cols, {feature: df[feature].where(replace_mask) for feature in group})

    null_data = df.assign(**null_cols)

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        return new_data_set

    p.__doc__ = learner_pred_fn_docstring("null_injector")

    log: LearnerLogType = {'null_injector': {
        "columns_to_inject": columns_to_inject,
        "proportion": proportion,
        "groups": groups
    }}

    return p, null_data, log


null_injector.__doc__ += learner_return_docstring("Null Injector")


@curry
@log_learner_time(learner_name='missing_warner')
def missing_warner(df: pd.DataFrame, cols_list: List[str],
                   new_column_name: str = "has_unexpected_missing",
                   detailed_warning: bool = False,
                   detailed_column_name: Optional[str] = None) -> LearnerReturnType:
    if (detailed_warning is False and detailed_column_name is not None) or \
            (detailed_warning is True and detailed_column_name is None):
        raise ValueError('Either detailed_warning and detailed_column_name should be defined or both should be False.')

    df_selected = df[cols_list]
    cols_without_missing = df_selected.loc[:, df_selected.isna().sum(axis=0) == 0].columns.tolist()

    def p(dataset: pd.DataFrame) -> pd.DataFrame:
        def detailed_assignment(df: pd.DataFrame, cols_to_check: List[str]) -> List[List[str]]:
            cols_with_missing = np.array([np.where(df[col].isna(), col, "") for col in cols_to_check]).T
            missing_by_row_list: List[List[str]] = [list(filter(None, x)) for x in cols_with_missing]
            if len(missing_by_row_list) == 0:
                return np.empty((0, 0)).tolist()
            else:
                return missing_by_row_list

        new_dataset = dataset.assign(**{new_column_name: lambda df: df[cols_without_missing].isna().sum(axis=1) > 0})
        if detailed_warning and detailed_column_name:
            missing_by_row_list = detailed_assignment(new_dataset, cols_without_missing)
            return new_dataset.assign(**{detailed_column_name: missing_by_row_list})
        else:
            return new_dataset

    p.__doc__ = learner_pred_fn_docstring("missing_warner")

    log: LearnerLogType = {"missing_warner": {
        "cols_list": cols_list,
        "cols_without_missing": cols_without_missing}
    }

    return p, df, log


missing_warner.__doc__ += learner_return_docstring("Missing Alerter")
```