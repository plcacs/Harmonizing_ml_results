from io import StringIO
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from flask_babel import gettext as __
from superset.common.chart_data import ChartDataResultFormat
from superset.extensions import event_logger
from superset.utils.core import extract_dataframe_dtypes, get_column_names, get_metric_names

def get_column_key(label: tuple, metrics: List[str]) -> tuple:
    parts = list(label)
    metric = parts[-1]
    parts[-1] = metrics.index(metric)
    return tuple(parts)

def pivot_df(
    df: pd.DataFrame,
    rows: List[str],
    columns: List[str],
    metrics: List[str],
    aggfunc: str = 'Sum',
    transpose_pivot: bool = False,
    combine_metrics: bool = False,
    show_rows_total: bool = False,
    show_columns_total: bool = False,
    apply_metrics_on_rows: bool = False
) -> pd.DataFrame:
    metric_name = __('Total (%(aggfunc)s)', aggfunc=aggfunc)
    if transpose_pivot:
        rows, columns = (columns, rows)
    if apply_metrics_on_rows:
        rows, columns = (columns, rows)
        axis = {'columns': 0, 'rows': 1}
    else:
        axis = {'columns': 1, 'rows': 0}
    df = df.fillna('SUPERSET_PANDAS_NAN')
    if rows or columns:
        df = df.pivot_table(
            index=rows,
            columns=columns,
            values=metrics,
            aggfunc=pivot_v2_aggfunc_map[aggfunc],
            margins=False
        )
    else:
        df.index = pd.Index([*df.index[:-1], metric_name], name='metric')
    if columns and (not rows):
        df = df.stack()
        if not isinstance(df, pd.DataFrame):
            df = df.to_frame()
        df = df.T
        df = df[metrics]
        df.index = pd.Index([*df.index[:-1], metric_name], name='metric')
    if combine_metrics and isinstance(df.columns, pd.MultiIndex):
        new_order = [*range(1, df.columns.nlevels), 0]
        df = df.reorder_levels(new_order, axis=1)
        decorated_columns = [(col, i) for i, col in enumerate(df.columns)]
        grouped_columns = sorted(decorated_columns, key=lambda t: get_column_key(t[0], metrics))
        indexes = [i for col, i in grouped_columns]
        df = df[df.columns[indexes]]
    elif rows:
        df = df[metrics]
    if aggfunc.endswith(' as Fraction of Total'):
        total = df.sum().sum()
        df = df.astype(total.dtypes) / total
    elif aggfunc.endswith(' as Fraction of Columns'):
        total = df.sum(axis=axis['rows'])
        df = df.astype(total.dtypes).div(total, axis=axis['columns'])
    elif aggfunc.endswith(' as Fraction of Rows'):
        total = df.sum(axis=axis['columns'])
        df = df.astype(total.dtypes).div(total, axis=axis['rows'])
    if not isinstance(df.index, pd.MultiIndex):
        df.index = pd.MultiIndex.from_tuples([(str(i),) for i in df.index])
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples([(str(i),) for i in df.columns])
    if show_rows_total:
        groups = df.columns
        if not apply_metrics_on_rows:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].replace('SUPERSET_PANDAS_NAN', np.nan, inplace=True)
                else:
                    df[col].replace('SUPERSET_PANDAS_NAN', 'nan', inplace=True)
        else:
            df.replace('SUPERSET_PANDAS_NAN', np.nan, inplace=True)
        for level in range(df.columns.nlevels):
            subgroups = {group[:level] for group in groups}
            for subgroup in subgroups:
                slice_ = df.columns.get_loc(subgroup)
                subtotal = pivot_v2_aggfunc_map[aggfunc](df.iloc[:, slice_], axis=1)
                depth = df.columns.nlevels - len(subgroup) - 1
                total = metric_name if level == 0 else __('Subtotal')
                subtotal_name = tuple([*subgroup, total, *[''] * depth])
                df.insert(int(slice_.stop), subtotal_name, subtotal)
    if rows and show_columns_total:
        groups = df.index
        for level in range(df.index.nlevels):
            subgroups = {group[:level] for group in groups}
            for subgroup in subgroups:
                slice_ = df.index.get_loc(subgroup)
                subtotal = pivot_v2_aggfunc_map[aggfunc](df.iloc[slice_, :].apply(pd.to_numeric, errors='coerce'), axis=0)
                depth = df.index.nlevels - len(subgroup) - 1
                total = metric_name if level == 0 else __('Subtotal')
                subtotal.name = tuple([*subgroup, total, *[''] * depth])
                df = pd.concat([df[:slice_.stop], subtotal.to_frame().T, df[slice_.stop:]])
    if apply_metrics_on_rows:
        df = df.T
    df.replace('SUPERSET_PANDAS_NAN', np.nan, inplace=True)
    df.rename(index={'SUPERSET_PANDAS_NAN': np.nan}, columns={'SUPERSET_PANDAS_NAN': np.nan}, inplace=True)
    return df

def list_unique_values(series: pd.Series) -> str:
    return ', '.join({str(v) for v in pd.Series.unique(series)})

pivot_v2_aggfunc_map: Dict[str, Any] = {
    'Count': pd.Series.count,
    'Count Unique Values': pd.Series.nunique,
    'List Unique Values': list_unique_values,
    'Sum': pd.Series.sum,
    'Average': pd.Series.mean,
    'Median': pd.Series.median,
    'Sample Variance': lambda series: pd.Series.var(series) if len(series) > 1 else 0,
    'Sample Standard Deviation': (lambda series: pd.Series.std(series) if len(series) > 1 else 0,),
    'Minimum': pd.Series.min,
    'Maximum': pd.Series.max,
    'First': lambda series: series[:1],
    'Last': lambda series: series[-1:],
    'Sum as Fraction of Total': pd.Series.sum,
    'Sum as Fraction of Rows': pd.Series.sum,
    'Sum as Fraction of Columns': pd.Series.sum,
    'Count as Fraction of Total': pd.Series.count,
    'Count as Fraction of Rows': pd.Series.count,
    'Count as Fraction of Columns': pd.Series.count
}

def pivot_table_v2(
    df: pd.DataFrame,
    form_data: Dict[str, Any],
    datasource: Optional['BaseDatasource'] = None
) -> pd.DataFrame:
    verbose_map = datasource.data['verbose_map'] if datasource else None
    return pivot_df(
        df,
        rows=get_column_names(form_data.get('groupbyRows'), verbose_map),
        columns=get_column_names(form_data.get('groupbyColumns'), verbose_map),
        metrics=get_metric_names(form_data['metrics'], verbose_map),
        aggfunc=form_data.get('aggregateFunction', 'Sum'),
        transpose_pivot=bool(form_data.get('transposePivot')),
        combine_metrics=bool(form_data.get('combineMetric')),
        show_rows_total=bool(form_data.get('rowTotals')),
        show_columns_total=bool(form_data.get('colTotals')),
        apply_metrics_on_rows=form_data.get('metricsLayout') == 'ROWS'
    )

def table(
    df: pd.DataFrame,
    form_data: Dict[str, Any],
    datasource: Optional['BaseDatasource'] = None
) -> pd.DataFrame:
    column_config = form_data.get('column_config', {})
    for column, config in column_config.items():
        if 'd3NumberFormat' in config:
            format_ = '{:' + config['d3NumberFormat'] + '}'
            try:
                df[column] = df[column].apply(format_.format)
            except Exception:
                pass
    return df

post_processors: Dict[str, Any] = {
    'pivot_table_v2': pivot_table_v2,
    'table': table
}

@event_logger.log_this
def apply_client_processing(
    result: Dict[str, Any],
    form_data: Optional[Dict[str, Any]] = None,
    datasource: Optional['BaseDatasource'] = None
) -> Dict[str, Any]:
    form_data = form_data or {}
    viz_type = form_data.get('viz_type')
    if viz_type not in post_processors:
        return result
    post_processor = post_processors[viz_type]
    for query in result['queries']:
        if query['result_format'] not in (rf.value for rf in ChartDataResultFormat):
            raise Exception(f"Result format {query['result_format']} not supported")
        data = query['data']
        if isinstance(data, str):
            data = data.strip()
        if not data:
            continue
        if query['result_format'] == ChartDataResultFormat.JSON:
            df = pd.DataFrame.from_dict(data)
        elif query['result_format'] == ChartDataResultFormat.CSV:
            df = pd.read_csv(StringIO(data))
        if datasource:
            df.rename(columns=datasource.data['verbose_map'], inplace=True)
        processed_df = post_processor(df, form_data, datasource)
        query['colnames'] = list(processed_df.columns)
        query['indexnames'] = list(processed_df.index)
        query['coltypes'] = extract_dataframe_dtypes(processed_df, datasource)
        query['rowcount'] = len(processed_df.index)
        processed_df.columns = [
            ' '.join((str(name) for name in column)).strip() if isinstance(column, tuple) else column
            for column in processed_df.columns
        ]
        processed_df.index = [
            ' '.join((str(name) for name in index)).strip() if isinstance(index, tuple) else index
            for index in processed_df.index
        ]
        if query['result_format'] == ChartDataResultFormat.JSON:
            query['data'] = processed_df.to_dict()
        elif query['result_format'] == ChartDataResultFormat.CSV:
            buf = StringIO()
            processed_df.to_csv(buf)
            buf.seek(0)
            query['data'] = buf.getvalue()
    return result
