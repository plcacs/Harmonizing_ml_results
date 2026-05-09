from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

class _GenericArrayFormatter:
    def __init__(self, obj: np.ndarray):
        self.obj = obj

    def get_result(self) -> List[str]:
        raise NotImplementedError

class FloatArrayFormatter(_GenericArrayFormatter):
    def get_result(self) -> List[str]:
        return [f' {val:.{get_option("display.precision") or 6}g}' for val in self.obj]

class _Timedelta64Formatter:
    def __init__(self, obj: np.ndarray, formatter: Optional[Any] = None):
        self.obj = obj
        self.formatter = formatter

    def get_result(self) -> List[str]:
        raise NotImplementedError

class _Datetime64Formatter:
    def __init__(self, obj: np.ndarray, formatter: Optional[Any] = None):
        self.obj = obj
        self.formatter = formatter

    def get_result(self) -> List[str]:
        raise NotImplementedError

class _Datetime64TZFormatter(_Datetime64Formatter):
    def get_result(self) -> List[str]:
        return [self._format_dttz(x) for x in self.obj]

    def _format_dttz(self, x: np.datetime64):
        tz = x.tzinfo
        if tz is None:
            return x.astype('datetime64[ns]').strftime('%Y-%m-%d %H:%M:%S')
        else:
            return x.strftime('%Y-%m-%d %H:%M:%S%z')

class _Timedelta64Formatter(_Timedelta64Formatter):
    def get_result(self) -> List[str]:
        return [self._format_td64(x) for x in self.obj]

    def _format_td64(self, x: np.timedelta64):
        if x == np.timedelta64(0, 'D'):
            return '0 days'
        elif x < 0:
            return f'-{abs(x).astype("timedelta64[D]").astype("O")}'
        else:
            return f'{x.astype("timedelta64[D]").astype("O")}'

class _Datetime64Formatter(_Datetime64Formatter):
    def get_result(self) -> List[str]:
        return [self._format_dttm(x) for x in self.obj]

    def _format_dttm(self, x: np.datetime64):
        if x == np.datetime64('NaT'):
            return 'NaT'
        else:
            return x.astype('datetime64[ns]').strftime('%Y-%m-%d %H:%M:%S')

class SeriesFormatter:
    def __init__(self, series: pd.Series, name: Optional[str] = None):
        self.series = series
        self.name = name

    def _get_footer(self) -> str:
        if self.name is not None:
            return f'Name: {self.name}, dtype: {self.series.dtype}'
        else:
            return f'dtype: {self.series.dtype}'

    def _format_value(self, value: Any) -> str:
        if isinstance(value, np.datetime64):
            return _Datetime64Formatter(value).get_result()[0]
        elif isinstance(value, np.timedelta64):
            return _Timedelta64Formatter(value).get_result()[0]
        elif isinstance(value, np.number):
            return f' {value:.{get_option("display.precision") or 6}g}'
        elif isinstance(value, np.bool_):
            return f' {value}'
        else:
            return repr(value)

    def get_result(self) -> List[str]:
        result = [self._format_value(x) for x in self.series]
        footer = self._get_footer()
        if len(result) > get_option('display.max_rows'):
            if get_option('display.max_rows') == 0:
                return result
            else:
                return result[:get_option('display.max_rows') - 1] + ['...'] + result[-1:]
        else:
            return result + [footer]

class DataFrameFormatter:
    def __init__(self, df: pd.DataFrame, max_rows: Optional[int] = None, min_rows: Optional[int] = None):
        self.df = df
        self.max_rows = max_rows
        self.min_rows = min_rows

    def _format_value(self, value: Any) -> str:
        if isinstance(value, np.datetime64):
            return _Datetime64Formatter(value).get_result()[0]
        elif isinstance(value, np.timedelta64):
            return _Timedelta64Formatter(value).get_result()[0]
        elif isinstance(value, np.number):
            return f' {value:.{get_option("display.precision") or 6}g}'
        elif isinstance(value, np.bool_):
            return f' {value}'
        else:
            return repr(value)

    def _format_row(self, row: pd.Series) -> str:
        return ' '.join([self._format_value(x) for x in row])

    def _format_index(self, index: pd.Index) -> str:
        return ' '.join([repr(x) for x in index])

    def _format_columns(self, columns: pd.Index) -> str:
        return ' '.join([repr(x) for x in columns])

    def _format_footer(self) -> str:
        return f'Length: {len(self.df.index)}, dtype: {self.df.index.dtype}'

    def _format_dimensions(self) -> str:
        if get_option('display.show_dimensions') == True:
            return f'{len(self.df.index)} rows x {len(self.df.columns)} columns'
        elif get_option('display.show_dimensions') == 'truncate':
            return f'{len(self.df.index)} rows x {len(self.df.columns)} columns'
        else:
            return ''

    def _format_value_counts(self, value_counts: pd.Series) -> str:
        return '\n'.join([f'{k}: {v}' for k, v in value_counts.items()])

    def _format_value_counts_index(self, value_counts: pd.Series) -> str:
        return '\n'.join([f'{k}: {v}' for k, v in value_counts.items()])

    def _format_value_counts_columns(self, value_counts: pd.Series) -> str:
        return '\n'.join([f'{k}: {v}' for k, v in value_counts.items()])

    def _format_value_counts_index_columns(self, value_counts: pd.Series) -> str:
        return '\n'.join([f'{k}: {v}' for k, v in value_counts.items()])

    def _format_index_value_counts(self) -> str:
        value_counts = self.df.index.value_counts()
        if len(value_counts) > get_option('display.max_rows'):
            return self._format_value_counts(value_counts[:get_option('display.max_rows')])
        else:
            return self._format_value_counts(value_counts)

    def _format_columns_value_counts(self) -> str:
        value_counts = self.df.columns.value_counts()
        if len(value_counts) > get_option('display.max_rows'):
            return self._format_value_counts(value_counts[:get_option('display.max_rows')])
        else:
            return self._format_value_counts(value_counts)

    def _format_index_columns_value_counts(self) -> str:
        value_counts = self.df.index.value_counts(self.df.columns)
        if len(value_counts) > get_option('display.max_rows'):
            return self._format_value_counts(value_counts[:get_option('display.max_rows')])
        else:
            return self._format_value_counts(value_counts)

    def _format_max_rows_fitted(self) -> int:
        if self.max_rows is None:
            return len(self.df.index)
        elif self.max_rows == 0:
            return 0
        elif self.max_rows < len(self.df.index):
            return self.max_rows
        else:
            return len(self.df.index)

    def _format_min_rows_fitted(self) -> int:
        if self.min_rows is None:
            return len(self.df.index)
        elif self.min_rows == 0:
            return 0
        elif self.min_rows > len(self.df.index):
            return len(self.df.index)
        elif self.min_rows < len(self.df.index):
            return self.min_rows
        else:
            return len(self.df.index)

    def _format_repr(self) -> str:
        max_rows = self._format_max_rows_fitted()
        min_rows = self._format_min_rows_fitted()
        if max_rows == 0:
            return ''
        elif max_rows == min_rows:
            return '\n'.join([self._format_row(self.df.iloc[:max_rows])])
        elif max_rows == 1:
            return '\n'.join([self._format_row(self.df.iloc[:1]), self._format_row(self.df.iloc[1:max_rows])])
        elif max_rows == 2:
            return '\n'.join([self._format_row(self.df.iloc[:1]), self._format_row(self.df.iloc[1:max_rows])])
        else:
            return '\n'.join([self._format_row(self.df.iloc[:max_rows - 1]), '...', self._format_row(self.df.iloc[max_rows - 1:])])

    def _format_repr_info(self) -> str:
        max_cols = get_option('display.max_columns')
        max_info_cols = get_option('display.max_info_columns')
        if max_cols == 0:
            return ''
        elif max_cols == 1:
            return f'   {self.df.columns[0]}  ...  {len(self.df.columns) - 1}\n'
        elif max_cols > len(self.df.columns):
            return f'   {self.df.columns}\n'
        else:
            return f'   {self.df.columns[:max_cols]}\n...'

    def _format_repr_info_nonverbose(self) -> str:
        max_cols = get_option('display.max_columns')
        max_info_cols = get_option('display.max_info_columns')
        if max_cols == 0:
            return ''
        elif max_cols == 1:
            return f'   {self.df.columns[0]}  ...  {len(self.df.columns) - 1}\n'
        elif max_cols > len(self.df.columns):
            return f'   {self.df.columns}\n'
        else:
            return f'   {self.df.columns[:max_cols]}\n   ...'

    def _format_repr_html(self) -> str:
        max_rows = self._format_max_rows_fitted()
        min_rows = self._format_min_rows_fitted()
        if max_rows == 0:
            return ''
        elif max_rows == 1:
            return f'<table><tr><td>{self._format_row(self.df.iloc[:1])}</td></tr></table>'
        elif max_rows == 2:
            return f'<table><tr><td>{self._format_row(self.df.iloc[:1])}</td></tr><tr><td>{self._format_row(self.df.iloc[1:max_rows])}</td></tr></table>'
        else:
            return f'<table><tr><td>{self._format_row(self.df.iloc[:max_rows - 1])}</td></tr><tr><td>...</td></tr><tr><td>{self._format_row(self.df.iloc[max_rows - 1:])}</td></tr></table>'

    def _format_repr_html_info(self) -> str:
        max_cols = get_option('display.max_columns')
        max_info_cols = get_option('display.max_info_columns')
        if max_cols == 0:
            return ''
        elif max_cols == 1:
            return f'<table><tr><td>{self._format_repr_info_nonverbose()}</td></tr></table>'
        elif max_cols > len(self.df.columns):
            return f'<table><tr><td>{self._format_repr_info()}</td></tr></table>'
        else:
            return f'<table><tr><td>{self._format_repr_info()}</td></tr></table>'

    def _format_repr_html_info_nonverbose(self) -> str:
        max_cols = get_option('display.max_columns')
        max_info_cols = get_option('display.max_info_columns')
        if max_cols == 0:
            return ''
        elif max_cols == 1:
            return f'<table><tr><td>{self._format_repr_info_nonverbose()}</td></tr></table>'
        elif max_cols > len(self.df.columns):
            return f'<table><tr><td>{self._format_repr_info()}</td></tr></table>'
        else:
            return f'<table><tr><td>{self._format_repr_info_nonverbose()}</td></tr></table>'

    def _format_repr_html_value_counts(self) -> str:
        max_rows = self._format_max_rows_fitted()
        min_rows = self._format_min_rows_fitted()
        if max_rows == 0:
            return ''
        elif max_rows == 1:
            return f'<table><tr><td>{self._format_value_counts_index()}</td></tr></table>'
        elif max_rows == 2:
            return f'<table><tr><td>{self._format_value_counts_index()}</td></tr><tr><td>{self._format_value_counts_index()}</td></tr></table>'
        else:
            return f'<table><tr><td>{self._format_value_counts_index()}</td></tr><tr><td>...</td></tr><tr><td>{self._format_value_counts_index()}</td></tr></table>'

    def _format_repr_html_value_counts_info(self) -> str:
        max_cols = get_option('display.max_columns')
        max_info_cols = get_option('display.max_info_columns')
        if max_cols == 0:
            return ''
        elif max_cols == 1:
            return f'<table><tr><td>{self._format_value_counts_columns()}</td></tr></table>'
        elif max_cols > len(self.df.columns):
            return f'<table><tr><td>{self._format_value_counts_index_columns()}</td></tr></table>'
        else:
            return f'<table><tr><td>{self._format_value_counts_index_columns()}</td></tr></table>'

    def _format_repr_html_value_counts_info_nonverbose(self) -> str:
        max_cols = get_option('display.max_columns')
        max_info_cols = get_option('display.max_info_columns')
        if max_cols == 0:
            return ''
        elif max_cols == 1:
            return f'<table><tr><td>{self._format_value_counts_columns()}</td></tr></table>'
        elif max_cols > len(self.df.columns):
            return f'<table><tr><td>{self._format_value_counts_index_columns()}</td></tr></table>'
        else:
            return f'<table><tr><td>{self._format_value_counts_columns()}</td></tr></table>'

    def _format_repr_html_index_columns_value_counts(self) -> str:
        max_rows = self._format_max_rows_fitted()
        min_rows = self._format_min_rows_fitted()
        if max_rows == 0:
            return ''
        elif max_rows == 1:
            return f'<table><tr><td>{self._format_value_counts_index_columns()}</td></tr></table>'
        elif max_rows == 2:
            return f'<table><tr><td>{self._format_value_counts_index_columns()}</td></tr><tr><td>{self._format_value_counts_index_columns()}</td></tr></table>'
        else:
            return f'<table><tr><td>{self._format_value_counts_index_columns()}</td></tr><tr><td>...</td></tr><tr><td>{self._format_value_counts_index_columns()}</td></tr></table>'

    def _format_repr_html_index_columns_value_counts_info(self) -> str:
        max_cols = get_option('display.max_columns')
        max_info_cols = get_option('display.max_info_columns')
        if max_cols == 0:
            return ''
        elif max_cols == 1:
            return f'<table><tr><td>{self._format_value_counts_index()}</td></tr></table>'
        elif max_cols > len(self.df.columns):
            return f'<table><tr><td>{self._format_value_counts_columns()}</td></tr></table>'
        else:
            return f'<table><tr><td>{self._format_value_counts_columns()}</td></tr></table>'

    def _format_repr_html_index_columns_value_counts_info_nonverbose(self) -> str:
        max_cols = get_option('display.max_columns')
        max_info_cols = get_option('display.max_info_columns')
        if max_cols == 0:
            return ''
        elif max_cols == 1:
            return f'<table><tr><td>{self._format_value_counts_index()}</td></tr></table>'
        elif max_cols > len(self.df.columns):
            return f'<table><tr><td>{self._format_value_counts_columns()}</td></tr></table>'
        else:
            return f'<table><tr><td>{self._format_value_counts_index()}</td></tr></table>'

    def _format_repr_html_index_columns_value_counts_nonverbose(self) -> str:
        max_rows = self._format_max_rows_fitted()
        min_rows = self._format_min_rows_fitted()
        if max_rows == 0:
            return ''
        elif max_rows == 1:
            return f'<table><tr><td>{self._format_value_counts_index()}</td></tr></table>'
        elif max_rows == 2:
            return f'<table><tr><td>{self._format_value_counts_index()}</td></tr><tr><td>{self._format_value_counts_index()}</td></tr></table>'
        else:
            return f'<table><tr><td>{self._format_value_counts_index()}</td></tr><tr><td>...</td></tr><tr><td>{self._format_value_counts_index()}</td></tr></table>'

    def get_result(self) -> str:
        if get_option('display.large_repr') == 'info':
            return self._format_repr_info()
        elif get_option('display.large_repr') == 'info-missing':
            return self._format_repr_info_nonverbose()
        elif get_option('display.large_repr') == 'info-null':
            return self._format_repr_html_info_nonverbose()
        elif get_option('display.large_repr') == 'html':
            return self._format_repr_html()
        elif get_option('display.large_repr') == 'html-index':
            return self._format_repr_html_index_columns_value_counts()
        elif get_option('display.large_repr') == 'html-index-nulls':
            return self._format_repr_html_index_columns_value_counts_info()
        elif get_option('display.large_repr') == 'html-index-non-numeric':
            return self._format_repr_html_index_columns_value_counts_nonverbose()
        elif get_option('display.large_repr') == 'html-value-counts':
            return self._format_repr_html_value_counts()
        elif get_option('display.large_repr') == 'html-value-counts-index':
            return self._format_repr_html_value_counts_index_columns()
        elif get_option('display.large_repr') == 'html-value-counts-nulls':
            return self._format_repr_html_value_counts_info()
        elif get_option('display.large_repr') == 'html-value-counts-non-numeric':
            return self._format_repr_html_value_counts_info_nonverbose()
        else:
            return self._format_repr()

def format_percentiles(percentiles: List[float]) -> List[str]:
    if not all(0 <= p <= 1 for p in percentiles):
        raise ValueError('percentiles should all be in the interval [0,1]')
    return [f'{p*100:.1f}%' for p in percentiles]

def get_adjustment() -> int:
    width = get_option('display.width')
    if width is None:
        return 0
    else:
        return width
