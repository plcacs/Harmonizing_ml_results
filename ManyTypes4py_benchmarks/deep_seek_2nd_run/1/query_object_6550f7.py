from __future__ import annotations
import logging
from datetime import datetime
from pprint import pformat
from typing import Any, NamedTuple, TYPE_CHECKING, Optional, List, Dict, Union, Tuple, Set
from flask import g
from flask_babel import gettext as _
from pandas import DataFrame
from superset import feature_flag_manager
from superset.common.chart_data import ChartDataResultType
from superset.exceptions import InvalidPostProcessingError, QueryClauseValidationException, QueryObjectValidationError
from superset.extensions import event_logger
from superset.sql_parse import sanitize_clause
from superset.superset_typing import Column, Metric, OrderBy
from superset.utils import json, pandas_postprocessing
from superset.utils.core import DTTM_ALIAS, find_duplicates, get_column_names, get_metric_names, is_adhoc_metric, QueryObjectFilterClause
from superset.utils.hashing import md5_sha_from_dict
from superset.utils.json import json_int_dttm_ser

if TYPE_CHECKING:
    from superset.connectors.sqla.models import BaseDatasource

logger = logging.getLogger(__name__)

class DeprecatedField(NamedTuple):
    old_name: str
    new_name: str

DEPRECATED_FIELDS: Tuple[DeprecatedField, ...] = (
    DeprecatedField(old_name='granularity_sqla', new_name='granularity'),
    DeprecatedField(old_name='groupby', new_name='columns'),
    DeprecatedField(old_name='timeseries_limit', new_name='series_limit'),
    DeprecatedField(old_name='timeseries_limit_metric', new_name='series_limit_metric')
)
DEPRECATED_EXTRAS_FIELDS: Tuple[DeprecatedField, ...] = (
    DeprecatedField(old_name='where', new_name='where'),
    DeprecatedField(old_name='having', new_name='having')
)

class QueryObject:
    """
    The query objects are constructed on the client.
    """

    def __init__(
        self,
        *,
        annotation_layers: Optional[List[Dict[str, Any]]] = None,
        applied_time_extras: Optional[Dict[str, Any]] = None,
        apply_fetch_values_predicate: bool = False,
        columns: Optional[List[Column]] = None,
        datasource: Optional[BaseDatasource] = None,
        extras: Optional[Dict[str, Any]] = None,
        filters: Optional[List[QueryObjectFilterClause]] = None,
        granularity: Optional[str] = None,
        is_rowcount: bool = False,
        is_timeseries: Optional[bool] = None,
        metrics: Optional[List[Metric]] = None,
        order_desc: bool = True,
        orderby: Optional[List[OrderBy]] = None,
        post_processing: Optional[List[Dict[str, Any]]] = None,
        row_limit: Optional[int] = None,
        row_offset: Optional[int] = None,
        series_columns: Optional[List[Column]] = None,
        series_limit: int = 0,
        series_limit_metric: Optional[Metric] = None,
        time_range: Optional[str] = None,
        time_shift: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        self._set_annotation_layers(annotation_layers)
        self.applied_time_extras = applied_time_extras or {}
        self.apply_fetch_values_predicate = apply_fetch_values_predicate or False
        self.columns = columns or []
        self.datasource = datasource
        self.extras = extras or {}
        self.filter = filters or []
        self.granularity = granularity
        self.is_rowcount = is_rowcount
        self._set_is_timeseries(is_timeseries)
        self._set_metrics(metrics)
        self.order_desc = order_desc
        self.orderby = orderby or []
        self._set_post_processing(post_processing)
        self.row_limit = row_limit
        self.row_offset = row_offset or 0
        self._init_series_columns(series_columns, metrics, is_timeseries)
        self.series_limit = series_limit
        self.series_limit_metric = series_limit_metric
        self.time_range = time_range
        self.time_shift = time_shift
        self.from_dttm: Optional[datetime] = kwargs.get('from_dttm')
        self.to_dttm: Optional[datetime] = kwargs.get('to_dttm')
        self.result_type: Optional[ChartDataResultType] = kwargs.get('result_type')
        self.time_offsets: List[str] = kwargs.get('time_offsets', [])
        self.inner_from_dttm: Optional[datetime] = kwargs.get('inner_from_dttm')
        self.inner_to_dttm: Optional[datetime] = kwargs.get('inner_to_dttm')
        self._rename_deprecated_fields(kwargs)
        self._move_deprecated_extra_fields(kwargs)

    def _set_annotation_layers(self, annotation_layers: Optional[List[Dict[str, Any]]]) -> None:
        self.annotation_layers = [layer for layer in annotation_layers or [] if layer['annotationType'] != 'FORMULA']

    def _set_is_timeseries(self, is_timeseries: Optional[bool]) -> None:
        self.is_timeseries = is_timeseries if is_timeseries is not None else DTTM_ALIAS in self.columns

    def _set_metrics(self, metrics: Optional[List[Metric]] = None) -> None:
        def is_str_or_adhoc(metric: Metric) -> bool:
            return isinstance(metric, str) or is_adhoc_metric(metric)
        self.metrics = metrics and [x if is_str_or_adhoc(x) else x['label'] for x in metrics]

    def _set_post_processing(self, post_processing: Optional[List[Dict[str, Any]]]) -> None:
        post_processing = post_processing or []
        self.post_processing = [post_proc for post_proc in post_processing if post_proc]

    def _init_series_columns(
        self,
        series_columns: Optional[List[Column]],
        metrics: Optional[List[Metric]],
        is_timeseries: Optional[bool]
    ) -> None:
        if series_columns:
            self.series_columns = series_columns
        elif is_timeseries and metrics:
            self.series_columns = self.columns
        else:
            self.series_columns = []

    def _rename_deprecated_fields(self, kwargs: Dict[str, Any]) -> None:
        for field in DEPRECATED_FIELDS:
            if field.old_name in kwargs:
                logger.warning('The field `%s` is deprecated, please use `%s` instead.', field.old_name, field.new_name)
                value = kwargs[field.old_name]
                if value:
                    if hasattr(self, field.new_name):
                        logger.warning('The field `%s` is already populated, replacing value with contents from `%s`.', field.new_name, field.old_name)
                    setattr(self, field.new_name, value)

    def _move_deprecated_extra_fields(self, kwargs: Dict[str, Any]) -> None:
        for field in DEPRECATED_EXTRAS_FIELDS:
            if field.old_name in kwargs:
                logger.warning('The field `%s` is deprecated and should be passed to `extras` via the `%s` property.', field.old_name, field.new_name)
                value = kwargs[field.old_name]
                if value:
                    if hasattr(self.extras, field.new_name):
                        logger.warning('The field `%s` is already populated in `extras`, replacing value with contents from `%s`.', field.new_name, field.old_name)
                    self.extras[field.new_name] = value

    @property
    def metric_names(self) -> List[str]:
        """Return metrics names (labels), coerce adhoc metrics to strings."""
        return get_metric_names(self.metrics or [])

    @property
    def column_names(self) -> List[str]:
        """Return column names (labels). Gives priority to groupbys if both groupbys
        and metrics are non-empty, otherwise returns column labels."""
        return get_column_names(self.columns)

    def validate(self, raise_exceptions: bool = True) -> Optional[QueryObjectValidationError]:
        """Validate query object"""
        try:
            self._validate_there_are_no_missing_series()
            self._validate_no_have_duplicate_labels()
            self._sanitize_filters()
            return None
        except QueryObjectValidationError as ex:
            if raise_exceptions:
                raise
            return ex

    def _validate_no_have_duplicate_labels(self) -> None:
        all_labels = self.metric_names + self.column_names
        if len(set(all_labels)) < len(all_labels):
            dup_labels = find_duplicates(all_labels)
            raise QueryObjectValidationError(_('Duplicate column/metric labels: %(labels)s. Please make sure all columns and metrics have a unique label.', labels=', '.join((f'"{x}"' for x in dup_labels))))

    def _sanitize_filters(self) -> None:
        for param in ('where', 'having'):
            clause = self.extras.get(param)
            if clause:
                try:
                    sanitized_clause = sanitize_clause(clause)
                    if sanitized_clause != clause:
                        self.extras[param] = sanitized_clause
                except QueryClauseValidationException as ex:
                    raise QueryObjectValidationError(ex.message) from ex

    def _validate_there_are_no_missing_series(self) -> None:
        missing_series = [col for col in self.series_columns if col not in self.columns]
        if missing_series:
            raise QueryObjectValidationError(_('The following entries in `series_columns` are missing in `columns`: %(columns)s. ', columns=', '.join((f'"{x}"' for x in missing_series))))

    def to_dict(self) -> Dict[str, Any]:
        query_object_dict = {
            'apply_fetch_values_predicate': self.apply_fetch_values_predicate,
            'columns': self.columns,
            'extras': self.extras,
            'filter': self.filter,
            'from_dttm': self.from_dttm,
            'granularity': self.granularity,
            'inner_from_dttm': self.inner_from_dttm,
            'inner_to_dttm': self.inner_to_dttm,
            'is_rowcount': self.is_rowcount,
            'is_timeseries': self.is_timeseries,
            'metrics': self.metrics,
            'order_desc': self.order_desc,
            'orderby': self.orderby,
            'row_limit': self.row_limit,
            'row_offset': self.row_offset,
            'series_columns': self.series_columns,
            'series_limit': self.series_limit,
            'series_limit_metric': self.series_limit_metric,
            'to_dttm': self.to_dttm,
            'time_shift': self.time_shift
        }
        return query_object_dict

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    def cache_key(self, **extra: Any) -> str:
        """
        The cache key is made out of the key/values from to_dict(), plus any
        other key/values in `extra`
        We remove datetime bounds that are hard values, and replace them with
        the use-provided inputs to bounds, which may be time-relative (as in
        "5 days ago" or "now").
        """
        cache_dict = self.to_dict()
        cache_dict.update(extra)
        if not self.apply_fetch_values_predicate:
            del cache_dict['apply_fetch_values_predicate']
        if self.datasource:
            cache_dict['datasource'] = self.datasource.uid
        if self.result_type:
            cache_dict['result_type'] = self.result_type
        if self.time_range:
            cache_dict['time_range'] = self.time_range
        if self.post_processing:
            cache_dict['post_processing'] = self.post_processing
        if self.time_offsets:
            cache_dict['time_offsets'] = self.time_offsets
        for k in ['from_dttm', 'to_dttm']:
            del cache_dict[k]
        annotation_fields = ['annotationType', 'descriptionColumns', 'intervalEndColumn', 'name', 'overrides', 'sourceType', 'timeColumn', 'titleColumn', 'value']
        annotation_layers = [{field: layer[field] for field in annotation_fields if field in layer} for layer in self.annotation_layers]
        if annotation_layers:
            cache_dict['annotation_layers'] = annotation_layers
        try:
            database = self.datasource.database
            if feature_flag_manager.is_feature_enabled('CACHE_IMPERSONATION') and database.impersonate_user or feature_flag_manager.is_feature_enabled('CACHE_QUERY_BY_USER'):
                if (key := database.db_engine_spec.get_impersonation_key(getattr(g, 'user', None))):
                    logger.debug('Adding impersonation key to QueryObject cache dict: %s', key)
                    cache_dict['impersonation_key'] = key
        except AttributeError:
            pass
        return md5_sha_from_dict(cache_dict, default=json_int_dttm_ser, ignore_nan=True)

    def exec_post_processing(self, df: DataFrame) -> DataFrame:
        """
        Perform post processing operations on DataFrame.

        :param df: DataFrame returned from database model.
        :return: new DataFrame to which all post processing operations have been
                 applied
        :raises QueryObjectValidationError: If the post processing operation
                 is incorrect
        """
        logger.debug('post_processing: \n %s', pformat(self.post_processing))
        with event_logger.log_context(f'{self.__class__.__name__}.post_processing'):
            for post_process in self.post_processing:
                operation = post_process.get('operation')
                if not operation:
                    raise InvalidPostProcessingError(_('`operation` property of post processing object undefined'))
                if not hasattr(pandas_postprocessing, operation):
                    raise InvalidPostProcessingError(_('Unsupported post processing operation: %(operation)s', type=operation))
                options = post_process.get('options', {})
                df = getattr(pandas_postprocessing, operation)(df, **options)
            return df
