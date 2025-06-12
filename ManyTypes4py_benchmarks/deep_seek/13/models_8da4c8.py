from __future__ import annotations
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Union, 
    DefaultDict, Sequence, Iterable, TYPE_CHECKING, cast
)
import builtins
import dataclasses
import logging
import re
from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import dateutil.parser
import numpy as np
import pandas as pd
import sqlalchemy as sa
from flask_appbuilder import Model
from flask_appbuilder.security.sqla.models import User
from flask_babel import gettext as __, lazy_gettext as _
from jinja2.exceptions import TemplateError
from markupsafe import escape, Markup
from sqlalchemy import (
    and_, Boolean, Column, DateTime, Enum, ForeignKey, inspect, 
    Integer, or_, String, Table as DBTable, Text, update
)
from sqlalchemy.engine.base import Connection
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    backref, foreign, Mapped, Query, reconstructor, relationship, 
    RelationshipProperty, Session
)
from sqlalchemy.orm.mapper import Mapper
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.sql import column, ColumnElement, literal_column, table
from sqlalchemy.sql.elements import ColumnClause, TextClause
from sqlalchemy.sql.selectable import Alias, TableClause
from superset import app, db, is_feature_enabled, security_manager
from superset.commands.dataset.exceptions import DatasetNotFoundError
from superset.common.db_query_status import QueryStatus
from superset.connectors.sqla.utils import (
    get_columns_description, get_physical_table_metadata, 
    get_virtual_table_metadata
)
from superset.constants import EMPTY_STRING, NULL_STRING
from superset.db_engine_specs.base import BaseEngineSpec, TimestampExpression
from superset.exceptions import (
    ColumnNotFoundException, DatasetInvalidPermissionEvaluationException, 
    QueryObjectValidationError, SupersetErrorException, 
    SupersetErrorsException, SupersetGenericDBErrorException, 
    SupersetSecurityException
)
from superset.jinja_context import BaseTemplateProcessor, ExtraCache, get_template_processor
from superset.models.annotations import Annotation
from superset.models.core import Database
from superset.models.helpers import (
    AuditMixinNullable, CertificationMixin, ExploreMixin, 
    ImportExportMixin, QueryResult
)
from superset.models.slice import Slice
from superset.sql_parse import Table
from superset.superset_typing import (
    AdhocColumn, AdhocMetric, FilterValue, FilterValues, 
    Metric, QueryObjectDict, ResultSetColumnType
)
from superset.utils import core as utils, json
from superset.utils.backports import StrEnum

if TYPE_CHECKING:
    from superset.db_engine_specs.base import BasicPropertiesType

config: Dict[str, Any] = app.config
metadata: sa.MetaData = Model.metadata
logger: logging.Logger = logging.getLogger(__name__)
ADVANCED_DATA_TYPES: List[str] = config['ADVANCED_DATA_TYPES']
VIRTUAL_TABLE_ALIAS: str = 'virtual_table'
ADDITIVE_METRIC_TYPES: Set[str] = {'count', 'sum', 'doubleSum'}
ADDITIVE_METRIC_TYPES_LOWER: Set[str] = {op.lower() for op in ADDITIVE_METRIC_TYPES}

@dataclass
class MetadataResult:
    added: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)

logger = logging.getLogger(__name__)
METRIC_FORM_DATA_PARAMS: List[str] = [
    'metric', 'metric_2', 'metrics', 'metrics_b', 'percent_metrics', 
    'secondary_metric', 'size', 'timeseries_limit_metric', 'x', 'y'
]
COLUMN_FORM_DATA_PARAMS: List[str] = [
    'all_columns', 'all_columns_x', 'columns', 'entity', 
    'groupby', 'order_by_cols', 'series'
]

class DatasourceKind(StrEnum):
    VIRTUAL = 'virtual'
    PHYSICAL = 'physical'

class BaseDatasource(AuditMixinNullable, ImportExportMixin):
    """A common interface to objects that are queryable
    (tables and datasources)"""
    __tablename__: Optional[str] = None
    baselink: Optional[str] = None
    owner_class: Optional[Any] = None
    query_language: Optional[str] = None
    is_rls_supported: bool = False

    @property
    def name(self) -> str:
        raise NotImplementedError()
    
    id: Column[int] = Column(Integer, primary_key=True)
    description: Column[Optional[str]] = Column(Text)
    default_endpoint: Column[Optional[str]] = Column(Text)
    is_featured: Column[bool] = Column(Boolean, default=False)
    filter_select_enabled: Column[bool] = Column(Boolean, default=True)
    offset: Column[int] = Column(Integer, default=0)
    cache_timeout: Column[Optional[int]] = Column(Integer)
    params: Column[Optional[str]] = Column(String(1000))
    perm: Column[Optional[str]] = Column(String(1000))
    schema_perm: Column[Optional[str]] = Column(String(1000))
    catalog_perm: Column[Optional[str]] = Column(String(1000), nullable=True, default=None)
    is_managed_externally: Column[bool] = Column(Boolean, nullable=False, default=False)
    external_url: Column[Optional[str]] = Column(Text, nullable=True)
    sql: Optional[str] = None
    extra_import_fields: List[str] = ['is_managed_externally', 'external_url']

    @property
    def kind(self) -> DatasourceKind:
        return DatasourceKind.VIRTUAL if self.sql else DatasourceKind.PHYSICAL

    @property
    def owners_data(self) -> List[Dict[str, Any]]:
        return [{
            'first_name': o.first_name, 
            'last_name': o.last_name, 
            'username': o.username, 
            'id': o.id
        } for o in self.owners]

    @property
    def is_virtual(self) -> bool:
        return self.kind == DatasourceKind.VIRTUAL

    @declared_attr
    def slices(self) -> RelationshipProperty[List[Slice]]:
        return relationship(
            'Slice', 
            overlaps='table', 
            primaryjoin=lambda: and_(
                foreign(Slice.datasource_id) == self.id, 
                foreign(Slice.datasource_type) == self.type
            )
        )
    
    columns: List[Any] = []
    metrics: List[Any] = []

    @property
    def type(self) -> str:
        raise NotImplementedError()

    @property
    def uid(self) -> str:
        """Unique id across datasource types"""
        return f'{self.id}__{self.type}'

    @property
    def column_names(self) -> List[str]:
        return sorted([c.column_name for c in self.columns], key=lambda x: x or '')

    @property
    def columns_types(self) -> Dict[str, Any]:
        return {c.column_name: c.type for c in self.columns}

    @property
    def main_dttm_col(self) -> str:
        return 'timestamp'

    @property
    def datasource_name(self) -> str:
        raise NotImplementedError()

    @property
    def connection(self) -> Optional[str]:
        """String representing the context of the Datasource"""
        return None

    @property
    def catalog(self) -> Optional[str]:
        """String representing the catalog of the Datasource (if it applies)"""
        return None

    @property
    def schema(self) -> Optional[str]:
        """String representing the schema of the Datasource (if it applies)"""
        return None

    @property
    def filterable_column_names(self) -> List[str]:
        return sorted([c.column_name for c in self.columns if c.filterable])

    @property
    def dttm_cols(self) -> List[str]:
        return []

    @property
    def url(self) -> str:
        return f'/{self.baselink}/edit/{self.id}'

    @property
    def explore_url(self) -> str:
        if self.default_endpoint:
            return self.default_endpoint
        return f'/explore/?datasource_type={self.type}&datasource_id={self.id}'

    @property
    def column_formats(self) -> Dict[str, str]:
        return {m.metric_name: m.d3format for m in self.metrics if m.d3format}

    @property
    def currency_formats(self) -> Dict[str, Any]:
        return {m.metric_name: m.currency_json for m in self.metrics if m.currency_json}

    def add_missing_metrics(self, metrics: List[Any]) -> None:
        existing_metrics = {m.metric_name for m in self.metrics}
        for metric in metrics:
            if metric.metric_name not in existing_metrics:
                metric.table_id = self.id
                self.metrics.append(metric)

    @property
    def short_data(self) -> Dict[str, Any]:
        """Data representation of the datasource sent to the frontend"""
        return {
            'edit_url': self.url,
            'id': self.id,
            'uid': self.uid,
            'catalog': self.catalog,
            'schema': self.schema or None,
            'name': self.name,
            'type': self.type,
            'connection': self.connection,
            'creator': str(self.created_by)
        }

    @property
    def select_star(self) -> Optional[str]:
        pass

    @property
    def order_by_choices(self) -> List[Tuple[str, str]]:
        choices = []
        for column_name in self.column_names:
            column_name = str(column_name or '')
            choices.append((json.dumps([column_name, True]), f'{column_name} ' + __('[asc]')))
            choices.append((json.dumps([column_name, False]), f'{column_name} ' + __('[desc]')))
        return choices

    @property
    def verbose_map(self) -> Dict[str, str]:
        verb_map = {'__timestamp': 'Time'}
        for o in self.metrics:
            if o.metric_name not in verb_map:
                verb_map[o.metric_name] = o.verbose_name or o.metric_name
        for o in self.columns:
            if o.column_name not in verb_map:
                verb_map[o.column_name] = o.verbose_name or o.column_name
        return verb_map

    @property
    def data(self) -> Dict[str, Any]:
        """Data representation of the datasource sent to the frontend"""
        return {
            'id': self.id,
            'uid': self.uid,
            'column_formats': self.column_formats,
            'currency_formats': self.currency_formats,
            'description': self.description,
            'database': self.database.data,
            'default_endpoint': self.default_endpoint,
            'filter_select': self.filter_select_enabled,
            'filter_select_enabled': self.filter_select_enabled,
            'name': self.name,
            'datasource_name': self.datasource_name,
            'table_name': self.datasource_name,
            'type': self.type,
            'catalog': self.catalog,
            'schema': self.schema or None,
            'offset': self.offset,
            'cache_timeout': self.cache_timeout,
            'params': self.params,
            'perm': self.perm,
            'edit_url': self.url,
            'sql': self.sql,
            'columns': [o.data for o in self.columns],
            'metrics': [o.data for o in self.metrics],
            'order_by_choices': self.order_by_choices,
            'owners': [owner.id for owner in self.owners],
            'verbose_map': self.verbose_map,
            'select_star': self.select_star
        }

    def data_for_slices(self, slices: List[Slice]) -> Dict[str, Any]:
        """
        The representation of the datasource containing only the required data
        to render the provided slices.

        Used to reduce the payload when loading a dashboard.
        """
        data = self.data
        metric_names: Set[str] = set()
        column_names: Set[str] = set()
        for slc in slices:
            form_data = slc.form_data
            for metric_param in METRIC_FORM_DATA_PARAMS:
                for metric in utils.as_list(form_data.get(metric_param) or []):
                    metric_names.add(utils.get_metric_name(metric))
                    if utils.is_adhoc_metric(metric):
                        column_ = metric.get('column') or {}
                        if (column_name := column_.get('column_name')):
                            column_names.add(column_name)
            column_names.update(
                (filter_['subject'] for filter_ in form_data.get('adhoc_filters') or [] 
                if filter_.get('clause') == 'WHERE' and filter_.get('subject')
            )
            column_names.update(
                (filter_config['column'] for filter_config in form_data.get('filter_configs') or [] 
                if 'column' in filter_config
            )
            try:
                query_context = slc.get_query_context()
            except DatasetNotFoundError:
                query_context = None
            if query_context:
                column_names.update([
                    utils.get_column_name(column_) 
                    for query in query_context.queries 
                    for column_ in query.columns
                ] or [])
            else:
                _columns = [
                    utils.get_column_name(column_) if utils.is_adhoc_column(column_) else column_ 
                    for column_param in COLUMN_FORM_DATA_PARAMS 
                    for column_ in utils.as_list(form_data.get(column_param) or [])
                ]
                column_names.update(_columns)
        filtered_metrics = [metric for metric in data['metrics'] if metric['metric_name'] in metric_names]
        filtered_columns = []
        column_types: Set[Any] = set()
        for column_ in data['columns']:
            generic_type = column_.get('type_generic')
            if generic_type is not None:
                column_types.add(generic_type)
            if column_['column_name'] in column_names:
                filtered_columns.append(column_)
        data['column_types'] = list(column_types)
        del data['description']
        data.update({'metrics': filtered_metrics})
        data.update({'columns': filtered_columns})
        all_columns = {
            column_['column_name']: column_['verbose_name'] or column_['column_name'] 
            for column_ in filtered_columns
        }
        verbose_map = {'__timestamp': 'Time'}
        verbose_map.update({
            metric['metric_name']: metric['verbose_name'] or metric['metric_name'] 
            for metric in filtered_metrics
        })
        verbose_map.update(all_columns)
        data['verbose_map'] = verbose_map
        data['column_names'] = set(all_columns.values()) | set(self.column_names)
        return data

    @staticmethod
    def filter_values_handler(
        values: Any, 
        operator: str, 
        target_generic_type: Any, 
        target_native_type: Optional[str] = None, 
        is_list_target: bool = False, 
        db_engine_spec: Optional[BaseEngineSpec] = None, 
        db_extra: Optional[Dict[str, Any]] = None
    ) -> Any:
        if values is None:
            return None

        def handle_single_value(value: Any) -> Any:
            if operator == utils.FilterOperator.TEMPORAL_RANGE:
                return value
            if isinstance(value, (float, int)) and target_generic_type == utils.GenericDataType.TEMPORAL and (target_native_type is not None) and (db_engine_spec is not None):
                value = db_engine_spec.convert_dttm(
                    target_type=target_native_type, 
                    dttm=datetime.utcfromtimestamp(value / 1000), 
                    db_extra=db_extra
                )
                value = literal_column(value)
            if isinstance(value, str):
                value = value.strip('\t\n')
                if target_generic_type == utils.GenericDataType.NUMERIC and operator not in {utils.FilterOperator.ILIKE, utils.FilterOperator.LIKE}:
                    return utils.cast_to_num(value)
                if value == NULL_STRING:
                    return None
                if value == EMPTY_STRING:
                    return ''
            if target_generic_type == utils.GenericDataType.BOOLEAN:
                return utils.cast_to_boolean(value)
            return value
        
        if isinstance(values, (list, tuple)):
            values = [handle_single_value(v) for v in values]
        else:
            values = handle_single_value(values)
        if is_list_target and (not isinstance(values, (tuple, list))):
            values = [values]
        elif not is_list_target and isinstance(values, (tuple, list)):
            values = values[0] if values else None
        return values

    def external_metadata(self) -> List[Dict[str, Any]]:
        """Returns column information from the external system"""
        raise NotImplementedError()

    def get_query_str(self, query_obj: Dict[str, Any]) -> str:
        """Returns a query as a string

        This is used to be displayed to the user so that they can
        understand what is taking place behind the scene"""
        raise NotImplementedError()

    def query(self, query_obj: Dict[str, Any]) -> QueryResult:
        """Executes the query and returns a dataframe

        query_obj is a dictionary representing Superset's query interface.
        Should return a ``superset.models.helpers.QueryResult``
        """
        raise NotImplementedError()

    @staticmethod
    def default_query(qry: Query) -> Query:
        return qry

    def get_column(self, column_name: str) -> Optional[Any]:
        if not column_name:
            return None
        for col in self.columns:
            if col.column_name == column_name:
                return col
        return None

   