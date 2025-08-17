from __future__ import annotations

import builtins
import dataclasses
import logging
import re
from collections import defaultdict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, Union

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
    and_,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    inspect,
    Integer,
    or_,
    String,
    Table as DBTable,
    Text,
    update,
)
from sqlalchemy.engine.base import Connection
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    backref,
    foreign,
    Mapped,
    Query,
    reconstructor,
    relationship,
    RelationshipProperty,
)
from sqlalchemy.orm.mapper import Mapper
from sqlalchemy.schema import UniqueConstraint
from sqlalchemy.sql import column, ColumnElement, literal_column, table
from sqlalchemy.sql.elements import ColumnClause, TextClause
from sqlalchemy.sql.expression import Label
from sqlalchemy.sql.selectable import Alias, TableClause

from superset import app, db, is_feature_enabled, security_manager
from superset.commands.dataset.exceptions import DatasetNotFoundError
from superset.common.db_query_status import QueryStatus
from superset.connectors.sqla.utils import (
    get_columns_description,
    get_physical_table_metadata,
    get_virtual_table_metadata,
)
from superset.constants import EMPTY_STRING, NULL_STRING
from superset.db_engine_specs.base import BaseEngineSpec, TimestampExpression
from superset.exceptions import (
    ColumnNotFoundException,
    DatasetInvalidPermissionEvaluationException,
    QueryObjectValidationError,
    SupersetErrorException,
    SupersetErrorsException,
    SupersetGenericDBErrorException,
    SupersetSecurityException,
)
from superset.jinja_context import (
    BaseTemplateProcessor,
    ExtraCache,
    get_template_processor,
)
from superset.models.annotations import Annotation
from superset.models.core import Database
from superset.models.helpers import (
    AuditMixinNullable,
    CertificationMixin,
    ExploreMixin,
    ImportExportMixin,
    QueryResult,
)
from superset.models.slice import Slice
from superset.sql_parse import Table
from superset.superset_typing import (
    AdhocColumn,
    AdhocMetric,
    FilterValue,
    FilterValues,
    Metric,
    QueryObjectDict,
    ResultSetColumnType,
)
from superset.utils import core as utils, json
from superset.utils.backports import StrEnum

config = app.config
metadata = Model.metadata  # pylint: disable=no-member
logger = logging.getLogger(__name__)
ADVANCED_DATA_TYPES = config["ADVANCED_DATA_TYPES"]
VIRTUAL_TABLE_ALIAS = "virtual_table"

# a non-exhaustive set of additive metrics
ADDITIVE_METRIC_TYPES: set[str] = {
    "count",
    "sum",
    "doubleSum",
}
ADDITIVE_METRIC_TYPES_LOWER: set[str] = {op.lower() for op in ADDITIVE_METRIC_TYPES}


@dataclass
class MetadataResult:
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)


logger = logging.getLogger(__name__)

METRIC_FORM_DATA_PARAMS: list[str] = [
    "metric",
    "metric_2",
    "metrics",
    "metrics_b",
    "percent_metrics",
    "secondary_metric",
    "size",
    "timeseries_limit_metric",
    "x",
    "y",
]

COLUMN_FORM_DATA_PARAMS: list[str] = [
    "all_columns",
    "all_columns_x",
    "columns",
    "entity",
    "groupby",
    "order_by_cols",
    "series",
]


class DatasourceKind(StrEnum):
    VIRTUAL = "virtual"
    PHYSICAL = "physical"


class BaseDatasource(AuditMixinNullable, ImportExportMixin):
    """A common interface to objects that are queryable
    (tables and datasources)"""

    __tablename__: Optional[str] = None  # {connector_name}_datasource
    baselink: Optional[str] = None  # url portion pointing to ModelView endpoint

    owner_class: Optional[User] = None

    # Used to do code highlighting when displaying the query in the UI
    query_language: Optional[str] = None

    # Only some datasources support Row Level Security
    is_rls_supported: bool = False

    @property
    def name(self) -> str:
        raise NotImplementedError()

    # Columns
    id: int = Column(Integer, primary_key=True)
    description: Any = Column(Text)
    default_endpoint: Any = Column(Text)
    is_featured: bool = Column(Boolean, default=False)  # TODO deprecating
    filter_select_enabled: bool = Column(Boolean, default=True)
    offset: int = Column(Integer, default=0)
    cache_timeout: Any = Column(Integer)
    params: Any = Column(String(1000))
    perm: Any = Column(String(1000))
    schema_perm: Any = Column(String(1000))
    catalog_perm: Any = Column(String(1000), nullable=True, default=None)
    is_managed_externally: bool = Column(Boolean, nullable=False, default=False)
    external_url: Any = Column(Text, nullable=True)

    sql: Optional[str] = None
    owners: list[User]
    update_from_object_fields: list[str]

    extra_import_fields: list[str] = ["is_managed_externally", "external_url"]

    @property
    def kind(self) -> DatasourceKind:
        return DatasourceKind.VIRTUAL if self.sql else DatasourceKind.PHYSICAL

    @property
    def owners_data(self) -> list[dict[str, Any]]:
        return [
            {
                "first_name": o.first_name,
                "last_name": o.last_name,
                "username": o.username,
                "id": o.id,
            }
            for o in self.owners
        ]

    @property
    def is_virtual(self) -> bool:
        return self.kind == DatasourceKind.VIRTUAL

    @declared_attr
    def slices(self) -> RelationshipProperty:
        return relationship(
            "Slice",
            overlaps="table",
            primaryjoin=lambda: and_(
                foreign(Slice.datasource_id) == self.id,
                foreign(Slice.datasource_type) == self.type,
            ),
        )

    columns: list[TableColumn] = []
    metrics: list[SqlMetric] = []

    @property
    def type(self) -> str:
        raise NotImplementedError()

    @property
    def uid(self) -> str:
        return f"{self.id}__{self.type}"

    @property
    def column_names(self) -> list[str]:
        return sorted([c.column_name for c in self.columns], key=lambda x: x or "")

    @property
    def columns_types(self) -> dict[str, str]:
        return {c.column_name: c.type for c in self.columns}

    @property
    def main_dttm_col(self) -> str:
        return "timestamp"

    @property
    def datasource_name(self) -> str:
        raise NotImplementedError()

    @property
    def connection(self) -> Optional[str]:
        return None

    @property
    def catalog(self) -> Optional[str]:
        return None

    @property
    def schema(self) -> Optional[str]:
        return None

    @property
    def filterable_column_names(self) -> list[str]:
        return sorted([c.column_name for c in self.columns if c.filterable])

    @property
    def dttm_cols(self) -> list[str]:
        return []

    @property
    def url(self) -> str:
        return f"/{self.baselink}/edit/{self.id}"

    @property
    def explore_url(self) -> str:
        if self.default_endpoint:
            return self.default_endpoint
        return f"/explore/?datasource_type={self.type}&datasource_id={self.id}"

    @property
    def column_formats(self) -> dict[str, Optional[str]]:
        return {m.metric_name: m.d3format for m in self.metrics if m.d3format}

    @property
    def currency_formats(self) -> dict[str, Optional[dict[str, Optional[str]]]]:
        return {m.metric_name: m.currency_json for m in self.metrics if m.currency_json}

    def add_missing_metrics(self, metrics: list[SqlMetric]) -> None:
        existing_metrics = {m.metric_name for m in self.metrics}
        for metric in metrics:
            if metric.metric_name not in existing_metrics:
                metric.table_id = self.id
                self.metrics.append(metric)

    @property
    def short_data(self) -> dict[str, Any]:
        return {
            "edit_url": self.url,
            "id": self.id,
            "uid": self.uid,
            "catalog": self.catalog,
            "schema": self.schema or None,
            "name": self.name,
            "type": self.type,
            "connection": self.connection,
            "creator": str(self.created_by),
        }

    @property
    def select_star(self) -> Optional[str]:
        pass

    @property
    def order_by_choices(self) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = []
        for column_name in self.column_names:
            column_name_str = str(column_name or "")
            choices.append(
                (json.dumps([column_name_str, True]), f"{column_name_str} " + __("[asc]"))
            )
            choices.append(
                (json.dumps([column_name_str, False]), f"{column_name_str} " + __("[desc]"))
            )
        return choices

    @property
    def verbose_map(self) -> dict[str, str]:
        verb_map: dict[str, str] = {"__timestamp": "Time"}
        for o in self.metrics:
            if o.metric_name not in verb_map:
                verb_map[o.metric_name] = o.verbose_name or o.metric_name
        for o in self.columns:
            if o.column_name not in verb_map:
                verb_map[o.column_name] = o.verbose_name or o.column_name
        return verb_map

    @property
    def data(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "uid": self.uid,
            "column_formats": self.column_formats,
            "currency_formats": self.currency_formats,
            "description": self.description,
            "database": self.database.data,  # pylint: disable=no-member
            "default_endpoint": self.default_endpoint,
            "filter_select": self.filter_select_enabled,
            "filter_select_enabled": self.filter_select_enabled,
            "name": self.name,
            "datasource_name": self.datasource_name,
            "table_name": self.datasource_name,
            "type": self.type,
            "catalog": self.catalog,
            "schema": self.schema or None,
            "offset": self.offset,
            "cache_timeout": self.cache_timeout,
            "params": self.params,
            "perm": self.perm,
            "edit_url": self.url,
            "sql": self.sql,
            "columns": [o.data for o in self.columns],
            "metrics": [o.data for o in self.metrics],
            "order_by_choices": self.order_by_choices,
            "owners": [owner.id for owner in self.owners],
            "verbose_map": self.verbose_map,
            "select_star": self.select_star,
        }

    def data_for_slices(self, slices: list[Slice]) -> dict[str, Any]:
        data: dict[str, Any] = self.data
        metric_names: set[str] = set()
        column_names: set[str] = set()
        for slc in slices:
            form_data: dict[str, Any] = slc.form_data
            for metric_param in METRIC_FORM_DATA_PARAMS:
                for metric in utils.as_list(form_data.get(metric_param) or []):
                    metric_names.add(utils.get_metric_name(metric))
                    if utils.is_adhoc_metric(metric):
                        column_ = metric.get("column") or {}
                        if (column_name := column_.get("column_name")) is not None:
                            column_names.add(column_name)
            column_names.update(
                filter_["subject"]
                for filter_ in form_data.get("adhoc_filters") or []
                if filter_.get("clause") == "WHERE" and filter_.get("subject")
            )
            column_names.update(
                filter_config["column"]
                for filter_config in form_data.get("filter_configs") or []
                if "column" in filter_config
            )
            try:
                query_context = slc.get_query_context()
            except DatasetNotFoundError:
                query_context = None
            if query_context:
                column_names.update(
                    [
                        utils.get_column_name(column_)
                        for query in query_context.queries
                        for column_ in query.columns
                    ]
                    or []
                )
            else:
                _columns = [
                    (
                        utils.get_column_name(column_)
                        if utils.is_adhoc_column(column_)
                        else column_
                    )
                    for column_param in COLUMN_FORM_DATA_PARAMS
                    for column_ in utils.as_list(form_data.get(column_param) or [])
                ]
                column_names.update(_columns)
        filtered_metrics = [
            metric
            for metric in data["metrics"]
            if metric["metric_name"] in metric_names
        ]
        filtered_columns: list[dict[str, Any]] = []
        column_types: set[utils.GenericDataType] = set()
        for column_ in data["columns"]:
            generic_type = column_.get("type_generic")
            if generic_type is not None:
                column_types.add(generic_type)
            if column_["column_name"] in column_names:
                filtered_columns.append(column_)
        data["column_types"] = list(column_types)
        del data["description"]
        data.update({"metrics": filtered_metrics})
        data.update({"columns": filtered_columns})
        all_columns: dict[str, str] = {
            column_["column_name"]: column_["verbose_name"] or column_["column_name"]
            for column_ in filtered_columns
        }
        verbose_map: dict[str, str] = {"__timestamp": "Time"}
        verbose_map.update(
            {
                metric["metric_name"]: metric["verbose_name"] or metric["metric_name"]
                for metric in filtered_metrics
            }
        )
        verbose_map.update(all_columns)
        data["verbose_map"] = verbose_map
        data["column_names"] = set(all_columns.values()) | set(self.column_names)
        return data

    @staticmethod
    def filter_values_handler(
        values: Optional[FilterValues],
        operator: str,
        target_generic_type: utils.GenericDataType,
        target_native_type: Optional[str] = None,
        is_list_target: bool = False,
        db_engine_spec: Optional[type[BaseEngineSpec]] = None,
        db_extra: Optional[dict[str, Any]] = None,
    ) -> Optional[FilterValues]:
        if values is None:
            return None

        def handle_single_value(value: Optional[FilterValue]) -> Optional[FilterValue]:
            if operator == utils.FilterOperator.TEMPORAL_RANGE:
                return value
            if (
                isinstance(value, (float, int))
                and target_generic_type == utils.GenericDataType.TEMPORAL
                and target_native_type is not None
                and db_engine_spec is not None
            ):
                value = db_engine_spec.convert_dttm(
                    target_type=target_native_type,
                    dttm=datetime.utcfromtimestamp(value / 1000),
                    db_extra=db_extra,
                )
                value = literal_column(value)
            if isinstance(value, str):
                value = value.strip("\t\n")
                if (
                    target_generic_type == utils.GenericDataType.NUMERIC
                    and operator
                    not in {utils.FilterOperator.ILIKE, utils.FilterOperator.LIKE}
                ):
                    return utils.cast_to_num(value)
                if value == NULL_STRING:
                    return None
                if value == EMPTY_STRING:
                    return ""
            if target_generic_type == utils.GenericDataType.BOOLEAN:
                return utils.cast_to_boolean(value)
            return value

        if isinstance(values, (list, tuple)):
            values = [handle_single_value(v) for v in values]  # type: ignore
        else:
            values = handle_single_value(values)
        if is_list_target and not isinstance(values, (tuple, list)):
            values = [values]  # type: ignore
        elif not is_list_target and isinstance(values, (tuple, list)):
            values = values[0] if values else None
        return values

    def external_metadata(self) -> list[ResultSetColumnType]:
        raise NotImplementedError()

    def get_query_str(self, query_obj: QueryObjectDict) -> str:
        raise NotImplementedError()

    def query(self, query_obj: QueryObjectDict) -> QueryResult:
        raise NotImplementedError()

    @staticmethod
    def default_query(qry: Query) -> Query:
        return qry

    def get_column(self, column_name: Optional[str]) -> Optional[TableColumn]:
        if not column_name:
            return None
        for col in self.columns:
            if col.column_name == column_name:
                return col
        return None

    @staticmethod
    def get_fk_many_from_list(
        object_list: list[dict[str, Any]],
        fkmany: list[Any],
        fkmany_class: type[TableColumn | SqlMetric],
        key_attr: str,
    ) -> list[Any]:
        object_dict = {o.get(key_attr): o for o in object_list}
        fkmany = [o for o in fkmany if getattr(o, key_attr) in object_dict]
        for fk in fkmany:
            obj = object_dict.get(getattr(fk, key_attr))
            if obj:
                for attr in fkmany_class.update_from_object_fields:
                    setattr(fk, attr, obj.get(attr))
        new_fks: list[Any] = []
        orm_keys = [getattr(o, key_attr) for o in fkmany]
        for obj in object_list:
            key = obj.get(key_attr)
            if key not in orm_keys:
                del obj["id"]
                orm_kwargs: dict[str, Any] = {}
                for k in obj:
                    if k in fkmany_class.update_from_object_fields and k in obj:
                        orm_kwargs[k] = obj[k]
                new_obj = fkmany_class(**orm_kwargs)
                new_fks.append(new_obj)
        fkmany += new_fks
        return fkmany

    def update_from_object(self, obj: dict[str, Any]) -> None:
        for attr in self.update_from_object_fields:
            setattr(self, attr, obj.get(attr))
        self.owners = obj.get("owners", [])
        metrics: list[dict[str, Any]] = obj["metrics"] if "metrics" in obj else []
        self.metrics = self.get_fk_many_from_list(metrics, self.metrics, SqlMetric, "metric_name")
        columns: list[dict[str, Any]] = obj["columns"] if "columns" in obj else []
        self.columns = self.get_fk_many_from_list(columns, self.columns, TableColumn, "column_name")

    def get_extra_cache_keys(self, query_obj: QueryObjectDict) -> list[Hashable]:
        return []

    def __hash__(self) -> int:
        return hash(self.uid)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseDatasource):
            return NotImplemented
        return self.uid == other.uid

    def raise_for_access(self) -> None:
        security_manager.raise_for_access(datasource=self)

    @classmethod
    def get_datasource_by_name(
        cls,
        datasource_name: str,
        catalog: Optional[str],
        schema: str,
        database_name: str,
    ) -> Optional[BaseDatasource]:
        raise NotImplementedError()

    def get_template_processor(self, **kwargs: Any) -> BaseTemplateProcessor:
        raise NotImplementedError()

    def text(self, clause: str) -> TextClause:
        raise NotImplementedError()

    def get_sqla_row_level_filters(
        self,
        template_processor: Optional[BaseTemplateProcessor] = None,
    ) -> list[TextClause]:
        template_processor = template_processor or self.get_template_processor()
        all_filters: list[TextClause] = []
        filter_groups: dict[Union[int, str], list[TextClause]] = defaultdict(list)
        try:
            for filter_ in security_manager.get_rls_filters(self):
                clause = self.text(f"({template_processor.process_template(filter_.clause)})")
                if filter_.group_key:
                    filter_groups[filter_.group_key].append(clause)
                else:
                    all_filters.append(clause)
            if is_feature_enabled("EMBEDDED_SUPERSET"):
                for rule in security_manager.get_guest_rls_filters(self):
                    clause = self.text(f"({template_processor.process_template(rule['clause'])})")
                    all_filters.append(clause)
            grouped_filters = [or_(*clauses) for clauses in filter_groups.values()]
            all_filters.extend(grouped_filters)
            return all_filters
        except TemplateError as ex:
            raise QueryObjectValidationError(
                _("Error in jinja expression in RLS filters: %(msg)s", msg=ex.message)
            ) from ex


class AnnotationDatasource(BaseDatasource):
    cache_timeout: int = 0
    changed_on: Any = None
    type: str = "annotation"
    column_names: list[str] = [
        "created_on",
        "changed_on",
        "id",
        "start_dttm",
        "end_dttm",
        "layer_id",
        "short_descr",
        "long_descr",
        "json_metadata",
        "created_by_fk",
        "changed_by_fk",
    ]

    def query(self, query_obj: QueryObjectDict) -> QueryResult:
        error_message: Optional[str] = None
        qry = db.session.query(Annotation)
        qry = qry.filter(Annotation.layer_id == query_obj["filter"][0]["val"])
        if query_obj["from_dttm"]:
            qry = qry.filter(Annotation.start_dttm >= query_obj["from_dttm"])
        if query_obj["to_dttm"]:
            qry = qry.filter(Annotation.end_dttm <= query_obj["to_dttm"])
        status: QueryStatus = QueryStatus.SUCCESS
        try:
            df = pd.read_sql_query(qry.statement, db.engine)
        except Exception as ex:  # pylint: disable=broad-except
            df = pd.DataFrame()
            status = QueryStatus.FAILED
            logger.exception(ex)
            error_message = utils.error_msg_from_exception(ex)
        return QueryResult(
            status=status,
            df=df,
            duration=timedelta(0),
            query="",
            error_message=error_message,
        )

    def get_query_str(self, query_obj: QueryObjectDict) -> str:
        raise NotImplementedError()

    def values_for_column(self, column_name: str, limit: int = 10000) -> list[Any]:
        raise NotImplementedError()


class TableColumn(AuditMixinNullable, ImportExportMixin, CertificationMixin, Model):
    __tablename__ = "table_columns"
    __table_args__ = (UniqueConstraint("table_id", "column_name"),)

    id: int = Column(Integer, primary_key=True)
    column_name: str = Column(String(255), nullable=False)
    verbose_name: Optional[str] = Column(String(1024))
    is_active: bool = Column(Boolean, default=True)
    type: Any = Column(Text)
    advanced_data_type: Any = Column(String(255))
    groupby: bool = Column(Boolean, default=True)
    filterable: bool = Column(Boolean, default=True)
    description: Any = Column(utils.MediumText())
    table_id: Any = Column(Integer, ForeignKey("tables.id", ondelete="CASCADE"))
    is_dttm: bool = Column(Boolean, default=False)
    expression: Any = Column(utils.MediumText())
    python_date_format: Any = Column(String(255))
    extra: Any = Column(Text)

    table: Mapped[SqlaTable] = relationship(
        "SqlaTable",
        back_populates="columns",
    )

    export_fields: list[str] = [
        "table_id",
        "column_name",
        "verbose_name",
        "is_dttm",
        "is_active",
        "type",
        "advanced_data_type",
        "groupby",
        "filterable",
        "expression",
        "description",
        "python_date_format",
        "extra",
    ]

    update_from_object_fields: list[str] = [s for s in export_fields if s not in ("table_id",)]
    export_parent: str = "table"

    def __init__(self, **kwargs: Any) -> None:
        self._database: Optional[Database] = kwargs.pop("database", None)
        super().__init__(**kwargs)

    @reconstructor
    def init_on_load(self) -> None:
        self._database = None

    def __repr__(self) -> str:
        return str(self.column_name)

    @property
    def is_boolean(self) -> bool:
        return self.type_generic == utils.GenericDataType.BOOLEAN

    @property
    def is_numeric(self) -> bool:
        return self.type_generic == utils.GenericDataType.NUMERIC

    @property
    def is_string(self) -> bool:
        return self.type_generic == utils.GenericDataType.STRING

    @property
    def is_temporal(self) -> bool:
        if self.is_dttm is not None:
            return self.is_dttm
        return self.type_generic == utils.GenericDataType.TEMPORAL

    @property
    def database(self) -> Database:
        return self.table.database if self.table else self._database  # type: ignore

    @property
    def db_engine_spec(self) -> type[BaseEngineSpec]:
        return self.database.db_engine_spec

    @property
    def db_extra(self) -> dict[str, Any]:
        return self.database.get_extra()

    @property
    def type_generic(self) -> Optional[utils.GenericDataType]:
        if self.is_dttm:
            return utils.GenericDataType.TEMPORAL
        column_spec = self.db_engine_spec.get_column_spec(self.type, db_extra=self.db_extra)
        return column_spec.generic_type if column_spec else None

    def get_sqla_col(
        self,
        label: Optional[str] = None,
        template_processor: Optional[BaseTemplateProcessor] = None,
    ) -> Column:
        label = label or self.column_name
        db_engine_spec: type[BaseEngineSpec] = self.db_engine_spec
        column_spec = db_engine_spec.get_column_spec(self.type, db_extra=self.db_extra)
        type_ = column_spec.sqla_type if column_spec else None
        if self.expression:
            expression: str = self.expression
            if template_processor:
                expression = template_processor.process_template(expression)
            col: ColumnClause = literal_column(expression, type_=type_)
        else:
            col = column(self.column_name, type_=type_)
        col = self.database.make_sqla_column_compatible(col, label)
        return col

    @property
    def datasource(self) -> RelationshipProperty:
        return self.table

    def get_timestamp_expression(
        self,
        time_grain: Optional[str],
        label: Optional[str] = None,
        template_processor: Optional[BaseTemplateProcessor] = None,
    ) -> Union[TimestampExpression, Label]:
        label = label or utils.DTTM_ALIAS
        pdf: Any = self.python_date_format
        is_epoch: bool = pdf in ("epoch_s", "epoch_ms")
        column_spec = self.db_engine_spec.get_column_spec(self.type, db_extra=self.db_extra)
        type_ = column_spec.sqla_type if column_spec else DateTime
        if not self.expression and not time_grain and not is_epoch:
            sqla_col: Column = column(self.column_name, type_=type_)
            return self.database.make_sqla_column_compatible(sqla_col, label)
        if self.expression:
            expression: str = self.expression
            if template_processor:
                expression = template_processor.process_template(expression)
            col = literal_column(expression, type_=type_)
        else:
            col = column(self.column_name, type_=type_)
        time_expr = self.db_engine_spec.get_timestamp_expr(col, pdf, time_grain)
        return self.database.make_sqla_column_compatible(time_expr, label)

    @property
    def data(self) -> dict[str, Any]:
        attrs: tuple[str, ...] = (
            "advanced_data_type",
            "certification_details",
            "certified_by",
            "column_name",
            "description",
            "expression",
            "filterable",
            "groupby",
            "id",
            "is_certified",
            "is_dttm",
            "python_date_format",
            "type",
            "type_generic",
            "verbose_name",
            "warning_markdown",
        )
        return {s: getattr(self, s) for s in attrs if hasattr(self, s)}


class SqlMetric(AuditMixinNullable, ImportExportMixin, CertificationMixin, Model):
    __tablename__ = "sql_metrics"
    __table_args__ = (UniqueConstraint("table_id", "metric_name"),)

    id: int = Column(Integer, primary_key=True)
    metric_name: str = Column(String(255), nullable=False)
    verbose_name: Optional[str] = Column(String(1024))
    metric_type: Any = Column(String(32))
    description: Any = Column(utils.MediumText())
    d3format: Optional[str] = Column(String(128))
    currency: Optional[str] = Column(String(128))
    warning_text: Any = Column(Text)
    table_id: Any = Column(Integer, ForeignKey("tables.id", ondelete="CASCADE"))
    expression: Any = Column(utils.MediumText(), nullable=False)
    extra: Any = Column(Text)

    table: Mapped[SqlaTable] = relationship(
        "SqlaTable",
        back_populates="metrics",
    )

    export_fields: list[str] = [
        "metric_name",
        "verbose_name",
        "metric_type",
        "table_id",
        "expression",
        "description",
        "d3format",
        "currency",
        "extra",
        "warning_text",
    ]
    update_from_object_fields: list[str] = [s for s in export_fields if s != "table_id"]
    export_parent: str = "table"

    def __repr__(self) -> str:
        return str(self.metric_name)

    def get_sqla_col(
        self,
        label: Optional[str] = None,
        template_processor: Optional[BaseTemplateProcessor] = None,
    ) -> Column:
        label = label or self.metric_name
        expression: str = self.expression
        if template_processor:
            expression = template_processor.process_template(expression)
        sqla_col: ColumnClause = literal_column(expression)
        return self.table.database.make_sqla_column_compatible(sqla_col, label)

    @property
    def perm(self) -> Optional[str]:
        return (
            ("{parent_name}.[{obj.metric_name}](id:{obj.id})").format(
                obj=self, parent_name=self.table.full_name
            )
            if self.table
            else None
        )

    def get_perm(self) -> Optional[str]:
        return self.perm

    @property
    def currency_json(self) -> Optional[dict[str, Optional[str]]]:
        try:
            return json.loads(self.currency or "{}") or None
        except (TypeError, json.JSONDecodeError) as exc:
            logger.error(
                "Unable to load currency json: %r. Leaving empty.", exc, exc_info=True
            )
            return None

    @property
    def data(self) -> dict[str, Any]:
        attrs: tuple[str, ...] = (
            "certification_details",
            "certified_by",
            "currency",
            "d3format",
            "description",
            "expression",
            "id",
            "is_certified",
            "metric_name",
            "warning_markdown",
            "warning_text",
            "verbose_name",
        )
        return {s: getattr(self, s) for s in attrs}


sqlatable_user: DBTable = DBTable(
    "sqlatable_user",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("ab_user.id", ondelete="CASCADE")),
    Column("table_id", Integer, ForeignKey("tables.id", ondelete="CASCADE")),
)


class SqlaTable(Model, BaseDatasource, ExploreMixin):
    type: str = "table"
    query_language: str = "sql"
    is_rls_supported: bool = True
    columns: Mapped[list[TableColumn]] = relationship(
        TableColumn,
        back_populates="table",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    metrics: Mapped[list[SqlMetric]] = relationship(
        SqlMetric,
        back_populates="table",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    metric_class: type[SqlMetric] = SqlMetric
    column_class: type[TableColumn] = TableColumn
    owner_class: type[User] = security_manager.user_model

    __tablename__ = "tables"
    __table_args__ = (UniqueConstraint("database_id", "catalog", "schema", "table_name"),)

    table_name: str = Column(String(250), nullable=False)
    main_dttm_col: Any = Column(String(250))
    database_id: Any = Column(Integer, ForeignKey("dbs.id"), nullable=False)
    fetch_values_predicate: Any = Column(Text)
    owners: list[User] = relationship(owner_class, secondary=sqlatable_user, backref="tables")
    database: Database = relationship(
        "Database",
        backref=backref("tables", cascade="all, delete-orphan"),
        foreign_keys=[database_id],
    )
    schema: Any = Column(String(255))
    catalog: Optional[str] = Column(String(256), nullable=True, default=None)
    sql: Any = Column(utils.MediumText())
    is_sqllab_view: bool = Column(Boolean, default=False)
    template_params: Any = Column(Text)
    extra: Any = Column(Text)
    normalize_columns: bool = Column(Boolean, default=False)
    always_filter_main_dttm: bool = Column(Boolean, default=False)

    baselink: str = "tablemodelview"

    export_fields: list[str] = [
        "table_name",
        "main_dttm_col",
        "description",
        "default_endpoint",
        "database_id",
        "offset",
        "cache_timeout",
        "catalog",
        "schema",
        "sql",
        "params",
        "template_params",
        "filter_select_enabled",
        "fetch_values_predicate",
        "extra",
        "normalize_columns",
        "always_filter_main_dttm",
    ]
    update_from_object_fields: list[str] = [f for f in export_fields if f != "database_id"]
    export_parent: str = "database"
    export_children: list[str] = ["metrics", "columns"]

    sqla_aggregations: dict[str, Any] = {
        "COUNT_DISTINCT": lambda column_name: sa.func.COUNT(sa.distinct(column_name)),
        "COUNT": sa.func.COUNT,
        "SUM": sa.func.SUM,
        "AVG": sa.func.AVG,
        "MIN": sa.func.MIN,
        "MAX": sa.func.MAX,
    }

    def __repr__(self) -> str:
        return self.name

    @property
    def db_extra(self) -> dict[str, Any]:
        return self.database.get_extra()

    @staticmethod
    def _apply_cte(sql: str, cte: Optional[str]) -> str:
        if cte:
            sql = f"{cte}\n{sql}"
        return sql

    @property
    def db_engine_spec(self) -> type[BaseEngineSpec]:
        return self.database.db_engine_spec

    @property
    def changed_by_name(self) -> str:
        if not self.changed_by:
            return ""
        return str(self.changed_by)

    @property
    def connection(self) -> str:
        return str(self.database)

    @property
    def description_markeddown(self) -> str:
        return utils.markdown(self.description)

    @property
    def datasource_name(self) -> str:
        return self.table_name

    @property
    def datasource_type(self) -> str:
        return self.type

    @property
    def database_name(self) -> str:
        return self.database.name

    @classmethod
    def get_datasource_by_name(
        cls,
        datasource_name: str,
        catalog: Optional[str],
        schema: Optional[str],
        database_name: str,
    ) -> Optional[SqlaTable]:
        schema_val: Optional[str] = schema or None
        query = (
            db.session.query(cls)
            .join(Database)
            .filter(cls.table_name == datasource_name)
            .filter(Database.database_name == database_name)
            .filter(cls.catalog == catalog)
        )
        for tbl in query.all():
            if schema_val == (tbl.schema or None):
                return tbl
        return None

    @property
    def link(self) -> Markup:
        name: str = escape(self.name)
        anchor: str = f'<a target="_blank" href="{self.explore_url}">{name}</a>'
        return Markup(anchor)

    def get_catalog_perm(self) -> Optional[str]:
        return security_manager.get_catalog_perm(
            self.database.database_name,
            self.catalog,
        )

    def get_schema_perm(self) -> Optional[str]:
        return security_manager.get_schema_perm(
            self.database.database_name,
            self.catalog,
            self.schema or None,
        )

    def get_perm(self) -> str:
        if self.database is None:
            raise DatasetInvalidPermissionEvaluationException()
        return f"[{self.database}].[{self.table_name}](id:{self.id})"

    @hybrid_property
    def name(self) -> str:
        return self.schema + "." + self.table_name if self.schema else self.table_name

    @property
    def full_name(self) -> str:
        return utils.get_datasource_full_name(
            self.database,
            self.table_name,
            catalog=self.catalog,
            schema=self.schema,
        )

    @property
    def dttm_cols(self) -> list[str]:
        l: list[str] = [c.column_name for c in self.columns if c.is_dttm]
        if self.main_dttm_col and self.main_dttm_col not in l:
            l.append(self.main_dttm_col)
        return l

    @property
    def num_cols(self) -> list[str]:
        return [c.column_name for c in self.columns if c.is_numeric]

    @property
    def any_dttm_col(self) -> Optional[str]:
        cols: list[str] = self.dttm_cols
        return cols[0] if cols else None

    @property
    def html(self) -> str:
        df: pd.DataFrame = pd.DataFrame((c.column_name, c.type) for c in self.columns)
        df.columns = ["field", "type"]
        return df.to_html(
            index=False,
            classes=("dataframe table table-striped table-bordered table-condensed"),
        )

    @property
    def sql_url(self) -> str:
        return self.database.sql_url + "?table_name=" + str(self.table_name)

    def external_metadata(self) -> list[ResultSetColumnType]:
        if self.sql:
            return get_virtual_table_metadata(dataset=self)
        return get_physical_table_metadata(
            database=self.database,
            table=Table(self.table_name, self.schema or None, self.catalog),
            normalize_columns=self.normalize_columns,
        )

    @property
    def time_column_grains(self) -> dict[str, Any]:
        return {
            "time_columns": self.dttm_cols,
            "time_grains": [grain.name for grain in self.database.grains()],
        }

    @property
    def select_star(self) -> Optional[str]:
        return self.database.select_star(
            Table(self.table_name, self.schema or None, self.catalog),
            show_cols=False,
            latest_partition=False,
        )

    @property
    def health_check_message(self) -> Optional[str]:
        check: Any = config["DATASET_HEALTH_CHECK"]
        return check(self) if check else None

    @property
    def granularity_sqla(self) -> list[tuple[Any, Any]]:
        return utils.choicify(self.dttm_cols)

    @property
    def time_grain_sqla(self) -> list[tuple[Any, Any]]:
        return [(g.duration, g.name) for g in self.database.grains() or []]

    @property
    def data(self) -> dict[str, Any]:
        data_: dict[str, Any] = super().data
        if self.type == "table":
            data_["granularity_sqla"] = self.granularity_sqla
            data_["time_grain_sqla"] = self.time_grain_sqla
            data_["main_dttm_col"] = self.main_dttm_col
            data_["fetch_values_predicate"] = self.fetch_values_predicate
            data_["template_params"] = self.template_params
            data_["is_sqllab_view"] = self.is_sqllab_view
            data_["health_check_message"] = self.health_check_message
            data_["extra"] = self.extra
            data_["owners"] = self.owners_data
            data_["always_filter_main_dttm"] = self.always_filter_main_dttm
            data_["normalize_columns"] = self.normalize_columns
        return data_

    @property
    def extra_dict(self) -> dict[str, Any]:
        try:
            return json.loads(self.extra)
        except (TypeError, json.JSONDecodeError):
            return {}

    def get_fetch_values_predicate(
        self, template_processor: Optional[BaseTemplateProcessor] = None
    ) -> TextClause:
        fetch_values_predicate: Any = self.fetch_values_predicate
        if template_processor:
            fetch_values_predicate = template_processor.process_template(fetch_values_predicate)
        try:
            return self.text(fetch_values_predicate)
        except TemplateError as ex:
            raise QueryObjectValidationError(
                _("Error in jinja expression in fetch values predicate: %(msg)s", msg=ex.message)
            ) from ex

    def get_template_processor(self, **kwargs: Any) -> BaseTemplateProcessor:
        return get_template_processor(table=self, database=self.database, **kwargs)

    def get_query_str(self, query_obj: QueryObjectDict) -> str:
        query_str_ext = self.get_query_str_extended(query_obj)
        all_queries = query_str_ext.prequeries + [query_str_ext.sql]
        return ";\n\n".join(all_queries) + ";"

    def get_sqla_table(self) -> TableClause:
        tbl: TableClause = table(self.table_name)
        if self.schema:
            tbl.schema = self.schema
        return tbl

    def get_from_clause(
        self, template_processor: Optional[BaseTemplateProcessor] = None
    ) -> tuple[TableClause | Alias, Optional[str]]:
        if not self.is_virtual:
            return self.get_sqla_table(), None
        return super().get_from_clause(template_processor)  # type: ignore

    def adhoc_metric_to_sqla(
        self,
        metric: AdhocMetric,
        columns_by_name: dict[str, TableColumn],
        template_processor: Optional[BaseTemplateProcessor] = None,
    ) -> ColumnElement:
        expression_type: Any = metric.get("expressionType")
        label: str = utils.get_metric_name(metric)
        if expression_type == utils.AdhocMetricExpressionType.SIMPLE:
            metric_column: dict[str, Any] = metric.get("column") or {}
            column_name: str = cast(str, metric_column.get("column_name"))
            table_column: Optional[TableColumn] = columns_by_name.get(column_name)
            if table_column:
                sqla_column: Column = table_column.get_sqla_col(template_processor=template_processor)
            else:
                sqla_column = column(column_name)
            sqla_metric: Any = self.sqla_aggregations[metric["aggregate"]](sqla_column)
        elif expression_type == utils.AdhocMetricExpressionType.SQL:
            try:
                expression: str = self._process_sql_expression(
                    expression=metric["sqlExpression"],
                    database_id=self.database_id,
                    engine=self.database.backend,
                    schema=self.schema,
                    template_processor=template_processor,
                )
            except SupersetSecurityException as ex:
                raise QueryObjectValidationError(ex.message) from ex
            sqla_metric = literal_column(expression)
        else:
            raise QueryObjectValidationError("Adhoc metric expressionType is invalid")
        return self.make_sqla_column_compatible(sqla_metric, label)

    def adhoc_column_to_sqla(
        self,
        col: AdhocColumn,
        force_type_check: bool = False,
        template_processor: Optional[BaseTemplateProcessor] = None,
    ) -> ColumnElement:
        label: str = utils.get_column_name(col)
        try:
            expression: str = self._process_sql_expression(
                expression=col["sqlExpression"],
                database_id=self.database_id,
                engine=self.database.backend,
                schema=self.schema,
                template_processor=template_processor,
            )
        except SupersetSecurityException as ex:
            raise QueryObjectValidationError(ex.message) from ex
        time_grain: Optional[str] = col.get("timeGrain")
        has_timegrain: bool = col.get("columnType") == "BASE_AXIS" and time_grain is not None
        is_dttm: bool = False
        pdf: Optional[str] = None
        if (col_in_metadata := self.get_column(expression)) is not None:
            sqla_column: Column = col_in_metadata.get_sqla_col(template_processor=template_processor)
            is_dttm = col_in_metadata.is_temporal
            pdf = col_in_metadata.python_date_format
        else:
            sqla_column = literal_column(expression)
            if has_timegrain or force_type_check:
                try:
                    tbl, _ = self.get_from_clause(template_processor)
                    qry = sa.select([sqla_column]).limit(1).select_from(tbl)
                    sql: str = self.database.compile_sqla_query(qry, catalog=self.catalog, schema=self.schema)
                    col_desc = get_columns_description(self.database, self.catalog, self.schema or None, sql)
                    if not col_desc:
                        raise SupersetGenericDBErrorException("Column not found")
                    is_dttm = col_desc[0]["is_dttm"]  # type: ignore
                except SupersetGenericDBErrorException as ex:
                    raise ColumnNotFoundException(message=str(ex)) from ex
        if is_dttm and has_timegrain:
            sqla_column = self.db_engine_spec.get_timestamp_expr(col=sqla_column, pdf=pdf, time_grain=time_grain)
        return self.make_sqla_column_compatible(sqla_column, label)

    def make_orderby_compatible(
        self, select_exprs: list[ColumnElement], orderby_exprs: list[ColumnElement]
    ) -> None:
        if self.db_engine_spec.allows_alias_to_source_column:
            return

        def is_alias_used_in_orderby(col: ColumnElement) -> bool:
            if not isinstance(col, Label):
                return False
            regexp = re.compile(f"\\(.*\\b{re.escape(col.name)}\\b.*\\)", re.IGNORECASE)
            return any(regexp.search(str(x)) for x in orderby_exprs)

        for col in select_exprs:
            if is_alias_used_in_orderby(col):
                col.name = f"{col.name}__"

    def text(self, clause: str) -> TextClause:
        return self.db_engine_spec.get_text_clause(clause)

    def _get_series_orderby(
        self,
        series_limit_metric: Metric,
        metrics_by_name: dict[str, SqlMetric],
        columns_by_name: dict[str, TableColumn],
        template_processor: Optional[BaseTemplateProcessor] = None,
    ) -> Column:
        if utils.is_adhoc_metric(series_limit_metric):
            assert isinstance(series_limit_metric, dict)
            ob: ColumnElement = self.adhoc_metric_to_sqla(series_limit_metric, columns_by_name)
        elif isinstance(series_limit_metric, str) and series_limit_metric in metrics_by_name:
            ob = metrics_by_name[series_limit_metric].get_sqla_col(template_processor=template_processor)
        else:
            raise QueryObjectValidationError(
                _("Metric '%(metric)s' does not exist", metric=series_limit_metric)
            )
        return ob

    def _normalize_prequery_result_type(
        self,
        row: pd.Series,
        dimension: str,
        columns_by_name: dict[str, TableColumn],
    ) -> Union[str, int, float, bool, Text]:
        value: Any = row[dimension]
        if isinstance(value, np.generic):
            value = value.item()
        column_ = columns_by_name.get(dimension)
        db_extra: dict[str, Any] = self.database.get_extra()
        if column_ and column_.type and column_.is_temporal and isinstance(value, str):
            sql: Optional[str] = self.db_engine_spec.convert_dttm(
                column_.type, dateutil.parser.parse(value), db_extra=db_extra
            )
            if sql:
                value = self.text(sql)
        return value

    def _get_top_groups(
        self,
        df: pd.DataFrame,
        dimensions: list[str],
        groupby_exprs: dict[str, Any],
        columns_by_name: dict[str, TableColumn],
    ) -> ColumnElement:
        groups: list[Any] = []
        for _unused, row in df.iterrows():
            group_terms: list[Any] = []
            for dimension in dimensions:
                value = self._normalize_prequery_result_type(row, dimension, columns_by_name)
                group_terms.append(groupby_exprs[dimension] == value)
            groups.append(and_(*group_terms))
        return or_(*groups)

    def query(self, query_obj: QueryObjectDict) -> QueryResult:
        qry_start_dttm: datetime = datetime.now()
        query_str_ext = self.get_query_str_extended(query_obj)
        sql: str = query_str_ext.sql
        status: QueryStatus = QueryStatus.SUCCESS
        errors: Optional[list[dict[str, Any]]] = None
        error_message: Optional[str] = None

        def assign_column_label(df: pd.DataFrame) -> pd.DataFrame:
            labels_expected: list[str] = query_str_ext.labels_expected
            if df is not None and not df.empty:
                if len(df.columns) < len(labels_expected):
                    raise QueryObjectValidationError(_("Db engine did not return all queried columns"))
                if len(df.columns) > len(labels_expected):
                    df = df.iloc[:, 0 : len(labels_expected)]
                df.columns = labels_expected
            return df

        try:
            df: pd.DataFrame = self.database.get_df(
                sql,
                self.catalog,
                self.schema or None,
                mutator=assign_column_label,
            )
        except (SupersetErrorException, SupersetErrorsException):
            raise
        except Exception as ex:  # pylint: disable=broad-except
            df = pd.DataFrame()
            status = QueryStatus.FAILED
            logger.warning("Query %s on schema %s failed", sql, self.schema, exc_info=True)
            db_engine_spec = self.db_engine_spec
            errors = [dataclasses.asdict(error) for error in db_engine_spec.extract_errors(ex)]
            error_message = utils.error_msg_from_exception(ex)
        return QueryResult(
            applied_template_filters=query_str_ext.applied_template_filters,
            applied_filter_columns=query_str_ext.applied_filter_columns,
            rejected_filter_columns=query_str_ext.rejected_filter_columns,
            status=status,
            df=df,
            duration=datetime.now() - qry_start_dttm,
            query=sql,
            errors=errors,
            error_message=error_message,
        )

    def get_sqla_table_object(self) -> Table:
        return self.database.get_table(Table(self.table_name, self.schema or None, self.catalog))

    def fetch_metadata(self) -> MetadataResult:
        new_columns: list[dict[str, Any]] = self.external_metadata()
        metrics = [
            SqlMetric(**metric)
            for metric in self.database.get_metrics(Table(self.table_name, self.schema or None, self.catalog))
        ]
        any_date_col: Optional[str] = None
        db_engine_spec: type[BaseEngineSpec] = self.db_engine_spec
        old_columns: list[TableColumn] = (
            db.session.query(TableColumn).filter(TableColumn.table_id == self.id).all()
            if self.id
            else self.columns
        )
        old_columns_by_name: dict[str, TableColumn] = {col.column_name: col for col in old_columns}
        results: MetadataResult = MetadataResult(
            removed=[col for col in old_columns_by_name if col not in {coldef["column_name"] for coldef in new_columns}]
        )
        columns: list[TableColumn] = []
        for col in new_columns:
            old_column: Optional[TableColumn] = old_columns_by_name.pop(col["column_name"], None)
            if not old_column:
                results.added.append(col["column_name"])
                new_column = TableColumn(
                    column_name=col["column_name"],
                    type=col["type"],
                    table=self,
                )
                new_column.is_dttm = new_column.is_temporal
                db_engine_spec.alter_new_orm_column(new_column)
            else:
                new_column = old_column
                if new_column.type != col["type"]:
                    results.modified.append(col["column_name"])
                new_column.type = col["type"]
                new_column.expression = ""
            new_column.groupby = True
            new_column.filterable = True
            columns.append(new_column)
            if not any_date_col and new_column.is_temporal:
                any_date_col = col["column_name"]
        columns.extend([col for col in old_columns if col.expression])
        self.columns = columns
        if not self.main_dttm_col:
            self.main_dttm_col = any_date_col
        self.add_missing_metrics(metrics)
        config["SQLA_TABLE_MUTATOR"](self)
        db.session.merge(self)
        return results

    @classmethod
    def query_datasources_by_name(
        cls,
        database: Database,
        datasource_name: str,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> list[SqlaTable]:
        filters: dict[str, Any] = {"database_id": database.id, "table_name": datasource_name}
        if catalog:
            filters["catalog"] = catalog
        if schema:
            filters["schema"] = schema
        return db.session.query(cls).filter_by(**filters).all()

    @classmethod
    def query_datasources_by_permissions(
        cls,
        database: Database,
        permissions: set[str],
        catalog_perms: set[str],
        schema_perms: set[str],
    ) -> list[SqlaTable]:
        filters = [
            method.in_(perms)
            for method, perms in zip(
                (SqlaTable.perm, SqlaTable.schema_perm, SqlaTable.catalog_perm),
                (permissions, schema_perms, catalog_perms),
                strict=False,
            )
            if perms
        ]
        return (
            db.session.query(cls)
            .filter_by(database_id=database.id)
            .filter(or_(*filters))
            .all()
        )

    @classmethod
    def get_eager_sqlatable_datasource(cls, datasource_id: int) -> SqlaTable:
        return (
            db.session.query(cls)
            .options(sa.orm.subqueryload(cls.columns), sa.orm.subqueryload(cls.metrics))
            .filter_by(id=datasource_id)
            .one()
        )

    @classmethod
    def get_all_datasources(cls) -> list[SqlaTable]:
        qry: Query = db.session.query(cls)
        qry = cls.default_query(qry)
        return qry.all()

    @staticmethod
    def default_query(qry: Query) -> Query:
        return qry.filter_by(is_sqllab_view=False)

    def has_extra_cache_key_calls(self, query_obj: QueryObjectDict) -> bool:
        templatable_statements: list[str] = []
        if self.sql:
            templatable_statements.append(self.sql)
        if self.fetch_values_predicate:
            templatable_statements.append(self.fetch_values_predicate)
        extras: dict[str, Any] = query_obj.get("extras", {})
        if "where" in extras:
            templatable_statements.append(extras["where"])
        if "having" in extras:
            templatable_statements.append(extras["having"])
        if columns := query_obj.get("columns"):
            calculated_columns: dict[str, Any] = {
                c.column_name: c.expression for c in self.columns if c.expression
            }
            for column_ in columns:
                if utils.is_adhoc_column(column_):
                    templatable_statements.append(column_["sqlExpression"])
                elif isinstance(column_, str) and column_ in calculated_columns:
                    templatable_statements.append(calculated_columns[column_])
        if metrics := query_obj.get("metrics"):
            metrics_by_name: dict[str, Any] = {
                m.metric_name: m.expression for m in self.metrics
            }
            for metric in metrics:
                if utils.is_adhoc_metric(metric) and (sql := metric.get("sqlExpression")):
                    templatable_statements.append(sql)
                elif isinstance(metric, str) and metric in metrics_by_name:
                    templatable_statements.append(metrics_by_name[metric])
        if self.is_rls_supported:
            templatable_statements += [f.clause for f in security_manager.get_rls_filters(self)]
        for statement in templatable_statements:
            if ExtraCache.regex.search(statement):
                return True
        return False

    def get_extra_cache_keys(self, query_obj: QueryObjectDict) -> list[Hashable]:
        extra_cache_keys: list[Hashable] = super().get_extra_cache_keys(query_obj)
        if self.has_extra_cache_key_calls(query_obj):
            sqla_query = self.get_sqla_query(**query_obj)
            extra_cache_keys += sqla_query.extra_cache_keys
        return list(set(extra_cache_keys))

    @property
    def quote_identifier(self) -> Callable[[str], str]:
        return self.database.quote_identifier

    @staticmethod
    def before_update(mapper: Mapper, connection: Connection, target: SqlaTable) -> None:
        target.load_database()
        security_manager.dataset_before_update(mapper, connection, target)

    @staticmethod
    def update_column(mapper: Mapper, connection: Connection, target: SqlMetric | TableColumn) -> None:
        session = inspect(target).session
        session.execute(update(SqlaTable).where(SqlaTable.id == target.table.id))

    @staticmethod
    def after_insert(mapper: Mapper, connection: Connection, target: SqlaTable) -> None:
        target.load_database()
        security_manager.dataset_after_insert(mapper, connection, target)

    @staticmethod
    def after_delete(mapper: Mapper, connection: Connection, sqla_table: SqlaTable) -> None:
        security_manager.dataset_after_delete(mapper, connection, sqla_table)

    def load_database(self: SqlaTable) -> None:
        if self.database_id and (not self.database or self.database.id != self.database_id):
            session = inspect(self).session
            self.database = session.query(Database).filter_by(id=self.database_id).one()


sa.event.listen(SqlaTable, "before_update", SqlaTable.before_update)
sa.event.listen(SqlaTable, "after_insert", SqlaTable.after_insert)
sa.event.listen(SqlaTable, "after_delete", SqlaTable.after_delete)
sa.event.listen(SqlMetric, "after_update", SqlaTable.update_column)
sa.event.listen(TableColumn, "after_update", SqlaTable.update_column)

RLSFilterRoles: DBTable = DBTable(
    "rls_filter_roles",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("role_id", Integer, ForeignKey("ab_role.id"), nullable=False),
    Column("rls_filter_id", Integer, ForeignKey("row_level_security_filters.id")),
)

RLSFilterTables: DBTable = DBTable(
    "rls_filter_tables",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("table_id", Integer, ForeignKey("tables.id")),
    Column("rls_filter_id", Integer, ForeignKey("row_level_security_filters.id")),
)


class RowLevelSecurityFilter(Model, AuditMixinNullable):
    __tablename__ = "row_level_security_filters"
    id: int = Column(Integer, primary_key=True)
    name: str = Column(String(255), unique=True, nullable=False)
    description: Any = Column(Text)
    filter_type: Any = Column(
        Enum(*[filter_type.value for filter_type in utils.RowLevelSecurityFilterType], name="filter_type_enum")
    )
    group_key: Optional[str] = Column(String(255), nullable=True)
    roles: list[Any] = relationship(
        security_manager.role_model,
        secondary=RLSFilterRoles,
        backref="row_level_security_filters",
    )
    tables: list[SqlaTable] = relationship(
        SqlaTable,
        overlaps="table",
        secondary=RLSFilterTables,
        backref="row_level_security_filters",
    )
    clause: str = Column(utils.MediumText(), nullable=False)