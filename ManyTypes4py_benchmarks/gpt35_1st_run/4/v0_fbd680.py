import logging
from typing import Any, Callable, Optional, Dict, List
import yaml
from flask_appbuilder import Model
from sqlalchemy.orm.session import make_transient
from superset import db
from superset.commands.base import BaseCommand
from superset.commands.database.exceptions import DatabaseNotFoundError
from superset.commands.dataset.exceptions import DatasetInvalidError
from superset.commands.importers.exceptions import IncorrectVersionError
from superset.connectors.sqla.models import BaseDatasource, SqlaTable, SqlMetric, TableColumn
from superset.models.core import Database
from superset.utils import json
from superset.utils.decorators import transaction
from superset.utils.dict_import_export import DATABASES_KEY
logger: logging.Logger = logging.getLogger(__name__)

def lookup_sqla_table(table: SqlaTable) -> Optional[SqlaTable]:
    return db.session.query(SqlaTable).join(Database).filter(SqlaTable.table_name == table.table_name, SqlaTable.schema == table.schema, Database.id == table.database_id).first()

def lookup_sqla_database(table: SqlaTable) -> Database:
    database = db.session.query(Database).filter_by(database_name=table.params_dict['database_name']).one_or_none()
    if database is None:
        raise DatabaseNotFoundError
    return database

def import_dataset(i_datasource: BaseDatasource, database_id: Optional[int] = None, import_time: Optional[str] = None) -> int:
    if isinstance(i_datasource, SqlaTable):
        lookup_database = lookup_sqla_database
        lookup_datasource = lookup_sqla_table
    else:
        raise DatasetInvalidError
    return import_datasource(i_datasource, lookup_database, lookup_datasource, import_time, database_id)

def lookup_sqla_metric(metric: SqlMetric) -> Optional[SqlMetric]:
    return db.session.query(SqlMetric).filter(SqlMetric.table_id == metric.table_id, SqlMetric.metric_name == metric.metric_name).first()

def import_metric(metric: SqlMetric) -> SqlMetric:
    return import_simple_obj(metric, lookup_sqla_metric)

def lookup_sqla_column(column: TableColumn) -> Optional[TableColumn]:
    return db.session.query(TableColumn).filter(TableColumn.table_id == column.table_id, TableColumn.column_name == column.column_name).first()

def import_column(column: TableColumn) -> TableColumn:
    return import_simple_obj(column, lookup_sqla_column)

def import_datasource(i_datasource: BaseDatasource, lookup_database: Callable[[BaseDatasource], Database], lookup_datasource: Callable[[BaseDatasource], Optional[BaseDatasource]], import_time: Optional[str] = None, database_id: Optional[int] = None) -> int:
    make_transient(i_datasource)
    logger.info('Started import of the datasource: %s', i_datasource.to_json())
    i_datasource.id = None
    i_datasource.database_id = database_id if database_id else getattr(lookup_database(i_datasource), 'id', None)
    i_datasource.alter_params(import_time=import_time)
    datasource = lookup_datasource(i_datasource)
    if datasource:
        datasource.override(i_datasource)
        db.session.flush()
    else:
        datasource = i_datasource.copy()
        db.session.add(datasource)
        db.session.flush()
    for metric in i_datasource.metrics:
        new_m = metric.copy()
        new_m.table_id = datasource.id
        logger.info('Importing metric %s from the datasource: %s', new_m.to_json(), i_datasource.full_name)
        imported_m = import_metric(new_m)
        if imported_m.metric_name not in [m.metric_name for m in datasource.metrics]:
            datasource.metrics.append(imported_m)
    for column in i_datasource.columns:
        new_c = column.copy()
        new_c.table_id = datasource.id
        logger.info('Importing column %s from the datasource: %s', new_c.to_json(), i_datasource.full_name)
        imported_c = import_column(new_c)
        if imported_c.column_name not in [c.column_name for c in datasource.columns]:
            datasource.columns.append(imported_c)
    db.session.flush()
    return datasource.id

def import_simple_obj(i_obj: Any, lookup_obj: Callable[[Any], Any]) -> Any:
    make_transient(i_obj)
    i_obj.id = None
    i_obj.table = None
    existing_column = lookup_obj(i_obj)
    i_obj.table = None
    if existing_column:
        existing_column.override(i_obj)
        db.session.flush()
        return existing_column
    db.session.add(i_obj)
    db.session.flush()
    return i_obj

def import_from_dict(data: Dict[str, Any], sync: Optional[List[str]] = None) -> None:
    if not sync:
        sync = []
    if isinstance(data, dict):
        logger.info('Importing %d %s', len(data.get(DATABASES_KEY, [])), DATABASES_KEY)
        for database in data.get(DATABASES_KEY, []):
            Database.import_from_dict(database, sync=sync)
    else:
        logger.info('Supplied object is not a dictionary.')

class ImportDatasetsCommand(BaseCommand):
    """
    Import datasources in YAML format.

    This is the original unversioned format used to export and import datasources
    in Superset.
    """

    def __init__(self, contents: Dict[str, str], *args: Any, **kwargs: Any) -> None:
        self.contents = contents
        self._configs: Dict[str, Any] = {}
        self.sync: List[str] = []
        if kwargs.get('sync_columns'):
            self.sync.append('columns')
        if kwargs.get('sync_metrics'):
            self.sync.append('metrics')

    @transaction()
    def run(self) -> None:
        self.validate()
        for file_name, config in self._configs.items():
            logger.info('Importing dataset from file %s', file_name)
            if isinstance(config, dict):
                import_from_dict(config, sync=self.sync)
            else:
                for dataset in config:
                    params = json.loads(dataset['params'])
                    database = db.session.query(Database).filter_by(database_name=params['database_name']).one()
                    dataset['database_id'] = database.id
                    SqlaTable.import_from_dict(dataset, sync=self.sync)

    def validate(self) -> None:
        for file_name, content in self.contents.items():
            try:
                config = yaml.safe_load(content)
            except yaml.parser.ParserError as ex:
                logger.exception('Invalid YAML file')
                raise IncorrectVersionError(f'{file_name} is not a valid YAML file') from ex
            if isinstance(config, dict):
                if DATABASES_KEY not in config:
                    raise IncorrectVersionError(f'{file_name} has no valid keys')
            elif isinstance(config, list):
                pass
            else:
                raise IncorrectVersionError(f'{file_name} is not a valid file')
            self._configs[file_name] = config
