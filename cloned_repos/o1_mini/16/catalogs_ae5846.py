from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Type, Union, List, Dict, Tuple, Optional
import sqlalchemy as sa
from alembic import op
from flask import current_app
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from superset import db, security_manager
from superset.db_engine_specs.base import GenericDBException
from superset.migrations.shared.security_converge import add_pvms, Permission, PermissionView, ViewMenu
from superset.models.core import Database

logger: logging.Logger = logging.getLogger('alembic')
Base: Any = declarative_base()


class SqlaTable(Base):
    __tablename__: str = 'tables'
    id: int = sa.Column(sa.Integer, primary_key=True)
    database_id: int = sa.Column(sa.Integer, nullable=False)
    perm: str = sa.Column(sa.String(1000))
    schema_perm: str = sa.Column(sa.String(1000))
    catalog_perm: Optional[str] = sa.Column(sa.String(1000), nullable=True, default=None)
    schema: str = sa.Column(sa.String(255))
    catalog: Optional[str] = sa.Column(sa.String(256), nullable=True, default=None)


class Query(Base):
    __tablename__: str = 'query'
    id: int = sa.Column(sa.Integer, primary_key=True)
    database_id: int = sa.Column(sa.Integer, nullable=False)
    catalog: Optional[str] = sa.Column(sa.String(256), nullable=True, default=None)


class SavedQuery(Base):
    __tablename__: str = 'saved_query'
    id: int = sa.Column(sa.Integer, primary_key=True)
    db_id: int = sa.Column(sa.Integer, nullable=False)
    catalog: Optional[str] = sa.Column(sa.String(256), nullable=True, default=None)


class TabState(Base):
    __tablename__: str = 'tab_state'
    id: int = sa.Column(sa.Integer, primary_key=True)
    database_id: int = sa.Column(sa.Integer, nullable=False)
    catalog: Optional[str] = sa.Column(sa.String(256), nullable=True, default=None)


class TableSchema(Base):
    __tablename__: str = 'table_schema'
    id: int = sa.Column(sa.Integer, primary_key=True)
    database_id: int = sa.Column(sa.Integer, nullable=False)
    catalog: Optional[str] = sa.Column(sa.String(256), nullable=True, default=None)


class Slice(Base):
    __tablename__: str = 'slices'
    id: int = sa.Column(sa.Integer, primary_key=True)
    datasource_id: Optional[int] = sa.Column(sa.Integer)
    datasource_type: str = sa.Column(sa.String(200))
    catalog_perm: Optional[str] = sa.Column(sa.String(1000), nullable=True, default=None)
    schema_perm: str = sa.Column(sa.String(1000))

ModelType: Type[Union[Query, SavedQuery, TabState, TableSchema]] = Union[Type[Query], Type[SavedQuery], Type[TabState], Type[TableSchema]]
MODELS: List[Tuple[Type[Base], str]] = [
    (Query, 'database_id'),
    (SavedQuery, 'db_id'),
    (TabState, 'database_id'),
    (TableSchema, 'database_id')
]


def get_known_schemas(database_name: str, session: Session) -> List[str]:
    """
    Read all known schemas from the existing schema permissions.
    """
    names: List[Tuple[str]] = session.query(ViewMenu.name) \
        .join(PermissionView, ViewMenu.id == PermissionView.view_menu_id) \
        .join(Permission, PermissionView.permission_id == Permission.id) \
        .filter(ViewMenu.name.like(f'[{database_name}]%'), Permission.name == 'schema_access') \
        .all()
    return sorted({name[0][1:-1].split('].[')[-1] for name in names})


def get_batch_size(session: Session) -> int:
    max_sqlite_in: int = 999
    return max_sqlite_in if session.bind.dialect.name == 'sqlite' else 1000000


def print_processed_batch(start_time: datetime, offset: int, total_rows: int, model: Type[Base], batch_size: int) -> None:
    """
    Print the progress of batch processing.

    This function logs the progress of processing a batch of rows from a model.
    It calculates the elapsed time since the start of the batch processing and
    logs the number of rows processed along with the percentage completion.

    Parameters:
        start_time (datetime): The start time of the batch processing.
        offset (int): The current offset in the batch processing.
        total_rows (int): The total number of rows to process.
        model (ModelType): The model being processed.
        batch_size (int): The size of the batch being processed.
    """
    elapsed_time: datetime = datetime.now() - start_time
    elapsed_seconds: float = elapsed_time.total_seconds()
    elapsed_formatted: str = f'{int(elapsed_seconds // 3600):02}:{int(elapsed_seconds % 3600 // 60):02}:{int(elapsed_seconds % 60):02}'
    rows_processed: int = min(offset + batch_size, total_rows)
    logger.info(
        f'{elapsed_formatted} - {rows_processed:,} of {total_rows:,} {model.__tablename__} rows processed '
        f'({rows_processed / total_rows * 100:.2f}%)'
    )


def update_catalog_column(session: Session, database: Database, catalog: Optional[str], downgrade: bool = False) -> None:
    """
    Update the `catalog` column in the specified models to the given catalog.

    This function iterates over a list of models defined by MODELS and updates
    the `catalog` columnto the specified catalog or None depending on the downgrade
    parameter. The update is performed in batches to optimize performance and reduce
    memory usage.

    Parameters:
        session (Session): The SQLAlchemy session to use for database operations.
        database (Database): The database instance containing the models to update.
        catalog (Optional[str]): The new catalog value to set in the `catalog` column or
            the default catalog if `downgrade` is True.
        downgrade (bool): If True, the `catalog` column is set to None where the
            catalog matches the specified catalog.
    """
    start_time: datetime = datetime.now()
    logger.info(f'Updating {database.database_name} models to catalog {catalog}')
    for model, column in MODELS:
        filter_condition = getattr(model, column) == database.id
        if downgrade:
            filter_condition &= model.catalog == catalog
        total_rows: int = session.query(sa.func.count(model.id)) \
            .filter(filter_condition) \
            .scalar() or 0
        logger.info(f'Total rows to be processed for {model.__tablename__}: {total_rows:,}')
        batch_size: int = get_batch_size(session)
        limit_value: int = min(batch_size, total_rows)
        for i in range(0, total_rows, batch_size):
            subquery = session.query(model.id) \
                .filter(filter_condition) \
                .order_by(model.id) \
                .offset(i) \
                .limit(limit_value) \
                .subquery()
            if session.bind.dialect.name == 'sqlite':
                ids_to_update: List[int] = [row.id for row in session.query(subquery.c.id).all()]
                if ids_to_update:
                    session.execute(
                        sa.update(model)
                        .where(model.id.in_(ids_to_update))
                        .values(catalog=None if downgrade else catalog)
                        .execution_options(synchronize_session=False)
                    )
            else:
                session.execute(
                    sa.update(model)
                    .where(model.id == subquery.c.id)
                    .values(catalog=None if downgrade else catalog)
                    .execution_options(synchronize_session=False)
                )
            print_processed_batch(start_time, i, total_rows, model, batch_size)


def update_schema_catalog_perms(
    session: Session,
    database: Database,
    catalog_perm: Optional[str],
    catalog: Optional[str],
    downgrade: bool = False
) -> None:
    """
    Update schema and catalog permissions for tables and charts in a given database.

    This function updates the `catalog`, `catalog_perm`, and `schema_perm` fields for
    tables and charts associated with the specified database. If `downgrade` is True,
    the `catalog` and `catalog_perm` fields are set to None, otherwise they are set
    to the provided `catalog` and `catalog_perm` values.

    Args:
        session (Session): The SQLAlchemy session to use for database operations.
        database (Database): The database object whose tables and charts will be updated.
        catalog_perm (Optional[str]): The new catalog permission to set.
        catalog (Optional[str]): The new catalog to set.
        downgrade (bool, optional): If True, reset the `catalog` and `catalog_perm` fields to None.
                                    Defaults to False.
    """
    mapping: Dict[int, str] = {}
    tables_query = session.query(SqlaTable).filter_by(database_id=database.id)
    if downgrade:
        tables_query = tables_query.filter_by(catalog=catalog)
    else:
        tables_query = tables_query.filter(SqlaTable.catalog == sa.null())
    for table in tables_query:
        schema_perm = security_manager.get_schema_perm(
            database.database_name,
            None if downgrade else catalog,
            table.schema
        )
        table.catalog = None if downgrade else catalog
        table.catalog_perm = catalog_perm
        table.schema_perm = schema_perm
        mapping[table.id] = schema_perm
    slices_query = session.query(Slice).join(SqlaTable, Slice.datasource_id == SqlaTable.id) \
        .join(Database, SqlaTable.database_id == Database.id) \
        .filter(Database.id == database.id, Slice.datasource_type == 'table')
    for chart in slices_query:
        if chart.datasource_id in mapping:
            chart.catalog_perm = catalog_perm
            chart.schema_perm = mapping[chart.datasource_id]


def delete_models_non_default_catalog(session: Session, database: Database, catalog: Optional[str]) -> None:
    """
    Delete models that are not in the default catalog.

    This function iterates over a list of models defined by MODELS and deletes
    the rows where the `catalog` column does not match the specified catalog.

    Parameters:
        session (Session): The SQLAlchemy session to use for database operations.
        database (Database): The database instance containing the models to delete.
        catalog (Optional[str]): The catalog to use to filter the models to delete.
    """
    start_time: datetime = datetime.now()
    logger.info(f'Deleting models not in the default catalog: {catalog}')
    for model, column in MODELS:
        filter_condition = getattr(model, column) == database.id
        filter_condition &= model.catalog != catalog
        total_rows: int = session.query(sa.func.count(model.id)) \
            .filter(filter_condition) \
            .scalar() or 0
        logger.info(f'Total rows to be processed for {model.__tablename__}: {total_rows:,}')
        batch_size: int = get_batch_size(session)
        limit_value: int = min(batch_size, total_rows)
        for i in range(0, total_rows, batch_size):
            subquery = session.query(model.id) \
                .filter(filter_condition) \
                .order_by(model.id) \
                .offset(i) \
                .limit(limit_value) \
                .subquery()
            if session.bind.dialect.name == 'sqlite':
                ids_to_delete: List[int] = [row.id for row in session.query(subquery.c.id).all()]
                if ids_to_delete:
                    session.execute(
                        sa.delete(model)
                        .where(model.id.in_(ids_to_delete))
                        .execution_options(synchronize_session=False)
                    )
            else:
                session.execute(
                    sa.delete(model)
                    .where(model.id == subquery.c.id)
                    .execution_options(synchronize_session=False)
                )
            print_processed_batch(start_time, i, total_rows, model, batch_size)


def upgrade_catalog_perms(engines: Optional[List[str]] = None) -> None:
    """
    Update models and permissions when catalogs are introduced in a DB engine spec.

    When an existing DB engine spec starts to support catalogs we need to:

        - Add `catalog_access` permissions for each catalog.
        - Rename existing `schema_access` permissions to include the default catalog.
        - Create `schema_access` permissions for each schema in the new catalogs.

    Also, for all the relevant existing models we need to:

        - Populate the `catalog` field with the default catalog.
        - Update `schema_perm` to include the default catalog.
        - Populate `catalog_perm` to include the default catalog.

    """
    bind = op.get_bind()
    session: Session = db.Session(bind=bind)
    databases: List[Database] = session.query(Database).all()
    for database in databases:
        db_engine_spec = database.db_engine_spec
        if engines and db_engine_spec.engine not in engines or not db_engine_spec.supports_catalog:
            continue
        try:
            default_catalog: Optional[str] = database.get_default_catalog()
        except GenericDBException as ex:
            logger.warning('Error fetching default catalog for database %s: %s', database.database_name, ex)
            continue
        if default_catalog:
            upgrade_database_catalogs(database, default_catalog, session)
    session.flush()


def upgrade_database_catalogs(database: Database, default_catalog: str, session: Session) -> None:
    """
    Upgrade a given database to support the default catalog.
    """
    catalog_perm: Optional[str] = security_manager.get_catalog_perm(database.database_name, default_catalog)
    pvms: Dict[str, Tuple[str, ...]] = {catalog_perm: ('catalog_access',)} if catalog_perm else {}
    new_schema_pvms: Dict[str, Tuple[str, ...]] = upgrade_schema_perms(database, default_catalog, session)
    pvms.update(new_schema_pvms)
    update_catalog_column(session, database, default_catalog, downgrade=False)
    update_schema_catalog_perms(session, database, catalog_perm, default_catalog, downgrade=False)
    if not current_app.config['CATALOGS_SIMPLIFIED_MIGRATION'] and not database.is_oauth2_enabled():
        new_catalog_pvms: Dict[str, Tuple[str, ...]] = add_non_default_catalogs(database, default_catalog, session)
        pvms.update(new_catalog_pvms)
    add_pvms(session, pvms)


def add_non_default_catalogs(database: Database, default_catalog: str, session: Session) -> Dict[str, Tuple[str, ...]]:
    """
    Add permissions for additional catalogs and their schemas.
    """
    try:
        catalogs: set[str] = {catalog for catalog in database.get_all_catalog_names() if catalog != default_catalog}
    except GenericDBException:
        return {}
    pvms: Dict[str, Tuple[str, ...]] = {}
    for catalog in catalogs:
        perm: Optional[str] = security_manager.get_catalog_perm(database.database_name, catalog)
        if perm:
            pvms[perm] = ('catalog_access',)
            new_schema_pvms: Dict[str, Tuple[str, ...]] = create_schema_perms(database, catalog)
            pvms.update(new_schema_pvms)
    return pvms


def upgrade_schema_perms(database: Database, default_catalog: str, session: Session) -> Dict[str, Tuple[str, ...]]:
    """
    Rename existing schema permissions to include the catalog.

    Schema permissions are stored (and processed) as strings, in the form:

        [database_name].[schema_name]

    When catalogs are first introduced for a DB engine spec we need to rename any
    existing permissions to the form:

        [database_name].[default_catalog_name].[schema_name]
    """
    schemas: List[str] = get_known_schemas(database.database_name, session)
    perms: Dict[str, Tuple[str, ...]] = {}
    for schema in schemas:
        current_perm: str = security_manager.get_schema_perm(database.database_name, None, schema)
        new_perm: str = security_manager.get_schema_perm(database.database_name, default_catalog, schema)
        existing_pvm: Optional[ViewMenu] = session.query(ViewMenu).filter_by(name=current_perm).one_or_none()
        if existing_pvm:
            if not session.query(ViewMenu).filter_by(name=new_perm).one_or_none():
                existing_pvm.name = new_perm
        elif new_perm:
            perms[new_perm] = ('schema_access',)
    return perms


def create_schema_perms(database: Database, catalog: str) -> Dict[str, Tuple[str, ...]]:
    """
    Create schema permissions for a given catalog.
    """
    try:
        schemas: List[str] = database.get_all_schema_names(catalog=catalog)
    except GenericDBException:
        return {}
    return {
        perm: ('schema_access',)
        for schema in schemas
        if (perm := security_manager.get_schema_perm(database.database_name, catalog, schema)) is not None
    }


def downgrade_catalog_perms(engines: Optional[List[str]] = None) -> None:
    """
    Reverse the process of `upgrade_catalog_perms`.

    This should:

        - Delete all `catalog_access` permissions.
        - Rename `schema_access` permissions in the default catalog to omit it.
        - Delete `schema_access` permissions for schemas not in the default catalog.

    Also, for models in the default catalog we should:

        - Populate the `catalog` field with `None`.
        - Update `schema_perm` to omit the default catalog.
        - Populate the `catalog_perm` field with `None`.

    WARNING: models (datasets and charts) not in the default catalog are deleted!
    """
    bind = op.get_bind()
    session: Session = db.Session(bind=bind)
    databases: List[Database] = session.query(Database).all()
    for database in databases:
        db_engine_spec = database.db_engine_spec
        if engines and db_engine_spec.engine not in engines or not db_engine_spec.supports_catalog:
            continue
        try:
            default_catalog: Optional[str] = database.get_default_catalog()
        except GenericDBException as ex:
            logger.warning('Error fetching default catalog for database %s: %s', database.database_name, ex)
            continue
        if default_catalog:
            downgrade_database_catalogs(database, default_catalog, session)
    session.flush()


def downgrade_database_catalogs(database: Database, default_catalog: str, session: Session) -> None:
    prefix: str = f'[{database.database_name}].%'
    pvms: List[PermissionView] = session.query(PermissionView) \
        .join(Permission, PermissionView.permission_id == Permission.id) \
        .join(ViewMenu, PermissionView.view_menu_id == ViewMenu.id) \
        .filter(Permission.name == 'catalog_access', ViewMenu.name.like(prefix)) \
        .all()
    for pvm in pvms:
        session.delete(pvm)
        session.delete(pvm.view_menu)
    downgrade_schema_perms(database, default_catalog, session)
    update_catalog_column(session, database, default_catalog, downgrade=True)
    update_schema_catalog_perms(session, database, None, default_catalog, downgrade=True)
    delete_models_non_default_catalog(session, database, default_catalog)
    tables: List[SqlaTable] = session.query(SqlaTable) \
        .filter(SqlaTable.database_id == database.id, SqlaTable.catalog != default_catalog) \
        .all()
    for table in tables:
        charts: List[Slice] = session.query(Slice) \
            .filter(Slice.datasource_id == table.id, Slice.datasource_type == 'table') \
            .all()
        for chart in charts:
            session.delete(chart)
        session.delete(table)
        pvm: Optional[PermissionView] = session.query(PermissionView) \
            .join(Permission, PermissionView.permission_id == Permission.id) \
            .join(ViewMenu, PermissionView.view_menu_id == ViewMenu.id) \
            .filter(Permission.name == 'datasource_access', ViewMenu.name == table.perm) \
            .one_or_none()
        if pvm:
            session.delete(pvm)
            session.delete(pvm.view_menu)
    session.flush()


def downgrade_schema_perms(database: Database, default_catalog: str, session: Session) -> None:
    """
    Rename default catalog schema permissions and delete other schema permissions.
    """
    prefix: str = f'[{database.database_name}].%'
    pvms: List[PermissionView] = session.query(PermissionView) \
        .join(Permission, PermissionView.permission_id == Permission.id) \
        .join(ViewMenu, PermissionView.view_menu_id == ViewMenu.id) \
        .filter(Permission.name == 'schema_access', ViewMenu.name.like(prefix)) \
        .all()
    pvms_to_delete: List[PermissionView] = []
    pvms_to_rename: List[Tuple[PermissionView, str]] = []
    for pvm in pvms:
        parts: List[str] = pvm.view_menu.name[1:-1].split('].[')
        if len(parts) != 3:
            logger.warning('Invalid schema permission: %s. Please fix manually', pvm.view_menu.name)
            continue
        database_name, catalog, schema = parts
        if catalog == default_catalog:
            new_name: str = security_manager.get_schema_perm(database_name, None, schema)
            if not session.query(ViewMenu).filter_by(name=new_name).one_or_none():
                pvms_to_rename.append((pvm, new_name))
        else:
            pvms_to_delete.append(pvm)
    for pvm, new_name in pvms_to_rename:
        pvm.view_menu.name = new_name
    for pvm in pvms_to_delete:
        session.delete(pvm)
        session.delete(pvm.view_menu)
