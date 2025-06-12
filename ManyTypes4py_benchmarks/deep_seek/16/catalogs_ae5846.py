from __future__ import annotations
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Type, Union
import sqlalchemy as sa
from alembic import op
from flask import current_app
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, Query as SQLAlchemyQuery
from superset import db, security_manager
from superset.db_engine_specs.base import GenericDBException
from superset.migrations.shared.security_converge import add_pvms, Permission, PermissionView, ViewMenu
from superset.models.core import Database
from sqlalchemy.sql.elements import BinaryExpression

logger = logging.getLogger('alembic')
Base = declarative_base()

class SqlaTable(Base):
    __tablename__ = 'tables'
    id = sa.Column(sa.Integer, primary_key=True)
    database_id = sa.Column(sa.Integer, nullable=False)
    perm = sa.Column(sa.String(1000))
    schema_perm = sa.Column(sa.String(1000))
    catalog_perm = sa.Column(sa.String(1000), nullable=True, default=None)
    schema = sa.Column(sa.String(255))
    catalog = sa.Column(sa.String(256), nullable=True, default=None)

class Query(Base):
    __tablename__ = 'query'
    id = sa.Column(sa.Integer, primary_key=True)
    database_id = sa.Column(sa.Integer, nullable=False)
    catalog = sa.Column(sa.String(256), nullable=True, default=None)

class SavedQuery(Base):
    __tablename__ = 'saved_query'
    id = sa.Column(sa.Integer, primary_key=True)
    db_id = sa.Column(sa.Integer, nullable=False)
    catalog = sa.Column(sa.String(256), nullable=True, default=None)

class TabState(Base):
    __tablename__ = 'tab_state'
    id = sa.Column(sa.Integer, primary_key=True)
    database_id = sa.Column(sa.Integer, nullable=False)
    catalog = sa.Column(sa.String(256), nullable=True, default=None)

class TableSchema(Base):
    __tablename__ = 'table_schema'
    id = sa.Column(sa.Integer, primary_key=True)
    database_id = sa.Column(sa.Integer, nullable=False)
    catalog = sa.Column(sa.String(256), nullable=True, default=None)

class Slice(Base):
    __tablename__ = 'slices'
    id = sa.Column(sa.Integer, primary_key=True)
    datasource_id = sa.Column(sa.Integer)
    datasource_type = sa.Column(sa.String(200))
    catalog_perm = sa.Column(sa.String(1000), nullable=True, default=None)
    schema_perm = sa.Column(sa.String(1000))

ModelType = Union[Type[Query], Type[SavedQuery], Type[TabState], Type[TableSchema]]
MODELS: List[Tuple[ModelType, str]] = [
    (Query, 'database_id'),
    (SavedQuery, 'db_id'),
    (TabState, 'database_id'),
    (TableSchema, 'database_id'),
]

def get_known_schemas(database_name: str, session: Session) -> List[str]:
    """
    Read all known schemas from the existing schema permissions.
    """
    names = session.query(ViewMenu.name).join(PermissionView, ViewMenu.id == PermissionView.view_menu_id).join(Permission, PermissionView.permission_id == Permission.id).filter(ViewMenu.name.like(f'[{database_name}]%'), Permission.name == 'schema_access').all()
    return sorted({name[0][1:-1].split('].[')[-1] for name in names})

def get_batch_size(session: Session) -> int:
    max_sqlite_in = 999
    return max_sqlite_in if session.bind.dialect.name == 'sqlite' else 1000000

def print_processed_batch(start_time: datetime, offset: int, total_rows: int, model: ModelType, batch_size: int) -> None:
    """
    Print the progress of batch processing.
    """
    elapsed_time = datetime.now() - start_time
    elapsed_seconds = elapsed_time.total_seconds()
    elapsed_formatted = f'{int(elapsed_seconds // 3600):02}:{int(elapsed_seconds % 3600 // 60):02}:{int(elapsed_seconds % 60):02}'
    rows_processed = min(offset + batch_size, total_rows)
    logger.info(f'{elapsed_formatted} - {rows_processed:,} of {total_rows:,} {model.__tablename__} rows processed ({rows_processed / total_rows * 100:.2f}%)')

def update_catalog_column(session: Session, database: Database, catalog: Optional[str], downgrade: bool = False) -> None:
    """
    Update the `catalog` column in the specified models to the given catalog.
    """
    start_time = datetime.now()
    logger.info(f'Updating {database.database_name} models to catalog {catalog}')
    for model, column in MODELS:
        total_rows = session.query(sa.func.count(model.id)).filter(getattr(model, column) == database.id).filter(model.catalog == catalog if downgrade else True).scalar()
        logger.info(f'Total rows to be processed for {model.__tablename__}: {total_rows:,}')
        batch_size = get_batch_size(session)
        limit_value = min(batch_size, total_rows)
        for i in range(0, total_rows, batch_size):
            subquery = session.query(model.id).filter(getattr(model, column) == database.id).filter(model.catalog == catalog if downgrade else True).order_by(model.id).offset(i).limit(limit_value).subquery()
            if session.bind.dialect.name == 'sqlite':
                ids_to_update = [row.id for row in session.query(subquery.c.id).all()]
                if ids_to_update:
                    session.execute(sa.update(model).where(model.id.in_(ids_to_update)).values(catalog=None if downgrade else catalog).execution_options(synchronize_session=False))
            else:
                session.execute(sa.update(model).where(model.id == subquery.c.id).values(catalog=None if downgrade else catalog).execution_options(synchronize_session=False))
            print_processed_batch(start_time, i, total_rows, model, batch_size)

def update_schema_catalog_perms(session: Session, database: Database, catalog_perm: Optional[str], catalog: Optional[str], downgrade: bool = False) -> None:
    """
    Update schema and catalog permissions for tables and charts in a given database.
    """
    mapping: Dict[int, str] = {}
    for table in session.query(SqlaTable).filter_by(database_id=database.id).filter_by(catalog=catalog if downgrade else None):
        schema_perm = security_manager.get_schema_perm(database.database_name, None if downgrade else catalog, table.schema)
        table.catalog = None if downgrade else catalog
        table.catalog_perm = catalog_perm
        table.schema_perm = schema_perm
        mapping[table.id] = schema_perm
    for chart in session.query(Slice).join(SqlaTable, Slice.datasource_id == SqlaTable.id).join(Database, SqlaTable.database_id == Database.id).filter(Database.id == database.id).filter(Slice.datasource_type == 'table'):
        if mapping.get(chart.datasource_id) is not None:
            chart.catalog_perm = catalog_perm
            chart.schema_perm = mapping[chart.datasource_id]

def delete_models_non_default_catalog(session: Session, database: Database, catalog: str) -> None:
    """
    Delete models that are not in the default catalog.
    """
    start_time = datetime.now()
    logger.info(f'Deleting models not in the default catalog: {catalog}')
    for model, column in MODELS:
        total_rows = session.query(sa.func.count(model.id)).filter(getattr(model, column) == database.id).filter(model.catalog != catalog).scalar()
        logger.info(f'Total rows to be processed for {model.__tablename__}: {total_rows:,}')
        batch_size = get_batch_size(session)
        limit_value = min(batch_size, total_rows)
        for i in range(0, total_rows, batch_size):
            subquery = session.query(model.id).filter(getattr(model, column) == database.id).filter(model.catalog != catalog).order_by(model.id).offset(i).limit(limit_value).subquery()
            if session.bind.dialect.name == 'sqlite':
                ids_to_delete = [row.id for row in session.query(subquery.c.id).all()]
                if ids_to_delete:
                    session.execute(sa.delete(model).where(model.id.in_(ids_to_delete)).execution_options(synchronize_session=False))
            else:
                session.execute(sa.delete(model).where(model.id == subquery.c.id).execution_options(synchronize_session=False))
            print_processed_batch(start_time, i, total_rows, model, batch_size)

def upgrade_catalog_perms(engines: Optional[Set[str]] = None) -> None:
    """
    Update models and permissions when catalogs are introduced in a DB engine spec.
    """
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for database in session.query(Database).all():
        db_engine_spec = database.db_engine_spec
        if engines and db_engine_spec.engine not in engines or not db_engine_spec.supports_catalog:
            continue
        try:
            default_catalog = database.get_default_catalog()
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
    catalog_perm = security_manager.get_catalog_perm(database.database_name, default_catalog)
    pvms: Dict[str, Tuple[str, ...]] = {catalog_perm: ('catalog_access',)} if catalog_perm else {}
    new_schema_pvms = upgrade_schema_perms(database, default_catalog, session)
    pvms.update(new_schema_pvms)
    update_catalog_column(session, database, default_catalog, False)
    update_schema_catalog_perms(session, database, catalog_perm, default_catalog, False)
    if not current_app.config['CATALOGS_SIMPLIFIED_MIGRATION'] and (not database.is_oauth2_enabled()):
        new_catalog_pvms = add_non_default_catalogs(database, default_catalog, session)
        pvms.update(new_catalog_pvms)
    add_pvms(session, pvms)

def add_non_default_catalogs(database: Database, default_catalog: str, session: Session) -> Dict[str, Tuple[str, ...]]:
    """
    Add permissions for additional catalogs and their schemas.
    """
    try:
        catalogs = {catalog for catalog in database.get_all_catalog_names() if catalog != default_catalog}
    except GenericDBException:
        return {}
    pvms: Dict[str, Tuple[str, ...]] = {}
    for catalog in catalogs:
        perm = security_manager.get_catalog_perm(database.database_name, catalog)
        if perm:
            pvms[perm] = ('catalog_access',)
            new_schema_pvms = create_schema_perms(database, catalog)
            pvms.update(new_schema_pvms)
    return pvms

def upgrade_schema_perms(database: Database, default_catalog: str, session: Session) -> Dict[str, Tuple[str, ...]]:
    """
    Rename existing schema permissions to include the catalog.
    """
    schemas = get_known_schemas(database.database_name, session)
    perms: Dict[str, Tuple[str, ...]] = {}
    for schema in schemas:
        current_perm = security_manager.get_schema_perm(database.database_name, None, schema)
        new_perm = security_manager.get_schema_perm(database.database_name, default_catalog, schema)
        if (existing_pvm := session.query(ViewMenu).filter_by(name=current_perm).one_or_none()):
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
        schemas = database.get_all_schema_names(catalog=catalog)
    except GenericDBException:
        return {}
    return {perm: ('schema_access',) for schema in schemas if (perm := security_manager.get_schema_perm(database.database_name, catalog, schema)) is not None}

def downgrade_catalog_perms(engines: Optional[Set[str]] = None) -> None:
    """
    Reverse the process of `upgrade_catalog_perms`.
    """
    bind = op.get_bind()
    session = db.Session(bind=bind)
    for database in session.query(Database).all():
        db_engine_spec = database.db_engine_spec
        if engines and db_engine_spec.engine not in engines or not db_engine_spec.supports_catalog:
            continue
        try:
            default_catalog = database.get_default_catalog()
        except GenericDBException as ex:
            logger.warning('Error fetching default catalog for database %s: %s', database.database_name, ex)
            continue
        if default_catalog:
            downgrade_database_catalogs(database, default_catalog, session)
    session.flush()

def downgrade_database_catalogs(database: Database, default_catalog: str, session: Session) -> None:
    prefix = f'[{database.database_name}].%'
    for pvm in session.query(PermissionView).join(Permission, PermissionView.permission_id == Permission.id).join(ViewMenu, PermissionView.view_menu_id == ViewMenu.id).filter(Permission.name == 'catalog_access', ViewMenu.name.like(prefix)).all():
        session.delete(pvm)
        session.delete(pvm.view_menu)
    downgrade_schema_perms(database, default_catalog, session)
    update_catalog_column(session, database, default_catalog, True)
    update_schema_catalog_perms(session, database, None, default_catalog, True)
    delete_models_non_default_catalog(session, database, default_catalog)
    for table in session.query(SqlaTable).filter(SqlaTable.database_id == database.id, SqlaTable.catalog != default_catalog):
        for chart in session.query(Slice).filter(Slice.datasource_id == table.id, Slice.datasource_type == 'table'):
            session.delete(chart)
        session.delete(table)
        pvm = session.query(PermissionView).join(Permission, PermissionView.permission_id == Permission.id).join(ViewMenu, PermissionView.view_menu_id == ViewMenu.id).filter(Permission.name == 'datasource_access', ViewMenu.name == table.perm).one()
        session.delete(pvm)
        session.delete(pvm.view_menu)
    session.flush()

def downgrade_schema_perms(database: Database, default_catalog: str, session: Session) -> None:
    """
    Rename default catalog schema permissions and delete other schema permissions.
    """
    prefix = f'[{database.database_name}].%'
    pvms = session.query(PermissionView).join(Permission, PermissionView.permission_id == Permission.id).join(ViewMenu, PermissionView.view_menu_id == ViewMenu.id).filter(Permission.name == 'schema_access', ViewMenu.name.like(prefix)).all()
    pvms_to_delete: List[PermissionView] = []
    pvms_to_rename: List[Tuple[PermissionView, str]] = []
    for pvm in pvms:
        parts = pvm.view_menu.name[1:-1].split('].[')
        if len(parts) != 3:
            logger.warning('Invalid schema permission: %s. Please fix manually', pvm.view_menu.name)
            continue
        database_name, catalog, schema = parts
        if catalog == default_catalog:
            new_name = security_manager.get_schema_perm(database_name, None, schema)
            if not session.query(ViewMenu).filter_by(name=new_name).one_or_none():
                pvms_to_rename.append((pvm, new_name))
        else:
            pvms_to_delete.append(pvm)
    for pvm in pvms_to_delete:
        session.delete(pvm)
        session.delete(pvm.view_menu)
    for pvm, new_name in pvms_to_rename:
        pvm.view_menu.name = new_name
