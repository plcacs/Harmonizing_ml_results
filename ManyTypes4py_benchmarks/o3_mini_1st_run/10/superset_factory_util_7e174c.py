import logging
from typing import Optional, List
from flask_appbuilder import Model
from flask_appbuilder.security.sqla.models import User
from superset import db
from superset.connectors.sqla.models import SqlaTable, sqlatable_user
from superset.models.core import Database
from superset.models.dashboard import Dashboard, dashboard_slices, dashboard_user, DashboardRoles
from superset.models.slice import Slice, slice_user
from tests.integration_tests.dashboards.dashboard_test_utils import random_slug, random_str, random_title

logger = logging.getLogger(__name__)

inserted_dashboards_ids: List[int] = []
inserted_databases_ids: List[int] = []
inserted_sqltables_ids: List[int] = []
inserted_slices_ids: List[int] = []


def create_dashboard_to_db(
    dashboard_title: Optional[str] = None,
    slug: Optional[str] = None,
    published: bool = False,
    owners: Optional[List[User]] = None,
    slices: Optional[List[Slice]] = None,
    css: str = '',
    json_metadata: str = '',
    position_json: str = ''
) -> Dashboard:
    dashboard: Dashboard = create_dashboard(dashboard_title, slug, published, owners, slices, css, json_metadata, position_json)
    insert_model(dashboard)
    inserted_dashboards_ids.append(dashboard.id)
    return dashboard


def create_dashboard(
    dashboard_title: Optional[str] = None,
    slug: Optional[str] = None,
    published: bool = False,
    owners: Optional[List[User]] = None,
    slices: Optional[List[Slice]] = None,
    css: str = '',
    json_metadata: str = '',
    position_json: str = ''
) -> Dashboard:
    dashboard_title = dashboard_title if dashboard_title is not None else random_title()
    slug = slug if slug is not None else random_slug()
    owners = owners if owners is not None else []
    slices = slices if slices is not None else []
    return Dashboard(
        dashboard_title=dashboard_title,
        slug=slug,
        published=published,
        owners=owners,
        css=css,
        position_json=position_json,
        json_metadata=json_metadata,
        slices=slices
    )


def insert_model(model: Model) -> None:
    db.session.add(model)
    db.session.commit()
    db.session.refresh(model)


def create_slice_to_db(
    name: Optional[str] = None,
    datasource_id: Optional[int] = None,
    owners: Optional[List[User]] = None
) -> Slice:
    slice_: Slice = create_slice(datasource_id, name=name, owners=owners)
    insert_model(slice_)
    inserted_slices_ids.append(slice_.id)
    return slice_


def create_slice(
    datasource_id: Optional[int] = None,
    datasource: Optional[SqlaTable] = None,
    name: Optional[str] = None,
    owners: Optional[List[User]] = None
) -> Slice:
    name = name if name is not None else random_str()
    owners = owners if owners is not None else []
    datasource_type: str = 'table'
    if datasource:
        return Slice(slice_name=name, table=datasource, owners=owners, datasource_type=datasource_type)
    datasource_id = datasource_id if datasource_id is not None else create_datasource_table_to_db(name=name + '_table').id
    return Slice(slice_name=name, datasource_id=datasource_id, owners=owners, datasource_type=datasource_type)


def create_datasource_table_to_db(
    name: Optional[str] = None,
    db_id: Optional[int] = None,
    owners: Optional[List[User]] = None
) -> SqlaTable:
    sqltable: SqlaTable = create_datasource_table(name, db_id, owners=owners)
    insert_model(sqltable)
    inserted_sqltables_ids.append(sqltable.id)
    return sqltable


def create_datasource_table(
    name: Optional[str] = None,
    db_id: Optional[int] = None,
    database: Optional[Database] = None,
    owners: Optional[List[User]] = None
) -> SqlaTable:
    name = name if name is not None else random_str()
    owners = owners if owners is not None else []
    if database:
        return SqlaTable(table_name=name, database=database, owners=owners)
    db_id = db_id if db_id is not None else create_database_to_db(name=name + '_db').id
    return SqlaTable(table_name=name, database_id=db_id, owners=owners)


def create_database_to_db(
    name: Optional[str] = None
) -> Database:
    database: Database = create_database(name)
    insert_model(database)
    inserted_databases_ids.append(database.id)
    return database


def create_database(
    name: Optional[str] = None
) -> Database:
    name = name if name is not None else random_str()
    return Database(database_name=name, sqlalchemy_uri='sqlite:///:memory:')


def delete_all_inserted_objects() -> None:
    delete_all_inserted_dashboards()
    delete_all_inserted_slices()
    delete_all_inserted_tables()
    delete_all_inserted_dbs()


def delete_all_inserted_dashboards() -> None:
    try:
        dashboards_to_delete: List[Dashboard] = db.session.query(Dashboard).filter(Dashboard.id.in_(inserted_dashboards_ids)).all()
        for dashboard in dashboards_to_delete:
            try:
                delete_dashboard(dashboard, False)
            except Exception:
                logger.error(f'failed to delete {dashboard.id}', exc_info=True)
                raise
        if len(inserted_dashboards_ids) > 0:
            db.session.commit()
            inserted_dashboards_ids.clear()
    except Exception:
        logger.error('delete_all_inserted_dashboards failed', exc_info=True)
        raise


def delete_dashboard(dashboard: Dashboard, do_commit: bool = False) -> None:
    logger.info(f'deleting dashboard {dashboard.id}')
    delete_dashboard_roles_associations(dashboard)
    delete_dashboard_users_associations(dashboard)
    delete_dashboard_slices_associations(dashboard)
    db.session.delete(dashboard)
    if do_commit:
        db.session.commit()


def delete_dashboard_users_associations(dashboard: Dashboard) -> None:
    db.session.execute(dashboard_user.delete().where(dashboard_user.c.dashboard_id == dashboard.id))


def delete_dashboard_roles_associations(dashboard: Dashboard) -> None:
    db.session.execute(DashboardRoles.delete().where(DashboardRoles.c.dashboard_id == dashboard.id))


def delete_dashboard_slices_associations(dashboard: Dashboard) -> None:
    db.session.execute(dashboard_slices.delete().where(dashboard_slices.c.dashboard_id == dashboard.id))


def delete_all_inserted_slices() -> None:
    try:
        slices_to_delete: List[Slice] = db.session.query(Slice).filter(Slice.id.in_(inserted_slices_ids)).all()
        for slice_ in slices_to_delete:
            try:
                delete_slice(slice_, False)
            except Exception:
                logger.error(f'failed to delete {slice_.id}', exc_info=True)
                raise
        if len(inserted_slices_ids) > 0:
            db.session.commit()
            inserted_slices_ids.clear()
    except Exception:
        logger.error('delete_all_inserted_slices failed', exc_info=True)
        raise


def delete_slice(slice_: Slice, do_commit: bool = False) -> None:
    logger.info(f'deleting slice {slice_.id}')
    delete_slice_users_associations(slice_)
    db.session.delete(slice_)
    if do_commit:
        db.session.commit()


def delete_slice_users_associations(slice_: Slice) -> None:
    db.session.execute(slice_user.delete().where(slice_user.c.slice_id == slice_.id))


def delete_all_inserted_tables() -> None:
    try:
        tables_to_delete: List[SqlaTable] = db.session.query(SqlaTable).filter(SqlaTable.id.in_(inserted_sqltables_ids)).all()
        for table in tables_to_delete:
            try:
                delete_sqltable(table, False)
            except Exception:
                logger.error(f'failed to delete {table.id}', exc_info=True)
                raise
        if len(inserted_sqltables_ids) > 0:
            db.session.commit()
            inserted_sqltables_ids.clear()
    except Exception:
        logger.error('delete_all_inserted_tables failed', exc_info=True)
        raise


def delete_sqltable(table: SqlaTable, do_commit: bool = False) -> None:
    logger.info(f'deleting table {table.id}')
    delete_table_users_associations(table)
    db.session.delete(table)
    if do_commit:
        db.session.commit()


def delete_table_users_associations(table: SqlaTable) -> None:
    db.session.execute(sqlatable_user.delete().where(sqlatable_user.c.table_id == table.id))


def delete_all_inserted_dbs() -> None:
    try:
        databases_to_delete: List[Database] = db.session.query(Database).filter(Database.id.in_(inserted_databases_ids)).all()
        for database in databases_to_delete:
            try:
                delete_database(database, False)
            except Exception:
                logger.error(f'failed to delete {database.id}', exc_info=True)
                raise
        if len(inserted_databases_ids) > 0:
            db.session.commit()
            inserted_databases_ids.clear()
    except Exception:
        logger.error('delete_all_inserted_databases failed', exc_info=True)
        raise


def delete_database(database: Database, do_commit: bool = False) -> None:
    logger.info(f'deleting database {database.id}')
    db.session.delete(database)
    if do_commit:
        db.session.commit()