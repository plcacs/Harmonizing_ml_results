from typing import Dict, List
from dbt.tests.fixtures.project import TestProjInfo
from dbt.tests.util import relation_from_name

def get_records(project: TestProjInfo, table: str, select: str = None, where: str = None) -> List[tuple]:
    table_name = relation_from_name(project.adapter, table)
    select_clause = select or '*'
    where_clause = where or '1 = 1'
    sql = f'\n        select {select_clause}\n        from {table_name}\n        where {where_clause}\n    '
    return [tuple(record) for record in project.run_sql(sql, fetch='all')]

def update_records(project: TestProjInfo, table: str, updates: Dict[str, str], where: str = None) -> None:
    table_name = relation_from_name(project.adapter, table)
    set_clause = ', '.join([' = '.join([field, expression]) for field, expression in updates.items()])
    where_clause = where or '1 = 1'
    sql = f'\n        update {table_name}\n        set {set_clause}\n        where {where_clause}\n    '
    project.run_sql(sql)

def insert_records(project: TestProjInfo, to_table: str, from_table: str, select: str, where: str = None) -> None:
    to_table_name = relation_from_name(project.adapter, to_table)
    from_table_name = relation_from_name(project.adapter, from_table)
    select_clause = select or '*'
    where_clause = where or '1 = 1'
    sql = f'\n        insert into {to_table_name}\n        select {select_clause}\n        from {from_table_name}\n        where {where_clause}\n    '
    project.run_sql(sql)

def delete_records(project: TestProjInfo, table: str, where: str = None) -> None:
    table_name = relation_from_name(project.adapter, table)
    where_clause = where or '1 = 1'
    sql = f'\n        delete from {table_name}\n        where {where_clause}\n    '
    project.run_sql(sql)

def clone_table(project: TestProjInfo, to_table: str, from_table: str, select: str, where: str = None) -> None:
    to_table_name = relation_from_name(project.adapter, to_table)
    from_table_name = relation_from_name(project.adapter, from_table)
    select_clause = select or '*'
    where_clause = where or '1 = 1'
    sql = f'drop table if exists {to_table_name}'
    project.run_sql(sql)
    sql = f'\n        create table {to_table_name} as\n        select {select_clause}\n        from {from_table_name}\n        where {where_clause}\n    '
    project.run_sql(sql)

def add_column(project: TestProjInfo, table: str, column: str, definition: str) -> None:
    if project.adapter.type() == 'bigquery' and 'varchar' in definition.lower():
        definition = 'string'
    table_name = relation_from_name(project.adapter, table)
    sql = f'\n        alter table {table_name}\n        add column {column} {definition}\n    '
    project.run_sql(sql)
