from typing import Dict, List, Tuple, Any, Optional, Union
from dbt.tests.fixtures.project import TestProjInfo
from dbt.tests.util import relation_from_name

def get_records(project: TestProjInfo, table: str, select: Optional[str] = None, where: Optional[str] = None) -> List[Tuple]:
    """
    Gets records from a single table in a dbt project

    Args:
        project: the dbt project that contains the table
        table: the name of the table without a schema
        select: the selection clause; defaults to all columns (*)
        where: the where clause to apply, if any; defaults to all records

    Returns:
        A list of records with each record as a tuple
    """
    table_name = relation_from_name(project.adapter, table)
    select_clause = select or '*'
    where_clause = where or '1 = 1'
    sql = f'\n        select {select_clause}\n        from {table_name}\n        where {where_clause}\n    '
    return [tuple(record) for record in project.run_sql(sql, fetch='all')]

def update_records(project: TestProjInfo, table: str, updates: Dict[str, str], where: Optional[str] = None) -> None:
    """
    Applies updates to a table in a dbt project

    Args:
        project: the dbt project that contains the table
        table: the name of the table without a schema
        updates: the updates to be applied in the form {'field_name': 'expression to be applied'}
        where: the where clause to apply, if any; defaults to all records
    """
    table_name = relation_from_name(project.adapter, table)
    set_clause = ', '.join([' = '.join([field, expression]) for field, expression in updates.items()])
    where_clause = where or '1 = 1'
    sql = f'\n        update {table_name}\n        set {set_clause}\n        where {where_clause}\n    '
    project.run_sql(sql)

def insert_records(project: TestProjInfo, to_table: str, from_table: str, select: Optional[str] = None, where: Optional[str] = None) -> None:
    """
    Inserts records from one table into another table in a dbt project

    Args:
        project: the dbt project that contains the table
        to_table: the name of the table, without a schema, in which the records will be inserted
        from_table: the name of the table, without a schema, which contains the records to be inserted
        select: the selection clause to apply on `from_table`; defaults to all columns (*)
        where: the where clause to apply on `from_table`, if any; defaults to all records
    """
    to_table_name = relation_from_name(project.adapter, to_table)
    from_table_name = relation_from_name(project.adapter, from_table)
    select_clause = select or '*'
    where_clause = where or '1 = 1'
    sql = f'\n        insert into {to_table_name}\n        select {select_clause}\n        from {from_table_name}\n        where {where_clause}\n    '
    project.run_sql(sql)

def delete_records(project: TestProjInfo, table: str, where: Optional[str] = None) -> None:
    """
    Deletes records from a table in a dbt project

    Args:
        project: the dbt project that contains the table
        table: the name of the table without a schema
        where: the where clause to apply, if any; defaults to all records
    """
    table_name = relation_from_name(project.adapter, table)
    where_clause = where or '1 = 1'
    sql = f'\n        delete from {table_name}\n        where {where_clause}\n    '
    project.run_sql(sql)

def clone_table(project: TestProjInfo, to_table: str, from_table: str, select: Optional[str] = None, where: Optional[str] = None) -> None:
    """
    Creates a new table based on another table in a dbt project

    Args:
        project: the dbt project that contains the table
        to_table: the name of the table, without a schema, to be created
        from_table: the name of the table, without a schema, to be cloned
        select: the selection clause to apply on `from_table`; defaults to all columns (*)
        where: the where clause to apply on `from_table`, if any; defaults to all records
    """
    to_table_name = relation_from_name(project.adapter, to_table)
    from_table_name = relation_from_name(project.adapter, from_table)
    select_clause = select or '*'
    where_clause = where or '1 = 1'
    sql = f'drop table if exists {to_table_name}'
    project.run_sql(sql)
    sql = f'\n        create table {to_table_name} as\n        select {select_clause}\n        from {from_table_name}\n        where {where_clause}\n    '
    project.run_sql(sql)

def add_column(project: TestProjInfo, table: str, column: str, definition: str) -> None:
    """
    Applies updates to a table in a dbt project

    Args:
        project: the dbt project that contains the table
        table: the name of the table without a schema
        column: the name of the new column
        definition: the definition of the new column, e.g. 'varchar(20) default null'
    """
    if project.adapter.type() == 'bigquery' and 'varchar' in definition.lower():
        definition = 'string'
    table_name = relation_from_name(project.adapter, table)
    sql = f'\n        alter table {table_name}\n        add column {column} {definition}\n    '
    project.run_sql(sql)
