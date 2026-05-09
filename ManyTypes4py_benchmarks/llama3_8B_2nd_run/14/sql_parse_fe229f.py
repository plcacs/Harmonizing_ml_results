from __future__ import annotations
import logging
import re
from collections.abc import Iterator
from typing import Any, cast, TYPE_CHECKING
import sqlparse
from flask_babel import gettext as __
from jinja2 import nodes, Template
from sqlalchemy import and_
from sqlparse import keywords
from sqlparse.lexer import Lexer
from sqlparse.sql import Function, Identifier, IdentifierList, Parenthesis, remove_quotes, Token, TokenList, Where
from sqlparse.tokens import Comment, CTE, DDL, DML, Keyword, Name, Punctuation, String, Whitespace, Wildcard
from sqlparse.utils import imt
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import QueryClauseValidationException, SupersetParseError, SupersetSecurityException
from superset.sql.parse import extract_tables_from_statement, SQLGLOT_DIALECTS, SQLScript, SQLStatement, Table
from superset.utils.backports import StrEnum
try:
    from sqloxide import parse_sql as sqloxide_parse
except (ImportError, ModuleNotFoundError):
    sqloxide_parse = None
if TYPE_CHECKING:
    from superset.models.core import Database
RESULT_OPERATIONS: set[str] = {'UNION', 'INTERSECT', 'EXCEPT', 'SELECT'}
ON_KEYWORD: str = 'ON'
PRECEDES_TABLE_NAME: set[str] = {'FROM', 'JOIN', 'DESCRIBE', 'WITH', 'LEFT JOIN', 'RIGHT JOIN'}
CTE_PREFIX: str = 'CTE__'
logger: logging.Logger = logging.getLogger(__name__)
lex: Lexer = Lexer.get_default_instance()
sqlparser_sql_regex: re.Pattern = keywords.SQL_REGEX
sqlparser_sql_regex.insert(25, ("'(|\\\\\\\\|\\\\|[^'])*'", sqlparse.tokens.String.Single))
lex.set_SQL_REGEX(sqlparser_sql_regex)

class CtasMethod(StrEnum, enum_member_names=['TABLE', 'VIEW']):
    ...

def _extract_limit_from_query(statement: SQLStatement) -> int | None:
    ...

def extract_top_from_query(statement: SQLStatement, top_keywords: set[str]) -> int | None:
    ...

def get_cte_remainder_query(sql: str) -> tuple[str, str]:
    ...

def check_sql_functions_exist(sql: str, function_list: set[str], engine: str = 'base') -> bool:
    ...

def strip_comments_from_sql(statement: str, engine: str = 'base') -> str:
    ...

class ParsedQuery:
    def __init__(self, sql_statement: str, strip_comments: bool = False, engine: str = 'base'):
        ...

    @property
    def tables(self) -> set[Table]:
        ...

    def _check_functions_exist_in_token(self, token: Token, functions: set[str]) -> bool:
        ...

    def check_functions_exist(self, functions: set[str]) -> bool:
        ...

    def _extract_tables_from_sql(self) -> set[Table]:
        ...

    @property
    def limit(self) -> int | None:
        ...

    def as_create_table(self, table_name: str, schema_name: str | None = None, overwrite: bool = False, method: CtasMethod = CtasMethod.TABLE) -> str:
        ...

    def set_or_update_query_limit(self, new_limit: int, force: bool = False) -> str:
        ...

def sanitize_clause(clause: str) -> str:
    ...

def has_table_query(expression: str, engine: str) -> bool:
    ...

def add_table_name(rls: TokenList, table: str) -> None:
    ...

def get_rls_for_table(candidate: Identifier, database_id: int, default_schema: str) -> TokenList | None:
    ...

def insert_rls_as_subquery(token_list: TokenList, database_id: int, default_schema: str) -> TokenList:
    ...

def insert_rls_in_predicate(token_list: TokenList, database_id: int, default_schema: str) -> TokenList:
    ...

SQLOXIDE_DIALECTS: dict[str, set[str]] = {'ansi': {'trino', 'trinonative', 'presto'}, 'hive': {'hive', 'databricks'}, 'ms': {'mssql'}, 'mysql': {'mysql'}, 'postgres': {'cockroachdb', 'hana', 'netezza', 'postgres', 'postgresql', 'redshift', 'vertica'}, 'snowflake': {'snowflake'}, 'sqlite': {'sqlite', 'gsheets', 'shillelagh'}, 'clickhouse': {'clickhouse'}}

def extract_table_references(sql_text: str, sqla_dialect: str, show_warning: bool = True) -> set[Table]:
    ...

def extract_tables_from_jinja_sql(sql: str, database: Database) -> set[Table]:
    ...
