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
RESULT_OPERATIONS = {'UNION', 'INTERSECT', 'EXCEPT', 'SELECT'}
ON_KEYWORD = 'ON'
PRECEDES_TABLE_NAME = {'FROM', 'JOIN', 'DESCRIBE', 'WITH', 'LEFT JOIN', 'RIGHT JOIN'}
CTE_PREFIX = 'CTE__'
logger = logging.getLogger(__name__)
lex = Lexer.get_default_instance()
sqlparser_sql_regex = keywords.SQL_REGEX
sqlparser_sql_regex.insert(25, ("'(''|\\\\\\\\|\\\\|[^'])*'", sqlparse.tokens.String.Single))
lex.set_SQL_REGEX(sqlparser_sql_regex)

class CtasMethod(StrEnum):
    TABLE = 'TABLE'
    VIEW = 'VIEW'

def _extract_limit_from_query(statement: SQLStatement) -> int:
    ...

def extract_top_from_query(statement: SQLStatement, top_keywords: set[str]) -> int:
    ...

def get_cte_remainder_query(sql: str) -> tuple[str, str]:
    ...

def check_sql_functions_exist(sql: str, function_list: set[str], engine: str = 'base') -> bool:
    ...

def strip_comments_from_sql(statement: str, engine: str = 'base') -> str:
    ...

class ParsedQuery:
    def __init__(self, sql_statement: str, strip_comments: bool = False, engine: str = 'base') -> None:
        ...

    def tables(self) -> set[Table]:
        ...

    def check_functions_exist(self, functions: set[str]) -> bool:
        ...

    def limit(self) -> int:
        ...

    def is_select(self) -> bool:
        ...

    def get_inner_cte_expression(self, tokens: list[Token]) -> list[Token]:
        ...

    def is_valid_ctas(self) -> bool:
        ...

    def is_valid_cvas(self) -> bool:
        ...

    def is_explain(self) -> bool:
        ...

    def is_show(self) -> bool:
        ...

    def is_set(self) -> bool:
        ...

    def is_unknown(self) -> bool:
        ...

    def stripped(self) -> str:
        ...

    def strip_comments(self) -> str:
        ...

    def get_statements(self) -> list[str]:
        ...

    @staticmethod
    def get_table(tlist: TokenList) -> Table:
        ...

    @staticmethod
    def _is_identifier(token: Token) -> bool:
        ...

    def as_create_table(self, table_name: str, schema_name: str = None, overwrite: bool = False, method: CtasMethod = CtasMethod.TABLE) -> str:
        ...

    def set_or_update_query_limit(self, new_limit: int, force: bool = False) -> str:
        ...

def sanitize_clause(clause: str) -> str:
    ...

class InsertRLSState(StrEnum):
    SCANNING = 'SCANNING'
    SEEN_SOURCE = 'SEEN_SOURCE'
    FOUND_TABLE = 'FOUND_TABLE'

def has_table_query(expression: str, engine: str) -> bool:
    ...

def add_table_name(rls: TokenList, table: Table) -> None:
    ...

def get_rls_for_table(candidate: Identifier, database_id: int, default_schema: str) -> TokenList:
    ...

def insert_rls_as_subquery(token_list: TokenList, database_id: int, default_schema: str) -> TokenList:
    ...

def insert_rls_in_predicate(token_list: TokenList, database_id: int, default_schema: str) -> TokenList:
    ...

SQLOXIDE_DIALECTS = {'ansi': {'trino', 'trinonative', 'presto'}, 'hive': {'hive', 'databricks'}, 'ms': {'mssql'}, 'mysql': {'mysql'}, 'postgres': {'cockroachdb', 'hana', 'netezza', 'postgres', 'postgresql', 'redshift', 'vertica'}, 'snowflake': {'snowflake'}, 'sqlite': {'sqlite', 'gsheets', 'shillelagh'}, 'clickhouse': {'clickhouse'}}
RE_JINJA_VAR = re.compile('\\{\\{[^\\{\\}]+\\}\\}')
RE_JINJA_BLOCK = re.compile('\\{[%#][^\\{\\}%#]+[%#]\\}')

def extract_table_references(sql_text: str, sqla_dialect: str, show_warning: bool = True) -> set[Table]:
    ...

def extract_tables_from_jinja_sql(sql: str, database: Database) -> set[Table]:
    ...
