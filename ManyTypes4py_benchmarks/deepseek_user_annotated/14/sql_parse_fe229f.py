# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=too-many-lines

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any, cast, TYPE_CHECKING, Optional, Union, List, Dict, Set, Tuple

import sqlparse
from flask_babel import gettext as __
from jinja2 import nodes, Template
from sqlalchemy import and_
from sqlparse import keywords
from sqlparse.lexer import Lexer
from sqlparse.sql import (
    Function,
    Identifier,
    IdentifierList,
    Parenthesis,
    remove_quotes,
    Token,
    TokenList,
    Where,
)
from sqlparse.tokens import (
    Comment,
    CTE,
    DDL,
    DML,
    Keyword,
    Name,
    Punctuation,
    String,
    Whitespace,
    Wildcard,
)
from sqlparse.utils import imt

from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import (
    QueryClauseValidationException,
    SupersetParseError,
    SupersetSecurityException,
)
from superset.sql.parse import (
    extract_tables_from_statement,
    SQLGLOT_DIALECTS,
    SQLScript,
    SQLStatement,
    Table,
)
from superset.utils.backports import StrEnum

try:
    from sqloxide import parse_sql as sqloxide_parse
except (ImportError, ModuleNotFoundError):
    sqloxide_parse = None

if TYPE_CHECKING:
    from superset.models.core import Database

RESULT_OPERATIONS: Set[str] = {"UNION", "INTERSECT", "EXCEPT", "SELECT"}
ON_KEYWORD: str = "ON"
PRECEDES_TABLE_NAME: Set[str] = {"FROM", "JOIN", "DESCRIBE", "WITH", "LEFT JOIN", "RIGHT JOIN"}
CTE_PREFIX: str = "CTE__"

logger: logging.Logger = logging.getLogger(__name__)

# TODO: Workaround for https://github.com/andialbrecht/sqlparse/issues/652.
# configure the Lexer to extend sqlparse
# reference: https://sqlparse.readthedocs.io/en/stable/extending/
lex: Lexer = Lexer.get_default_instance()
sqlparser_sql_regex: List[Tuple[str, Any]] = keywords.SQL_REGEX
sqlparser_sql_regex.insert(25, (r"'(''|\\\\|\\|[^'])*'", sqlparse.tokens.String.Single))
lex.set_SQL_REGEX(sqlparser_sql_regex)


class CtasMethod(StrEnum):
    TABLE: str = "TABLE"
    VIEW: str = "VIEW"


def _extract_limit_from_query(statement: TokenList) -> Optional[int]:
    """
    Extract limit clause from SQL statement.

    :param statement: SQL statement
    :return: Limit extracted from query, None if no limit present in statement
    """
    idx, _ = statement.token_next_by(m=(Keyword, "LIMIT"))
    if idx is not None:
        _, token = statement.token_next(idx=idx)
        if token:
            if isinstance(token, IdentifierList):
                # In case of "LIMIT <offset>, <limit>", find comma and extract
                # first succeeding non-whitespace token
                idx, _ = token.token_next_by(m=(sqlparse.tokens.Punctuation, ","))
                _, token = token.token_next(idx=idx)
            if token and token.ttype == sqlparse.tokens.Literal.Number.Integer:
                return int(token.value)
    return None


def extract_top_from_query(statement: TokenList, top_keywords: Set[str]) -> Optional[int]:
    """
    Extract top clause value from SQL statement.

    :param statement: SQL statement
    :param top_keywords: keywords that are considered as synonyms to TOP
    :return: top value extracted from query, None if no top value present in statement
    """
    str_statement: str = str(statement)
    str_statement = str_statement.replace("\n", " ").replace("\r", "")
    token: List[str] = str_statement.rstrip().split(" ")
    token = [part for part in token if part]
    top: Optional[int] = None
    for i, part in enumerate(token):
        if part.upper() in top_keywords and len(token) - 1 > i:
            try:
                top = int(token[i + 1])
            except ValueError:
                top = None
            break
    return top


def get_cte_remainder_query(sql: str) -> Tuple[Optional[str], str]:
    """
    parse the SQL and return the CTE and rest of the block to the caller

    :param sql: SQL query
    :return: CTE and remainder block to the caller
    """
    cte: Optional[str] = None
    remainder: str = sql
    stmt: TokenList = sqlparse.parse(sql)[0]

    # The first meaningful token for CTE will be with WITH
    idx, token = stmt.token_next(-1, skip_ws=True, skip_cm=True)
    if not (token and token.ttype == CTE):
        return cte, remainder
    idx, token = stmt.token_next(idx)
    idx = stmt.token_index(token) + 1

    # extract rest of the SQLs after CTE
    remainder = "".join(str(token) for token in stmt.tokens[idx:]).strip()
    cte = f"WITH {token.value}"

    return cte, remainder


def check_sql_functions_exist(
    sql: str,
    function_list: Set[str],
    engine: str = "base",
) -> bool:
    """
    Check if the SQL statement contains any of the specified functions.

    :param sql: The SQL statement
    :param function_list: The list of functions to search for
    :param engine: The engine to use for parsing the SQL statement
    """
    return ParsedQuery(sql, engine=engine).check_functions_exist(function_list)


def strip_comments_from_sql(statement: str, engine: str = "base") -> str:
    """
    Strips comments from a SQL statement, does a simple test first
    to avoid always instantiating the expensive ParsedQuery constructor

    This is useful for engines that don't support comments

    :param statement: A string with the SQL statement
    :return: SQL statement without comments
    """
    return (
        ParsedQuery(statement, engine=engine).strip_comments()
        if "--" in statement
        else statement
    )


class ParsedQuery:
    def __init__(
        self,
        sql_statement: str,
        strip_comments: bool = False,
        engine: str = "base",
    ) -> None:
        if strip_comments:
            sql_statement = sqlparse.format(sql_statement, strip_comments=True)

        self.sql: str = sql_statement
        self._engine: str = engine
        self._dialect: Optional[str] = SQLGLOT_DIALECTS.get(engine) if engine else None
        self._tables: Set[Table] = set()
        self._alias_names: Set[str] = set()
        self._limit: Optional[int] = None

        logger.debug("Parsing with sqlparse statement: %s", self.sql)
        self._parsed: List[TokenList] = sqlparse.parse(self.stripped())
        for statement in self._parsed:
            self._limit = _extract_limit_from_query(statement)

    @property
    def tables(self) -> Set[Table]:
        if not self._tables:
            self._tables = self._extract_tables_from_sql()
        return self._tables

    def _check_functions_exist_in_token(
        self, token: Token, functions: Set[str]
    ) -> bool:
        if (
            isinstance(token, Function)
            and token.get_name() is not None
            and token.get_name().lower() in functions
        ):
            return True
        if hasattr(token, "tokens"):
            for inner_token in token.tokens:
                if self._check_functions_exist_in_token(inner_token, functions):
                    return True
        return False

    def check_functions_exist(self, functions: Set[str]) -> bool:
        """
        Check if the SQL statement contains any of the specified functions.

        :param functions: A set of functions to search for
        :return: True if the statement contains any of the specified functions
        """
        for statement in self._parsed:
            for token in statement.tokens:
                if self._check_functions_exist_in_token(token, functions):
                    return True
        return False

    def _extract_tables_from_sql(self) -> Set[Table]:
        """
        Extract all table references in a query.

        Note: this uses sqlglot, since it's better at catching more edge cases.
        """
        try:
            statements: List[Any] = [
                statement._parsed  # pylint: disable=protected-access
                for statement in SQLScript(self.stripped(), self._engine).statements
            ]
        except SupersetParseError as ex:
            logger.warning("Unable to parse SQL (%s): %s", self._dialect, self.sql)
            raise SupersetSecurityException(
                SupersetError(
                    error_type=SupersetErrorType.QUERY_SECURITY_ACCESS_ERROR,
                    message=__(
                        "You may have an error in your SQL statement. {message}"
                    ).format(message=ex.error.message),
                    level=ErrorLevel.ERROR,
                )
            ) from ex

        return {
            table
            for statement in statements
            for table in extract_tables_from_statement(statement, self._dialect)
            if statement
        }

    @property
    def limit(self) -> Optional[int]:
        return self._limit

    def _get_cte_tables(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "with" not in parsed:
            return []
        return parsed["with"].get("cte_tables", [])

    def _check_cte_is_select(self, oxide_parse: List[Dict[str, Any]]) -> bool:
        """
        Check if a oxide parsed CTE contains only SELECT statements

        :param oxide_parse: parsed CTE
        :return: True if CTE is a SELECT statement
        """

        def is_body_select(body: Dict[str, Any]) -> bool:
            if op := body.get("SetOperation"):
                return is_body_select(op["left"]) and is_body_select(op["right"])
            return all(key == "Select" for key in body.keys())

        for query in oxide_parse:
            parsed_query = query["Query"]
            cte_tables = self._get_cte_tables(parsed_query)
            for cte_table in cte_tables:
                is_select = is_body_select(cte_table["query"]["body"])
                if not is_select:
                    return False
        return True

    def is_select(self) -> bool:  # noqa: C901
        # make sure we strip comments; prevents a bug with comments in the CTE
        parsed: List[TokenList] = sqlparse.parse(self.strip_comments())
        seen_select: bool = False

        for statement in parsed:
            # Check if this is a CTE
            if statement.is_group and statement[0].ttype == Keyword.CTE:
                if sqloxide_parse is not None:
                    try:
                        if not self._check_cte_is_select(
                            sqloxide_parse(self.strip_comments(), dialect="ansi")
                        ):
                            return False
                    except ValueError:
                        # sqloxide was not able to parse the query, so let's continue with  # noqa: E501
                        # sqlparse
                        pass
                inner_cte = self.get_inner_cte_expression(statement.tokens) or []
                # Check if the inner CTE is a not a SELECT
                if any(token.ttype == DDL for token in inner_cte) or any(
                    token.ttype == DML and token.normalized != "SELECT"
                    for token in inner_cte
                ):
                    return False

            if statement.get_type() == "SELECT":
                seen_select = True
                continue

            if statement.get_type() != "UNKNOWN":
                return False

            # for `UNKNOWN`, check all DDL/DML explicitly: only `SELECT` DML is allowed,
            # and no DDL is allowed
            if any(token.ttype == DDL for token in statement) or any(
                token.ttype == DML and token.normalized != "SELECT"
                for token in statement
            ):
                return False

            if imt(statement.tokens[0], m=(Keyword, "USE")):
                continue

            # return false on `EXPLAIN`, `SET`, `SHOW`, etc.
            if imt(statement.tokens[0], t=Keyword):
                return False

            if not any(
                token.ttype == DML and token.normalized == "SELECT"
                for token in statement
            ):
                return False

        return seen_select

    def get_inner_cte_expression(self, tokens: TokenList) -> Optional[TokenList]:
        for token in tokens:
            if self._is_identifier(token):
                for identifier_token in token.tokens:
                    if (
                        isinstance(identifier_token, Parenthesis)
                        and identifier_token.is_group
                    ):
                        return identifier_token.tokens
        return None

    def is_valid_ctas(self) -> bool:
        parsed: List[TokenList] = sqlparse.parse(self.strip_comments())
        return parsed[-1].get_type() == "SELECT"

    def is_valid_cvas(self) -> bool:
        parsed: List[TokenList] = sqlparse.parse(self.strip_comments())
        return len(parsed) == 1 and parsed[0].get_type() == "SELECT"

    def is_explain(self) -> bool:
        # Remove comments
        statements_without_comments: str = sqlparse.format(
            self.stripped(), strip_comments=True
        )

        # Explain statements will only be the first statement
        return statements_without_comments.upper().startswith("EXPLAIN")

    def is_show(self) -> bool:
        # Remove comments
        statements_without_comments: str = sqlparse.format(
            self.stripped(), strip_comments=True
        )
        # Show statements will only be the first statement
        return statements_without_comments.upper().startswith("SHOW")

    def is_set(self) -> bool:
        # Remove comments
        statements_without_comments: str = sqlparse.format(
            self.stripped(), strip_comments=True
        )
        # Set statements will only be the first statement
        return statements_without_comments.upper().startswith("SET")

    def is_unknown(self) -> bool:
        return self._parsed[0].get_type() == "UNKNOWN"

    def stripped(self) -> str:
        return self.sql.strip(" \t\r\n;")

    def strip_comments(self) -> str:
        return sqlparse.format(self.stripped(), strip_comments=True)

    def get_statements(self) -> List[str]:
        """Returns a list of SQL statements as strings, stripped"""
        statements: List[str] = []
        for statement in self._parsed:
            if statement:
                sql: str = str(statement).strip(" \n;\t")
                if sql:
                    statements.append(sql)
        return statements

    @staticmethod
    def get_table(tlist: TokenList) -> Optional[Table]:
        """
        Return the table if valid, i.e., conforms to the [[catalog.]schema.]table
        construct.

        :param tlist: The SQL tokens
        :returns: The table if the name conforms
        """

        # Strip the alias if present.
        idx: int = len(tlist.tokens)

        if tlist.has_alias():
            ws_idx, _ = tlist.token_next_by(t=Whitespace)

            if ws_idx != -1:
                idx = ws_idx

        tokens = tlist.tokens[:idx]

        if (
            len(tokens) in (1, 3, 5)
            and all(imt(token, t=[Name, String]) for token in tokens[::2])
            and all(imt(token, m=(Punctuation, ".")) for token in tokens[1::2])
        ):
            return Table(*[remove_quotes(token.value) for token in tokens[::-2]])

        return None

    @staticmethod
    def _is_identifier(token: Token) -> bool:
        return isinstance(token, (IdentifierList, Identifier))

    def as_create_table(
        self,
        table_name: str,
        schema_name: Optional[str] = None,
        overwrite: bool = False,
        method: CtasMethod = CtasMethod.TABLE,
    ) -> str:
        """Reformats the query into the create table as query.

        Works only for the single select SQL statements, in all other cases
        the sql query is not modified.
        :param table_name: table that will contain the results of the query execution
        :param schema_name: schema name for the target table
        :param overwrite: table_name will be dropped if true
        :param method: method for the CTA query, currently view or table creation
        :return: Create table as query
        """
        exec_sql: str = ""
        sql: str = self.stripped()
        # TODO(bkyryliuk): quote full_table_name
        full_table_name: str = f"{schema_name}.{table_name}" if schema_name else table_name
        if overwrite:
            exec_sql = f"DROP {method} IF EXISTS {full_table_name};\n"
        exec_sql += f"CREATE {method} {full_table_name} AS \n{sql}"
        return exec_sql

    def set_or_update_query