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

import logging
import re
from re import Pattern
from typing import Any, Optional, Union, Dict, List, Tuple, Type
from urllib import parse

from flask_babel import gettext as __
from sqlalchemy import Float, Integer, Numeric, types
from sqlalchemy.engine.url import URL
from sqlalchemy.sql.type_api import TypeEngine

from superset.db_engine_specs.mysql import MySQLEngineSpec
from superset.errors import SupersetErrorType
from superset.models.core import Database
from superset.utils.core import GenericDataType

# Regular expressions to catch custom errors
CONNECTION_ACCESS_DENIED_REGEX: Pattern[str] = re.compile(
    "Access denied for user '(?P<username>.*?)'"
)
CONNECTION_UNKNOWN_DATABASE_REGEX: Pattern[str] = re.compile(
    "Unknown database '(?P<database>.*?)'"
)

logger: logging.Logger = logging.getLogger(__name__)


class TINYINT(Integer):
    __visit_name__: str = "TINYINT"


class LARGEINT(Integer):
    __visit_name__: str = "LARGEINT"


class DOUBLE(Float):
    __visit_name__: str = "DOUBLE"


class HLL(Numeric):
    __visit_name__: str = "HLL"


class BITMAP(Numeric):
    __visit_name__: str = "BITMAP"


class PERCENTILE(Numeric):
    __visit_name__: str = "PERCENTILE"


class ARRAY(TypeEngine):
    __visit_name__: str = "ARRAY"

    @property
    def python_type(self) -> Optional[Type[List[Any]]]:
        return list


class MAP(TypeEngine):
    __visit_name__: str = "MAP"

    @property
    def python_type(self) -> Optional[Type[Dict[Any, Any]]]:
        return dict


class STRUCT(TypeEngine):
    __visit_name__: str = "STRUCT"

    @property
    def python_type(self) -> Optional[Type[Any]]:
        return None


class StarRocksEngineSpec(MySQLEngineSpec):
    engine: str = "starrocks"
    engine_name: str = "StarRocks"

    default_driver: str = "starrocks"
    sqlalchemy_uri_placeholder: str = (
        "starrocks://user:password@host:port/catalog.db[?key=value&key=value...]"
    )

    column_type_mappings: Tuple[Tuple[Pattern[str], TypeEngine, GenericDataType], ...] = (
        (
            re.compile(r"^tinyint", re.IGNORECASE),
            TINYINT(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^largeint", re.IGNORECASE),
            LARGEINT(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^decimal.*", re.IGNORECASE),
            types.DECIMAL(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^double", re.IGNORECASE),
            DOUBLE(),
            GenericDataType.NUMERIC,
        ),
        (
            re.compile(r"^varchar(\((\d+)\))*$", re.IGNORECASE),
            types.VARCHAR(),
            GenericDataType.STRING,
        ),
        (
            re.compile(r"^char(\((\d+)\))*$", re.IGNORECASE),
            types.CHAR(),
            GenericDataType.STRING,
        ),
        (
            re.compile(r"^json", re.IGNORECASE),
            types.JSON(),
            GenericDataType.STRING,
        ),
        (
            re.compile(r"^binary.*", re.IGNORECASE),
            types.String(),
            GenericDataType.STRING,
        ),
        (
            re.compile(r"^percentile", re.IGNORECASE),
            PERCENTILE(),
            GenericDataType.STRING,
        ),
        (re.compile(r"^hll", re.IGNORECASE), HLL(), GenericDataType.STRING),
        (re.compile(r"^bitmap", re.IGNORECASE), BITMAP(), GenericDataType.STRING),
        (re.compile(r"^array.*", re.IGNORECASE), ARRAY(), GenericDataType.STRING),
        (re.compile(r"^map.*", re.IGNORECASE), MAP(), GenericDataType.STRING),
        (re.compile(r"^struct.*", re.IGNORECASE), STRUCT(), GenericDataType.STRING),
    )

    custom_errors: Dict[Pattern[str], Tuple[str, SupersetErrorType, Dict[str, Any]]] = {
        CONNECTION_ACCESS_DENIED_REGEX: (
            __('Either the username "%(username)s" or the password is incorrect.'),
            SupersetErrorType.CONNECTION_ACCESS_DENIED_ERROR,
            {"invalid": ["username", "password"]},
        ),
        CONNECTION_UNKNOWN_DATABASE_REGEX: (
            __('Unable to connect to database "%(database)s".'),
            SupersetErrorType.CONNECTION_UNKNOWN_DATABASE_ERROR,
            {"invalid": ["database"]},
        ),
    }

    @classmethod
    def adjust_engine_params(
        cls,
        uri: URL,
        connect_args: Dict[str, Any],
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> Tuple[URL, Dict[str, Any]]:
        database: Optional[str] = uri.database
        if schema and database:
            schema = parse.quote(schema, safe="")
            if "." in database:
                database = database.split(".")[0] + "." + schema
            else:
                database = "default_catalog." + schema
            uri = uri.set(database=database)

        return uri, connect_args

    @classmethod
    def get_schema_from_engine_params(
        cls,
        sqlalchemy_uri: URL,
        connect_args: Dict[str, Any],
    ) -> Optional[str]:
        """
        Return the configured schema.

        For StarRocks the SQLAlchemy URI looks like this:

            starrocks://localhost:9030/catalog.schema

        """
        database: str = sqlalchemy_uri.database.strip("/")

        if "." not in database:
            return None

        return parse.unquote(database.split(".")[1])

    @classmethod
    def get_url_for_impersonation(
        cls,
        url: URL,
        impersonate_user: bool,
        username: Optional[str] = None,
        access_token: Optional[str] = None,
    ) -> URL:
        """
        Return a modified URL with the username set.

        :param url: SQLAlchemy URL object
        :param impersonate_user: Flag indicating if impersonation is enabled
        :param username: Effective username
        :param access_token: Personal access token
        """
        # Leave URL unchanged. We will impersonate with the pre-query below.
        return url

    @classmethod
    def get_prequeries(
        cls,
        database: Database,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
    ) -> List[str]:
        """
        Return pre-session queries.

        These are currently used as an alternative to ``adjust_engine_params`` for
        databases where the selected schema cannot be specified in the SQLAlchemy URI or
        connection arguments.

        For example, in order to specify a default schema in RDS we need to run a query
        at the beginning of the session:

            sql> set search_path = my_schema;

        """
        if database.impersonate_user:
            username: Optional[str] = database.get_effective_user(database.url_object)

            if username:
                return [f'EXECUTE AS "{username}" WITH NO REVERT;']

        return []
