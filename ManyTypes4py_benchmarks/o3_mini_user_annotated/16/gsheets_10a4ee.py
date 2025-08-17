#!/usr/bin/env python3
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

from __future__ import annotations

import logging
import re
from re import Pattern
from typing import Any, TYPE_CHECKING, TypedDict

import pandas as pd
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from flask import g
from flask_babel import gettext as __
from marshmallow import fields, Schema
from marshmallow.exceptions import ValidationError
from requests import Session
from shillelagh.adapters.api.gsheets.lib import SCOPES
from shillelagh.exceptions import UnauthenticatedError
from sqlalchemy.engine import create_engine
from sqlalchemy.engine.url import URL

from superset import db, security_manager
from superset.databases.schemas import encrypted_field_properties, EncryptedString
from superset.db_engine_specs.shillelagh import ShillelaghEngineSpec
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import SupersetException
from superset.utils import json

if TYPE_CHECKING:
    from superset.models.core import Database
    from superset.sql_parse import Table

_logger: logging.Logger = logging.getLogger()

EXAMPLE_GSHEETS_URL: str = (
    "https://docs.google.com/spreadsheets/d/"
    "1LcWZMsdCl92g7nA-D6qGRqg1T5TiHyuKJUY1u9XAnsk/edit#gid=0"
)

SYNTAX_ERROR_REGEX: Pattern[str] = re.compile('SQLError: near "(?P<server_error>.*?)": syntax error')

ma_plugin: MarshmallowPlugin = MarshmallowPlugin()


class GSheetsParametersSchema(Schema):
    catalog = fields.Dict()  # type: ignore
    service_account_info = EncryptedString(
        required=False,
        metadata={
            "description": "Contents of GSheets JSON credentials.",
            "field_name": "service_account_info",
        },
    )


class GSheetsParametersType(TypedDict):
    service_account_info: str
    catalog: dict[str, str] | None


class GSheetsPropertiesType(TypedDict):
    parameters: GSheetsParametersType
    catalog: dict[str, str]


class GSheetsEngineSpec(ShillelaghEngineSpec):
    """Engine for Google spreadsheets"""

    engine_name: str = "Google Sheets"
    engine: str = "gsheets"
    allows_joins: bool = True
    allows_subqueries: bool = True

    parameters_schema: Schema = GSheetsParametersSchema()
    default_driver: str = "apsw"
    sqlalchemy_uri_placeholder: str = "gsheets://"

    # when editing the database, mask this field in `encrypted_extra`
    # pylint: disable=invalid-name
    encrypted_extra_sensitive_fields: set[str] = {"$.service_account_info.private_key"}

    custom_errors: dict[Pattern[str], tuple[str, SupersetErrorType, dict[str, Any]]] = {
        SYNTAX_ERROR_REGEX: (
            __(
                'Please check your query for syntax errors near "%(server_error)s". '
                "Then, try running your query again.",
            ),
            SupersetErrorType.SYNTAX_ERROR,
            {},
        ),
    }

    supports_file_upload: bool = True

    # OAuth 2.0
    supports_oauth2: bool = True
    oauth2_scope: str = " ".join(SCOPES)
    oauth2_authorization_request_uri: str = (
        "https://accounts.google.com/o/oauth2/v2/auth"
    )
    oauth2_token_request_uri: str = "https://oauth2.googleapis.com/token"
    oauth2_exception = UnauthenticatedError

    @classmethod
    def get_url_for_impersonation(
        cls,
        url: URL,
        impersonate_user: bool,
        username: str | None,
        access_token: str | None,
    ) -> URL:
        if not impersonate_user:
            return url

        if username is not None:
            user = security_manager.find_user(username=username)
            if user and user.email:
                url = url.update_query_dict({"subject": user.email})

        if access_token:
            url = url.update_query_dict({"access_token": access_token})

        return url

    @classmethod
    def get_extra_table_metadata(
        cls,
        database: Database,
        table: Table,
    ) -> dict[str, Any]:
        with database.get_raw_connection(
            catalog=table.catalog,
            schema=table.schema,
        ) as conn:
            cursor = conn.cursor()
            cursor.execute(f'SELECT GET_METADATA("{table.table}")')
            results = cursor.fetchone()[0]
        try:
            metadata: dict[str, Any] = json.loads(results)
        except Exception:
            metadata = {}

        return {"metadata": metadata.get("extra", {})}

    @classmethod
    def build_sqlalchemy_uri(
        cls,
        _: GSheetsParametersType,
        encrypted_extra: dict[str, Any] | None = None,
    ) -> str:
        return "gsheets://"

    @classmethod
    def get_parameters_from_uri(
        cls,
        uri: str,
        encrypted_extra: dict[str, Any] | None = None,
    ) -> Any:
        if encrypted_extra:
            return {**encrypted_extra}

        raise ValidationError("Invalid service credentials")

    @classmethod
    def parameters_json_schema(cls) -> Any:
        """
        Return configuration parameters as OpenAPI.
        """
        if not cls.parameters_schema:
            return None

        spec: APISpec = APISpec(
            title="Database Parameters",
            version="1.0.0",
            openapi_version="3.0.0",
            plugins=[ma_plugin],
        )

        ma_plugin.init_spec(spec)
        ma_plugin.converter.add_attribute_function(encrypted_field_properties)
        spec.components.schema(cls.__name__, schema=cls.parameters_schema)
        return spec.to_dict()["components"]["schemas"][cls.__name__]

    @classmethod
    def validate_parameters(
        cls,
        properties: GSheetsPropertiesType,
    ) -> list[SupersetError]:
        errors: list[SupersetError] = []

        parameters: dict[str, Any] = properties.get("parameters", {})
        if parameters and parameters.get("catalog"):
            table_catalog: dict[str, Any] = parameters.get("catalog", {})
        else:
            table_catalog = properties.get("catalog", {})

        encrypted_credentials: Any = parameters.get("service_account_info") or "{}"

        if isinstance(encrypted_credentials, str):
            encrypted_credentials = json.loads(encrypted_credentials)

        if not table_catalog:
            errors.append(
                SupersetError(
                    message="Sheet name is required",
                    error_type=SupersetErrorType.CONNECTION_MISSING_PARAMETERS_ERROR,
                    level=ErrorLevel.WARNING,
                    extra={"catalog": {"idx": 0, "name": True}},
                ),
            )
            return errors

        subject: str | None = g.user.email if g.user else None

        engine = create_engine(
            "gsheets://",
            service_account_info=encrypted_credentials,
            subject=subject,
        )
        conn = engine.connect()
        idx: int = 0

        for name, url in table_catalog.items():
            if not name:
                errors.append(
                    SupersetError(
                        message="Sheet name is required",
                        error_type=SupersetErrorType.CONNECTION_MISSING_PARAMETERS_ERROR,
                        level=ErrorLevel.WARNING,
                        extra={"catalog": {"idx": idx, "name": True}},
                    ),
                )
                return errors

            if not url:
                errors.append(
                    SupersetError(
                        message="URL is required",
                        error_type=SupersetErrorType.CONNECTION_MISSING_PARAMETERS_ERROR,
                        level=ErrorLevel.WARNING,
                        extra={"catalog": {"idx": idx, "url": True}},
                    ),
                )
                return errors

            try:
                results = conn.execute(f'SELECT * FROM "{url}" LIMIT 1')
                results.fetchall()
            except Exception:
                errors.append(
                    SupersetError(
                        message=(
                            "The URL could not be identified. Please check for typos "
                            "and make sure that ‘Type of Google Sheets allowed’ "
                            "selection matches the input."
                        ),
                        error_type=SupersetErrorType.TABLE_DOES_NOT_EXIST_ERROR,
                        level=ErrorLevel.WARNING,
                        extra={"catalog": {"idx": idx, "url": True}},
                    ),
                )
            idx += 1
        return errors

    @staticmethod
    def _do_post(
        session: Session,
        url: str,
        body: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        _logger.info("POST %s", url)
        _logger.debug(body)
        response = session.post(
            url,
            json=body,
            **kwargs,
        )
        payload: dict[str, Any] = response.json()
        _logger.debug(payload)

        if "error" in payload:
            raise SupersetException(payload["error"]["message"])

        return payload

    @classmethod
    def df_to_sql(
        cls,
        database: Database,
        table: Table,
        df: pd.DataFrame,
        to_sql_kwargs: dict[str, Any],
    ) -> None:
        from shillelagh.backends.apsw.dialects.base import get_adapter_for_table_name

        extra: dict[str, Any] = database.get_extra()
        engine_params: dict[str, Any] = extra.setdefault("engine_params", {})
        catalog: dict[str, Any] = engine_params.setdefault("catalog", {})

        spreadsheet_url: str | None = catalog.get(table.table)
        if spreadsheet_url and "if_exists" in to_sql_kwargs:
            if to_sql_kwargs["if_exists"] == "append":
                raise SupersetException("Append operation not currently supported")
            if to_sql_kwargs["if_exists"] == "fail":
                raise SupersetException("Table already exists")
            if to_sql_kwargs["if_exists"] == "replace":
                pass

        with cls.get_engine(
            database,
            catalog=table.catalog,
            schema=table.schema,
        ) as engine:
            with engine.connect() as conn:
                adapter = get_adapter_for_table_name(
                    conn,
                    spreadsheet_url or EXAMPLE_GSHEETS_URL,
                )
                session: Session = adapter._get_session()  # type: ignore

        if spreadsheet_url:
            spreadsheet_id: str = adapter._spreadsheet_id  # type: ignore
            range_: str = adapter._sheet_name  # type: ignore
            url_post: str = (
                "https://sheets.googleapis.com/v4/spreadsheets/"
                f"{spreadsheet_id}/values/{range_}:clear"
            )
            cls._do_post(session, url_post, {})
        else:
            payload: dict[str, Any] = cls._do_post(
                session,
                "https://sheets.googleapis.com/v4/spreadsheets",
                {"properties": {"title": table.table}},
            )
            spreadsheet_id = payload["spreadsheetId"]
            range_ = payload["sheets"][0]["properties"]["title"]
            spreadsheet_url = payload["spreadsheetUrl"]

        data: list[list[Any]] = df.fillna("").values.tolist()
        data.insert(0, df.columns.values.tolist())
        body: dict[str, Any] = {
            "range": range_,
            "majorDimension": "ROWS",
            "values": data,
        }
        url_append: str = (
            "https://sheets.googleapis.com/v4/spreadsheets/"
            f"{spreadsheet_id}/values/{range_}:append"
        )
        cls._do_post(
            session,
            url_append,
            body,
            params={"valueInputOption": "USER_ENTERED"},
        )

        catalog[table.table] = spreadsheet_url
        database.extra = json.dumps(extra)
        db.session.add(database)
        db.session.commit()