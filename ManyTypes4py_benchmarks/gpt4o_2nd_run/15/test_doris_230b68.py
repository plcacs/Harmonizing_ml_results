from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import Mock
import pytest
from sqlalchemy import JSON, types
from sqlalchemy.engine.url import URL, make_url
from superset.db_engine_specs.doris import (
    AggState,
    ARRAY,
    BITMAP,
    DOUBLE,
    HLL,
    LARGEINT,
    MAP,
    QuantileState,
    STRUCT,
    TINYINT,
)
from superset.utils.core import GenericDataType
from tests.unit_tests.db_engine_specs.utils import assert_column_spec


@pytest.mark.parametrize(
    'native_type,sqla_type,attrs,generic_type,is_dttm',
    [
        ('tinyint', TINYINT, None, GenericDataType.NUMERIC, False),
        ('largeint', LARGEINT, None, GenericDataType.NUMERIC, False),
        ('decimal(38,18)', types.DECIMAL, None, GenericDataType.NUMERIC, False),
        ('decimalv3(38,18)', types.DECIMAL, None, GenericDataType.NUMERIC, False),
        ('double', DOUBLE, None, GenericDataType.NUMERIC, False),
        ('char(10)', types.CHAR, None, GenericDataType.STRING, False),
        ('varchar(65533)', types.VARCHAR, None, GenericDataType.STRING, False),
        ('binary', types.BINARY, None, GenericDataType.STRING, False),
        ('text', types.TEXT, None, GenericDataType.STRING, False),
        ('string', types.String, None, GenericDataType.STRING, False),
        ('datetimev2', types.DateTime, None, GenericDataType.STRING, False),
        ('datev2', types.Date, None, GenericDataType.STRING, False),
        ('array<varchar(65533)>', ARRAY, None, GenericDataType.STRING, False),
        ('map<string,int>', MAP, None, GenericDataType.STRING, False),
        ('struct<int,string>', STRUCT, None, GenericDataType.STRING, False),
        ('json', JSON, None, GenericDataType.STRING, False),
        ('jsonb', JSON, None, GenericDataType.STRING, False),
        ('bitmap', BITMAP, None, GenericDataType.STRING, False),
        ('hll', HLL, None, GenericDataType.STRING, False),
        ('quantile_state', QuantileState, None, GenericDataType.STRING, False),
        ('agg_state', AggState, None, GenericDataType.STRING, False),
    ],
)
def test_get_column_spec(
    native_type: str,
    sqla_type: types.TypeEngine,
    attrs: Optional[Dict[str, Any]],
    generic_type: GenericDataType,
    is_dttm: bool,
) -> None:
    from superset.db_engine_specs.doris import DorisEngineSpec as spec
    assert_column_spec(spec, native_type, sqla_type, attrs, generic_type, is_dttm)


@pytest.mark.parametrize(
    'sqlalchemy_uri,connect_args,return_schema,return_connect_args',
    [
        (
            'doris://user:password@host/db1',
            {'param1': 'some_value'},
            'internal.information_schema',
            {'param1': 'some_value'},
        ),
        (
            'pydoris://user:password@host/db1',
            {'param1': 'some_value'},
            'internal.information_schema',
            {'param1': 'some_value'},
        ),
        (
            'doris://user:password@host/catalog1.db1',
            {'param1': 'some_value'},
            'catalog1.information_schema',
            {'param1': 'some_value'},
        ),
        (
            'pydoris://user:password@host/catalog1.db1',
            {'param1': 'some_value'},
            'catalog1.information_schema',
            {'param1': 'some_value'},
        ),
    ],
)
def test_adjust_engine_params(
    sqlalchemy_uri: str,
    connect_args: Dict[str, Any],
    return_schema: str,
    return_connect_args: Dict[str, Any],
) -> None:
    from superset.db_engine_specs.doris import DorisEngineSpec
    url: URL = make_url(sqlalchemy_uri)
    returned_url, returned_connect_args = DorisEngineSpec.adjust_engine_params(
        url, connect_args
    )
    assert returned_url.database == return_schema
    assert returned_connect_args == return_connect_args


@pytest.mark.parametrize(
    'url,expected_schema',
    [
        ('doris://localhost:9030/hive.test', 'test'),
        ('doris://localhost:9030/hive', None),
    ],
)
def test_get_schema_from_engine_params(url: str, expected_schema: Optional[str]) -> None:
    """
    Test the ``get_schema_from_engine_params`` method.
    """
    from superset.db_engine_specs.doris import DorisEngineSpec
    assert DorisEngineSpec.get_schema_from_engine_params(
        make_url(url), {}
    ) == expected_schema


@pytest.mark.parametrize(
    'database_value,expected_catalog',
    [
        ('catalog1.schema1', 'catalog1'),
        ('catalog1', 'catalog1'),
        (None, None),
    ],
)
def test_get_default_catalog(
    database_value: Optional[str], expected_catalog: Optional[str]
) -> None:
    """
    Test the ``get_default_catalog`` method.
    """
    from superset.db_engine_specs.doris import DorisEngineSpec
    from superset.models.core import Database
    database: Mock = Mock(spec=Database)
    database.url_object.database = database_value
    assert DorisEngineSpec.get_default_catalog(database) == expected_catalog


@pytest.mark.parametrize(
    'mock_catalogs,expected_result',
    [
        (
            [
                Mock(CatalogName='catalog1'),
                Mock(CatalogName='catalog2'),
                Mock(CatalogName='catalog3'),
            ],
            {'catalog1', 'catalog2', 'catalog3'},
        ),
        ([Mock(CatalogName='single_catalog')], {'single_catalog'}),
        ([], set()),
    ],
)
def test_get_catalog_names(
    mock_catalogs: List[Mock], expected_result: Set[str]
) -> None:
    """
    Test the ``get_catalog_names`` method.
    """
    from superset.db_engine_specs.doris import DorisEngineSpec
    from superset.models.core import Database
    database: Mock = Mock(spec=Database)
    inspector: Mock = Mock()
    inspector.bind.execute.return_value = mock_catalogs
    catalogs: Set[str] = DorisEngineSpec.get_catalog_names(database, inspector)
    inspector.bind.execute.assert_called_once_with('SHOW CATALOGS')
    assert catalogs == expected_result
