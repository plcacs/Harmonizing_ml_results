from typing import Any, Dict, List, Optional, Tuple, Union
import pytest
from pytest_mock import MockerFixture
from sqlalchemy import JSON, types
from sqlalchemy.engine.url import URL, make_url
from superset.db_engine_specs.starrocks import ARRAY, BITMAP, DOUBLE, HLL, LARGEINT, MAP, PERCENTILE, STRUCT, TINYINT
from superset.utils.core import GenericDataType
from tests.unit_tests.db_engine_specs.utils import assert_column_spec

@pytest.mark.parametrize('native_type,sqla_type,attrs,generic_type,is_dttm', [('tinyint', TINYINT, None, GenericDataType.NUMERIC, False), ('largeint', LARGEINT, None, GenericDataType.NUMERIC, False), ('decimal(38,18)', types.DECIMAL, None, GenericDataType.NUMERIC, False), ('double', DOUBLE, None, GenericDataType.NUMERIC, False), ('char(10)', types.CHAR, None, GenericDataType.STRING, False), ('varchar(65533)', types.VARCHAR, None, GenericDataType.STRING, False), ('binary', types.String, None, GenericDataType.STRING, False), ('array<varchar(65533)>', ARRAY, None, GenericDataType.STRING, False), ('map<string,int>', MAP, None, GenericDataType.STRING, False), ('struct<int,string>', STRUCT, None, GenericDataType.STRING, False), ('json', JSON, None, GenericDataType.STRING, False), ('bitmap', BITMAP, None, GenericDataType.STRING, False), ('hll', HLL, None, GenericDataType.STRING, False), ('percentile', PERCENTILE, None, GenericDataType.STRING, False)])
def test_get_column_spec(native_type: str, sqla_type: Any, attrs: Optional[Dict[str, Any]], generic_type: GenericDataType, is_dttm: bool) -> None:
    from superset.db_engine_specs.starrocks import StarRocksEngineSpec as spec
    assert_column_spec(spec, native_type, sqla_type, attrs, generic_type, is_dttm)

@pytest.mark.parametrize('sqlalchemy_uri,connect_args,return_schema,return_connect_args', [('starrocks://user:password@host/db1', {'param1': 'some_value'}, 'db1', {'param1': 'some_value'}), ('starrocks://user:password@host/catalog1.db1', {'param1': 'some_value'}, 'catalog1.db1', {'param1': 'some_value'})])
def test_adjust_engine_params(sqlalchemy_uri: str, connect_args: Dict[str, Any], return_schema: str, return_connect_args: Dict[str, Any]) -> None:
    from superset.db_engine_specs.starrocks import StarRocksEngineSpec
    url = make_url(sqlalchemy_uri)
    returned_url, returned_connect_args = StarRocksEngineSpec.adjust_engine_params(url, connect_args)
    assert returned_url.database == return_schema
    assert returned_connect_args == return_connect_args

def test_get_schema_from_engine_params() -> None:
    """
    Test the ``get_schema_from_engine_params`` method.
    """
    from superset.db_engine_specs.starrocks import StarRocksEngineSpec
    assert StarRocksEngineSpec.get_schema_from_engine_params(make_url('starrocks://localhost:9030/hive.default'), {}) == 'default'
    assert StarRocksEngineSpec.get_schema_from_engine_params(make_url('starrocks://localhost:9030/hive'), {}) is None

def test_impersonation_username(mocker: MockerFixture) -> None:
    """
    Test impersonation and make sure that `get_url_for_impersonation` leaves the URL
    unchanged and that `get_prequeries` returns the appropriate impersonation query.
    """
    from superset.db_engine_specs.starrocks import StarRocksEngineSpec
    database = mocker.MagicMock()
    database.impersonate_user = True
    database.get_effective_user.return_value = 'alice'
    assert StarRocksEngineSpec.get_url_for_impersonation(url=make_url('starrocks://service_user@localhost:9030/hive.default'), impersonate_user=True, username='alice', access_token=None) == make_url('starrocks://service_user@localhost:9030/hive.default')
    assert StarRocksEngineSpec.get_prequeries(database) == ['EXECUTE AS "alice" WITH NO REVERT;']

def test_impersonation_disabled(mocker: MockerFixture) -> None:
    """
    Test that impersonation is not applied when the feature is disabled in
    `get_url_for_impersonation` and `get_prequeries`.
    """
    from superset.db_engine_specs.starrocks import StarRocksEngineSpec
    database = mocker.MagicMock()
    database.impersonate_user = False
    database.get_effective_user.return_value = 'alice'
    assert StarRocksEngineSpec.get_url_for_impersonation(url=make_url('starrocks://service_user@localhost:9030/hive.default'), impersonate_user=False, username='alice', access_token=None) == make_url('starrocks://service_user@localhost:9030/hive.default')
    assert StarRocksEngineSpec.get_prequeries(database) == []
