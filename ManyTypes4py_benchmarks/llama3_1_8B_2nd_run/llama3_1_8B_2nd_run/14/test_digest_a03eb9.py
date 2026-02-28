from __future__ import annotations
from contextlib import nullcontext
from typing import Any, TYPE_CHECKING
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
from flask_appbuilder.security.sqla.models import User
from superset.connectors.sqla.models import BaseDatasource, SqlaTable
from superset.tasks.exceptions import InvalidExecutorError
from superset.tasks.types import Executor, ExecutorType, FixedExecutor
from superset.utils.core import DatasourceType, override_user

if TYPE_CHECKING:
    from superset.models.dashboard import Dashboard
    from superset.models.slice import Slice

_DEFAULT_DASHBOARD_KWARGS: dict[str, Any] = {'id': 1, 'dashboard_title': 'My Title', 'slices': [{'id': 1, 'slice_name': 'My Chart'}], 'position_json': '{"a": "b"}', 'css': 'background-color: lightblue;', 'json_metadata': '{"c": "d"}'}
_DEFAULT_CHART_KWARGS: dict[str, Any] = {'id': 2, 'params': {'a': 'b'}}

def custom_dashboard_func(dashboard: Dashboard, executor_type: ExecutorType, executor: Executor) -> str:
    return f'{dashboard.id}.{executor_type.value}.{executor}'

def custom_chart_func(chart: Slice, executor_type: ExecutorType, executor: Executor) -> str:
    return f'{chart.id}.{executor_type.value}.{executor}'

def prepare_datasource_mock(datasource_conf: dict[str, Any], spec: type) -> MagicMock:
    datasource = MagicMock(spec=spec)
    datasource.id = 1
    datasource.type = DatasourceType.TABLE
    datasource.is_rls_supported = datasource_conf.get('is_rls_supported', False)
    datasource.get_sqla_row_level_filters = datasource_conf.get('get_sqla_row_level_filters', MagicMock(return_value=[]))
    return datasource

@pytest.mark.parametrize('dashboard_overrides,execute_as,has_current_user,use_custom_digest,rls_datasources,expected_result', [
    (None, [FixedExecutor('admin')], False, False, [], '71452fee8ffbd8d340193d611bcd4559'),
    (None, [ExecutorType.CURRENT_USER], True, False, [], '209dc060ac19271b8708731e3b8280f5'),
    ({'dashboard_title': 'My Other Title'}, [ExecutorType.CURRENT_USER], True, False, [], '209dc060ac19271b8708731e3b8280f5'),
    ({'id': 2}, [ExecutorType.CURRENT_USER], True, False, [], '06a4144466dbd5ffad0c3c2225e96296'),
    ({'slices': [{'id': 2, 'slice_name': 'My Other Chart'}]}, [ExecutorType.CURRENT_USER], True, False, [], 'a823ece9563895ccb14f3d9095e84f7a'),
    ({'position_json': {'b': 'c'}}, [ExecutorType.CURRENT_USER], True, False, [], '33c5475f92a904925ab3ef493526e5b5'),
    ({'css': 'background-color: darkblue;'}, [ExecutorType.CURRENT_USER], True, False, [], 'cec57345e6402c0d4b3caee5cfaa0a03'),
    ({'json_metadata': {'d': 'e'}}, [ExecutorType.CURRENT_USER], True, False, [], '5380dcbe94621a0759b09554404f3d02'),
    (None, [ExecutorType.CURRENT_USER], True, False, [{'is_rls_supported': True, 'get_sqla_row_level_filters': MagicMock(return_value=['filter1'])}], '4138959f275c1991466cafcfb190fd72'),
    (None, [ExecutorType.CURRENT_USER], True, False, [{'is_rls_supported': True, 'get_sqla_row_level_filters': MagicMock(return_value=['filter1', 'filter2'])}, {'is_rls_supported': True, 'get_sqla_row_level_filters': MagicMock(return_value=['filter3', 'filter4'])}], '80d3bfcc7144bccdba8c718cf49b6420'),
    (None, [ExecutorType.CURRENT_USER], True, False, [{'is_rls_supported': False, 'get_sqla_row_level_filters': MagicMock(return_value=[])}, {'is_rls_supported': True, 'get_sqla_row_level_filters': MagicMock(return_value=['filter1', 'filter2'])}], 'e8fc68cd5aba22a5f1acf06164bfc0f4'),
    (None, [ExecutorType.CURRENT_USER], False, False, [], None),
    (None, [ExecutorType.FIXED_USER], False, False, [], InvalidExecutorError())
])
def test_dashboard_digest(dashboard_overrides: dict[str, Any] | None, execute_as: list[Executor], has_current_user: bool, use_custom_digest: bool, rls_datasources: list[dict[str, Any]], expected_result: str | Exception):
    from superset import app, security_manager
    from superset.models.dashboard import Dashboard
    from superset.models.slice import Slice
    from superset.thumbnails.digest import get_dashboard_digest
    kwargs = {**_DEFAULT_DASHBOARD_KWARGS, **(dashboard_overrides or {})}
    slices = [Slice(**slice_kwargs) for slice_kwargs in kwargs.pop('slices')]
    dashboard = Dashboard(**kwargs, slices=slices)
    datasources = []
    for rls_source in rls_datasources:
        datasource = prepare_datasource_mock(rls_source, BaseDatasource)
        datasources.append(datasource)
    user = None
    if has_current_user:
        user = User(id=1, username='1')
    func = custom_dashboard_func if use_custom_digest else None
    with patch.dict(app.config, {'THUMBNAIL_EXECUTORS': execute_as, 'THUMBNAIL_DASHBOARD_DIGEST_FUNC': func}), patch.object(type(dashboard), 'datasources', new_callable=PropertyMock, return_value=datasources), patch.object(security_manager, 'find_user', return_value=user), override_user(user):
        cm = pytest.raises(type(expected_result)) if isinstance(expected_result, Exception) else nullcontext()
        with cm:
            assert get_dashboard_digest(dashboard=dashboard) == expected_result

@pytest.mark.parametrize('chart_overrides,execute_as,has_current_user,use_custom_digest,rls_datasource,expected_result', [
    (None, [FixedExecutor('admin')], False, False, None, '47d852b5c4df211c115905617bb722c1'),
    (None, [ExecutorType.CURRENT_USER], True, False, None, '4f8109d3761e766e650af514bb358f10'),
    (None, [ExecutorType.CURRENT_USER], True, True, None, '2.current_user.1'),
    (None, [ExecutorType.CURRENT_USER], True, False, {'is_rls_supported': True, 'get_sqla_row_level_filters': MagicMock(return_value=['filter1'])}, '61e70336c27eb97fb050328a0b050373'),
    (None, [ExecutorType.CURRENT_USER], True, False, {'is_rls_supported': True, 'get_sqla_row_level_filters': MagicMock(return_value=['filter1', 'filter2'])}, '95c7cefde8cb519f005f33bfb33cb196'),
    (None, [ExecutorType.CURRENT_USER], True, False, {'is_rls_supported': False, 'get_sqla_row_level_filters': MagicMock(return_value=[])}, '4f8109d3761e766e650af514bb358f10'),
    (None, [ExecutorType.CURRENT_USER], False, False, None, None),
    (None, [ExecutorType.FIXED_USER], False, False, None, InvalidExecutorError())
])
def test_chart_digest(chart_overrides: dict[str, Any] | None, execute_as: list[Executor], has_current_user: bool, use_custom_digest: bool, rls_datasource: dict[str, Any] | None, expected_result: str | Exception):
    from superset import app, security_manager
    from superset.models.slice import Slice
    from superset.thumbnails.digest import get_chart_digest
    datasource = None
    if rls_datasource:
        datasource = prepare_datasource_mock(rls_datasource, SqlaTable)
    kwargs = {**_DEFAULT_CHART_KWARGS, **(chart_overrides or {})}
    chart = Slice(**kwargs)
    user = None
    if has_current_user:
        user = User(id=1, username='1')
    func = custom_chart_func if use_custom_digest else None
    with patch.dict(app.config, {'THUMBNAIL_EXECUTORS': execute_as, 'THUMBNAIL_CHART_DIGEST_FUNC': func}), patch.object(type(chart), 'datasource', new_callable=PropertyMock, return_value=datasource), patch.object(security_manager, 'find_user', return_value=user), override_user(user):
        cm = pytest.raises(type(expected_result)) if isinstance(expected_result, Exception) else nullcontext()
        with cm:
            assert get_chart_digest(chart=chart) == expected_result
