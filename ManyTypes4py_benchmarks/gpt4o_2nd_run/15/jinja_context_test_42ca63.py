from __future__ import annotations
from typing import Any, List, Dict, Optional
import pytest
from freezegun import freeze_time
from pytest_mock import MockerFixture
from sqlalchemy.dialects import mysql
from sqlalchemy.dialects.postgresql import dialect
from superset import app
from superset.commands.dataset.exceptions import DatasetNotFoundError
from superset.connectors.sqla.models import SqlaTable, SqlMetric, TableColumn
from superset.exceptions import SupersetTemplateException
from superset.jinja_context import dataset_macro, ExtraCache, metric_macro, safe_proxy, TimeFilter, WhereInMacro
from superset.models.core import Database
from superset.models.slice import Slice
from superset.utils import json

def test_filter_values_adhoc_filters() -> None:
    with app.test_request_context(data={'form_data': json.dumps({'adhoc_filters': [{'clause': 'WHERE', 'comparator': 'foo', 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'name'}]})}):
        cache = ExtraCache()
        assert cache.filter_values('name') == ['foo']
        assert cache.applied_filters == ['name']
    with app.test_request_context(data={'form_data': json.dumps({'adhoc_filters': [{'clause': 'WHERE', 'comparator': ['foo', 'bar'], 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'name'}]})}):
        cache = ExtraCache()
        assert cache.filter_values('name') == ['foo', 'bar']
        assert cache.applied_filters == ['name']

def test_filter_values_extra_filters() -> None:
    with app.test_request_context(data={'form_data': json.dumps({'extra_filters': [{'col': 'name', 'op': 'in', 'val': 'foo'}]})}):
        cache = ExtraCache()
        assert cache.filter_values('name') == ['foo']
        assert cache.applied_filters == ['name']

def test_filter_values_default() -> None:
    cache = ExtraCache()
    assert cache.filter_values('name', 'foo') == ['foo']
    assert cache.removed_filters == []

def test_filter_values_remove_not_present() -> None:
    cache = ExtraCache()
    assert cache.filter_values('name', remove_filter=True) == []
    assert cache.removed_filters == []

def test_filter_values_no_default() -> None:
    cache = ExtraCache()
    assert cache.filter_values('name') == []

def test_get_filters_adhoc_filters() -> None:
    with app.test_request_context(data={'form_data': json.dumps({'adhoc_filters': [{'clause': 'WHERE', 'comparator': 'foo', 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'name'}]})}):
        cache = ExtraCache()
        assert cache.get_filters('name') == [{'op': 'IN', 'col': 'name', 'val': ['foo']}]
        assert cache.removed_filters == []
        assert cache.applied_filters == ['name']
    with app.test_request_context(data={'form_data': json.dumps({'adhoc_filters': [{'clause': 'WHERE', 'comparator': ['foo', 'bar'], 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'name'}]})}):
        cache = ExtraCache()
        assert cache.get_filters('name') == [{'op': 'IN', 'col': 'name', 'val': ['foo', 'bar']}]
        assert cache.removed_filters == []
    with app.test_request_context(data={'form_data': json.dumps({'adhoc_filters': [{'clause': 'WHERE', 'comparator': ['foo', 'bar'], 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'name'}]})}):
        cache = ExtraCache()
        assert cache.get_filters('name', remove_filter=True) == [{'op': 'IN', 'col': 'name', 'val': ['foo', 'bar']}]
        assert cache.removed_filters == ['name']
        assert cache.applied_filters == ['name']

def test_get_filters_remove_not_present() -> None:
    cache = ExtraCache()
    assert cache.get_filters('name', remove_filter=True) == []
    assert cache.removed_filters == []

def test_url_param_query() -> None:
    with app.test_request_context(query_string={'foo': 'bar'}):
        cache = ExtraCache()
        assert cache.url_param('foo') == 'bar'

def test_url_param_default() -> None:
    with app.test_request_context():
        cache = ExtraCache()
        assert cache.url_param('foo', 'bar') == 'bar'

def test_url_param_no_default() -> None:
    with app.test_request_context():
        cache = ExtraCache()
        assert cache.url_param('foo') is None

def test_url_param_form_data() -> None:
    with app.test_request_context(query_string={'form_data': json.dumps({'url_params': {'foo': 'bar'}})}):
        cache = ExtraCache()
        assert cache.url_param('foo') == 'bar'

def test_url_param_escaped_form_data() -> None:
    with app.test_request_context(query_string={'form_data': json.dumps({'url_params': {'foo': "O'Brien"}})}):
        cache = ExtraCache(dialect=dialect())
        assert cache.url_param('foo') == "O''Brien"

def test_url_param_escaped_default_form_data() -> None:
    with app.test_request_context(query_string={'form_data': json.dumps({'url_params': {'foo': "O'Brien"}})}):
        cache = ExtraCache(dialect=dialect())
        assert cache.url_param('bar', "O'Malley") == "O''Malley"

def test_url_param_unescaped_form_data() -> None:
    with app.test_request_context(query_string={'form_data': json.dumps({'url_params': {'foo': "O'Brien"}})}):
        cache = ExtraCache(dialect=dialect())
        assert cache.url_param('foo', escape_result=False) == "O'Brien"

def test_url_param_unescaped_default_form_data() -> None:
    with app.test_request_context(query_string={'form_data': json.dumps({'url_params': {'foo': "O'Brien"}})}):
        cache = ExtraCache(dialect=dialect())
        assert cache.url_param('bar', "O'Malley", escape_result=False) == "O'Malley"

def test_safe_proxy_primitive() -> None:
    def func(input_: Any) -> Any:
        return input_
    assert safe_proxy(func, 'foo') == 'foo'

def test_safe_proxy_dict() -> None:
    def func(input_: Any) -> Any:
        return input_
    assert safe_proxy(func, {'foo': 'bar'}) == {'foo': 'bar'}

def test_safe_proxy_lambda() -> None:
    def func(input_: Any) -> Any:
        return input_
    with pytest.raises(SupersetTemplateException):
        safe_proxy(func, lambda: 'bar')

def test_safe_proxy_nested_lambda() -> None:
    def func(input_: Any) -> Any:
        return input_
    with pytest.raises(SupersetTemplateException):
        safe_proxy(func, {'foo': lambda: 'bar'})

def test_user_macros(mocker: MockerFixture) -> None:
    mock_g = mocker.patch('superset.utils.core.g')
    mock_cache_key_wrapper = mocker.patch('superset.jinja_context.ExtraCache.cache_key_wrapper')
    mock_g.user.id = 1
    mock_g.user.username = 'my_username'
    mock_g.user.email = 'my_email@test.com'
    cache = ExtraCache()
    assert cache.current_user_id() == 1
    assert cache.current_username() == 'my_username'
    assert cache.current_user_email() == 'my_email@test.com'
    assert mock_cache_key_wrapper.call_count == 3

def test_user_macros_without_cache_key_inclusion(mocker: MockerFixture) -> None:
    mock_g = mocker.patch('superset.utils.core.g')
    mock_cache_key_wrapper = mocker.patch('superset.jinja_context.ExtraCache.cache_key_wrapper')
    mock_g.user.id = 1
    mock_g.user.username = 'my_username'
    mock_g.user.email = 'my_email@test.com'
    cache = ExtraCache()
    assert cache.current_user_id(False) == 1
    assert cache.current_username(False) == 'my_username'
    assert cache.current_user_email(False) == 'my_email@test.com'
    assert mock_cache_key_wrapper.call_count == 0

def test_user_macros_without_user_info(mocker: MockerFixture) -> None:
    mock_g = mocker.patch('superset.utils.core.g')
    mock_g.user = None
    cache = ExtraCache()
    assert cache.current_user_id() is None
    assert cache.current_username() is None
    assert cache.current_user_email() is None

def test_where_in() -> None:
    where_in = WhereInMacro(mysql.dialect())
    assert where_in([1, 'b', 3]) == "(1, 'b', 3)"
    assert where_in([1, 'b', 3], '"') == "(1, 'b', 3)\n-- WARNING: the `mark` parameter was removed from the `where_in` macro for security reasons\n"
    assert where_in(["O'Malley's"]) == "('O''Malley''s')"

def test_dataset_macro(mocker: MockerFixture) -> None:
    mocker.patch('superset.connectors.sqla.models.security_manager.get_guest_rls_filters', return_value=[])
    columns = [TableColumn(column_name='ds', is_dttm=1, type='TIMESTAMP'), TableColumn(column_name='num_boys', type='INTEGER'), TableColumn(column_name='revenue', type='INTEGER'), TableColumn(column_name='expenses', type='INTEGER'), TableColumn(column_name='profit', type='INTEGER', expression='revenue-expenses')]
    metrics = [SqlMetric(metric_name='cnt', expression='COUNT(*)')]
    dataset = SqlaTable(table_name='old_dataset', columns=columns, metrics=metrics, main_dttm_col='ds', default_endpoint='https://www.youtube.com/watch?v=dQw4w9WgXcQ', database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), offset=-8, description='This is the description', is_featured=1, cache_timeout=3600, schema='my_schema', sql=None, params=json.dumps({'remote_id': 64, 'database_name': 'examples', 'import_time': 1606677834}), perm=None, filter_select_enabled=1, fetch_values_predicate='foo IN (1, 2)', is_sqllab_view=0, template_params=json.dumps({'answer': '42'}), schema_perm=None, extra=json.dumps({'warning_markdown': '*WARNING*'}))
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = dataset
    mocker.patch('superset.connectors.sqla.models.security_manager.get_guest_rls_filters', return_value=[])
    space = ' '
    assert dataset_macro(1) == f'(\nSELECT ds AS ds, num_boys AS num_boys, revenue AS revenue, expenses AS expenses, revenue-expenses AS profit{space}\nFROM my_schema.old_dataset\n) AS dataset_1'
    assert dataset_macro(1, include_metrics=True) == f'(\nSELECT ds AS ds, num_boys AS num_boys, revenue AS revenue, expenses AS expenses, revenue-expenses AS profit, COUNT(*) AS cnt{space}\nFROM my_schema.old_dataset GROUP BY ds, num_boys, revenue, expenses, revenue-expenses\n) AS dataset_1'
    assert dataset_macro(1, include_metrics=True, columns=['ds']) == f'(\nSELECT ds AS ds, COUNT(*) AS cnt{space}\nFROM my_schema.old_dataset GROUP BY ds\n) AS dataset_1'
    DatasetDAO.find_by_id.return_value = None
    with pytest.raises(DatasetNotFoundError) as excinfo:
        dataset_macro(1)
    assert str(excinfo.value) == 'Dataset 1 not found!'

def test_dataset_macro_mutator_with_comments(mocker: MockerFixture) -> None:
    def mutator(sql: str) -> str:
        return f'-- begin\n{sql}\n-- end'
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id().get_query_str_extended().sql = mutator('SELECT 1')
    assert dataset_macro(1) == '(\n-- begin\nSELECT 1\n-- end\n) AS dataset_1'

def test_metric_macro_with_dataset_id(mocker: MockerFixture) -> None:
    mock_get_form_data = mocker.patch('superset.views.utils.get_form_data')
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = SqlaTable(table_name='test_dataset', metrics=[SqlMetric(metric_name='count', expression='COUNT(*)')], database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), schema='my_schema', sql=None)
    assert metric_macro('count', 1) == 'COUNT(*)'
    mock_get_form_data.assert_not_called()

def test_metric_macro_recursive(mocker: MockerFixture) -> None:
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {'datasource': {'id': 1}}
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = SqlaTable(table_name='test_dataset', metrics=[SqlMetric(metric_name='a', expression='COUNT(*)'), SqlMetric(metric_name='b', expression="{{ metric('a') }}"), SqlMetric(metric_name='c', expression="{{ metric('b') }}")], database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), schema='my_schema', sql=None)
    assert metric_macro('c', 1) == 'COUNT(*)'

def test_metric_macro_recursive_compound(mocker: MockerFixture) -> None:
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {'datasource': {'id': 1}}
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = SqlaTable(table_name='test_dataset', metrics=[SqlMetric(metric_name='a', expression='SUM(*)'), SqlMetric(metric_name='b', expression='COUNT(*)'), SqlMetric(metric_name='c', expression="{{ metric('a') }} / {{ metric('b') }}")], database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), schema='my_schema', sql=None)
    assert metric_macro('c', 1) == 'SUM(*) / COUNT(*)'

def test_metric_macro_recursive_cyclic(mocker: MockerFixture) -> None:
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {'datasource': {'id': 1}}
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = SqlaTable(table_name='test_dataset', metrics=[SqlMetric(metric_name='a', expression="{{ metric('c') }}"), SqlMetric(metric_name='b', expression="{{ metric('a') }}"), SqlMetric(metric_name='c', expression="{{ metric('b') }}")], database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), schema='my_schema', sql=None)
    with pytest.raises(SupersetTemplateException) as excinfo:
        metric_macro('c', 1)
    assert str(excinfo.value) == 'Cyclic metric macro detected'

def test_metric_macro_recursive_infinite(mocker: MockerFixture) -> None:
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {'datasource': {'id': 1}}
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = SqlaTable(table_name='test_dataset', metrics=[SqlMetric(metric_name='a', expression="{{ metric('a') }}")], database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), schema='my_schema', sql=None)
    with pytest.raises(SupersetTemplateException) as excinfo:
        metric_macro('a', 1)
    assert str(excinfo.value) == 'Cyclic metric macro detected'

def test_metric_macro_with_dataset_id_invalid_key(mocker: MockerFixture) -> None:
    mock_get_form_data = mocker.patch('superset.views.utils.get_form_data')
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = SqlaTable(table_name='test_dataset', metrics=[SqlMetric(metric_name='count', expression='COUNT(*)')], database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), schema='my_schema', sql=None)
    with pytest.raises(SupersetTemplateException) as excinfo:
        metric_macro('blah', 1)
    assert str(excinfo.value) == 'Metric ``blah`` not found in test_dataset.'
    mock_get_form_data.assert_not_called()

def test_metric_macro_invalid_dataset_id(mocker: MockerFixture) -> None:
    mock_get_form_data = mocker.patch('superset.views.utils.get_form_data')
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = None
    with pytest.raises(DatasetNotFoundError) as excinfo:
        metric_macro('macro_key', 100)
    assert str(excinfo.value) == 'Dataset ID 100 not found.'
    mock_get_form_data.assert_not_called()

def test_metric_macro_no_dataset_id_no_context(mocker: MockerFixture) -> None:
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {}
    with app.test_request_context():
        with pytest.raises(SupersetTemplateException) as excinfo:
            metric_macro('macro_key')
        assert str(excinfo.value) == 'Please specify the Dataset ID for the ``macro_key`` metric in the Jinja macro.'
        DatasetDAO.find_by_id.assert_not_called()

def test_metric_macro_no_dataset_id_with_context_missing_info(mocker: MockerFixture) -> None:
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {'queries': []}
    with app.test_request_context(data={'form_data': json.dumps({'adhoc_filters': [{'clause': 'WHERE', 'comparator': 'foo', 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'name'}]})}):
        with pytest.raises(SupersetTemplateException) as excinfo:
            metric_macro('macro_key')
        assert str(excinfo.value) == 'Please specify the Dataset ID for the ``macro_key`` metric in the Jinja macro.'
        DatasetDAO.find_by_id.assert_not_called()

def test_metric_macro_no_dataset_id_with_context_datasource_id(mocker: MockerFixture) -> None:
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = SqlaTable(table_name='test_dataset', metrics=[SqlMetric(metric_name='macro_key', expression='COUNT(*)')], database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), schema='my_schema', sql=None)
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {}
    with app.test_request_context(data={'form_data': json.dumps({'queries': [{'url_params': {'datasource_id': 1}}]})}):
        assert metric_macro('macro_key') == 'COUNT(*)'
    mock_g.form_data = {'queries': [{'url_params': {'datasource_id': 1}}]}
    with app.test_request_context():
        assert metric_macro('macro_key') == 'COUNT(*)'

def test_metric_macro_no_dataset_id_with_context_datasource_id_none(mocker: MockerFixture) -> None:
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {}
    with app.test_request_context(data={'form_data': json.dumps({'queries': [{'url_params': {'datasource_id': None}}]})}):
        with pytest.raises(SupersetTemplateException) as excinfo:
            metric_macro('macro_key')
        assert str(excinfo.value) == 'Please specify the Dataset ID for the ``macro_key`` metric in the Jinja macro.'
    mock_g.form_data = {'queries': [{'url_params': {'datasource_id': None}}]}
    with app.test_request_context():
        with pytest.raises(SupersetTemplateException) as excinfo:
            metric_macro('macro_key')
        assert str(excinfo.value) == 'Please specify the Dataset ID for the ``macro_key`` metric in the Jinja macro.'

def test_metric_macro_no_dataset_id_with_context_chart_id(mocker: MockerFixture) -> None:
    ChartDAO = mocker.patch('superset.daos.chart.ChartDAO')
    ChartDAO.find_by_id.return_value = Slice(datasource_id=1)
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = SqlaTable(table_name='test_dataset', metrics=[SqlMetric(metric_name='macro_key', expression='COUNT(*)')], database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), schema='my_schema', sql=None)
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {}
    with app.test_request_context(data={'form_data': json.dumps({'queries': [{'url_params': {'slice_id': 1}}]})}):
        assert metric_macro('macro_key') == 'COUNT(*)'
    mock_g.form_data = {'queries': [{'url_params': {'slice_id': 1}}]}
    with app.test_request_context():
        assert metric_macro('macro_key') == 'COUNT(*)'

def test_metric_macro_no_dataset_id_with_context_slice_id_none(mocker: MockerFixture) -> None:
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {}
    with app.test_request_context(data={'form_data': json.dumps({'queries': [{'url_params': {'slice_id': None}}]})}):
        with pytest.raises(SupersetTemplateException) as excinfo:
            metric_macro('macro_key')
        assert str(excinfo.value) == 'Please specify the Dataset ID for the ``macro_key`` metric in the Jinja macro.'
    mock_g.form_data = {'queries': [{'url_params': {'slice_id': None}}]}
    with app.test_request_context():
        with pytest.raises(SupersetTemplateException) as excinfo:
            metric_macro('macro_key')
        assert str(excinfo.value) == 'Please specify the Dataset ID for the ``macro_key`` metric in the Jinja macro.'

def test_metric_macro_no_dataset_id_with_context_deleted_chart(mocker: MockerFixture) -> None:
    ChartDAO = mocker.patch('superset.daos.chart.ChartDAO')
    ChartDAO.find_by_id.return_value = None
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {}
    with app.test_request_context(data={'form_data': json.dumps({'queries': [{'url_params': {'slice_id': 1}}]})}):
        with pytest.raises(SupersetTemplateException) as excinfo:
            metric_macro('macro_key')
        assert str(excinfo.value) == 'Please specify the Dataset ID for the ``macro_key`` metric in the Jinja macro.'
    mock_g.form_data = {'queries': [{'url_params': {'slice_id': 1}}]}
    with app.test_request_context():
        with pytest.raises(SupersetTemplateException) as excinfo:
            metric_macro('macro_key')
        assert str(excinfo.value) == 'Please specify the Dataset ID for the ``macro_key`` metric in the Jinja macro.'

def test_metric_macro_no_dataset_id_available_in_request_form_data(mocker: MockerFixture) -> None:
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = SqlaTable(table_name='test_dataset', metrics=[SqlMetric(metric_name='macro_key', expression='COUNT(*)')], database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), schema='my_schema', sql=None)
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {}
    with app.test_request_context(data={'form_data': json.dumps({'datasource': {'id': 1}})}):
        assert metric_macro('macro_key') == 'COUNT(*)'
    mock_g.form_data = {'datasource': '1__table'}
    with app.test_request_context():
        assert metric_macro('macro_key') == 'COUNT(*)'

@pytest.mark.parametrize('description,args,kwargs,sqlalchemy_uri,queries,time_filter,removed_filters,applied_filters', [('Missing time_range and filter will return a No filter result', [], {'target_type': 'TIMESTAMP'}, 'postgresql://mydb', [{}], TimeFilter(from_expr=None, to_expr=None, time_range='No filter'), [], []), ('Missing time range and filter with default value will return a result with the defaults', [], {'default': 'Last week', 'target_type': 'TIMESTAMP'}, 'postgresql://mydb', [{}], TimeFilter(from_expr="TO_TIMESTAMP('2024-08-27 00:00:00.000000', 'YYYY-MM-DD HH24:MI:SS.US')", to_expr="TO_TIMESTAMP('2024-09-03 00:00:00.000000', 'YYYY-MM-DD HH24:MI:SS.US')", time_range='Last week'), [], []), ('Time range is extracted with the expected format, and default is ignored', [], {'default': 'Last month', 'target_type': 'TIMESTAMP'}, 'postgresql://mydb', [{'time_range': 'Last week'}], TimeFilter(from_expr="TO_TIMESTAMP('2024-08-27 00:00:00.000000', 'YYYY-MM-DD HH24:MI:SS.US')", to_expr="TO_TIMESTAMP('2024-09-03 00:00:00.000000', 'YYYY-MM-DD HH24:MI:SS.US')", time_range='Last week'), [], []), ('Filter is extracted with the native format of the column (TIMESTAMP)', ['dttm'], {}, 'postgresql://mydb', [{'filters': [{'col': 'dttm', 'op': 'TEMPORAL_RANGE', 'val': 'Last week'}]}], TimeFilter(from_expr="TO_TIMESTAMP('2024-08-27 00:00:00.000000', 'YYYY-MM-DD HH24:MI:SS.US')", to_expr="TO_TIMESTAMP('2024-09-03 00:00:00.000000', 'YYYY-MM-DD HH24:MI:SS.US')", time_range='Last week'), [], ['dttm']), ('Filter is extracted with the native format of the column (DATE)', ['dt'], {'remove_filter': True}, 'postgresql://mydb', [{'filters': [{'col': 'dt', 'op': 'TEMPORAL_RANGE', 'val': 'Last week'}]}], TimeFilter(from_expr="TO_DATE('2024-08-27', 'YYYY-MM-DD')", to_expr="TO_DATE('2024-09-03', 'YYYY-MM-DD')", time_range='Last week'), ['dt'], ['dt']), ('Filter is extracted with the overridden format (TIMESTAMP to DATE)', ['dttm'], {'target_type': 'DATE', 'remove_filter': True}, 'trino://mydb', [{'filters': [{'col': 'dttm', 'op': 'TEMPORAL_RANGE', 'val': 'Last month'}]}], TimeFilter(from_expr="DATE '2024-08-03'", to_expr="DATE '2024-09-03'", time_range='Last month'), ['dttm'], ['dttm']), ('Filter is formatted with the custom format, ignoring target_type', ['dttm'], {'target_type': 'DATE', 'strftime': '%Y%m%d', 'remove_filter': True}, 'trino://mydb', [{'filters': [{'col': 'dttm', 'op': 'TEMPORAL_RANGE', 'val': 'Last month'}]}], TimeFilter(from_expr='20240803', to_expr='20240903', time_range='Last month'), ['dttm'], ['dttm'])])
def test_get_time_filter(description: str, args: List[Any], kwargs: Dict[str, Any], sqlalchemy_uri: str, queries: List[Dict[str, Any]], time_filter: TimeFilter, removed_filters: List[str], applied_filters: List[str]) -> None:
    columns = [TableColumn(column_name='dt', is_dttm=1, type='DATE'), TableColumn(column_name='dttm', is_dttm=1, type='TIMESTAMP')]
    database = Database(database_name='my_database', sqlalchemy_uri=sqlalchemy_uri)
    table = SqlaTable(table_name='my_dataset', columns=columns, main_dttm_col='dt', database=database)
    with freeze_time('2024-09-03'), app.test_request_context(json={'queries': queries}):
        cache = ExtraCache(database=database, table=table)
        assert cache.get_time_filter(*args, **kwargs) == time_filter, description
        assert cache.removed_filters == removed_filters
        assert cache.applied_filters == applied_filters
