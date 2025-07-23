from __future__ import annotations
from typing import Any, Optional, List, Dict, Union, Callable, Tuple, cast
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
    """
    Test the ``filter_values`` macro with ``adhoc_filters``.
    """
    with app.test_request_context(data={'form_data': json.dumps({'adhoc_filters': [{'clause': 'WHERE', 'comparator': 'foo', 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'name'}]})}):
        cache = ExtraCache()
        assert cache.filter_values('name') == ['foo']
        assert cache.applied_filters == ['name']
    with app.test_request_context(data={'form_data': json.dumps({'adhoc_filters': [{'clause': 'WHERE', 'comparator': ['foo', 'bar'], 'expressionType': 'SIMPLE', 'operator': 'in', 'subject': 'name'}]})}):
        cache = ExtraCache()
        assert cache.filter_values('name') == ['foo', 'bar']
        assert cache.applied_filters == ['name']

def test_filter_values_extra_filters() -> None:
    """
    Test the ``filter_values`` macro with ``extra_filters``.
    """
    with app.test_request_context(data={'form_data': json.dumps({'extra_filters': [{'col': 'name', 'op': 'in', 'val': 'foo'}]})}):
        cache = ExtraCache()
        assert cache.filter_values('name') == ['foo']
        assert cache.applied_filters == ['name']

def test_filter_values_default() -> None:
    """
    Test the ``filter_values`` macro with a default value.
    """
    cache = ExtraCache()
    assert cache.filter_values('name', 'foo') == ['foo']
    assert cache.removed_filters == []

def test_filter_values_remove_not_present() -> None:
    """
    Test the ``filter_values`` macro without a match and ``remove_filter`` set to True.
    """
    cache = ExtraCache()
    assert cache.filter_values('name', remove_filter=True) == []
    assert cache.removed_filters == []

def test_filter_values_no_default() -> None:
    """
    Test calling the ``filter_values`` macro without a match.
    """
    cache = ExtraCache()
    assert cache.filter_values('name') == []

def test_get_filters_adhoc_filters() -> None:
    """
    Test the ``get_filters`` macro.
    """
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
    """
    Test the ``get_filters`` macro without a match and ``remove_filter`` set to True.
    """
    cache = ExtraCache()
    assert cache.get_filters('name', remove_filter=True) == []
    assert cache.removed_filters == []

def test_url_param_query() -> None:
    """
    Test the ``url_param`` macro.
    """
    with app.test_request_context(query_string={'foo': 'bar'}):
        cache = ExtraCache()
        assert cache.url_param('foo') == 'bar'

def test_url_param_default() -> None:
    """
    Test the ``url_param`` macro with a default value.
    """
    with app.test_request_context():
        cache = ExtraCache()
        assert cache.url_param('foo', 'bar') == 'bar'

def test_url_param_no_default() -> None:
    """
    Test the ``url_param`` macro without a match.
    """
    with app.test_request_context():
        cache = ExtraCache()
        assert cache.url_param('foo') is None

def test_url_param_form_data() -> None:
    """
    Test the ``url_param`` with ``url_params`` in ``form_data``.
    """
    with app.test_request_context(query_string={'form_data': json.dumps({'url_params': {'foo': 'bar'}})}):
        cache = ExtraCache()
        assert cache.url_param('foo') == 'bar'

def test_url_param_escaped_form_data() -> None:
    """
    Test the ``url_param`` with ``url_params`` in ``form_data`` returning
    an escaped value with a quote.
    """
    with app.test_request_context(query_string={'form_data': json.dumps({'url_params': {'foo': "O'Brien"}})}):
        cache = ExtraCache(dialect=dialect())
        assert cache.url_param('foo') == "O''Brien"

def test_url_param_escaped_default_form_data() -> None:
    """
    Test the ``url_param`` with default value containing an escaped quote.
    """
    with app.test_request_context(query_string={'form_data': json.dumps({'url_params': {'foo': "O'Brien"}})}):
        cache = ExtraCache(dialect=dialect())
        assert cache.url_param('bar', "O'Malley") == "O''Malley"

def test_url_param_unescaped_form_data() -> None:
    """
    Test the ``url_param`` with ``url_params`` in ``form_data`` returning
    an un-escaped value with a quote.
    """
    with app.test_request_context(query_string={'form_data': json.dumps({'url_params': {'foo': "O'Brien"}})}):
        cache = ExtraCache(dialect=dialect())
        assert cache.url_param('foo', escape_result=False) == "O'Brien"

def test_url_param_unescaped_default_form_data() -> None:
    """
    Test the ``url_param`` with default value containing an un-escaped quote.
    """
    with app.test_request_context(query_string={'form_data': json.dumps({'url_params': {'foo': "O'Brien"}})}):
        cache = ExtraCache(dialect=dialect())
        assert cache.url_param('bar', "O'Malley", escape_result=False) == "O'Malley"

def test_safe_proxy_primitive() -> None:
    """
    Test the ``safe_proxy`` helper with a function returning a ``str``.
    """

    def func(input_: str) -> str:
        return input_
    assert safe_proxy(func, 'foo') == 'foo'

def test_safe_proxy_dict() -> None:
    """
    Test the ``safe_proxy`` helper with a function returning a ``dict``.
    """

    def func(input_: Dict[str, str]) -> Dict[str, str]:
        return input_
    assert safe_proxy(func, {'foo': 'bar'}) == {'foo': 'bar'}

def test_safe_proxy_lambda() -> None:
    """
    Test the ``safe_proxy`` helper with a function returning a ``lambda``.
    Should raise ``SupersetTemplateException``.
    """

    def func(input_: Callable[[], str]) -> Callable[[], str]:
        return input_
    with pytest.raises(SupersetTemplateException):
        safe_proxy(func, lambda: 'bar')

def test_safe_proxy_nested_lambda() -> None:
    """
    Test the ``safe_proxy`` helper with a function returning a ``dict``
    containing ``lambda`` value. Should raise ``SupersetTemplateException``.
    """

    def func(input_: Dict[str, Callable[[], str]]) -> Dict[str, Callable[[], str]]:
        return input_
    with pytest.raises(SupersetTemplateException):
        safe_proxy(func, {'foo': lambda: 'bar'})

def test_user_macros(mocker: MockerFixture) -> None:
    """
    Test all user macros:
        - ``current_user_id``
        - ``current_username``
        - ``current_user_email``
    """
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
    """
    Test all user macros with ``add_to_cache_keys`` set to ``False``.
    """
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
    """
    Test all user macros when no user info is available.
    """
    mock_g = mocker.patch('superset.utils.core.g')
    mock_g.user = None
    cache = ExtraCache()
    assert cache.current_user_id() == None
    assert cache.current_username() == None
    assert cache.current_user_email() == None

def test_where_in() -> None:
    """
    Test the ``where_in`` Jinja2 filter.
    """
    where_in = WhereInMacro(mysql.dialect())
    assert where_in([1, 'b', 3]) == "(1, 'b', 3)"
    assert where_in([1, 'b', 3], '"') == "(1, 'b', 3)\n-- WARNING: the `mark` parameter was removed from the `where_in` macro for security reasons\n"
    assert where_in(["O'Malley's"]) == "('O''Malley''s')"

def test_dataset_macro(mocker: MockerFixture) -> None:
    """
    Test the ``dataset_macro`` macro.
    """
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
    """
    Test ``dataset_macro`` when the mutator adds comment.
    """

    def mutator(sql: str) -> str:
        """
        A simple mutator that wraps the query in comments.
        """
        return f'-- begin\n{sql}\n-- end'
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id().get_query_str_extended().sql = mutator('SELECT 1')
    assert dataset_macro(1) == '(\n-- begin\nSELECT 1\n-- end\n) AS dataset_1'

def test_metric_macro_with_dataset_id(mocker: MockerFixture) -> None:
    """
    Test the ``metric_macro`` when passing a dataset ID.
    """
    mock_get_form_data = mocker.patch('superset.views.utils.get_form_data')
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = SqlaTable(table_name='test_dataset', metrics=[SqlMetric(metric_name='count', expression='COUNT(*)')], database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), schema='my_schema', sql=None)
    assert metric_macro('count', 1) == 'COUNT(*)'
    mock_get_form_data.assert_not_called()

def test_metric_macro_recursive(mocker: MockerFixture) -> None:
    """
    Test the ``metric_macro`` when the definition is recursive.
    """
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {'datasource': {'id': 1}}
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = SqlaTable(table_name='test_dataset', metrics=[SqlMetric(metric_name='a', expression='COUNT(*)'), SqlMetric(metric_name='b', expression="{{ metric('a') }}"), SqlMetric(metric_name='c', expression="{{ metric('b') }}")], database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), schema='my_schema', sql=None)
    assert metric_macro('c', 1) == 'COUNT(*)'

def test_metric_macro_recursive_compound(mocker: MockerFixture) -> None:
    """
    Test the ``metric_macro`` when the definition is compound.
    """
    mock_g = mocker.patch('superset.jinja_context.g')
    mock_g.form_data = {'datasource': {'id': 1}}
    DatasetDAO = mocker.patch('superset.daos.dataset.DatasetDAO')
    DatasetDAO.find_by_id.return_value = SqlaTable(table_name='test_dataset', metrics=[SqlMetric(metric_name='a', expression='SUM(*)'), SqlMetric(metric_name='b', expression='COUNT(*)'), SqlMetric(metric_name='c', expression="{{ metric('a') }} / {{ metric('b') }}")], database=Database(database_name='my_database', sqlalchemy_uri='sqlite://'), schema='my_schema', sql=None)
    assert metric_macro('c', 1) == 'SUM(*) /