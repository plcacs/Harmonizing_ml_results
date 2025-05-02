import contextlib
import logging
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Set, Tuple, Union, cast
import msgpack
import pyarrow as pa
from flask import flash, g, has_request_context, redirect, request
from flask_appbuilder.security.sqla import models as ab_models
from flask_appbuilder.security.sqla.models import User
from flask_babel import _
from sqlalchemy.exc import NoResultFound
from werkzeug.wrappers.response import Response
from superset import app, dataframe, db, result_set, viz
from superset.common.db_query_status import QueryStatus
from superset.daos.datasource import DatasourceDAO
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.exceptions import CacheLoadError, SerializationError, SupersetException, SupersetSecurityException
from superset.extensions import cache_manager, feature_flag_manager, security_manager
from superset.legacy import update_time_range
from superset.models.core import Database
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.models.sql_lab import Query
from superset.superset_typing import FormData
from superset.utils import json
from superset.utils.core import DatasourceType
from superset.utils.decorators import stats_timing
from superset.viz import BaseViz

logger: logging.Logger = logging.getLogger(__name__)
stats_logger: Any = app.config['STATS_LOGGER']
REJECTED_FORM_DATA_KEYS: List[str] = []
if not feature_flag_manager.is_feature_enabled('ENABLE_JAVASCRIPT_CONTROLS'):
    REJECTED_FORM_DATA_KEYS = ['js_tooltip', 'js_onclick_href', 'js_data_mutator']

def sanitize_datasource_data(datasource_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if datasource_data:
        datasource_database = datasource_data.get('database')
        if datasource_database:
            datasource_database['parameters'] = {}
    return datasource_data

def bootstrap_user_data(user: User, include_perms: bool = False) -> Dict[str, Any]:
    if user.is_anonymous:
        payload: Dict[str, Any] = {}
        user.roles = (security_manager.find_role('Public'),)
    elif security_manager.is_guest_user(user):
        payload = {'username': user.username, 'firstName': user.first_name, 'lastName': user.last_name, 'isActive': user.is_active, 'isAnonymous': user.is_anonymous}
    else:
        payload = {'username': user.username, 'firstName': user.first_name, 'lastName': user.last_name, 'userId': user.id, 'isActive': user.is_active, 'isAnonymous': user.is_anonymous, 'createdOn': user.created_on.isoformat(), 'email': user.email}
    if include_perms:
        roles, permissions = get_permissions(user)
        payload['roles'] = roles
        payload['permissions'] = permissions
    return payload

def get_permissions(user: User) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    if not user.roles:
        raise AttributeError('User object does not have roles')
    data_permissions: DefaultDict[str, Set[str]] = defaultdict(set)
    roles_permissions: Dict[str, Any] = security_manager.get_user_roles_permissions(user)
    for _, permissions in roles_permissions.items():
        for permission in permissions:
            if permission[0] in ('datasource_access', 'database_access'):
                data_permissions[permission[0]].add(permission[1])
    transformed_permissions: DefaultDict[str, List[str]] = defaultdict(list)
    for perm in data_permissions:
        transformed_permissions[perm] = list(data_permissions[perm])
    return (roles_permissions, transformed_permissions)

def get_viz(form_data: FormData, datasource_type: DatasourceType, datasource_id: int, force: bool = False, force_cached: bool = False) -> BaseViz:
    viz_type: str = form_data.get('viz_type', 'table')
    datasource = DatasourceDAO.get_datasource(DatasourceType(datasource_type), datasource_id)
    viz_obj: BaseViz = viz.viz_types[viz_type](datasource, form_data=form_data, force=force, force_cached=force_cached)
    return viz_obj

def loads_request_json(request_json_data: str) -> Dict[str, Any]:
    try:
        return json.loads(request_json_data)
    except (TypeError, json.JSONDecodeError):
        return {}

def get_form_data(slice_id: Optional[int] = None, use_slice_data: bool = False, initial_form_data: Optional[FormData] = None) -> Tuple[FormData, Optional[Slice]]:
    form_data: FormData = initial_form_data or {}
    if has_request_context():
        json_data: Dict[str, Any] = request.get_json(cache=True) if request.is_json else {}
        first_query: Optional[Dict[str, Any]] = json_data['queries'][0] if 'queries' in json_data and json_data['queries'] else None
        add_sqllab_custom_filters(form_data)
        request_form_data: Optional[str] = request.form.get('form_data')
        request_args_data: Optional[str] = request.args.get('form_data')
        if first_query:
            form_data.update(first_query)
        if request_form_data:
            parsed_form_data: Dict[str, Any] = loads_request_json(request_form_data)
            queries: Optional[List[Dict[str, Any]]] = parsed_form_data.get('queries')
            if isinstance(queries, list):
                form_data.update(queries[0])
            else:
                form_data.update(parsed_form_data)
        if request_args_data:
            form_data.update(loads_request_json(request_args_data))
    if not form_data and hasattr(g, 'form_data'):
        form_data = g.form_data
        json_data = form_data['queries'][0] if 'queries' in form_data else {}
        form_data.update(json_data)
    form_data = {k: v for k, v in form_data.items() if k not in REJECTED_FORM_DATA_KEYS}
    slice_id = form_data.get('slice_id') or slice_id
    slc: Optional[Slice] = None
    valid_keys: List[str] = ['slice_id', 'extra_filters', 'adhoc_filters', 'viz_type']
    valid_slice_id: bool = all((key in valid_keys for key in form_data))
    if slice_id and (use_slice_data or valid_slice_id):
        slc = db.session.query(Slice).filter_by(id=slice_id).one_or_none()
        if slc:
            slice_form_data: FormData = slc.form_data.copy()
            slice_form_data.update(form_data)
            form_data = slice_form_data
    update_time_range(form_data)
    return (form_data, slc)

def add_sqllab_custom_filters(form_data: FormData) -> None:
    """
    SQLLab can include a "filters" attribute in the templateParams.
    The filters attribute is a list of filters to include in the
    request. Useful for testing templates in SQLLab.
    """
    try:
        data: Dict[str, Any] = json.loads(request.data)
        if isinstance(data, dict):
            params_str: Optional[str] = data.get('templateParams')
            if isinstance(params_str, str):
                params: Dict[str, Any] = json.loads(params_str)
                if isinstance(params, dict):
                    filters: Optional[List[Dict[str, Any]]] = params.get('_filters')
                    if filters:
                        form_data.update({'filters': filters})
    except (TypeError, json.JSONDecodeError):
        data = {}

def get_datasource_info(datasource_id: Optional[int], datasource_type: Optional[str], form_data: FormData) -> Tuple[int, str]:
    """
    Compatibility layer for handling of datasource info

    datasource_id & datasource_type used to be passed in the URL
    directory, now they should come as part of the form_data,

    This function allows supporting both without duplicating code

    :param datasource_id: The datasource ID
    :param datasource_type: The datasource type
    :param form_data: The URL form data
    :returns: The datasource ID and type
    :raises SupersetException: If the datasource no longer exists
    """
    if '__' in (datasource := form_data.get('datasource', '')):
        datasource_id, datasource_type = datasource.split('__')
        if datasource_id == 'None':
            datasource_id = None
    if not datasource_id:
        raise SupersetException(_('The dataset associated with this chart no longer exists'))
    datasource_id = int(datasource_id)
    return (datasource_id, datasource_type)

def apply_display_max_row_limit(sql_results: Dict[str, Any], rows: Optional[int] = None) -> Dict[str, Any]:
    """
    Given a `sql_results` nested structure, applies a limit to the number of rows

    `sql_results` here is the nested structure coming out of sql_lab.get_sql_results, it
    contains metadata about the query, as well as the data set returned by the query.
    This method limits the number of rows adds a `displayLimitReached: True` flag to the
    metadata.

    :param sql_results: The results of a sql query from sql_lab.get_sql_results
    :param rows: The number of rows to apply a limit to
    :returns: The mutated sql_results structure
    """
    display_limit: int = rows or app.config['DISPLAY_MAX_ROW']
    if display_limit and sql_results['status'] == QueryStatus.SUCCESS and (display_limit < sql_results['query']['rows']):
        sql_results['data'] = sql_results['data'][:display_limit]
        sql_results['displayLimitReached'] = True
    return sql_results

CONTAINER_TYPES: List[str] = ['COLUMN', 'GRID', 'TABS', 'TAB', 'ROW']

def get_dashboard_extra_filters(slice_id: int, dashboard_id: int) -> List[Dict[str, Any]]:
    dashboard: Optional[Dashboard] = db.session.query(Dashboard).filter_by(id=dashboard_id).one_or_none()
    if dashboard is None or not dashboard.json_metadata or (not dashboard.slices) or (not any((slc for slc in dashboard.slices if slc.id == slice_id))):
        return []
    with contextlib.suppress(json.JSONDecodeError):
        json_metadata: Dict[str, Any] = json.loads(dashboard.json_metadata)
        default_filters: Dict[str, Any] = json.loads(json_metadata.get('default_filters', 'null'))
        if not default_filters:
            return []
        filter_scopes: Dict[str, Any] = json_metadata.get('filter_scopes', {})
        layout: Dict[str, Any] = json.loads(dashboard.position_json or '{}')
        if isinstance(layout, dict) and isinstance(filter_scopes, dict) and isinstance(default_filters, dict):
            return build_extra_filters(layout, filter_scopes, default_filters, slice_id)
    return []

def build_extra_filters(layout: Dict[str, Any], filter_scopes: Dict[str, Any], default_filters: Dict[str, Any], slice_id: int) -> List[Dict[str, Any]]:
    extra_filters: List[Dict[str, Any]] = []
    for filter_id, columns in default_filters.items():
        filter_slice: Optional[Slice] = db.session.query(Slice).filter_by(id=filter_id).one_or_none()
        filter_configs: List[Dict[str, Any]] = []
        if filter_slice:
            filter_configs = json.loads(filter_slice.params or '{}').get('filter_configs') or []
        scopes_by_filter_field: Dict[str, Any] = filter_scopes.get(filter_id, {})
        for col, val in columns.items():
            if not val:
                continue
            current_field_scopes: Dict[str, Any] = scopes_by_filter_field.get(col, {})
            scoped_container_ids: List[str] = current_field_scopes.get('scope', ['ROOT_ID'])
            immune_slice_ids: List[int] = current_field_scopes.get('immune', [])
            for container_id in scoped_container_ids:
                if slice_id not in immune_slice_ids and is_slice_in_container(layout, container_id, slice_id):
                    for filter_config in filter_configs:
                        if filter_config['column'] == col:
                            is_multiple: bool = filter_config['multiple']
                            if not is_multiple and isinstance(val, list):
                                val = val[0]
                            elif is_multiple and (not isinstance(val, list)):
                                val = [val]
                            break
                    extra_filters.append({'col': col, 'op': 'in' if isinstance(val, list) else '==', 'val': val})
    return extra_filters

def is_slice_in_container(layout: Dict[str, Any], container_id: str, slice_id: int) -> bool:
    if container_id == 'ROOT_ID':
        return True
    node: Dict[str, Any] = layout[container_id]
    node_type: str = node.get('type')
    if node_type == 'CHART' and node.get('meta', {}).get('chartId') == slice_id:
        return True
    if node_type in CONTAINER_TYPES:
        children: List[str] = node.get('children', [])
        return any((is_slice_in_container(layout, child_id, slice_id) for child_id in children))
    return False

def check_resource_permissions(check_perms: Callable[..., None]) -> Callable[..., Any]:
    """
    A decorator for checking permissions on a request using the passed-in function.
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:

        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            check_perms(*args, **kwargs)
            return f(*args, **kwargs)
        return wrapper
    return decorator

def check_explore_cache_perms(_self: Any, cache_key: str) -> None:
    """
    Loads async explore_json request data from cache and performs access check

    :param _self: the Superset view instance
    :param cache_key: the cache key passed into /explore_json/data/
    :raises SupersetSecurityException: If the user cannot access the resource
    """
    cached: Optional[Dict[str, Any]] = cache_manager.cache.get(cache_key)
    if not cached:
        raise CacheLoadError('Cached data not found')
    check_datasource_perms(_self, form_data=cached['form_data'])

def check_datasource_perms(_self: Any, datasource_type: Optional[str] = None, datasource_id: Optional[int] = None, **kwargs: Any) -> None:
    """
    Check if user can access a cached response from explore_json.

    This function takes `self` since it must have the same signature as the
    the decorated method.

    :param datasource_type: The datasource type
    :param datasource_id: The datasource ID
    :raises SupersetSecurityException: If the user cannot access the resource
    """
    form_data: FormData = kwargs['form_data'] if 'form_data' in kwargs else get_form_data()[0]
    try:
        datasource_id, datasource_type = get_datasource_info(datasource_id, datasource_type, form_data)
    except SupersetException as ex:
        raise SupersetSecurityException(SupersetError(error_type=SupersetErrorType.FAILED_FETCHING_DATASOURCE_INFO_ERROR, level=ErrorLevel.ERROR, message=str(ex))) from ex
    if datasource_type is None:
        raise SupersetSecurityException(SupersetError(error_type=SupersetErrorType.UNKNOWN_DATASOURCE_TYPE_ERROR, level=ErrorLevel.ERROR, message=_('Could not determine datasource type')))
    try:
        viz_obj: BaseViz = get_viz(datasource_type=datasource_type, datasource_id=datasource_id, form_data=form_data, force=False)
    except NoResultFound as ex:
        raise SupersetSecurityException(SupersetError(error_type=SupersetErrorType.UNKNOWN_DATASOURCE_TYPE_ERROR, level=ErrorLevel.ERROR, message=_('Could not find viz object'))) from ex
    viz_obj.raise_for_access()

def _deserialize_results_payload(payload: bytes, query: Query, use_msgpack: bool = False) -> Dict[str, Any]:
    logger.debug('Deserializing from msgpack: %r', use_msgpack)
    if use_msgpack:
        with stats_timing('sqllab.query.results_backend_msgpack_deserialize', stats_logger):
            ds_payload: Dict[str, Any] = msgpack.loads(payload, raw=False)
        with stats_timing('sqllab.query.results_backend_pa_deserialize', stats_logger):
            try:
                reader: pa.BufferReader = pa.BufferReader(ds_payload['data'])
                pa_table: pa.Table = pa.ipc.open_stream(reader).read_all()
            except pa.ArrowSerializationError as ex:
                raise SerializationError('Unable to deserialize table') from ex
        df: Any = result_set.SupersetResultSet.convert_table_to_df(pa_table)
        ds_payload['data'] = dataframe.df_to_records(df) or []
        for column in ds_payload['selected_columns']:
            if 'name' in column:
                column['column_name'] = column.get('name')
        db_engine_spec: Any = query.database.db_engine_spec
        all_columns: List[Dict[str, Any]], data: List[Dict[str, Any]], expanded_columns: List[Dict[str, Any]] = db_engine_spec.expand_data(ds_payload['selected_columns'], ds_payload['data'])
        ds_payload.update({'data': data, 'columns': all_columns, 'expanded_columns': expanded_columns})
        return ds_payload
    with stats_timing('sqllab.query.results_backend_json_deserialize', stats_log