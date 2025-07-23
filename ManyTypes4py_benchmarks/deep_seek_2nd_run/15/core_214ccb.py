from __future__ import annotations
import contextlib
import logging
from datetime import datetime
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from urllib import parse
from flask import abort, flash, g, redirect, request, Response
from flask_appbuilder import expose
from flask_appbuilder.security.decorators import has_access, has_access_api, permission_name
from flask_babel import gettext as __, lazy_gettext as _
from sqlalchemy.exc import SQLAlchemyError
from superset import app, appbuilder, conf, db, event_logger, is_feature_enabled, security_manager
from superset.async_events.async_query_manager import AsyncQueryTokenException
from superset.commands.chart.exceptions import ChartNotFoundError
from superset.commands.chart.warm_up_cache import ChartWarmUpCacheCommand
from superset.commands.dashboard.exceptions import DashboardAccessDeniedError
from superset.commands.dashboard.permalink.get import GetDashboardPermalinkCommand
from superset.commands.dataset.exceptions import DatasetNotFoundError
from superset.commands.explore.form_data.create import CreateFormDataCommand
from superset.commands.explore.form_data.get import GetFormDataCommand
from superset.commands.explore.form_data.parameters import CommandParameters
from superset.commands.explore.permalink.get import GetExplorePermalinkCommand
from superset.common.chart_data import ChartDataResultFormat, ChartDataResultType
from superset.connectors.sqla.models import BaseDatasource, SqlaTable
from superset.daos.chart import ChartDAO
from superset.daos.datasource import DatasourceDAO
from superset.dashboards.permalink.exceptions import DashboardPermalinkGetFailedError
from superset.exceptions import CacheLoadError, SupersetException, SupersetSecurityException
from superset.explore.permalink.exceptions import ExplorePermalinkGetFailedError
from superset.extensions import async_query_manager, cache_manager
from superset.models.core import Database
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from superset.models.sql_lab import Query
from superset.models.user_attributes import UserAttribute
from superset.superset_typing import FlaskResponse
from superset.utils import core as utils, json
from superset.utils.cache import etag_cache
from superset.utils.core import DatasourceType, get_user_id, ReservedUrlParameters
from superset.views.base import api, BaseSupersetView, common_bootstrap_payload, CsvResponse, data_payload_response, deprecated, generate_download_headers, json_error_response, json_success
from superset.views.error_handling import handle_api_exception
from superset.views.utils import bootstrap_user_data, check_datasource_perms, check_explore_cache_perms, check_resource_permissions, get_datasource_info, get_form_data, get_viz, loads_request_json, redirect_with_flash, sanitize_datasource_data
from superset.viz import BaseViz

config = app.config
SQLLAB_QUERY_COST_ESTIMATE_TIMEOUT: int = config['SQLLAB_QUERY_COST_ESTIMATE_TIMEOUT']
stats_logger: Any = config['STATS_LOGGER']
logger: logging.Logger = logging.getLogger(__name__)
DATASOURCE_MISSING_ERR: str = __('The data source seems to have been deleted')
USER_MISSING_ERR: str = __('The user seems to have been deleted')
PARAMETER_MISSING_ERR: str = __('Please check your template parameters for syntax errors and make sure they match across your SQL query and Set Parameters. Then, try running your query again.')
SqlResults = Dict[str, Any]

class Superset(BaseSupersetView):
    """The base views for Superset!"""
    logger: logging.Logger = logging.getLogger(__name__)

    @has_access
    @event_logger.log_this
    @expose('/slice/<int:slice_id>/')
    def slice(self, slice_id: int) -> FlaskResponse:
        _, slc = get_form_data(slice_id, use_slice_data=True)
        if not slc:
            abort(404)
        form_data = parse.quote(json.dumps({'slice_id': slice_id}))
        endpoint = f'/explore/?form_data={form_data}'
        if ReservedUrlParameters.is_standalone_mode():
            endpoint += f'&{ReservedUrlParameters.STANDALONE}=true'
        return redirect(endpoint)

    def get_query_string_response(self, viz_obj: BaseViz) -> FlaskResponse:
        query: Optional[str] = None
        try:
            if (query_obj := viz_obj.query_obj()):
                query = viz_obj.datasource.get_query_str(query_obj)
        except Exception as ex:
            err_msg = utils.error_msg_from_exception(ex)
            logger.exception(err_msg)
            return json_error_response(err_msg)
        if not query:
            query = 'No query.'
        return self.json_response({'query': query, 'language': viz_obj.datasource.query_language})

    def get_raw_results(self, viz_obj: BaseViz) -> FlaskResponse:
        payload = viz_obj.get_df_payload()
        if viz_obj.has_error(payload):
            return json_error_response(payload=payload, status=400)
        return self.json_response({'data': payload['df'].to_dict('records'), 'colnames': payload.get('colnames'), 'coltypes': payload.get('coltypes'), 'rowcount': payload.get('rowcount'), 'sql_rowcount': payload.get('sql_rowcount')})

    def get_samples(self, viz_obj: BaseViz) -> FlaskResponse:
        return self.json_response(viz_obj.get_samples())

    @staticmethod
    def send_data_payload_response(viz_obj: BaseViz, payload: Dict[str, Any]) -> FlaskResponse:
        return data_payload_response(*viz_obj.payload_json_and_has_error(payload))

    def generate_json(self, viz_obj: BaseViz, response_type: Optional[str] = None) -> FlaskResponse:
        if response_type == ChartDataResultFormat.CSV:
            return CsvResponse(viz_obj.get_csv(), headers=generate_download_headers('csv'))
        if response_type == ChartDataResultType.QUERY:
            return self.get_query_string_response(viz_obj)
        if response_type == ChartDataResultType.RESULTS:
            return self.get_raw_results(viz_obj)
        if response_type == ChartDataResultType.SAMPLES:
            return self.get_samples(viz_obj)
        payload = viz_obj.get_payload()
        return self.send_data_payload_response(viz_obj, payload)

    @event_logger.log_this
    @api
    @has_access_api
    @handle_api_exception
    @permission_name('explore_json')
    @expose('/explore_json/data/<cache_key>', methods=('GET',))
    @check_resource_permissions(check_explore_cache_perms)
    @deprecated(eol_version='5.0.0')
    def explore_json_data(self, cache_key: str) -> FlaskResponse:
        try:
            cached = cache_manager.cache.get(cache_key)
            if not cached:
                raise CacheLoadError('Cached data not found')
            form_data = cached.get('form_data')
            response_type = cached.get('response_type')
            g.form_data = form_data
            datasource_id, datasource_type = get_datasource_info(None, None, form_data)
            viz_obj = get_viz(datasource_type=cast(str, datasource_type), datasource_id=datasource_id, form_data=form_data, force_cached=True)
            return self.generate_json(viz_obj, response_type)
        except SupersetException as ex:
            return json_error_response(utils.error_msg_from_exception(ex), 400)

    @api
    @has_access_api
    @handle_api_exception
    @event_logger.log_this
    @expose('/explore_json/<datasource_type>/<int:datasource_id>/', methods=('GET', 'POST'))
    @expose('/explore_json/', methods=('GET', 'POST'))
    @etag_cache()
    @check_resource_permissions(check_datasource_perms)
    @deprecated(eol_version='5.0.0')
    def explore_json(self, datasource_type: Optional[str] = None, datasource_id: Optional[int] = None) -> FlaskResponse:
        response_type = ChartDataResultFormat.JSON.value
        responses = list(ChartDataResultFormat)
        responses.extend(list(ChartDataResultType))
        for response_option in responses:
            if request.args.get(response_option) == 'true':
                response_type = response_option
                break
        if response_type == ChartDataResultFormat.CSV and (not security_manager.can_access('can_csv', 'Superset')):
            return json_error_response(_("You don't have the rights to download as csv"), status=403)
        form_data = get_form_data()[0]
        try:
            datasource_id, datasource_type = get_datasource_info(datasource_id, datasource_type, form_data)
            force = request.args.get('force') == 'true'
            if is_feature_enabled('GLOBAL_ASYNC_QUERIES') and response_type == ChartDataResultFormat.JSON:
                with contextlib.suppress(CacheLoadError):
                    viz_obj = get_viz(datasource_type=cast(str, datasource_type), datasource_id=datasource_id, form_data=form_data, force_cached=True, force=force)
                    payload = viz_obj.get_payload()
                    if payload is not None:
                        return self.send_data_payload_response(viz_obj, payload)
                try:
                    async_channel_id = async_query_manager.parse_channel_id_from_request(request)
                    job_metadata = async_query_manager.submit_explore_json_job(async_channel_id, form_data, response_type, force, get_user_id())
                except AsyncQueryTokenException:
                    return json_error_response('Not authorized', 401)
                return json_success(json.dumps(job_metadata), status=202)
            viz_obj = get_viz(datasource_type=cast(str, datasource_type), datasource_id=datasource_id, form_data=form_data, force=force)
            return self.generate_json(viz_obj, response_type)
        except SupersetException as ex:
            return json_error_response(utils.error_msg_from_exception(ex), 400)

    @staticmethod
    def get_redirect_url() -> str:
        redirect_url = request.url.replace('/superset/explore', '/explore')
        form_data_key = None
        if (request_form_data := request.args.get('form_data')):
            parsed_form_data = loads_request_json(request_form_data)
            slice_id = parsed_form_data.get('slice_id', int(request.args.get('slice_id', 0)))
            if (datasource := parsed_form_data.get('datasource')):
                datasource_id, datasource_type = datasource.split('__')
                parameters = CommandParameters(datasource_id=datasource_id, datasource_type=datasource_type, chart_id=slice_id, form_data=request_form_data)
                form_data_key = CreateFormDataCommand(parameters).run()
        if form_data_key:
            url = parse.urlparse(redirect_url)
            query = parse.parse_qs(url.query)
            query.pop('form_data')
            query['form_data_key'] = [form_data_key]
            url = url._replace(query=parse.urlencode(query, True))
            redirect_url = parse.urlunparse(url)
        url = parse.urlparse(redirect_url)
        return f'{url.path}?{url.query}' if url.query else url.path

    @has_access
    @event_logger.log_this
    @expose('/explore/<datasource_type>/<int:datasource_id>/', methods=('GET', 'POST'))
    @expose('/explore/', methods=('GET', 'POST'))
    @deprecated()
    def explore(self, datasource_type: Optional[str] = None, datasource_id: Optional[int] = None, key: Optional[str] = None) -> FlaskResponse:
        if request.method == 'GET':
            return redirect(Superset.get_redirect_url())
        initial_form_data: Dict[str, Any] = {}
        form_data_key = request.args.get('form_data_key')
        if key is not None:
            command = GetExplorePermalinkCommand(key)
            try:
                if (permalink_value := command.run()):
                    state = permalink_value['state']
                    initial_form_data = state['formData']
                    url_params = state.get('urlParams')
                    if url_params:
                        initial_form_data['url_params'] = dict(url_params)
                else:
                    return json_error_response(_('Error: permalink state not found'), status=404)
            except (ChartNotFoundError, ExplorePermalinkGetFailedError) as ex:
                flash(__('Error: %(msg)s', msg=ex.message), 'danger')
                return redirect('/chart/list/')
        elif form_data_key:
            parameters = CommandParameters(key=form_data_key)
            value = GetFormDataCommand(parameters).run()
            initial_form_data = json.loads(value) if value else {}
        if not initial_form_data:
            slice_id = request.args.get('slice_id')
            dataset_id = request.args.get('dataset_id')
            if slice_id:
                initial_form_data['slice_id'] = slice_id
                if form_data_key:
                    flash(_('Form data not found in cache, reverting to chart metadata.'))
            elif dataset_id:
                initial_form_data['datasource'] = f'{dataset_id}__table'
                if form_data_key:
                    flash(_('Form data not found in cache, reverting to dataset metadata.'))
        form_data, slc = get_form_data(use_slice_data=True, initial_form_data=initial_form_data)
        query_context = request.form.get('query_context')
        try:
            datasource_id, datasource_type = get_datasource_info(datasource_id, datasource_type, form_data)
        except SupersetException:
            datasource_id = None
            datasource_type = SqlaTable.type
        datasource = None
        if datasource_id is not None:
            with contextlib.suppress(DatasetNotFoundError):
                datasource = DatasourceDAO.get_datasource(DatasourceType('table'), datasource_id)
        datasource_name = datasource.name if datasource else _('[Missing Dataset]')
        viz_type = form_data.get('viz_type')
        if not viz_type and datasource and datasource.default_endpoint:
            return redirect(datasource.default_endpoint)
        selectedColumns: List[Dict[str, Any]] = []
        if 'selectedColumns' in form_data:
            selectedColumns = form_data.pop('selectedColumns')
        if 'viz_type' not in form_data:
            form_data['viz_type'] = app.config['DEFAULT_VIZ_TYPE']
            if app.config['DEFAULT_VIZ_TYPE'] == 'table':
                all_columns = []
                for x in selectedColumns:
                    all_columns.append(x['name'])
                form_data['all_columns'] = all_columns
        slice_add_perm = security_manager.can_access('can_write', 'Chart')
        slice_overwrite_perm = security_manager.is_owner(slc) if slc else False
        slice_download_perm = security_manager.can_access('can_csv', 'Superset')
        form_data['datasource'] = str(datasource_id) + '__' + cast(str, datasource_type)
        utils.convert_legacy_filters_into_adhoc(form_data)
        utils.merge_extra_filters(form_data)
        if request.method == 'GET':
            utils.merge_request_params(form_data, request.args)
        action = request.args.get('action')
        if action == 'overwrite' and (not slice_overwrite_perm):
            return json_error_response(_("You don't have the rights to alter this chart"), status=403)
        if action == 'saveas' and (not slice_add_perm):
            return json_error_response(_("You don't have the rights to create a chart"), status=403)
        if action in ('saveas', 'overwrite') and datasource:
            return self.save_or_overwrite_slice(slc, slice_add_perm, slice_overwrite_perm, slice_download_perm, datasource.id, datasource.type, datasource.name, query_context)
        standalone_mode = ReservedUrlParameters.is_standalone_mode()
        force = request.args.get('force') in {'force', '1', 'true'}
        dummy_datasource_data = {'type': datasource_type, 'name': datasource_name, 'columns': [], 'metrics': [], 'database': {'id': 0, 'backend': ''}}
        try:
            datasource_data = datasource.data if datasource else dummy_datasource_data
        except (SupersetException, SQLAlchemyError):
            datasource_data = dummy_datasource_data
        if datasource:
            datasource_data['owners'] = datasource.owners_data
            if isinstance(datasource, Query):
                datasource_data['columns'] = datasource.columns
        bootstrap_data = {'can_add': slice_add_perm, 'datasource': sanitize_datasource_data(datasource_data), 'form_data': form_data, 'datasource_id': datasource_id, 'datasource_type': datasource_type, 'slice': slc.data if slc else None, 'standalone': standalone_mode, 'force': force, 'user': bootstrap_user_data(g.user, include_perms=True), 'forced_height': request.args.get('height'), 'common': common_bootstrap_payload()}
        if slc:
            title = slc.slice_name
        elif datasource:
            table_name = datasource.table_name if datasource_type == 'table' else datasource.datasource_name
            title = _('Explore - %(table)s', table=table_name)
        else:
            title = _('Explore')
        return self.render_template('superset/basic.html', bootstrap_data=json.dumps(bootstrap_data, default=json.pessimistic_json_iso_dttm_ser), entry='explore', title=title, standalone_mode=standalone_mode)

    @staticmethod
    def save_or_overwrite_slice(slc: Optional[Slice], slice_add_perm: bool, slice_overwrite_perm: bool, slice_download_perm: bool, datasource_id: int, datasource_type: str, datasource_name: str, query_context: Optional[str] = None) -> FlaskResponse:
        slice_name = request.args.get('slice_name')
        action = request.args.get('action')
        form