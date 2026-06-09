from typing import Any

# === Third-party dependency: flask ===
# Used symbols: Response, g

# === Third-party dependency: flask_appbuilder.security.sqla ===
# Used symbols: models

# === Third-party dependency: flask_testing ===
# Used symbols: TestCase

# === Third-party dependency: pandas ===
# Used symbols: DataFrame

# === Unresolved dependency: rison ===
# Used unresolved symbols: dumps

# === Third-party dependency: sqlalchemy.dialects.mysql ===
# Used symbols: dialect

# === Third-party dependency: sqlalchemy.engine.interfaces ===
class Dialect(EventTarget): ...

# === Third-party dependency: sqlalchemy.sql ===
# Used symbols: func

# === Internal dependency: superset ===
from superset.extensions import db
from superset.extensions import security_manager

# === Internal dependency: superset.app ===
def create_app(superset_config_module=..., superset_app_root=...): ...

# === Internal dependency: superset.connectors.sqla.models ===
class BaseDatasource(AuditMixinNullable, ImportExportMixin): ...
class SqlaTable(CoreDataset, BaseDatasource, ExploreMixin): ...

# === Internal dependency: superset.models.core ===
class Database(CoreDatabase, AuditMixinNullable, ImportExportMixin): ...

# === Internal dependency: superset.models.dashboard ===
class Dashboard(CoreDashboard, AuditMixinNullable, ImportExportMixin):
    ...

# === Internal dependency: superset.models.slice ===
class Slice(CoreChart, AuditMixinNullable, ImportExportMixin): ...

# === Internal dependency: superset.sql.parse ===
class CTASMethod(enum.Enum): ...

# === Internal dependency: superset.utils.core ===
def get_example_default_schema(): ...
def shortid(): ...

# === Internal dependency: superset.utils.database ===
def get_example_database(): ...

# === Internal dependency: superset.utils.json ===
def loads(obj, encoding=..., allow_nan=..., object_hook=...): ...

# === Internal dependency: superset.views.base_api ===
class BaseSupersetModelRestApi(BaseSupersetApiMixin, ModelRestApi): ...

# === Internal dependency: tests.integration_tests.constants ===
ADMIN_USERNAME = 'admin'

# === Internal dependency: tests.integration_tests.fixtures.importexport ===
database_metadata_config = {'version': '1.0.0', 'type': 'Database', 'timestamp': '2020-11-04T21:27:44.423819+00:00'}
dataset_metadata_config = {'version': '1.0.0', 'type': 'SqlaTable', 'timestamp': '2020-11-04T21:27:44.423819+00:00'}
chart_metadata_config = {'version': '1.0.0', 'type': 'Slice', 'timestamp': '2020-11-04T21:27:44.423819+00:00'}
dashboard_metadata_config = {'version': '1.0.0', 'type': 'Dashboard', 'timestamp': '2020-11-04T21:27:44.423819+00:00'}
metadata_files = {'database': database_metadata_config, 'dataset': dataset_metadata_config, 'chart': chart_metadata_config, 'dashboard': dashboard_metadata_config}
database_config = {'allow_csv_upload': True, 'allow_ctas': True, 'allow_cvas': True, 'allow_dml': True, 'allow_run_async': False, 'cache_timeout': None, 'database_name': 'imported_database', 'expose_in_sqllab': True, ...}
dataset_config = {'table_name': 'imported_dataset', 'main_dttm_col': None, 'currency_code_column': 'currency', 'description': 'This is a dataset that was exported', 'default_endpoint': '', 'offset': 66, 'cache_timeout': 55, 'schema': '', ...}
chart_config = {'slice_name': 'Deck Path', 'viz_type': 'deck_path', 'params': {'color_picker': {'a': 1, 'b': 135, 'g': 122, 'r': 0}, 'datasource': '12__table', 'js_columns': ['color'], 'js_data_mutator': 'data => data.map(d => ({\\n    ...d,\\n    color: colors.hexToRGB(d.extraProps.color)\\n}));', 'js_onclick_href': '', 'js_tooltip': '', 'line_column': 'path_json', 'line_type': 'json', ...}, 'query_context': '{"datasource":{"id":12,"type":"table"},"force":false,"queries":[{"time_range":" : ","filters":[],"extras":{"time_grain_sqla":null,"having":"","where":""},"applied_time_extras":{},"columns":[],"metrics":[],"annotation_layers":[],"row_limit":5000,"timeseries_limit":0,"order_desc":true,"url_params":{},"custom_params":{},"custom_form_data":{}}],"result_format":"json","result_type":"full"}', 'cache_timeout': None, 'uuid': '0c23747a-6528-4629-97bf-e4b78d3b9df1', 'version': '1.0.0', 'dataset_uuid': '10808100-158b-42c4-842e-f32b99d88dfb'}
dashboard_config = {'dashboard_title': 'Test dash', 'description': None, 'css': '', 'slug': None, 'uuid': 'c4b28c4e-a1fe-4cf8-a5ac-d6f11d6fdd51', 'position': {'CHART-SVAlICPOSJ': {'children': [], 'id': 'CHART-SVAlICPOSJ', 'meta': {'chartId': 83, 'height': 50, 'sliceName': 'Number of California Births', 'uuid': '0c23747a-6528-4629-97bf-e4b78d3b9df1', 'width': 4}, 'parents': ['ROOT_ID', 'GRID_ID', 'ROW-dP_CHaK2q'], 'type': 'CHART'}, 'DASHBOARD_VERSION_KEY': 'v2', 'GRID_ID': {'children': ['ROW-dP_CHaK2q'], 'id': 'GRID_ID', 'parents': ['ROOT_ID'], 'type': 'GRID'}, 'HEADER_ID': {'id': 'HEADER_ID', 'meta': {'text': 'Test dash'}, 'type': 'HEADER'}, 'ROOT_ID': {'children': ['GRID_ID'], 'id': 'ROOT_ID', 'type': 'ROOT'}, 'ROW-dP_CHaK2q': {'children': ['CHART-SVAlICPOSJ'], 'id': 'ROW-dP_CHaK2q', 'meta': {'0': 'ROOT_ID', 'background': 'BACKGROUND_TRANSPARENT'}, 'parents': ['ROOT_ID', 'GRID_ID'], 'type': 'ROW'}}, 'metadata': {'timed_refresh_immune_slices': [83], 'filter_scopes': {'83': {'region': {'scope': ['ROOT_ID'], 'immune': [83]}}}, 'expanded_slices': {'83': True}, 'refresh_frequency': 0, 'default_filters': '{}', 'color_scheme': None, 'remote_id': 7, 'import_time': 1604342885}, 'version': '1.0.0'}

# === Internal dependency: tests.integration_tests.test_app ===
superset_config_module = environ.get(...)
app = create_app(...)

# === Third-party dependency: yaml ===
def safe_dump(data, stream = ..., **kwds) -> Any: ...