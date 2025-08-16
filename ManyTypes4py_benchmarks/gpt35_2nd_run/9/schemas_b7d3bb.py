from typing import Any, Mapping, Union, List, Dict

get_delete_ids_schema: Dict[str, Union[str, Dict[str, str]]] = {'type': 'array', 'items': {'type': 'integer'}}
get_export_ids_schema: Dict[str, Union[str, Dict[str, str]]] = {'type': 'array', 'items': {'type': 'integer'}}
get_fav_star_ids_schema: Dict[str, Union[str, Dict[str, str]]] = {'type': 'array', 'items': {'type': 'integer'}}
thumbnail_query_schema: Dict[str, Union[str, Dict[str, str]]] = {'type': 'object', 'properties': {'force': {'type': 'boolean'}}}
width_height_schema: Dict[str, Union[str, Dict[str, str]]] = {'type': 'array', 'items': {'type': 'integer'}}
screenshot_query_schema: Dict[str, Union[str, Dict[str, Union[bool, str, List[int]]]]] = {'type': 'object', 'properties': {'force': {'type': 'boolean'}, 'permalink': {'type': 'string'}, 'window_size': width_height_schema, 'thumb_size': width_height_schema}}
openapi_spec_methods_override: Dict[str, Dict[str, Dict[str, str]]] = {'get': {'get': {'summary': 'Get a dashboard detail information'}}, 'get_list': {'get': {'summary': 'Get a list of dashboards', 'description': 'Gets a list of dashboards, use Rison or JSON query parameters for filtering, sorting, pagination and  for selecting specific columns and metadata.'}}, 'info': {'get': {'summary': 'Get metadata information about this API resource'}}, 'related': {'get': {'description': 'Get a list of all possible owners for a dashboard.'}}}

def validate_json(value: Any) -> None:
def validate_json_metadata(value: Any) -> None:

class SharedLabelsColorsField(fields.Field):
    def _deserialize(self, value: Any, attr: str, data: Mapping[str, Any], **kwargs: Any) -> Union[List[str], List[Dict[str, Any]]]:

class DashboardJSONMetadataSchema(Schema):
    native_filter_configuration: List[Dict[str, Any]]
    chart_configuration: Dict[str, Any]
    global_chart_configuration: Dict[str, Any]
    timed_refresh_immune_slices: List[int]
    filter_scopes: Dict[str, Any]
    expanded_slices: Dict[str, Any]
    refresh_frequency: int
    default_filters: str
    stagger_refresh: bool
    stagger_time: int
    color_scheme: str
    color_namespace: str
    positions: Dict[str, Any]
    label_colors: Dict[str, Any]
    shared_label_colors: SharedLabelsColorsField
    map_label_colors: Dict[str, Any]
    color_scheme_domain: List[str]
    cross_filters_enabled: bool
    import_time: int
    remote_id: int
    filter_bar_orientation: str
    native_filter_migration: Dict[str, Any]

class UserSchema(Schema):
    id: int
    username: str
    first_name: str
    last_name: str

class RolesSchema(Schema):
    id: int
    name: str

class TagSchema(Schema):
    id: int
    name: str
    type: TagType

class DashboardGetResponseSchema(Schema):
    id: int
    slug: str
    url: str
    dashboard_title: str
    thumbnail_url: str
    published: bool
    css: str
    json_metadata: str
    position_json: str
    certified_by: str
    certification_details: str
    changed_by_name: str
    changed_by: UserSchema
    changed_on: datetime
    created_by: UserSchema
    charts: List[str]
    owners: List[UserSchema]
    roles: List[RolesSchema]
    tags: List[TagSchema]
    changed_on_humanized: str
    created_on_humanized: str
    is_managed_externally: bool

class DatabaseSchema(Schema):
    id: int
    name: str
    backend: str
    allows_subquery: bool
    allows_cost_estimate: bool
    allows_virtual_table_explore: bool
    disable_data_preview: bool
    disable_drill_to_detail: bool
    allow_multi_catalog: bool
    explore_database_id: int

class DashboardDatasetSchema(Schema):
    id: int
    uid: str
    column_formats: Dict[str, Any]
    currency_formats: Dict[str, Any]
    database: DatabaseSchema
    default_endpoint: str
    filter_select: bool
    filter_select_enabled: bool
    is_sqllab_view: bool
    name: str
    datasource_name: str
    table_name: str
    type: str
    schema: str
    offset: int
    cache_timeout: int
    params: str
    perm: str
    edit_url: str
    sql: str
    select_star: str
    main_dttm_col: str
    health_check_message: str
    fetch_values_predicate: str
    template_params: str
    owners: List[Dict[str, Any]]
    columns: List[Dict[str, Any]]
    column_types: List[int]
    column_names: List[str]
    metrics: List[Dict[str, Any]]
    order_by_choices: List[List[str]]
    verbose_map: Dict[str, str]
    time_grain_sqla: List[List[str]]
    granularity_sqla: List[List[str]]
    normalize_columns: bool
    always_filter_main_dttm: bool

class TabSchema(Schema):
    children: List[TabSchema]
    value: str
    title: str
    parents: List[str]

class TabsPayloadSchema(Schema):
    all_tabs: Dict[str, str]
    tab_tree: List[TabSchema]

class BaseDashboardSchema(Schema):

class DashboardPostSchema(BaseDashboardSchema):
    dashboard_title: str
    slug: str
    owners: List[int]
    roles: List[int]
    position_json: str
    css: str
    json_metadata: str
    published: bool
    certified_by: str
    certification_details: str
    is_managed_externally: bool
    external_url: str

class DashboardCopySchema(Schema):
    dashboard_title: str
    css: str
    json_metadata: str
    duplicate_slices: bool

class DashboardPutSchema(BaseDashboardSchema):
    dashboard_title: str
    slug: str
    owners: List[int]
    roles: List[int]
    position_json: str
    css: str
    json_metadata: str
    published: bool
    certified_by: str
    certification_details: str
    is_managed_externally: bool
    external_url: str
    tags: List[int]

class DashboardNativeFiltersConfigUpdateSchema(BaseDashboardSchema):
    deleted: List[str]
    modified: List[Dict[str, Any]]
    reordered: List[str]

class DashboardColorsConfigUpdateSchema(BaseDashboardSchema):
    color_namespace: str
    color_scheme: str
    map_label_colors: Dict[str, Any]
    shared_label_colors: SharedLabelsColorsField
    label_colors: Dict[str, Any]
    color_scheme_domain: List[str]

class DashboardScreenshotPostSchema(Schema):
    dataMask: Dict[str, Any]
    activeTabs: List[str]
    anchor: str
    urlParams: List[Tuple[str, str]]

class ChartFavStarResponseResult(Schema):
    id: int
    value: bool

class GetFavStarIdsSchema(Schema):
    result: List[ChartFavStarResponseResult]

class ImportV1DashboardSchema(Schema):
    dashboard_title: str
    description: str
    css: str
    slug: str
    uuid: UUID
    position: Dict[str, Any]
    metadata: Dict[str, Any]
    version: str
    is_managed_externally: bool
    external_url: str
    certified_by: str
    certification_details: str
    published: bool

class EmbeddedDashboardConfigSchema(Schema):
    allowed_domains: List[str]

class EmbeddedDashboardResponseSchema(Schema):
    uuid: str
    allowed_domains: List[str]
    dashboard_id: str
    changed_on: datetime
    changed_by: UserSchema

class DashboardCacheScreenshotResponseSchema(Schema):
    cache_key: str
    dashboard_url: str
    image_url: str
    task_status: str
    task_updated_at: str

class CacheScreenshotSchema(Schema):
    dataMask: Dict[str, Any]
    activeTabs: List[str]
    anchor: str
    urlParams: List[List[str]]
