import re
from typing import Any, Dict, List, Optional, Tuple, Union
from marshmallow import fields, post_dump, post_load, pre_load, Schema
from marshmallow.validate import Length, ValidationError
from superset import security_manager
from superset.tags.models import TagType
from superset.utils import json


get_delete_ids_schema: Dict[str, Any] = {'type': 'array', 'items': {'type': 'integer'}}
get_export_ids_schema: Dict[str, Any] = {'type': 'array', 'items': {'type': 'integer'}}
get_fav_star_ids_schema: Dict[str, Any] = {'type': 'array', 'items': {'type': 'integer'}}
thumbnail_query_schema: Dict[str, Any] = {'type': 'object', 'properties': {'force': {'type': 'boolean'}}}
width_height_schema: Dict[str, Any] = {'type': 'array', 'items': {'type': 'integer'}}
screenshot_query_schema: Dict[str, Any] = {
    'type': 'object',
    'properties': {
        'force': {'type': 'boolean'},
        'permalink': {'type': 'string'},
        'window_size': width_height_schema,
        'thumb_size': width_height_schema
    }
}
dashboard_title_description: str = 'A title for the dashboard.'
slug_description: str = 'Unique identifying part for the web address of the dashboard.'
owners_description: str = 'Owner are users ids allowed to delete or change this dashboard. If left empty you will be one of the owners of the dashboard.'
roles_description: str = 'Roles is a list which defines access to the dashboard. These roles are always applied in addition to restrictions on dataset level access. If no roles defined then the dashboard is available to all roles.'
position_json_description: str = 'This json object describes the positioning of the widgets in the dashboard. It is dynamically generated when adjusting the widgets size and positions by using drag & drop in the dashboard view'
css_description: str = 'Override CSS for the dashboard.'
json_metadata_description: str = 'This JSON object is generated dynamically when clicking the save or overwrite button in the dashboard view. It is exposed here for reference and for power users who may want to alter  specific parameters.'
published_description: str = 'Determines whether or not this dashboard is visible in the list of all dashboards.'
charts_description: str = "The names of the dashboard's charts. Names are used for legacy reasons."
certified_by_description: str = 'Person or group that has certified this dashboard'
certification_details_description: str = 'Details of the certification'
tags_description: str = 'Tags to be associated with the dashboard'
openapi_spec_methods_override: Dict[str, Any] = {
    'get': {'get': {'summary': 'Get a dashboard detail information'}},
    'get_list': {
        'get': {
            'summary': 'Get a list of dashboards',
            'description': 'Gets a list of dashboards, use Rison or JSON query parameters for filtering, sorting, pagination and  for selecting specific columns and metadata.'
        }
    },
    'info': {'get': {'summary': 'Get metadata information about this API resource'}},
    'related': {'get': {'description': 'Get a list of all possible owners for a dashboard.'}}
}


def validate_json(value: str) -> None:
    try:
        json.validate_json(value)
    except json.JSONDecodeError as ex:
        raise ValidationError('JSON not valid') from ex


def validate_json_metadata(value: Optional[str]) -> None:
    if not value:
        return
    try:
        value_obj = json.loads(value)
    except json.JSONDecodeError as ex:
        raise ValidationError('JSON not valid') from ex
    errors = DashboardJSONMetadataSchema().validate(value_obj, partial=False)
    if errors:
        raise ValidationError(errors)


class SharedLabelsColorsField(fields.Field):
    """
    A custom field that accepts either a list of strings or a dictionary.
    """

    def _deserialize(
        self, value: Any, attr: Optional[str], data: Optional[Dict[str, Any]], **kwargs: Any
    ) -> Union[List[str], Dict[str, Any]]:
        if isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                return value
        elif isinstance(value, dict):
            return []
        raise ValidationError('Not a valid list')


class DashboardJSONMetadataSchema(Schema):
    native_filter_configuration: Optional[List[Dict[str, Any]]] = fields.List(fields.Dict(), allow_none=True)
    chart_configuration: Dict[str, Any] = fields.Dict()
    global_chart_configuration: Dict[str, Any] = fields.Dict()
    timed_refresh_immune_slices: List[int] = fields.List(fields.Integer())
    filter_scopes: Dict[str, Any] = fields.Dict()
    expanded_slices: Dict[str, Any] = fields.Dict()
    refresh_frequency: Optional[int] = fields.Integer()
    default_filters: Optional[str] = fields.Str()
    stagger_refresh: Optional[bool] = fields.Boolean()
    stagger_time: Optional[int] = fields.Integer()
    color_scheme: Optional[str] = fields.Str(allow_none=True)
    color_namespace: Optional[str] = fields.Str(allow_none=True)
    positions: Optional[Dict[str, Any]] = fields.Dict(allow_none=True)
    label_colors: Dict[str, Any] = fields.Dict()
    shared_label_colors: Union[List[str], Dict[str, Any]] = SharedLabelsColorsField()
    map_label_colors: Dict[str, Any] = fields.Dict()
    color_scheme_domain: List[str] = fields.List(fields.Str())
    cross_filters_enabled: bool = fields.Boolean(dump_default=True)
    import_time: Optional[int] = fields.Integer()
    remote_id: Optional[int] = fields.Integer()
    filter_bar_orientation: Optional[str] = fields.Str(allow_none=True)
    native_filter_migration: Dict[str, Any] = fields.Dict()

    @pre_load
    def remove_show_native_filters(self, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """
        Remove ``show_native_filters`` from the JSON metadata.

        This field was removed in https://github.com/apache/superset/pull/23228, but might
        be present in old exports.
        """
        if 'show_native_filters' in data:
            del data['show_native_filters']
        return data


class UserSchema(Schema):
    id: fields.Int = fields.Int()
    username: fields.Str = fields.String()
    first_name: fields.Str = fields.String()
    last_name: fields.Str = fields.String()


class RolesSchema(Schema):
    id: fields.Int = fields.Int()
    name: fields.Str = fields.String()


class TagSchema(Schema):
    id: fields.Int = fields.Int()
    name: fields.Str = fields.String()
    type: TagType = fields.Enum(TagType, by_value=True)


class DashboardGetResponseSchema(Schema):
    id: fields.Int = fields.Int()
    slug: fields.Str = fields.String()
    url: fields.Str = fields.String()
    dashboard_title: fields.Str = fields.String(metadata={'description': dashboard_title_description})
    thumbnail_url: Optional[str] = fields.String(allow_none=True)
    published: bool = fields.Boolean()
    css: str = fields.String(metadata={'description': css_description})
    json_metadata: str = fields.String(metadata={'description': json_metadata_description})
    position_json: str = fields.String(metadata={'description': position_json_description})
    certified_by: str = fields.String(metadata={'description': certified_by_description})
    certification_details: str = fields.String(metadata={'description': certification_details_description})
    changed_by_name: str = fields.String()
    changed_by: UserSchema = fields.Nested(UserSchema(exclude=['username']))
    changed_on: fields.DateTime = fields.DateTime()
    created_by: UserSchema = fields.Nested(UserSchema(exclude=['username']))
    charts: List[str] = fields.List(fields.String(metadata={'description': charts_description}))
    owners: List[UserSchema] = fields.List(fields.Nested(UserSchema(exclude=['username'])))
    roles: List[RolesSchema] = fields.List(fields.Nested(RolesSchema))
    tags: List[TagSchema] = fields.Nested(TagSchema, many=True)
    changed_on_humanized: str = fields.String(data_key='changed_on_delta_humanized')
    created_on_humanized: str = fields.String(data_key='created_on_delta_humanized')
    is_managed_externally: Optional[bool] = fields.Boolean(allow_none=True, dump_default=False)

    @post_dump
    def post_dump(self, serialized: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if security_manager.is_guest_user():
            serialized.pop('owners', None)
            serialized.pop('changed_by_name', None)
            serialized.pop('changed_by', None)
        return serialized


class DatabaseSchema(Schema):
    id: fields.Int = fields.Int()
    name: fields.Str = fields.String()
    backend: fields.Str = fields.String()
    allows_subquery: bool = fields.Bool()
    allows_cost_estimate: bool = fields.Bool()
    allows_virtual_table_explore: bool = fields.Bool()
    disable_data_preview: bool = fields.Bool()
    disable_drill_to_detail: bool = fields.Bool()
    allow_multi_catalog: bool = fields.Bool()
    explore_database_id: Optional[int] = fields.Int()


class DashboardDatasetSchema(Schema):
    id: fields.Int = fields.Int()
    uid: fields.Str = fields.Str()
    column_formats: Dict[str, Any] = fields.Dict()
    currency_formats: Dict[str, Any] = fields.Dict()
    database: DatabaseSchema = fields.Nested(DatabaseSchema)
    default_endpoint: Optional[str] = fields.String()
    filter_select: bool = fields.Bool()
    filter_select_enabled: bool = fields.Bool()
    is_sqllab_view: bool = fields.Bool()
    name: str = fields.Str()
    datasource_name: str = fields.Str()
    table_name: str = fields.Str()
    type: str = fields.Str()
    schema: str = fields.Str()
    offset: Optional[int] = fields.Int()
    cache_timeout: Optional[int] = fields.Int()
    params: Optional[str] = fields.Str()
    perm: Optional[str] = fields.Str()
    edit_url: Optional[str] = fields.Str()
    sql: Optional[str] = fields.Str()
    select_star: Optional[str] = fields.Str()
    main_dttm_col: Optional[str] = fields.Str()
    health_check_message: Optional[str] = fields.Str()
    fetch_values_predicate: Optional[str] = fields.Str()
    template_params: Optional[str] = fields.Str()
    owners: List[Dict[str, Any]] = fields.List(fields.Dict())
    columns: List[Dict[str, Any]] = fields.List(fields.Dict())
    column_types: List[int] = fields.List(fields.Int())
    column_names: List[str] = fields.List(fields.Str())
    metrics: List[Dict[str, Any]] = fields.List(fields.Dict())
    order_by_choices: List[List[str]] = fields.List(fields.List(fields.Str()))
    verbose_map: Dict[str, str] = fields.Dict(fields.Str(), fields.Str())
    time_grain_sqla: List[List[str]] = fields.List(fields.List(fields.Str()))
    granularity_sqla: List[List[str]] = fields.List(fields.List(fields.Str()))
    normalize_columns: bool = fields.Bool()
    always_filter_main_dttm: bool = fields.Bool()

    @post_dump
    def post_dump(self, serialized: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if security_manager.is_guest_user():
            serialized.pop('owners', None)
            serialized.pop('database', None)
        return serialized


class TabSchema(Schema):
    children: List['TabSchema'] = fields.List(fields.Nested(lambda: TabSchema()))
    value: str = fields.Str()
    title: str = fields.Str()
    parents: List[str] = fields.List(fields.Str())


class TabsPayloadSchema(Schema):
    all_tabs: Dict[str, str] = fields.Dict(keys=fields.String(), values=fields.String())
    tab_tree: List[TabSchema] = fields.List(fields.Nested(lambda: TabSchema()))


class BaseDashboardSchema(Schema):

    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if 'slug' in data and data['slug']:
            data['slug'] = data['slug'].strip()
            data['slug'] = data['slug'].replace(' ', '-')
            data['slug'] = re.sub(r'[^\w\-]+', '', data['slug'])
        return data


class DashboardPostSchema(BaseDashboardSchema):
    dashboard_title: Optional[str] = fields.String(
        metadata={'description': dashboard_title_description},
        allow_none=True,
        validate=Length(0, 500)
    )
    slug: Optional[str] = fields.String(
        metadata={'description': slug_description},
        allow_none=True,
        validate=Length(1, 255)
    )
    owners: List[int] = fields.List(
        fields.Integer(metadata={'description': owners_description})
    )
    roles: List[int] = fields.List(
        fields.Integer(metadata={'description': roles_description})
    )
    position_json: str = fields.String(
        metadata={'description': position_json_description},
        validate=validate_json
    )
    css: str = fields.String(metadata={'description': css_description})
    json_metadata: str = fields.String(
        metadata={'description': json_metadata_description},
        validate=validate_json_metadata
    )
    published: bool = fields.Boolean(metadata={'description': published_description})
    certified_by: Optional[str] = fields.String(
        metadata={'description': certified_by_description},
        allow_none=True
    )
    certification_details: Optional[str] = fields.String(
        metadata={'description': certification_details_description},
        allow_none=True
    )
    is_managed_externally: Optional[bool] = fields.Boolean(
        allow_none=True,
        dump_default=False
    )
    external_url: Optional[str] = fields.String(allow_none=True)


class DashboardCopySchema(Schema):
    dashboard_title: Optional[str] = fields.String(
        metadata={'description': dashboard_title_description},
        allow_none=True,
        validate=Length(0, 500)
    )
    css: str = fields.String(metadata={'description': css_description})
    json_metadata: str = fields.String(
        metadata={'description': json_metadata_description},
        validate=validate_json_metadata,
        required=True
    )
    duplicate_slices: bool = fields.Boolean(
        metadata={'description': 'Whether or not to also copy all charts on the dashboard'}
    )


class DashboardPutSchema(BaseDashboardSchema):
    dashboard_title: Optional[str] = fields.String(
        metadata={'description': dashboard_title_description},
        allow_none=True,
        validate=Length(0, 500)
    )
    slug: Optional[str] = fields.String(
        metadata={'description': slug_description},
        allow_none=True,
        validate=Length(0, 255)
    )
    owners: Optional[List[int]] = fields.List(
        fields.Integer(metadata={'description': owners_description}),
        allow_none=True
    )
    roles: Optional[List[int]] = fields.List(
        fields.Integer(metadata={'description': roles_description}),
        allow_none=True
    )
    position_json: Optional[str] = fields.String(
        metadata={'description': position_json_description},
        allow_none=True,
        validate=validate_json
    )
    css: Optional[str] = fields.String(
        metadata={'description': css_description},
        allow_none=True
    )
    json_metadata: Optional[str] = fields.String(
        metadata={'description': json_metadata_description},
        allow_none=True,
        validate=validate_json_metadata
    )
    published: Optional[bool] = fields.Boolean(
        metadata={'description': published_description},
        allow_none=True
    )
    certified_by: Optional[str] = fields.String(
        metadata={'description': certified_by_description},
        allow_none=True
    )
    certification_details: Optional[str] = fields.String(
        metadata={'description': certification_details_description},
        allow_none=True
    )
    is_managed_externally: Optional[bool] = fields.Boolean(
        allow_none=True,
        dump_default=False
    )
    external_url: Optional[str] = fields.String(allow_none=True)
    tags: Optional[List[int]] = fields.List(
        fields.Integer(metadata={'description': tags_description}),
        allow_none=True
    )


class DashboardNativeFiltersConfigUpdateSchema(BaseDashboardSchema):
    deleted: List[str] = fields.List(fields.String(), allow_none=False)
    modified: List[Any] = fields.List(fields.Raw(), allow_none=False)
    reordered: List[str] = fields.List(fields.String(), allow_none=False)


class DashboardColorsConfigUpdateSchema(BaseDashboardSchema):
    color_namespace: Optional[str] = fields.String(allow_none=True)
    color_scheme: Optional[str] = fields.String(allow_none=True)
    map_label_colors: Dict[str, Any] = fields.Dict(allow_none=False)
    shared_label_colors: Union[List[str], Dict[str, Any]] = SharedLabelsColorsField()
    label_colors: Dict[str, Any] = fields.Dict(allow_none=False)
    color_scheme_domain: List[str] = fields.List(fields.String(), allow_none=False)


class DashboardScreenshotPostSchema(Schema):
    dataMask: Optional[Dict[str, Any]] = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        metadata={'description': 'An object representing the data mask.'}
    )
    activeTabs: Optional[List[str]] = fields.List(
        fields.Str(),
        metadata={'description': 'A list representing active tabs.'}
    )
    anchor: Optional[str] = fields.String(
        metadata={'description': 'A string representing the anchor.'}
    )
    urlParams: Optional[List[Tuple[str, str]]] = fields.List(
        fields.Tuple((fields.Str(), fields.Str())),
        metadata={'description': 'A list of tuples, each containing two strings.'}
    )


class ChartFavStarResponseResult(Schema):
    id: int = fields.Integer(metadata={'description': 'The Chart id'})
    value: bool = fields.Boolean(metadata={'description': 'The FaveStar value'})


class GetFavStarIdsSchema(Schema):
    result: List[ChartFavStarResponseResult] = fields.List(
        fields.Nested(ChartFavStarResponseResult),
        metadata={'description': 'A list of results for each corresponding chart in the request'}
    )


class ImportV1DashboardSchema(Schema):
    dashboard_title: str = fields.String(required=True)
    description: Optional[str] = fields.String(allow_none=True)
    css: Optional[str] = fields.String(allow_none=True)
    slug: Optional[str] = fields.String(allow_none=True)
    uuid: str = fields.UUID(required=True)
    position: Dict[str, Any] = fields.Dict()
    metadata: Dict[str, Any] = fields.Dict()
    version: str = fields.String(required=True)
    is_managed_externally: Optional[bool] = fields.Boolean(allow_none=True, dump_default=False)
    external_url: Optional[str] = fields.String(allow_none=True)
    certified_by: Optional[str] = fields.String(allow_none=True)
    certification_details: Optional[str] = fields.String(allow_none=True)
    published: Optional[bool] = fields.Boolean(allow_none=True)


class EmbeddedDashboardConfigSchema(Schema):
    allowed_domains: List[str] = fields.List(fields.String(), required=True)


class EmbeddedDashboardResponseSchema(Schema):
    uuid: str = fields.String()
    allowed_domains: List[str] = fields.List(fields.String())
    dashboard_id: str = fields.String()
    changed_on: fields.DateTime = fields.DateTime()
    changed_by: UserSchema = fields.Nested(UserSchema)


class DashboardCacheScreenshotResponseSchema(Schema):
    cache_key: str = fields.String(metadata={'description': 'The cache key'})
    dashboard_url: str = fields.String(metadata={'description': 'The url to render the dashboard'})
    image_url: str = fields.String(metadata={'description': 'The url to fetch the screenshot'})
    task_status: str = fields.String(metadata={'description': 'The status of the async screenshot'})
    task_updated_at: str = fields.String(metadata={'description': 'The timestamp of the last change in status'})


class CacheScreenshotSchema(Schema):
    dataMask: Optional[Dict[str, Any]] = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=False
    )
    activeTabs: Optional[List[str]] = fields.List(
        fields.Str(),
        required=False
    )
    anchor: Optional[str] = fields.Str(
        required=False
    )
    urlParams: Optional[List[List[str]]] = fields.List(
        fields.List(fields.Str(), validate=lambda x: len(x) == 2),
        required=False
    )
