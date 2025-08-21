import re
from typing import Any, Mapping, Optional, Dict, List, Union
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
        'thumb_size': width_height_schema,
    },
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
            'description': 'Gets a list of dashboards, use Rison or JSON query parameters for filtering, sorting, pagination and  for selecting specific columns and metadata.',
        }
    },
    'info': {'get': {'summary': 'Get metadata information about this API resource'}},
    'related': {'get': {'description': 'Get a list of all possible owners for a dashboard.'}},
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
        self,
        value: Any,
        attr: Optional[str],
        data: Optional[Mapping[str, Any]],
        **kwargs: Any,
    ) -> List[str]:
        if isinstance(value, list):
            if all((isinstance(item, str) for item in value)):
                return value
        elif isinstance(value, dict):
            return []
        raise ValidationError('Not a valid list')


class DashboardJSONMetadataSchema(Schema):
    native_filter_configuration = fields.List(fields.Dict(), allow_none=True)
    chart_configuration = fields.Dict()
    global_chart_configuration = fields.Dict()
    timed_refresh_immune_slices = fields.List(fields.Integer())
    filter_scopes = fields.Dict()
    expanded_slices = fields.Dict()
    refresh_frequency = fields.Integer()
    default_filters = fields.Str()
    stagger_refresh = fields.Boolean()
    stagger_time = fields.Integer()
    color_scheme = fields.Str(allow_none=True)
    color_namespace = fields.Str(allow_none=True)
    positions = fields.Dict(allow_none=True)
    label_colors = fields.Dict()
    shared_label_colors = SharedLabelsColorsField()
    map_label_colors = fields.Dict()
    color_scheme_domain = fields.List(fields.Str())
    cross_filters_enabled = fields.Boolean(dump_default=True)
    import_time = fields.Integer()
    remote_id = fields.Integer()
    filter_bar_orientation = fields.Str(allow_none=True)
    native_filter_migration = fields.Dict()

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
    id = fields.Int()
    username = fields.String()
    first_name = fields.String()
    last_name = fields.String()


class RolesSchema(Schema):
    id = fields.Int()
    name = fields.String()


class TagSchema(Schema):
    id = fields.Int()
    name = fields.String()
    type = fields.Enum(TagType, by_value=True)


class DashboardGetResponseSchema(Schema):
    id = fields.Int()
    slug = fields.String()
    url = fields.String()
    dashboard_title = fields.String(metadata={'description': dashboard_title_description})
    thumbnail_url = fields.String(allow_none=True)
    published = fields.Boolean()
    css = fields.String(metadata={'description': css_description})
    json_metadata = fields.String(metadata={'description': json_metadata_description})
    position_json = fields.String(metadata={'description': position_json_description})
    certified_by = fields.String(metadata={'description': certified_by_description})
    certification_details = fields.String(metadata={'description': certification_details_description})
    changed_by_name = fields.String()
    changed_by = fields.Nested(UserSchema(exclude=['username']))
    changed_on = fields.DateTime()
    created_by = fields.Nested(UserSchema(exclude=['username']))
    charts = fields.List(fields.String(metadata={'description': charts_description}))
    owners = fields.List(fields.Nested(UserSchema(exclude=['username'])))
    roles = fields.List(fields.Nested(RolesSchema))
    tags = fields.Nested(TagSchema, many=True)
    changed_on_humanized = fields.String(data_key='changed_on_delta_humanized')
    created_on_humanized = fields.String(data_key='created_on_delta_humanized')
    is_managed_externally = fields.Boolean(allow_none=True, dump_default=False)

    @post_dump()
    def post_dump(self, serialized: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if security_manager.is_guest_user():
            del serialized['owners']
            del serialized['changed_by_name']
            del serialized['changed_by']
        return serialized


class DatabaseSchema(Schema):
    id = fields.Int()
    name = fields.String()
    backend = fields.String()
    allows_subquery = fields.Bool()
    allows_cost_estimate = fields.Bool()
    allows_virtual_table_explore = fields.Bool()
    disable_data_preview = fields.Bool()
    disable_drill_to_detail = fields.Bool()
    allow_multi_catalog = fields.Bool()
    explore_database_id = fields.Int()


class DashboardDatasetSchema(Schema):
    id = fields.Int()
    uid = fields.Str()
    column_formats = fields.Dict()
    currency_formats = fields.Dict()
    database = fields.Nested(DatabaseSchema)
    default_endpoint = fields.String()
    filter_select = fields.Bool()
    filter_select_enabled = fields.Bool()
    is_sqllab_view = fields.Bool()
    name = fields.Str()
    datasource_name = fields.Str()
    table_name = fields.Str()
    type = fields.Str()
    schema = fields.Str()
    offset = fields.Int()
    cache_timeout = fields.Int()
    params = fields.Str()
    perm = fields.Str()
    edit_url = fields.Str()
    sql = fields.Str()
    select_star = fields.Str()
    main_dttm_col = fields.Str()
    health_check_message = fields.Str()
    fetch_values_predicate = fields.Str()
    template_params = fields.Str()
    owners = fields.List(fields.Dict())
    columns = fields.List(fields.Dict())
    column_types = fields.List(fields.Int())
    column_names = fields.List(fields.Str())
    metrics = fields.List(fields.Dict())
    order_by_choices = fields.List(fields.List(fields.Str()))
    verbose_map = fields.Dict(fields.Str(), fields.Str())
    time_grain_sqla = fields.List(fields.List(fields.Str()))
    granularity_sqla = fields.List(fields.List(fields.Str()))
    normalize_columns = fields.Bool()
    always_filter_main_dttm = fields.Bool()

    @post_dump()
    def post_dump(self, serialized: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if security_manager.is_guest_user():
            del serialized['owners']
            del serialized['database']
        return serialized


class TabSchema(Schema):
    children = fields.List(fields.Nested(lambda: TabSchema()))
    value = fields.Str()
    title = fields.Str()
    parents = fields.List(fields.Str())


class TabsPayloadSchema(Schema):
    all_tabs = fields.Dict(keys=fields.String(), values=fields.String())
    tab_tree = fields.List(fields.Nested(lambda: TabSchema))


class BaseDashboardSchema(Schema):
    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if data.get('slug'):
            data['slug'] = data['slug'].strip()
            data['slug'] = data['slug'].replace(' ', '-')
            data['slug'] = re.sub('[^\\w\\-]+', '', data['slug'])
        return data


class DashboardPostSchema(BaseDashboardSchema):
    dashboard_title = fields.String(metadata={'description': dashboard_title_description}, allow_none=True, validate=Length(0, 500))
    slug = fields.String(metadata={'description': slug_description}, allow_none=True, validate=[Length(1, 255)])
    owners = fields.List(fields.Integer(metadata={'description': owners_description}))
    roles = fields.List(fields.Integer(metadata={'description': roles_description}))
    position_json = fields.String(metadata={'description': position_json_description}, validate=validate_json)
    css = fields.String(metadata={'description': css_description})
    json_metadata = fields.String(metadata={'description': json_metadata_description}, validate=validate_json_metadata)
    published = fields.Boolean(metadata={'description': published_description})
    certified_by = fields.String(metadata={'description': certified_by_description}, allow_none=True)
    certification_details = fields.String(metadata={'description': certification_details_description}, allow_none=True)
    is_managed_externally = fields.Boolean(allow_none=True, dump_default=False)
    external_url = fields.String(allow_none=True)


class DashboardCopySchema(Schema):
    dashboard_title = fields.String(metadata={'description': dashboard_title_description}, allow_none=True, validate=Length(0, 500))
    css = fields.String(metadata={'description': css_description})
    json_metadata = fields.String(metadata={'description': json_metadata_description}, validate=validate_json_metadata, required=True)
    duplicate_slices = fields.Boolean(metadata={'description': 'Whether or not to also copy all charts on the dashboard'})


class DashboardPutSchema(BaseDashboardSchema):
    dashboard_title = fields.String(metadata={'description': dashboard_title_description}, allow_none=True, validate=Length(0, 500))
    slug = fields.String(metadata={'description': slug_description}, allow_none=True, validate=Length(0, 255))
    owners = fields.List(fields.Integer(metadata={'description': owners_description}, allow_none=True))
    roles = fields.List(fields.Integer(metadata={'description': roles_description}, allow_none=True))
    position_json = fields.String(metadata={'description': position_json_description}, allow_none=True, validate=validate_json)
    css = fields.String(metadata={'description': css_description}, allow_none=True)
    json_metadata = fields.String(metadata={'description': json_metadata_description}, allow_none=True, validate=validate_json_metadata)
    published = fields.Boolean(metadata={'description': published_description}, allow_none=True)
    certified_by = fields.String(metadata={'description': certified_by_description}, allow_none=True)
    certification_details = fields.String(metadata={'description': certification_details_description}, allow_none=True)
    is_managed_externally = fields.Boolean(allow_none=True, dump_default=False)
    external_url = fields.String(allow_none=True)
    tags = fields.List(fields.Integer(metadata={'description': tags_description}, allow_none=True))


class DashboardNativeFiltersConfigUpdateSchema(BaseDashboardSchema):
    deleted = fields.List(fields.String(), allow_none=False)
    modified = fields.List(fields.Raw(), allow_none=False)
    reordered = fields.List(fields.String(), allow_none=False)


class DashboardColorsConfigUpdateSchema(BaseDashboardSchema):
    color_namespace = fields.String(allow_none=True)
    color_scheme = fields.String(allow_none=True)
    map_label_colors = fields.Dict(allow_none=False)
    shared_label_colors = SharedLabelsColorsField()
    label_colors = fields.Dict(allow_none=False)
    color_scheme_domain = fields.List(fields.String(), allow_none=False)


class DashboardScreenshotPostSchema(Schema):
    dataMask = fields.Dict(keys=fields.Str(), values=fields.Raw(), metadata={'description': 'An object representing the data mask.'})
    activeTabs = fields.List(fields.Str(), metadata={'description': 'A list representing active tabs.'})
    anchor = fields.String(metadata={'description': 'A string representing the anchor.'})
    urlParams = fields.List(fields.Tuple((fields.Str(), fields.Str())), metadata={'description': 'A list of tuples, each containing two strings.'})


class ChartFavStarResponseResult(Schema):
    id = fields.Integer(metadata={'description': 'The Chart id'})
    value = fields.Boolean(metadata={'description': 'The FaveStar value'})


class GetFavStarIdsSchema(Schema):
    result = fields.List(fields.Nested(ChartFavStarResponseResult), metadata={'description': 'A list of results for each corresponding chart in the request'})


class ImportV1DashboardSchema(Schema):
    dashboard_title = fields.String(required=True)
    description = fields.String(allow_none=True)
    css = fields.String(allow_none=True)
    slug = fields.String(allow_none=True)
    uuid = fields.UUID(required=True)
    position = fields.Dict()
    metadata = fields.Dict()
    version = fields.String(required=True)
    is_managed_externally = fields.Boolean(allow_none=True, dump_default=False)
    external_url = fields.String(allow_none=True)
    certified_by = fields.String(allow_none=True)
    certification_details = fields.String(allow_none=True)
    published = fields.Boolean(allow_none=True)


class EmbeddedDashboardConfigSchema(Schema):
    allowed_domains = fields.List(fields.String(), required=True)


class EmbeddedDashboardResponseSchema(Schema):
    uuid = fields.String()
    allowed_domains = fields.List(fields.String())
    dashboard_id = fields.String()
    changed_on = fields.DateTime()
    changed_by = fields.Nested(UserSchema)


class DashboardCacheScreenshotResponseSchema(Schema):
    cache_key = fields.String(metadata={'description': 'The cache key'})
    dashboard_url = fields.String(metadata={'description': 'The url to render the dashboard'})
    image_url = fields.String(metadata={'description': 'The url to fetch the screenshot'})
    task_status = fields.String(metadata={'description': 'The status of the async screenshot'})
    task_updated_at = fields.String(metadata={'description': 'The timestamp of the last change in status'})


class CacheScreenshotSchema(Schema):
    dataMask = fields.Dict(keys=fields.Str(), values=fields.Raw(), required=False)
    activeTabs = fields.List(fields.Str(), required=False)
    anchor = fields.Str(required=False)
    urlParams = fields.List(fields.List(fields.Str(), validate=lambda x: len(x) == 2), required=False)