# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from marshmallow import fields, post_dump, post_load, pre_load, Schema
from marshmallow.validate import Length, ValidationError

from superset import security_manager
from superset.tags.models import TagType
from superset.utils import json

get_delete_ids_schema: Dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
get_export_ids_schema: Dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
get_fav_star_ids_schema: Dict[str, Any] = {"type": "array", "items": {"type": "integer"}}
thumbnail_query_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {"force": {"type": "boolean"}},
}
width_height_schema: Dict[str, Any] = {
    "type": "array",
    "items": {"type": "integer"},
}
screenshot_query_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "force": {"type": "boolean"},
        "permalink": {"type": "string"},
        "window_size": width_height_schema,
        "thumb_size": width_height_schema,
    },
}
dashboard_title_description: str = "A title for the dashboard."
slug_description: str = "Unique identifying part for the web address of the dashboard."
owners_description: str = (
    "Owner are users ids allowed to delete or change this dashboard. "
    "If left empty you will be one of the owners of the dashboard."
)
roles_description: str = (
    "Roles is a list which defines access to the dashboard. "
    "These roles are always applied in addition to restrictions on dataset "
    "level access. "
    "If no roles defined then the dashboard is available to all roles."
)
position_json_description: str = (
    "This json object describes the positioning of the widgets "
    "in the dashboard. It is dynamically generated when "
    "adjusting the widgets size and positions by using "
    "drag & drop in the dashboard view"
)
css_description: str = "Override CSS for the dashboard."
json_metadata_description: str = (
    "This JSON object is generated dynamically when clicking "
    "the save or overwrite button in the dashboard view. "
    "It is exposed here for reference and for power users who may want to alter "
    " specific parameters."
)
published_description: str = (
    "Determines whether or not this dashboard is visible in "
    "the list of all dashboards."
)
charts_description: str = (
    "The names of the dashboard's charts. Names are used for legacy reasons."
)
certified_by_description: str = "Person or group that has certified this dashboard"
certification_details_description: str = "Details of the certification"
tags_description: str = "Tags to be associated with the dashboard"

openapi_spec_methods_override: Dict[str, Dict[str, Dict[str, str]]] = {
    "get": {"get": {"summary": "Get a dashboard detail information"}},
    "get_list": {
        "get": {
            "summary": "Get a list of dashboards",
            "description": "Gets a list of dashboards, use Rison or JSON query "
            "parameters for filtering, sorting, pagination and "
            " for selecting specific columns and metadata.",
        }
    },
    "info": {"get": {"summary": "Get metadata information about this API resource"}},
    "related": {
        "get": {"description": "Get a list of all possible owners for a dashboard."}
    },
}


def validate_json(value: Union[bytes, bytearray, str]) -> None:
    try:
        json.validate_json(value)
    except json.JSONDecodeError as ex:
        raise ValidationError("JSON not valid") from ex


def validate_json_metadata(value: Union[bytes, bytearray, str]) -> None:
    if not value:
        return
    try:
        value_obj = json.loads(value)
    except json.JSONDecodeError as ex:
        raise ValidationError("JSON not valid") from ex
    errors = DashboardJSONMetadataSchema().validate(value_obj, partial=False)
    if errors:
        raise ValidationError(errors)


class SharedLabelsColorsField(fields.Field):
    """
    A custom field that accepts either a list of strings or a dictionary.
    """

    def _deserialize(
        self,
        value: Union[List[str], Dict[str, str]],
        attr: Optional[str],
        data: Optional[Mapping[str, Any]],
        **kwargs: Dict[str, Any],
    ) -> List[str]:
        if isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                return value
        elif isinstance(value, dict):
            # Enforce list (for backward compatibility)
            return []

        raise ValidationError("Not a valid list")


class DashboardJSONMetadataSchema(Schema):
    # native_filter_configuration is for dashboard-native filters
    native_filter_configuration: fields.List = fields.List(fields.Dict(), allow_none=True)
    # chart_configuration for now keeps data about cross-filter scoping for charts
    chart_configuration: fields.Dict = fields.Dict()
    # global_chart_configuration keeps data about global cross-filter scoping
    # for charts - can be overridden by chart_configuration for each chart
    global_chart_configuration: fields.Dict = fields.Dict()
    timed_refresh_immune_slices: fields.List = fields.List(fields.Integer())
    # deprecated wrt dashboard-native filters
    filter_scopes: fields.Dict = fields.Dict()
    expanded_slices: fields.Dict = fields.Dict()
    refresh_frequency: fields.Integer = fields.Integer()
    # deprecated wrt dashboard-native filters
    default_filters: fields.Str = fields.Str()
    stagger_refresh: fields.Boolean = fields.Boolean()
    stagger_time: fields.Integer = fields.Integer()
    color_scheme: fields.Str = fields.Str(allow_none=True)
    color_namespace: fields.Str = fields.Str(allow_none=True)
    positions: fields.Dict = fields.Dict(allow_none=True)
    label_colors: fields.Dict = fields.Dict()
    shared_label_colors: SharedLabelsColorsField = SharedLabelsColorsField()
    map_label_colors: fields.Dict = fields.Dict()
    color_scheme_domain: fields.List = fields.List(fields.Str())
    cross_filters_enabled: fields.Boolean = fields.Boolean(dump_default=True)
    # used for v0 import/export
    import_time: fields.Integer = fields.Integer()
    remote_id: fields.Integer = fields.Integer()
    filter_bar_orientation: fields.Str = fields.Str(allow_none=True)
    native_filter_migration: fields.Dict = fields.Dict()

    @pre_load
    def remove_show_native_filters(  # pylint: disable=unused-argument
        self,
        data: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Remove ``show_native_filters`` from the JSON metadata.

        This field was removed in https://github.com/apache/superset/pull/23228, but might
        be present in old exports.
        """  # noqa: E501
        if "show_native_filters" in data:
            del data["show_native_filters"]

        return data


class UserSchema(Schema):
    id: fields.Int = fields.Int()
    username: fields.String = fields.String()
    first_name: fields.String = fields.String()
    last_name: fields.String = fields.String()


class RolesSchema(Schema):
    id: fields.Int = fields.Int()
    name: fields.String = fields.String()


class TagSchema(Schema):
    id: fields.Int = fields.Int()
    name: fields.String = fields.String()
    type: fields.Enum = fields.Enum(TagType, by_value=True)


class DashboardGetResponseSchema(Schema):
    id: fields.Int = fields.Int()
    slug: fields.String = fields.String()
    url: fields.String = fields.String()
    dashboard_title: fields.String = fields.String(
        metadata={"description": dashboard_title_description}
    )
    thumbnail_url: fields.String = fields.String(allow_none=True)
    published: fields.Boolean = fields.Boolean()
    css: fields.String = fields.String(metadata={"description": css_description})
    json_metadata: fields.String = fields.String(metadata={"description": json_metadata_description})
    position_json: fields.String = fields.String(metadata={"description": position_json_description})
    certified_by: fields.String = fields.String(metadata={"description": certified_by_description})
    certification_details: fields.String = fields.String(
        metadata={"description": certification_details_description}
    )
    changed_by_name: fields.String = fields.String()
    changed_by: fields.Nested = fields.Nested(UserSchema(exclude=["username"]))
    changed_on: fields.DateTime = fields.DateTime()
    created_by: fields.Nested = fields.Nested(UserSchema(exclude=["username"]))
    charts: fields.List = fields.List(fields.String(metadata={"description": charts_description}))
    owners: fields.List = fields.List(fields.Nested(UserSchema(exclude=["username"])))
    roles: fields.List = fields.List(fields.Nested(RolesSchema))
    tags: fields.Nested = fields.Nested(TagSchema, many=True)
    changed_on_humanized: fields.String = fields.String(data_key="changed_on_delta_humanized")
    created_on_humanized: fields.String = fields.String(data_key="created_on_delta_humanized")
    is_managed_externally: fields.Boolean = fields.Boolean(allow_none=True, dump_default=False)

    # pylint: disable=unused-argument
    @post_dump()
    def post_dump(self, serialized: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if security_manager.is_guest_user():
            del serialized["owners"]
            del serialized["changed_by_name"]
            del serialized["changed_by"]
        return serialized


class DatabaseSchema(Schema):
    id: fields.Int = fields.Int()
    name: fields.String = fields.String()
    backend: fields.String = fields.String()
    allows_subquery: fields.Bool = fields.Bool()
    allows_cost_estimate: fields.Bool = fields.Bool()
    allows_virtual_table_explore: fields.Bool = fields.Bool()
    disable_data_preview: fields.Bool = fields.Bool()
    disable_drill_to_detail: fields.Bool = fields.Bool()
    allow_multi_catalog: fields.Bool = fields.Bool()
    explore_database_id: fields.Int = fields.Int()


class DashboardDatasetSchema(Schema):
    id: fields.Int = fields.Int()
    uid: fields.Str = fields.Str()
    column_formats: fields.Dict = fields.Dict()
    currency_formats: fields.Dict = fields.Dict()
    database: fields.Nested = fields.Nested(DatabaseSchema)
    default_endpoint: fields.String = fields.String()
    filter_select: fields.Bool = fields.Bool()
    filter_select_enabled: fields.Bool = fields.Bool()
    is_sqllab_view: fields.Bool = fields.Bool()
    name: fields.Str = fields.Str()
    datasource_name: fields.Str = fields.Str()
    table_name: fields.Str = fields.Str()
    type: fields.Str = fields.Str()
    schema: fields.Str = fields.Str()
    offset: fields.Int = fields.Int()
    cache_timeout: fields.Int = fields.Int()
    params: fields.Str = fields.Str()
    perm: fields.Str = fields.Str()
    edit_url: fields.Str = fields.Str()
    sql: fields.Str = fields.Str()
    select_star: fields.Str = fields.Str()
    main_dttm_col: fields.Str = fields.Str()
    health_check_message: fields.Str = fields.Str()
    fetch_values_predicate: fields.Str = fields.Str()
    template_params: fields.Str = fields.Str()
    owners: fields.List = fields.List(fields.Dict())
    columns: fields.List = fields.List(fields.Dict())
    column_types: fields.List = fields.List(fields.Int())
    column_names: fields.List = fields.List(fields.Str())
    metrics: fields.List = fields.List(fields.Dict())
    order_by_choices: fields.List = fields.List(fields.List(fields.Str()))
    verbose_map: fields.Dict = fields.Dict(fields.Str(), fields.Str())
    time_grain_sqla: fields.List = fields.List(fields.List(fields.Str()))
    granularity_sqla: fields.List = fields.List(fields.List(fields.Str()))
    normalize_columns: fields.Bool = fields.Bool()
    always_filter_main_dttm: fields.Bool = fields.Bool()

    # pylint: disable=unused-argument
    @post_dump()
    def post_dump(self, serialized: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if security_manager.is_guest_user():
            del serialized["owners"]
            del serialized["database"]
        return serialized


class TabSchema(Schema):
    # pylint: disable=W0108
    children: fields.List = fields.List(fields.Nested(lambda: TabSchema()))
    value: fields.Str = fields.Str()
    title: fields.Str = fields.Str()
    parents: fields.List = fields.List(fields.Str())


class TabsPayloadSchema(Schema):
    all_tabs: fields.Dict = fields.Dict(keys=fields.String(), values=fields.String())
    tab_tree: fields.List = fields.List(fields.Nested(lambda: TabSchema))


class BaseDashboardSchema(Schema):
    # pylint: disable=unused-argument
    @post_load
    def post_load(self, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        if data.get("slug"):
            data["slug"] = data["slug"].strip()
            data["slug"] = data["slug"].replace(" ", "-")
            data["slug"] = re.sub(r"[^\w\-]+", "", data["slug"])
        return data


class DashboardPostSchema(BaseDashboardSchema):
    dashboard_title: fields.String = fields.String(
        metadata={"description": dashboard_title_description},
        allow_none=True,
        validate=Length(0, 500),
    )
    slug: fields.String = fields.String(
        metadata={"description": slug_description},
        allow_none=True,
        validate=[Length(1, 255)],
    )
    owners: fields.List = fields.List(fields.Integer(metadata={"description": owners_description}))
    roles: fields.List = fields.List(fields.Integer(metadata={"description": roles_description}))
    position_json: fields.String = fields.String(
        metadata={"description": position_json_description}, validate=validate_json
    )
    css: fields.String = fields.String(metadata={"description": css_description})
    json_metadata: fields.String = fields.String(
        metadata={"description": json_metadata_description},
        validate=validate_json_metadata,
    )
    published: fields.Boolean = fields.Boolean(metadata={"description": published_description})
    certified_by: fields.String = fields.String(
        metadata={"description": certified_by_description}, allow_none=True
    )
    certification_details: fields.String = fields.String(
        metadata={"description": certification_details_description}, allow_none=True
    )
    is_managed_externally: fields.Boolean = fields.Boolean(allow_none=True, dump_default=False)
    external_url: fields.String = fields.String(allow_none=True)


class DashboardCopySchema(Schema):
    dashboard_title: fields.String = fields.String(
        metadata={"description": dashboard_title_description},
        allow_none=True,
        validate=Length(0, 500),
    )
    css: fields.String = fields.String(metadata={"description": css_description})
    json_metadata: fields.String = fields.String(
        metadata={"description": json_metadata_description},
        validate=validate_json_metadata,
        required=True,
    )
    duplicate_slices: fields.Boolean = fields.Boolean(
        metadata={
            "description": "Whether or not to also copy all charts on the dashboard"
        }
    )


class DashboardPutSchema(BaseDashboardSchema):
    dashboard_title: fields.String = fields.String(
        metadata={"description": dashboard_title_description},
        allow_none=True,
        validate=Length(0, 500),
    )
    slug: fields.String = fields.String(
        metadata={"description": slug_description},
        allow_none=True,
        validate=Length(0, 255),
    )
    owners: fields.List = fields.List(
        fields.Integer(metadata={"description": owners_description}, allow_none=True)
    )
    roles: fields.List = fields.List(
        fields.Integer(metadata={"description": roles_description}, allow_none=True)
    )
    position_json: fields.String = fields.String(
        metadata={"description": position_json_description},
        allow_none=True,
        validate=validate_json,
    )
    css: fields.String = fields.String(metadata={"description": css_description}, allow_none=True)
    json_metadata: fields.String = fields.String(
        metadata={"description": json_metadata_description},
        allow_none=True,
        validate=validate_json_metadata,
    )
    published: fields.Boolean = fields.Boolean(
        metadata={"description": published_description}, allow_none=True
    )
    certified_by: fields.String = fields.String(
        metadata={"description": certified_by_description}, allow_none=True
    )
    certification_details: fields.String = fields.String(
        metadata={"description": certification_details_description}, allow_none=True
    )
    is_managed_externally: fields.Boolean = fields.Boolean(allow_none=True, dump_default=False)
    external_url: fields.String = fields.String(allow_none=True)
    tags: fields.List = fields.List(
        fields.Integer(metadata={"description": tags_description}, allow_none=True)
    )


class