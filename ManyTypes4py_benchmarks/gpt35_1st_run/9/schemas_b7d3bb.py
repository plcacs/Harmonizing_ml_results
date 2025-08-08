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
class DashboardJSONMetadataSchema(Schema):
class UserSchema(Schema):
class RolesSchema(Schema):
class TagSchema(Schema):
class DashboardGetResponseSchema(Schema):
class DatabaseSchema(Schema):
class DashboardDatasetSchema(Schema):
class TabSchema(Schema):
class TabsPayloadSchema(Schema):
class BaseDashboardSchema(Schema):
class DashboardPostSchema(BaseDashboardSchema):
class DashboardCopySchema(Schema):
class DashboardPutSchema(BaseDashboardSchema):
class DashboardNativeFiltersConfigUpdateSchema(BaseDashboardSchema):
class DashboardColorsConfigUpdateSchema(BaseDashboardSchema):
class DashboardScreenshotPostSchema(Schema):
class ChartFavStarResponseResult(Schema):
class GetFavStarIdsSchema(Schema):
class ImportV1DashboardSchema(Schema):
class EmbeddedDashboardConfigSchema(Schema):
class EmbeddedDashboardResponseSchema(Schema):
class DashboardCacheScreenshotResponseSchema(Schema):
class CacheScreenshotSchema(Schema):
