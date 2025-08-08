from typing import TypeVar, cast

def is_analytics_ready(realm: Realm) -> bool:
def render_stats(request: HttpRequest, data_url_suffix: str, realm: Optional[Realm], *, title: Optional[str] = None, analytics_ready: bool = True) -> HttpResponse:
def stats(request: HttpRequest) -> HttpResponse:
def stats_for_realm(request: HttpRequest, *, realm_str: str) -> HttpResponse:
def stats_for_remote_realm(request: HttpRequest, *, remote_server_id: int, remote_realm_id: int) -> HttpResponse:
def get_chart_data_for_realm(request: HttpRequest, user_profile: UserProfile, /, *, realm_str: str, chart_name: str, min_length: Optional[NonNegativeInt] = None, start: Optional[datetime] = None, end: Optional[datetime] = None) -> HttpResponse:
def get_chart_data_for_stream(request: HttpRequest, user_profile: UserProfile, *, stream_id: int, chart_name: str, min_length: Optional[NonNegativeInt] = None, start: Optional[datetime] = None, end: Optional[datetime] = None) -> HttpResponse:
def get_chart_data_for_remote_realm(request: HttpRequest, user_profile: UserProfile, /, *, remote_server_id: int, remote_realm_id: int, chart_name: str, min_length: Optional[NonNegativeInt] = None, start: Optional[datetime] = None, end: Optional[datetime] = None) -> HttpResponse:
def stats_for_installation(request: HttpRequest) -> HttpResponse:
def stats_for_remote_installation(request: HttpRequest, remote_server_id: int) -> HttpResponse:
def get_chart_data_for_installation(request: HttpRequest, user_profile: UserProfile, /, *, chart_name: str, min_length: Optional[NonNegativeInt] = None, start: Optional[datetime] = None, end: Optional[datetime] = None) -> HttpResponse:
def get_chart_data_for_remote_installation(request: HttpRequest, user_profile: UserProfile, /, *, remote_server_id: int, chart_name: str, min_length: Optional[NonNegativeInt] = None, start: Optional[datetime] = None, end: Optional[datetime] = None) -> HttpResponse:
def get_chart_data(request: HttpRequest, user_profile: UserProfile, *, chart_name: str, min_length: Optional[NonNegativeInt] = None, start: Optional[datetime] = None, end: Optional[datetime] = None) -> HttpResponse:
def do_get_chart_data(request: HttpRequest, user_profile: UserProfile, *, chart_name: str, min_length: Optional[NonNegativeInt] = None, start: Optional[datetime] = None, end: Optional[datetime] = None, realm: Optional[Realm] = None, for_installation: bool = False, remote: bool = False, remote_realm_id: Optional[int] = None, server: Optional[RemoteZulipServer] = None, stream: Optional[Stream] = None) -> HttpResponse:
def sort_by_totals(value_arrays: dict[str, dict[Optional[str], list[int]]) -> list[str]:
def sort_client_labels(data: dict[str, dict[str, list[int]]) -> list[str]:
def table_filtered_to_id(table: Type[CountT], key_id: int) -> QuerySet[CountT]:
def client_label_map(name: str) -> str:
def rewrite_client_arrays(value_arrays: dict[str, list[int]]) -> dict[str, list[int]]:
def get_time_series_by_subgroup(stat: CountStat, table: Type[BaseCount], key_id: int, end_times: list[datetime], subgroup_to_label: dict[CountStat, dict[Optional[str], str]], include_empty_subgroups: bool) -> dict[str, list[int]]:
