    def __init__(self, df: DataFrame = DataFrame(), query: str = '', annotation_data: dict = None, applied_template_filters: list = None, applied_filter_columns: list = None, rejected_filter_columns: list = None, status: QueryStatus = None, error_message: str = None, is_loaded: bool = False, stacktrace: str = None, is_cached: bool = None, cache_dttm: Any = None, cache_value: Any = None, sql_rowcount: int = None) -> None:

    def set_query_result(self, key: str, query_result: QueryResult, annotation_data: dict = None, force_query: bool = False, timeout: int = None, datasource_uid: Any = None, region: CacheRegion = CacheRegion.DEFAULT) -> None:

    @classmethod
    def get(cls, key: str, region: CacheRegion = CacheRegion.DEFAULT, force_query: bool = False, force_cached: bool = False) -> QueryCacheManager:

    @staticmethod
    def set(key: str, value: Any, timeout: int = None, datasource_uid: Any = None, region: CacheRegion = CacheRegion.DEFAULT) -> None:

    @staticmethod
    def delete(key: str, region: CacheRegion = CacheRegion.DEFAULT) -> None:

    @staticmethod
    def has(key: str, region: CacheRegion = CacheRegion.DEFAULT) -> bool:
