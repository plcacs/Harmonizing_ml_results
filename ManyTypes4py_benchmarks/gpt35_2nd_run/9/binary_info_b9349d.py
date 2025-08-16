    def __init__(self, bucket_name: str, object_key: str, yara_analyzer: YaraAnalyzer) -> None:
    def _download_from_s3(self) -> None:
    def __enter__(self) -> 'BinaryInfo':
    def __exit__(self, exception_type: Any, exception_value: Any, traceback: Any) -> None:
    @property
    def matched_rule_ids(self) -> Set[str]:
    @property
    def filepath(self) -> str:
    def save_matches_and_alert(self, analyzer_version: str, dynamo_table_name: str, sns_topic_arn: str, sns_enabled: bool = True) -> None:
    def publish_negative_match_result(self, sns_topic_arn: str) -> None:
    def summary(self) -> Dict[str, Any]:
