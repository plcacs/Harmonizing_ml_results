def _update_function_tags(
    self, function_arn: str, requested_tags: Dict[str, str]
) -> None:
    remote_tags = self._client('lambda').list_tags(Resource=function_arn)[
        'Tags'
    ]
    self._remove_unrequested_remote_tags(
        function_arn, requested_tags, remote_tags
    )
    self._add_missing_or_differing_value_requested_tags(
        function_arn, requested_tags, remote_tags
    )
