    def __init__(self, word_tag_delimiter: str = DEFAULT_WORD_TAG_DELIMITER, token_delimiter: str = None, token_indexers: Dict[str, TokenIndexer] = None, **kwargs: Any) -> None:
    def _read(self, file_path: str) -> Iterable[Instance]:
    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
