    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None, tokenizer: Tokenizer = None, segment_sentences: bool = False, max_sequence_length: int = None, skip_label_indexing: bool = False, text_key: str = 'text', label_key: str = 'label', **kwargs: Any) -> None:
    
    def _read(self, file_path: str) -> Iterable[Instance]:
    
    def _truncate(self, tokens: List[str]) -> List[str]:
    
    def text_to_instance(self, text: str, label: Union[str, int] = None) -> Instance:
    
    def apply_token_indexers(self, instance: Instance) -> None:
