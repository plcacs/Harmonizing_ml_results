    def _is_divider(line: str) -> bool:
        empty_line: bool = line.strip() == ''
        if empty_line:
            return True
        else:
            first_token: str = line.split()[0]
            if first_token == '-DOCSTART-':
                return True
            else:
                return False

    def __init__(self, token_indexers: Optional[Dict[str, TokenIndexer]] = None, tag_label: str = 'ner', feature_labels: Sequence[str] = (), convert_to_coding_scheme: Optional[str] = None, label_namespace: str = 'labels', **kwargs):
    
    def _read(self, file_path: str) -> Iterable[Instance]:
    
    def text_to_instance(self, tokens: List[Token], pos_tags: Optional[List[str]] = None, chunk_tags: Optional[List[str]] = None, ner_tags: Optional[List[str]] = None) -> Instance:
