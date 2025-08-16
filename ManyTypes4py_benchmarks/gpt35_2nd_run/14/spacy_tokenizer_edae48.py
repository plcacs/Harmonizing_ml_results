    def __init__(self, language: str = 'en_core_web_sm', pos_tags: bool = True, parse: bool = False, ner: bool = False, keep_spacy_tokens: bool = False, split_on_spaces: bool = False, start_tokens: Optional[List[str]] = None, end_tokens: Optional[List[str]] = None) -> None:
    
    def _sanitize(self, tokens: List[Doc]) -> List[Token]:
    
    def batch_tokenize(self, texts: List[str]) -> List[List[Token]]:
    
    def tokenize(self, text: str) -> List[Token]:
    
    def _to_params(self) -> dict:
