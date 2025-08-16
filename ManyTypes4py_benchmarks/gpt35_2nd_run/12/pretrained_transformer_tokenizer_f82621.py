    def __init__(self, model_name: str, add_special_tokens: bool = True, max_length: Optional[int] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None, verification_tokens: Optional[Tuple[str, str]] = None) -> None:
    
    def _reverse_engineer_special_tokens(self, token_a: str, token_b: str, model_name: str, tokenizer_kwargs: Dict[str, Any]) -> None:
    
    def tokenizer_lowercases(self, tokenizer: PreTrainedTokenizer) -> bool:
    
    def tokenize(self, text: str) -> List[Token]:
    
    def _estimate_character_indices(self, text: str, token_ids: List[int]) -> List[Optional[Tuple[int, int]]]:
    
    def _intra_word_tokenize(self, string_tokens: List[str]) -> Tuple[List[Token], List[Optional[Tuple[int, int]]]]:
    
    def _increment_offsets(self, offsets: List[Optional[Tuple[int, int]]], increment: int) -> List[Optional[Tuple[int, int]]]:
    
    def intra_word_tokenize(self, string_tokens: List[str]) -> Tuple[List[Token], List[Optional[Tuple[int, int]]]]:
    
    def intra_word_tokenize_sentence_pair(self, string_tokens_a: List[str], string_tokens_b: List[str]) -> Tuple[List[Token], List[Optional[Tuple[int, int]]], List[Optional[Tuple[int, int]]]]:
    
    def add_special_tokens(self, tokens1: List[Token], tokens2: Optional[List[Token]]) -> List[Token]:
    
    def num_special_tokens_for_sequence(self) -> int:
    
    def num_special_tokens_for_pair(self) -> int:
    
    def _to_params(self) -> Dict[str, Any]:
