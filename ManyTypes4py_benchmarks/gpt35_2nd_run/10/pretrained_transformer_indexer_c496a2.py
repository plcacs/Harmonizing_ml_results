    def __init__(self, model_name: str, namespace: str = 'tags', max_length: Optional[int] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
    def _add_encoding_to_vocabulary_if_needed(self, vocab: Vocabulary) -> None:
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, Any]:
    def indices_to_tokens(self, indexed_tokens: Dict[str, List[int]], vocabulary: Vocabulary) -> List[Token]:
    def _extract_token_and_type_ids(self, tokens: List[Token]) -> Tuple[List[int], List[int]]:
    def _postprocess_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
    def get_empty_token_list(self) -> Dict[str, List[Any]]:
    def as_padded_tensor_dict(self, tokens: Dict[str, List[Any]], padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
