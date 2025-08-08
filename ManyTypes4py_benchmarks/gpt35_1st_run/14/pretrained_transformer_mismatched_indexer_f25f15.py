class PretrainedTransformerMismatchedIndexer(TokenIndexer):
    def __init__(self, model_name: str, namespace: str = 'tags', max_length: Optional[int] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
    def get_empty_token_list(self) -> IndexedTokenList:
    def as_padded_tensor_dict(self, tokens: Dict[str, Any], padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
    def __eq__(self, other: Any) -> bool:
