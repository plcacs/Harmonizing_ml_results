    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:
        pass

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, Any]:
        ...

    def indices_to_tokens(self, indexed_tokens: IndexedTokenList, vocabulary: Vocabulary) -> List[Token]:
        ...

    def get_empty_token_list(self) -> Dict[str, List[Any]]:
        ...

    def as_padded_tensor_dict(self, tokens: Dict[str, List[Any]], padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
        ...
