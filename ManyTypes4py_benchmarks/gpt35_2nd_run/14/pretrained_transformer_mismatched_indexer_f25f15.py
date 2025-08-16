    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> IndexedTokenList:
    def as_padded_tensor_dict(self, tokens: Dict[str, Any], padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
