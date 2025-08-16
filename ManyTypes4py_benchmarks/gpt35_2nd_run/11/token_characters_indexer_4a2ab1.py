    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[List[int]]]:
    def get_padding_lengths(self, indexed_tokens: Dict[str, List[List[int]]]) -> Dict[str, int]:
    def as_padded_tensor_dict(self, tokens: Dict[str, List[List[int]]], padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
