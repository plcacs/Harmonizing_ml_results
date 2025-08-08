    def __init__(self, namespace: str = 'token_characters', character_tokenizer: CharacterTokenizer = CharacterTokenizer(), start_tokens: List[str] = None, end_tokens: List[str] = None, min_padding_length: int = 0, token_min_padding_length: int = 0) -> None:

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[List[int]]]:

    def get_padding_lengths(self, indexed_tokens: Dict[str, List[List[int]]]) -> Dict[str, int]:

    def as_padded_tensor_dict(self, tokens: Dict[str, List[List[int]]], padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:

    def get_empty_token_list(self) -> Dict[str, List[List[int]]]:
