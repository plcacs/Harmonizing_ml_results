    def __init__(self, tokens_to_add: Dict[str, int] = None):
    
    def convert_word_to_char_ids(self, word: str) -> List[int]:
    
    def __eq__(self, other: object) -> bool:
    
    def __init__(self, namespace: str = 'elmo_characters', tokens_to_add: Dict[str, int] = None, token_min_padding_length: int = 0):
    
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:
    
    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[List[int]]]:
    
    def as_padded_tensor_dict(self, tokens: Dict[str, List[List[int]], padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
