def _make_bos_eos(character: int, padding_character: int, beginning_of_word_character: int, end_of_word_character: int, max_word_length: int) -> List[int]:

class ELMoCharacterMapper:
    max_word_length: int = 50
    beginning_of_sentence_character: int = 256
    end_of_sentence_character: int = 257
    beginning_of_word_character: int = 258
    end_of_word_character: int = 259
    padding_character: int = 260
    beginning_of_sentence_characters: List[int] = _make_bos_eos(beginning_of_sentence_character, padding_character, beginning_of_word_character, end_of_word_character, max_word_length)
    end_of_sentence_characters: List[int] = _make_bos_eos(end_of_sentence_character, padding_character, beginning_of_word_character, end_of_word_character, max_word_length)
    bos_token: str = '<S>'
    eos_token: str = '</S>'

    def __init__(self, tokens_to_add: Dict[str, int] = None):

    def convert_word_to_char_ids(self, word: str) -> List[int]:

@TokenIndexer.register('elmo_characters')
class ELMoTokenCharactersIndexer(TokenIndexer):
    def __init__(self, namespace: str = 'elmo_characters', tokens_to_add: Dict[str, int] = None, token_min_padding_length: int = 0):

    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]) -> None:

    def get_empty_token_list(self) -> Dict[str, List[List[int]]]:

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[List[int]]]:

    def as_padded_tensor_dict(self, tokens: Dict[str, List[List[int]]], padding_lengths: Dict[str, int]) -> Dict[str, torch.Tensor]:
