from typing import Dict, List, Optional, Any, Union
import torch
from torch import Tensor
from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList
from allennlp.data.vocabulary import Vocabulary


def _make_bos_eos(character: int, padding_character: int, beginning_of_word_character: int, end_of_word_character: int, max_word_length: int) -> List[int]:
    char_ids: List[int] = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


class ELMoCharacterMapper:
    """
    Maps individual tokens to sequences of character ids, compatible with ELMo.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.

    We allow to add optional additional special tokens with designated
    character ids with `tokens_to_add`.
    """
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

    def __init__(self, tokens_to_add: Optional[Dict[str, int]] = None) -> None:
        self.tokens_to_add: Dict[str, int] = tokens_to_add or {}

    def convert_word_to_char_ids(self, word: str) -> List[int]:
        if word in self.tokens_to_add:
            char_ids: List[int] = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            char_ids[1] = self.tokens_to_add[word]
            char_ids[2] = ELMoCharacterMapper.end_of_word_character
        elif word == ELMoCharacterMapper.bos_token:
            char_ids = ELMoCharacterMapper.beginning_of_sentence_characters
        elif word == ELMoCharacterMapper.eos_token:
            char_ids = ELMoCharacterMapper.end_of_sentence_characters
        else:
            word_encoded: bytes = word.encode('utf-8', 'ignore')[:ELMoCharacterMapper.max_word_length - 2]
            char_ids = [ELMoCharacterMapper.padding_character] * ELMoCharacterMapper.max_word_length
            char_ids[0] = ELMoCharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = ELMoCharacterMapper.end_of_word_character
        return [c + 1 for c in char_ids]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented  # type: ignore


@TokenIndexer.register('elmo_characters')
class ELMoTokenCharactersIndexer(TokenIndexer):
    """
    Convert a token to an array of character ids to compute ELMo representations.

    Registered as a `TokenIndexer` with name "elmo_characters".

    # Parameters

    namespace : `str`, optional (default=`elmo_characters`)
    tokens_to_add : `Dict[str, int]`, optional (default=`None`)
        If not None, then provides a mapping of special tokens to character
        ids. When using pre-trained models, then the character id must be
        less then 261, and we recommend using un-used ids (e.g. 1-32).
    token_min_padding_length : `int`, optional (default=`0`)
        See :class:`TokenIndexer`.
    """

    def __init__(self, namespace: str = 'elmo_characters', tokens_to_add: Optional[Dict[str, int]] = None, token_min_padding_length: int = 0) -> None:
        super().__init__(token_min_padding_length)
        self._namespace: str = namespace
        self._mapper: ELMoCharacterMapper = ELMoCharacterMapper(tokens_to_add)

    def count_vocab_items(self, token: Token, counter: Dict[str, int]) -> None:
        pass

    def get_empty_token_list(self) -> Dict[str, List[List[int]]]:
        return {'elmo_tokens': []}

    def tokens_to_indices(self, tokens: List[Token], vocabulary: Vocabulary) -> Dict[str, List[List[int]]]:
        return {'elmo_tokens': [self._mapper.convert_word_to_char_ids(t.ensure_text()) for t in tokens]}

    def as_padded_tensor_dict(self, tokens: Dict[str, List[List[int]]], padding_lengths: Dict[str, int]) -> Dict[str, Tensor]:
        def padding_token() -> List[int]:
            return [0] * ELMoCharacterMapper.max_word_length

        padded_tokens: List[List[List[int]]] = pad_sequence_to_length(
            tokens['elmo_tokens'],
            padding_lengths['elmo_tokens'],
            default_value=padding_token
        )
        tensor_dict: Dict[str, Tensor] = {}
        tensor_dict['elmo_tokens'] = torch.LongTensor(padded_tokens)
        return tensor_dict