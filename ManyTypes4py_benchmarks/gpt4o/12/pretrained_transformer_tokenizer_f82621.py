import copy
import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple, Iterable
from transformers import PreTrainedTokenizer
from allennlp.common.util import sanitize_wordpiece
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)

@Tokenizer.register('pretrained_transformer')
class PretrainedTransformerTokenizer(Tokenizer):
    def __init__(self, model_name: str, add_special_tokens: bool = True, max_length: Optional[int] = None, tokenizer_kwargs: Optional[Dict[str, Any]] = None, verification_tokens: Optional[Tuple[str, str]] = None) -> None:
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        else:
            tokenizer_kwargs = tokenizer_kwargs.copy()
        tokenizer_kwargs.setdefault('use_fast', True)
        self._tokenizer_kwargs = tokenizer_kwargs
        self._model_name = model_name
        from allennlp.common import cached_transformers
        self.tokenizer: PreTrainedTokenizer = cached_transformers.get_tokenizer(self._model_name, add_special_tokens=False, **self._tokenizer_kwargs)
        self._add_special_tokens = add_special_tokens
        self._max_length = max_length
        self._tokenizer_lowercases = self.tokenizer_lowercases(self.tokenizer)
        if verification_tokens is None:
            try:
                self._reverse_engineer_special_tokens('a', 'b', model_name, tokenizer_kwargs)
            except AssertionError:
                self._reverse_engineer_special_tokens('1', '2', model_name, tokenizer_kwargs)
        else:
            token_a, token_b = verification_tokens
            self._reverse_engineer_special_tokens(token_a, token_b, model_name, tokenizer_kwargs)

    def _reverse_engineer_special_tokens(self, token_a: str, token_b: str, model_name: str, tokenizer_kwargs: Optional[Dict[str, Any]]) -> None:
        self.sequence_pair_start_tokens: List[Token] = []
        self.sequence_pair_mid_tokens: List[Token] = []
        self.sequence_pair_end_tokens: List[Token] = []
        self.sequence_pair_first_token_type_id: Optional[int] = None
        self.sequence_pair_second_token_type_id: Optional[int] = None
        self.single_sequence_start_tokens: List[Token] = []
        self.single_sequence_end_tokens: List[Token] = []
        self.single_sequence_token_type_id: Optional[int] = None
        from allennlp.common import cached_transformers
        tokenizer_with_special_tokens: PreTrainedTokenizer = cached_transformers.get_tokenizer(model_name, add_special_tokens=True, **tokenizer_kwargs or {})
        dummy_output = tokenizer_with_special_tokens.encode_plus(token_a, token_b, add_special_tokens=True, return_token_type_ids=True, return_attention_mask=False)
        if len(dummy_output['token_type_ids']) != len(dummy_output['input_ids']):
            logger.warning('Tokenizer library did not return valid token type ids. We will assume they are all zero.')
            dummy_output['token_type_ids'] = [0] * len(dummy_output['input_ids'])
        dummy_a = self.tokenizer.encode(token_a, add_special_tokens=False)[0]
        assert dummy_a in dummy_output['input_ids']
        dummy_b = self.tokenizer.encode(token_b, add_special_tokens=False)[0]
        assert dummy_b in dummy_output['input_ids']
        assert dummy_a != dummy_b
        seen_dummy_a = False
        seen_dummy_b = False
        for token_id, token_type_id in zip(dummy_output['input_ids'], dummy_output['token_type_ids']):
            if token_id == dummy_a:
                if seen_dummy_a or seen_dummy_b:
                    raise ValueError('Cannot auto-determine the number of special tokens added.')
                seen_dummy_a = True
                assert self.sequence_pair_first_token_type_id is None or self.sequence_pair_first_token_type_id == token_type_id, 'multiple different token type ids found for the first sequence'
                self.sequence_pair_first_token_type_id = token_type_id
                continue
            if token_id == dummy_b:
                if seen_dummy_b:
                    raise ValueError('Cannot auto-determine the number of special tokens added.')
                seen_dummy_b = True
                assert self.sequence_pair_second_token_type_id is None or self.sequence_pair_second_token_type_id == token_type_id, 'multiple different token type ids found for the second sequence'
                self.sequence_pair_second_token_type_id = token_type_id
                continue
            token = Token(tokenizer_with_special_tokens.convert_ids_to_tokens(token_id), text_id=token_id, type_id=token_type_id)
            if not seen_dummy_a:
                self.sequence_pair_start_tokens.append(token)
            elif not seen_dummy_b:
                self.sequence_pair_mid_tokens.append(token)
            else:
                self.sequence_pair_end_tokens.append(token)
        assert len(self.sequence_pair_start_tokens) + len(self.sequence_pair_mid_tokens) + len(self.sequence_pair_end_tokens) == self.tokenizer.num_special_tokens_to_add(pair=True)
        dummy_output = tokenizer_with_special_tokens.encode_plus(token_a, add_special_tokens=True, return_token_type_ids=True, return_attention_mask=False)
        if len(dummy_output['token_type_ids']) != len(dummy_output['input_ids']):
            logger.warning('Tokenizer library did not return valid token type ids. We will assume they are all zero.')
            dummy_output['token_type_ids'] = [0] * len(dummy_output['input_ids'])
        seen_dummy_a = False
        for token_id, token_type_id in zip(dummy_output['input_ids'], dummy_output['token_type_ids']):
            if token_id == dummy_a:
                if seen_dummy_a:
                    raise ValueError('Cannot auto-determine the number of special tokens added.')
                seen_dummy_a = True
                assert self.single_sequence_token_type_id is None or self.single_sequence_token_type_id == token_type_id, 'multiple different token type ids found for the sequence'
                self.single_sequence_token_type_id = token_type_id
                continue
            token = Token(tokenizer_with_special_tokens.convert_ids_to_tokens(token_id), text_id=token_id, type_id=token_type_id)
            if not seen_dummy_a:
                self.single_sequence_start_tokens.append(token)
            else:
                self.single_sequence_end_tokens.append(token)
        assert len(self.single_sequence_start_tokens) + len(self.single_sequence_end_tokens) == self.tokenizer.num_special_tokens_to_add(pair=False)

    @staticmethod
    def tokenizer_lowercases(tokenizer: PreTrainedTokenizer) -> bool:
        tokenized = tokenizer.tokenize('A')
        detokenized = ' '.join(tokenized)
        return 'a' in detokenized

    def tokenize(self, text: str) -> List[Token]:
        max_length = self._max_length
        if max_length is not None and (not self._add_special_tokens):
            max_length += self.num_special_tokens_for_sequence()
        encoded_tokens = self.tokenizer.encode_plus(text=text, add_special_tokens=True, max_length=max_length, truncation=True if max_length is not None else False, return_tensors=None, return_offsets_mapping=self.tokenizer.is_fast, return_attention_mask=False, return_token_type_ids=True, return_special_tokens_mask=True)
        token_ids, token_type_ids, special_tokens_mask, token_offsets = (encoded_tokens['input_ids'], encoded_tokens['token_type_ids'], encoded_tokens['special_tokens_mask'], encoded_tokens.get('offset_mapping'))
        if token_offsets is None:
            token_offsets = self._estimate_character_indices(text, token_ids)
        tokens = []
        for token_id, token_type_id, special_token_mask, offsets in zip(token_ids, token_type_ids, special_tokens_mask, token_offsets):
            if not self._add_special_tokens and special_token_mask == 1:
                continue
            if offsets is None or offsets[0] >= offsets[1]:
                start = None
                end = None
            else:
                start, end = offsets
            tokens.append(Token(text=self.tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=False), text_id=token_id, type_id=token_type_id, idx=start, idx_end=end))
        return tokens

    def _estimate_character_indices(self, text: str, token_ids: List[int]) -> List[Optional[Tuple[int, int]]]:
        token_texts = [sanitize_wordpiece(t) for t in self.tokenizer.convert_ids_to_tokens(token_ids)]
        token_offsets = [None] * len(token_ids)
        if self._tokenizer_lowercases:
            text = text.lower()
            token_texts = [t.lower() for t in token_texts]
        min_allowed_skipped_whitespace = 3
        allowed_skipped_whitespace = min_allowed_skipped_whitespace
        text_index = 0
        token_index = 0
        while text_index < len(text) and token_index < len(token_ids):
            token_text = token_texts[token_index]
            token_start_index = text.find(token_text, text_index)
            if token_start_index < 0:
                token_index += 1
                allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                continue
            non_whitespace_chars_skipped = sum((1 for c in text[text_index:token_start_index] if not c.isspace()))
            if non_whitespace_chars_skipped > allowed_skipped_whitespace:
                token_index += 1
                allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                continue
            allowed_skipped_whitespace = min_allowed_skipped_whitespace
            token_offsets[token_index] = (token_start_index, token_start_index + len(token_text))
            text_index = token_start_index + len(token_text)
            token_index += 1
        return token_offsets

    def _intra_word_tokenize(self, string_tokens: List[str]) -> Tuple[List[Token], List[Optional[Tuple[int, int]]]]:
        tokens = []
        offsets = []
        for token_string in string_tokens:
            wordpieces = self.tokenizer.encode_plus(token_string, add_special_tokens=False, return_tensors=None, return_offsets_mapping=False, return_attention_mask=False)
            wp_ids = wordpieces['input_ids']
            if len(wp_ids) > 0:
                offsets.append((len(tokens), len(tokens) + len(wp_ids) - 1))
                tokens.extend((Token(text=wp_text, text_id=wp_id) for wp_id, wp_text in zip(wp_ids, self.tokenizer.convert_ids_to_tokens(wp_ids))))
            else:
                offsets.append(None)
        return (tokens, offsets)

    @staticmethod
    def _increment_offsets(offsets: List[Optional[Tuple[int, int]]], increment: int) -> List[Optional[Tuple[int, int]]]:
        return [None if offset is None else (offset[0] + increment, offset[1] + increment) for offset in offsets]

    def intra_word_tokenize(self, string_tokens: List[str]) -> Tuple[List[Token], List[Optional[Tuple[int, int]]]]:
        tokens, offsets = self._intra_word_tokenize(string_tokens)
        tokens = self.add_special_tokens(tokens)
        offsets = self._increment_offsets(offsets, len(self.single_sequence_start_tokens))
        return (tokens, offsets)

    def intra_word_tokenize_sentence_pair(self, string_tokens_a: List[str], string_tokens_b: List[str]) -> Tuple[List[Token], List[Optional[Tuple[int, int]]], List[Optional[Tuple[int, int]]]]:
        tokens_a, offsets_a = self._intra_word_tokenize(string_tokens_a)
        tokens_b, offsets_b = self._intra_word_tokenize(string_tokens_b)
        offsets_b = self._increment_offsets(offsets_b, len(self.sequence_pair_start_tokens) + len(tokens_a) + len(self.sequence_pair_mid_tokens))
        tokens_a = self.add_special_tokens(tokens_a, tokens_b)
        offsets_a = self._increment_offsets(offsets_a, len(self.sequence_pair_start_tokens))
        return (tokens_a, offsets_a, offsets_b)

    def add_special_tokens(self, tokens1: List[Token], tokens2: Optional[List[Token]] = None) -> List[Token]:
        def with_new_type_id(tokens: List[Token], type_id: int) -> List[Token]:
            return [dataclasses.replace(t, type_id=type_id) for t in tokens]
        tokens2 = copy.deepcopy(tokens2)
        if tokens2 is None:
            return self.single_sequence_start_tokens + with_new_type_id(tokens1, self.single_sequence_token_type_id) + self.single_sequence_end_tokens
        else:
            return self.sequence_pair_start_tokens + with_new_type_id(tokens1, self.sequence_pair_first_token_type_id) + self.sequence_pair_mid_tokens + with_new_type_id(tokens2, self.sequence_pair_second_token_type_id) + self.sequence_pair_end_tokens

    def num_special_tokens_for_sequence(self) -> int:
        return len(self.single_sequence_start_tokens) + len(self.single_sequence_end_tokens)

    def num_special_tokens_for_pair(self) -> int:
        return len(self.sequence_pair_start_tokens) + len(self.sequence_pair_mid_tokens) + len(self.sequence_pair_end_tokens)

    def _to_params(self) -> Dict[str, Any]:
        return {'type': 'pretrained_transformer', 'model_name': self._model_name, 'add_special_tokens': self._add_special_tokens, 'max_length': self._max_length, 'tokenizer_kwargs': self._tokenizer_kwargs}
