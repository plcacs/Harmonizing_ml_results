from typing import Dict, List, Union, Iterator, Optional
import logging
import json
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
from allennlp.data.tokenizers.token import Token
logger = logging.getLogger(__name__)


@DatasetReader.register('text_classification_json')
class TextClassificationJsonReader(DatasetReader):

    def __init__(self, token_indexers=None, tokenizer=None,
        segment_sentences=False, max_sequence_length=None,
        skip_label_indexing=False, text_key='text', label_key='label', **kwargs
        ):
        super().__init__(manual_distributed_sharding=True,
            manual_multiprocess_sharding=True, **kwargs)
        self._tokenizer: Tokenizer = tokenizer or SpacyTokenizer()
        self._segment_sentences: bool = segment_sentences
        self._max_sequence_length: Optional[int] = max_sequence_length
        self._skip_label_indexing: bool = skip_label_indexing
        self._token_indexers: Dict[str, TokenIndexer] = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        self._text_key: str = text_key
        self._label_key: str = label_key
        self._sentence_segmenter: Optional[SpacySentenceSplitter
            ] = SpacySentenceSplitter() if self._segment_sentences else None

    def _read(self, file_path):
        with open(cached_path(file_path), 'r') as data_file:
            for line in self.shard_iterable(data_file.readlines()):
                if not line:
                    continue
                items: Dict[str, Union[str, int]] = json.loads(line)
                text: str = items[self._text_key]
                label: Optional[Union[str, int]] = items.get(self._label_key)
                if label is not None:
                    if self._skip_label_indexing:
                        try:
                            label = int(label)
                        except ValueError:
                            raise ValueError(
                                'Labels must be integers if skip_label_indexing is True.'
                                )
                    else:
                        label = str(label)
                yield self.text_to_instance(text=text, label=label)

    def _truncate(self, tokens):
        if self._max_sequence_length is not None and len(tokens
            ) > self._max_sequence_length:
            tokens = tokens[:self._max_sequence_length]
        return tokens

    def text_to_instance(self, text, label=None):
        fields: Dict[str, Field] = {}
        if self._segment_sentences and self._sentence_segmenter is not None:
            sentences: List[Field] = []
            sentence_splits: List[str
                ] = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                word_tokens: List[Token] = self._tokenizer.tokenize(sentence)
                if self._max_sequence_length is not None:
                    word_tokens = self._truncate(word_tokens)
                sentences.append(TextField(word_tokens))
            fields['tokens'] = ListField(sentences)
        else:
            tokens: List[Token] = self._tokenizer.tokenize(text)
            if self._max_sequence_length is not None:
                tokens = self._truncate(tokens)
            fields['tokens'] = TextField(tokens)
        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=self.
                _skip_label_indexing)
        return Instance(fields)

    def apply_token_indexers(self, instance):
        if self._segment_sentences:
            for text_field in cast(ListField, instance.fields['tokens']
                ).fields:
                text_field.token_indexers = self._token_indexers
        else:
            cast(TextField, instance.fields['tokens']
                ).token_indexers = self._token_indexers
