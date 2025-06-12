from typing import Dict, List, Optional, Any
import logging
import json
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter
logger = logging.getLogger(__name__)

@DatasetReader.register('text_classification_json')
class TextClassificationJsonReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        tokenizer: Optional[Tokenizer] = None,
        segment_sentences: bool = False,
        max_sequence_length: Optional[int] = None,
        skip_label_indexing: bool = False,
        text_key: str = 'text',
        label_key: str = 'label',
        **kwargs: Any
    ) -> None:
        super().__init__(manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._segment_sentences = segment_sentences
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._text_key = text_key
        self._label_key = label_key
        if self._segment_sentences:
            self._sentence_segmenter = SpacySentenceSplitter()

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(cached_path(file_path), 'r') as data_file:
            for line in self.shard_iterable(data_file.readlines()):
                if not line:
                    continue
                items = json.loads(line)
                text = items[self._text_key]
                label = items.get(self._label_key)
                if label is not None:
                    if self._skip_label_indexing:
                        try:
                            label = int(label)
                        except ValueError:
                            raise ValueError('Labels must be integers if skip_label_indexing is True.')
                    else:
                        label = str(label)
                yield self.text_to_instance(text=text, label=label)

    def _truncate(self, tokens: List[Any]) -> List[Any]:
        if len(tokens) > self._max_sequence_length:
            tokens = tokens[:self._max_sequence_length]
        return tokens

    def text_to_instance(self, text: str, label: Optional[Union[str, int]] = None) -> Instance:
        fields: Dict[str, Field] = {}
        if self._segment_sentences:
            sentences: List[TextField] = []
            sentence_splits = self._sentence_segmenter.split_sentences(text)
            for sentence in sentence_splits:
                word_tokens = self._tokenizer.tokenize(sentence)
                if self._max_sequence_length is not None:
                    word_tokens = self._truncate(word_tokens)
                sentences.append(TextField(word_tokens))
            fields['tokens'] = ListField(sentences)
        else:
            tokens = self._tokenizer.tokenize(text)
            if self._max_sequence_length is not None:
                tokens = self._truncate(tokens)
            fields['tokens'] = TextField(tokens)
        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=self._skip_label_indexing)
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        if self._segment_sentences:
            for text_field in instance.fields['tokens']:
                text_field._token_indexers = self._token_indexers
        else:
            instance.fields['tokens']._token_indexers = self._token_indexers
