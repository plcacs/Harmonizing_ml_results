from typing import Dict, List, Optional, Sequence, Iterable, Any
import itertools
import logging
import warnings
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader, PathOrStr
from allennlp.data.dataset_readers.dataset_utils import to_bioul
from allennlp.data.fields import TextField, SequenceLabelField, Field, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)

def _is_divider(line: str) -> bool:
    empty_line: bool = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token: str = line.split()[0]
        if first_token == '-DOCSTART-':
            return True
        else:
            return False

@DatasetReader.register('conll2003')
class Conll2003DatasetReader(DatasetReader):
    _VALID_LABELS = {'ner', 'pos', 'chunk'}

    def __init__(
        self,
        token_indexers: Optional[Dict[str, TokenIndexer]] = None,
        tag_label: str = 'ner',
        feature_labels: Sequence[str] = (),
        convert_to_coding_scheme: Optional[str] = None,
        label_namespace: str = 'labels',
        **kwargs: Any
    ) -> None:
        if 'coding_scheme' in kwargs:
            warnings.warn('`coding_scheme` is deprecated.', DeprecationWarning)
            coding_scheme: Any = kwargs.pop('coding_scheme')
            if coding_scheme not in ('IOB1', 'BIOUL'):
                raise ConfigurationError('unknown coding_scheme: {}'.format(coding_scheme))
            if coding_scheme == 'IOB1':
                convert_to_coding_scheme = None
            else:
                convert_to_coding_scheme = coding_scheme
        super().__init__(manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs)
        self._token_indexers: Dict[str, TokenIndexer] = token_indexers or {'tokens': SingleIdTokenIndexer()}
        if tag_label is not None and tag_label not in self._VALID_LABELS:
            raise ConfigurationError('unknown tag label type: {}'.format(tag_label))
        for label in feature_labels:
            if label not in self._VALID_LABELS:
                raise ConfigurationError('unknown feature label type: {}'.format(label))
        if convert_to_coding_scheme not in (None, 'BIOUL'):
            raise ConfigurationError('unknown convert_to_coding_scheme: {}'.format(convert_to_coding_scheme))
        self.tag_label: str = tag_label
        self.feature_labels: set = set(feature_labels)
        self.convert_to_coding_scheme: Optional[str] = convert_to_coding_scheme
        self.label_namespace: str = label_namespace
        self._original_coding_scheme: str = 'IOB1'

    def _read(self, file_path: PathOrStr) -> Iterable[Instance]:
        file_path = cached_path(file_path)
        with open(file_path, 'r') as data_file:
            logger.info('Reading instances from lines in file at: %s', file_path)
            line_chunks: Iterable[Iterable[str]] = (
                lines for is_divider, lines in itertools.groupby(data_file, _is_divider) if not is_divider
            )
            for lines in self.shard_iterable(line_chunks):
                fields_list: List[List[str]] = [line.strip().split() for line in lines]
                # Transpose the list of lists.
                fields: List[List[str]] = [list(field) for field in zip(*fields_list)]
                tokens_, pos_tags, chunk_tags, ner_tags = fields  # type: List[str], List[str], List[str], List[str]
                tokens: List[Token] = [Token(token) for token in tokens_]
                yield self.text_to_instance(tokens, pos_tags, chunk_tags, ner_tags)

    def text_to_instance(
        self,
        tokens: List[Token],
        pos_tags: Optional[List[str]] = None,
        chunk_tags: Optional[List[str]] = None,
        ner_tags: Optional[List[str]] = None
    ) -> Instance:
        sequence: TextField = TextField(tokens)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        instance_fields['metadata'] = MetadataField({'words': [x.text for x in tokens]})
        if self.convert_to_coding_scheme == 'BIOUL':
            coded_chunks: Optional[List[str]] = to_bioul(chunk_tags, encoding=self._original_coding_scheme) if chunk_tags is not None else None
            coded_ner: Optional[List[str]] = to_bioul(ner_tags, encoding=self._original_coding_scheme) if ner_tags is not None else None
        else:
            coded_chunks = chunk_tags
            coded_ner = ner_tags
        if 'pos' in self.feature_labels:
            if pos_tags is None:
                raise ConfigurationError('Dataset reader was specified to use pos_tags as features. Pass them to text_to_instance.')
            instance_fields['pos_tags'] = SequenceLabelField(pos_tags, sequence, 'pos_tags')
        if 'chunk' in self.feature_labels:
            if coded_chunks is None:
                raise ConfigurationError('Dataset reader was specified to use chunk tags as features. Pass them to text_to_instance.')
            instance_fields['chunk_tags'] = SequenceLabelField(coded_chunks, sequence, 'chunk_tags')
        if 'ner' in self.feature_labels:
            if coded_ner is None:
                raise ConfigurationError('Dataset reader was specified to use NER tags as  features. Pass them to text_to_instance.')
            instance_fields['ner_tags'] = SequenceLabelField(coded_ner, sequence, 'ner_tags')
        if self.tag_label == 'ner' and coded_ner is not None:
            instance_fields['tags'] = SequenceLabelField(coded_ner, sequence, self.label_namespace)
        elif self.tag_label == 'pos' and pos_tags is not None:
            instance_fields['tags'] = SequenceLabelField(pos_tags, sequence, self.label_namespace)
        elif self.tag_label == 'chunk' and coded_chunks is not None:
            instance_fields['tags'] = SequenceLabelField(coded_chunks, sequence, self.label_namespace)
        return Instance(instance_fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields['tokens']._token_indexers = self._token_indexers