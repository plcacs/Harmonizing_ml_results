from typing import Dict, List, Any, Optional, Iterator
import logging
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)
DEFAULT_WORD_TAG_DELIMITER: str = '###'


@DatasetReader.register('sequence_tagging')
class SequenceTaggingDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    ```
    WORD###TAG [TAB] WORD###TAG [TAB] ..... 
    ```

    and converts it into a `Dataset` suitable for sequence tagging. You can also specify
    alternative delimiters in the constructor.

    Registered as a `DatasetReader` with name "sequence_tagging".

    # Parameters

    word_tag_delimiter: `str`, optional (default=`"###"`)
        The text that separates each WORD from its TAG.
    token_delimiter: `str`, optional (default=`None`)
        The text that separates each WORD-TAG pair from the next pair. If `None`
        then the line will just be split on whitespace.
    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    """

    def __init__(self,
                 word_tag_delimiter: str = DEFAULT_WORD_TAG_DELIMITER,
                 token_delimiter: Optional[str] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None,
                 **kwargs: Any) -> None:
        super().__init__(manual_distributed_sharding=True, manual_multiprocess_sharding=True, **kwargs)
        self._token_indexers: Dict[str, TokenIndexer] = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._word_tag_delimiter: str = word_tag_delimiter
        self._token_delimiter: Optional[str] = token_delimiter
        self._params: Dict[str, Any] = {'word_tag_delimiter': self._word_tag_delimiter,
                                        'token_delimiter': self._token_delimiter,
                                        'token_indexers': self._token_indexers}
        self._params.update(kwargs)

    def _read(self, file_path: str) -> Iterator[Instance]:
        file_path = cached_path(file_path)
        with open(file_path, 'r') as data_file:
            logger.info('Reading instances from lines in file at: %s', file_path)
            for line in self.shard_iterable(data_file):
                line = line.strip('\n')
                if not line:
                    continue
                tokens_and_tags: List[List[str]] = [pair.rsplit(self._word_tag_delimiter, 1) 
                                                      for pair in line.split(self._token_delimiter)]
                tokens: List[Token] = [Token(token) for token, tag in tokens_and_tags]
                tags: List[str] = [tag for token, tag in tokens_and_tags]
                yield self.text_to_instance(tokens, tags)

    def text_to_instance(self, tokens: List[Token], tags: Optional[List[str]] = None) -> Instance:
        """
        We take `pre-tokenized` input here, because we don't have a tokenizer in this class.
        """
        fields: Dict[str, Field] = {}
        sequence: TextField = TextField(tokens)
        fields['tokens'] = sequence
        fields['metadata'] = MetadataField({'words': [x.text for x in tokens]})
        if tags is not None:
            fields['tags'] = SequenceLabelField(tags, sequence)
        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        instance.fields['tokens']._token_indexers = self._token_indexers

    def _to_params(self) -> Dict[str, Any]:
        return self._params
