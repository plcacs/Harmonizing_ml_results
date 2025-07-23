from typing import Dict, List, Any, Iterator, Optional, Iterable
import logging
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
logger = logging.getLogger(__name__)
DEFAULT_WORD_TAG_DELIMITER = '###'

@DatasetReader.register('sequence_tagging')
class SequenceTaggingDatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    