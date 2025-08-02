from typing import Dict, List, Optional, Sequence, Iterable, Set, Iterator, Any, Union, cast
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
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == '-DOCSTART-':
            return True
        else:
            return False

@DatasetReader.register('conll2003')
class Conll2003DatasetReader(DatasetReader):
    """
    Reads instances from a pretokenised file where each line is in the following format:

    