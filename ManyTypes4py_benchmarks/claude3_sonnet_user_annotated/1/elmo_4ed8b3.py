import json
import logging
import warnings
from typing import Any, Dict, List, Union, Optional, Tuple

import numpy
import torch

from torch.nn.modules import Dropout

from allennlp.common import FromParams
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.batch import Batch
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import (
    ELMoCharacterMapper,
    ELMoTokenCharactersIndexer,
)
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn.util import (
    add_sentence_boundary_token_ids,
    get_device_of,
    remove_sentence_boundaries,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py


logger = logging.getLogger(__name__)


class Elmo(torch.nn.Module, FromParams):
    """
    Compute ELMo representations using a pre-trained bidirectional language model.

    See "Deep contextualized word representations", Peters et al. for details.

    This module takes character id input and computes `num_output_representations` different layers
    of ELMo representations.  Typically `num_output_representations` is 1 or 2.  For example, in
    the case of the SRL model in the above paper, `num_output_representations=1` where ELMo was included at
    the input token representation layer.  In the case of the SQuAD model, `num_output_representations=2`
    as ELMo was also included at the GRU output layer.

    In the implementation below, we learn separate scalar weights for each output layer,
    but only run the biLM once on each input sequence for efficiency.

    # Parameters

    options_file : `str`, required.
        ELMo JSON options file
    weight_file : `str`, required.
        ELMo hdf5 weight file
    num_output_representations : `int`, required.
        The number of ELMo representation to output with
        different linear weighted combination of the 3 layers (i.e.,
        character-convnet output, 1st lstm output, 2nd lstm output).
    requires_grad : `bool`, optional
        If True, compute gradient of ELMo parameters for fine tuning.
    do_layer_norm : `bool`, optional, (default = `False`).
        Should we apply layer normalization (passed to `ScalarMix`)?
    dropout : `float`, optional, (default = `0.5`).
        The dropout to be applied to the ELMo representations.
    vocab_to_cache : `List[str]`, optional, (default = `None`).
        A list of words to pre-compute and cache character convolutions
        for. If you use this option, Elmo expects that you pass word
        indices of shape (batch_size, timesteps) to forward, instead
        of character indices. If you use this option and pass a word which
        wasn't pre-cached, this will break.
    keep_sentence_boundaries : `bool`, optional, (default = `False`)
        If True, the representation of the sentence boundary tokens are
        not removed.
    scalar_mix_parameters : `List[float]`, optional, (default = `None`)
        If not `None`, use these scalar mix parameters to weight the representations
        produced by different layers. These mixing weights are not updated during
        training. The mixing weights here should be the unnormalized (i.e., pre-softmax)
        weights. So, if you wanted to use only the 1st layer of a 2-layer ELMo,
        you can set this to [-9e10, 1, -9e10 ].
    module : `torch.nn.Module`, optional, (default = `None`).
        If provided, then use this module instead of the pre-trained ELMo biLM.
        If using this option, then pass `None` for both `options_file`
        and `weight_file`.  The module must provide a public attribute
        `num_layers` with the number of internal layers and its `forward`
        method must return a `dict` with `activations` and `mask` keys
        (see `_ElmoBilm` for an example).  Note that `requires_grad` is also
        ignored with this option.
    """

    def __init__(
        self,
        options_file: str,
        weight_file: str,
        num_output_representations: int,
        requires_grad: bool = False,
        do_layer_norm: bool = False,
        dropout: float = 0.5,
        vocab_to_cache: Optional[List[str]] = None,
        keep_sentence_boundaries: bool = False,
        scalar_mix_parameters: Optional[List[float]] = None,
        module: Optional[torch.nn.Module] = None,
    ) -> None:
        super().__init__()

        logger.info("Initializing ELMo")
        if module is not None:
            if options_file is not None or weight_file is not None:
                raise ConfigurationError("Don't provide options_file or weight_file with module")
            self._elmo_lstm = module
        else:
            self._elmo_lstm = _ElmoBiLm(  # type: ignore
                options_file,
                weight_file,
                requires_grad=requires_grad,
                vocab_to_cache=vocab_to_cache,
            )
        self._has_cached_vocab = vocab_to_cache is not None
        self._keep_sentence_boundaries = keep_sentence_boundaries
        self._dropout = Dropout(p=dropout)
        self._scalar_mixes: List[ScalarMix] = []
        for k in range(num_output_representations):
            scalar_mix = ScalarMix(
                self._elmo_lstm.num_layers,  # type: ignore
                do_layer_norm=do_layer_norm,
                initial_scalar_parameters=scalar_mix_parameters,
                trainable=scalar_mix_parameters is None,
            )
            self.add_module("scalar_mix_{}".format(k), scalar_mix)
            self._scalar_mixes.append(scalar_mix)

    def get_output_dim(self) -> int:
        return self._elmo_lstm.get_output_dim()

    def forward(
        self, inputs: torch.Tensor, word_inputs: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        # Parameters

        inputs : `torch.Tensor`, required.
        Shape `(batch_size, timesteps, 50)` of character ids representing the current batch.
        word_inputs : `torch.Tensor`, required.
            If you passed a cached vocab, you can in addition pass a tensor of shape
            `(batch_size, timesteps)`, which represent word ids which have been pre-cached.

        # Returns

        `Dict[str, Union[torch.Tensor, List[torch.Tensor]]]`
            A dict with the following keys:
            - `'elmo_representations'` (`List[torch.Tensor]`) :
              A `num_output_representations` list of ELMo representations for the input sequence.
              Each representation is shape `(batch_size, timesteps, embedding_dim)`
            - `'mask'` (`torch.BoolTensor`) :
              Shape `(batch_size, timesteps)` long tensor with sequence mask.
        """
        # reshape the input if needed
        original_shape = inputs.size()
        if len(original_shape) > 3:
            timesteps, num_characters = original_shape[-2:]
            reshaped_inputs = inputs.view(-1, timesteps, num_characters)
        else:
            reshaped_inputs = inputs

        if word_inputs is not None:
            original_word_size = word_inputs.size()
            if self._has_cached_vocab and len(original_word_size) > 2:
                reshaped_word_inputs = word_inputs.view(-1, original_word_size[-1])
            elif not self._has_cached_vocab:
                logger.warning(
                    "Word inputs were passed to ELMo but it does not have a cached vocab."
                )
                reshaped_word_inputs = None
            else:
                reshaped_word_inputs = word_inputs
        else:
            reshaped_word_inputs = word_inputs

        # run the biLM
        bilm_output = self._elmo_lstm(reshaped_inputs, reshaped_word_inputs)  # type: ignore
        layer_activations = bilm_output["activations"]
        mask_with_bos_eos = bilm_output["mask"]

        # compute the elmo representations
        representations = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, "scalar_mix_{}".format(i))
            representation_with_bos_eos = scalar_mix(layer_activations, mask_with_bos_eos)
            if self._keep_sentence_boundaries:
                processed_representation = representation_with_bos_eos
                processed_mask = mask_with_bos_eos
            else:
                representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(
                    representation_with_bos_eos, mask_with_bos_eos
                )
                processed_representation = representation_without_bos_eos
                processed_mask = mask_without_bos_eos
            representations.append(self._dropout(processed_representation))

        # reshape if necessary
        if word_inputs is not None and len(original_word_size) > 2:
            mask = processed_mask.view(original_word_size)
            elmo_representations = [
                representation.view(original_word_size + (-1,))
                for representation in representations
            ]
        elif len(original_shape) > 3:
            mask = processed_mask.view(original_shape[:-1])
            elmo_representations = [
                representation.view(original_shape[:-1] + (-1,))
                for representation in representations
            ]
        else:
            mask = processed_mask
            elmo_representations = representations

        return {"elmo_representations": elmo_representations, "mask": mask}


def batch_to_ids(batch: List[List[str]]) -> torch.Tensor:
    """
    Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters
    (len(batch), max sentence length, max word length).

    # Parameters

    batch : `List[List[str]]`, required
        A list of tokenized sentences.

    # Returns

        A tensor of padded character ids.
    """
    instances = []
    indexer = ELMoTokenCharactersIndexer()
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {"character_ids": indexer})
        instance = Instance({"elmo": field})
        instances.append(instance)

    dataset = Batch(instances)
    vocab = Vocabulary()
    dataset.index_instances(vocab)
    return dataset.as_tensor_dict()["elmo"]["character_ids"]["elmo_tokens"]


class _ElmoCharacterEncoder(torch.nn.Module):
    """
    Compute context insensitive token representation using pretrained biLM.

    This embedder has input character ids of size (batch_size, sequence_length, 50)
    and returns (batch_size, sequence_length + 2, embedding_dim), where embedding_dim
    is specified in the options file (typically 512).

    We add special entries at the beginning and end of each sequence corresponding
    to <S> and </S>, the beginning and end of sentence tokens.

    Note: this is a lower level class useful for advanced usage.  Most users should
    use `ElmoTokenEmbedder` or `allennlp.modules.Elmo` instead.

    # Parameters

    options_file : `str`
        ELMo JSON options file
    weight_file : `str`
        ELMo hdf5 weight file
    requires_grad : `bool`, optional, (default = `False`).
        If True, compute gradient of ELMo parameters for fine tuning.


    The relevant section of the options file is something like:

    