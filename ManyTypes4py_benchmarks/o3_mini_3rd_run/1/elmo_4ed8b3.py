#!/usr/bin/env python3
import json
import logging
import warnings
from typing import Any, Dict, List, Optional, Union, Iterator
import numpy
import torch
from torch import Tensor
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
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn.util import add_sentence_boundary_token_ids, get_device_of, remove_sentence_boundaries

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=FutureWarning)
    import h5py

logger = logging.getLogger(__name__)


class Elmo(torch.nn.Module, FromParams):
    """
    Compute ELMo representations using a pre-trained bidirectional language model.
    """

    def __init__(self,
                 options_file: Optional[str],
                 weight_file: Optional[str],
                 num_output_representations: int,
                 requires_grad: bool = False,
                 do_layer_norm: bool = False,
                 dropout: float = 0.5,
                 vocab_to_cache: Optional[List[str]] = None,
                 keep_sentence_boundaries: bool = False,
                 scalar_mix_parameters: Optional[List[float]] = None,
                 module: Optional[torch.nn.Module] = None) -> None:
        super().__init__()
        logger.info('Initializing ELMo')
        if module is not None:
            if options_file is not None or weight_file is not None:
                raise ConfigurationError("Don't provide options_file or weight_file with module")
            self._elmo_lstm = module
        else:
            # type: ignore
            self._elmo_lstm = _ElmoBiLm(options_file, weight_file, requires_grad=requires_grad, vocab_to_cache=vocab_to_cache)
        self._has_cached_vocab: bool = vocab_to_cache is not None
        self._keep_sentence_boundaries: bool = keep_sentence_boundaries
        self._dropout: Dropout = Dropout(p=dropout)
        self._scalar_mixes: List[ScalarMix] = []
        for k in range(num_output_representations):
            scalar_mix = ScalarMix(self._elmo_lstm.num_layers,
                                   do_layer_norm=do_layer_norm,
                                   initial_scalar_parameters=scalar_mix_parameters,
                                   trainable=scalar_mix_parameters is None)
            self.add_module('scalar_mix_{}'.format(k), scalar_mix)
            self._scalar_mixes.append(scalar_mix)

    def get_output_dim(self) -> int:
        return self._elmo_lstm.get_output_dim()

    def forward(self, inputs: Tensor, word_inputs: Optional[Tensor] = None) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        Computes the ELMo representations.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch_size, timesteps, 50) of character ids.
        word_inputs : Optional[torch.Tensor]
            Optionally, a tensor of shape (batch_size, timesteps) representing pre-cached word ids.

        Returns
        -------
        Dict[str, Union[torch.Tensor, List[torch.Tensor]]]
            A dict with keys:
              - 'elmo_representations': a list of tensors with shape (batch_size, timesteps, embedding_dim)
              - 'mask': a torch.BoolTensor with shape (batch_size, timesteps)
        """
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
                logger.warning('Word inputs were passed to ELMo but it does not have a cached vocab.')
                reshaped_word_inputs = None
            else:
                reshaped_word_inputs = word_inputs
        else:
            reshaped_word_inputs = word_inputs

        bilm_output: Dict[str, Any] = self._elmo_lstm(reshaped_inputs, reshaped_word_inputs)
        layer_activations: List[Tensor] = bilm_output['activations']
        mask_with_bos_eos: Tensor = bilm_output['mask']
        representations: List[Tensor] = []
        for i in range(len(self._scalar_mixes)):
            scalar_mix = getattr(self, 'scalar_mix_{}'.format(i))
            representation_with_bos_eos: Tensor = scalar_mix(layer_activations, mask_with_bos_eos)
            if self._keep_sentence_boundaries:
                processed_representation = representation_with_bos_eos
                processed_mask = mask_with_bos_eos
            else:
                representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(representation_with_bos_eos,
                                                                                                  mask_with_bos_eos)
                processed_representation = representation_without_bos_eos
                processed_mask = mask_without_bos_eos
            representations.append(self._dropout(processed_representation))
        if word_inputs is not None and len(original_word_size) > 2:
            mask = processed_mask.view(original_word_size)
            elmo_representations = [representation.view(original_word_size + (-1,)) for representation in representations]
        elif len(original_shape) > 3:
            mask = processed_mask.view(original_shape[:-1])
            elmo_representations = [representation.view(original_shape[:-1] + (-1,)) for representation in representations]
        else:
            mask = processed_mask
            elmo_representations = representations
        return {'elmo_representations': elmo_representations, 'mask': mask}


def batch_to_ids(batch: List[List[str]]) -> Tensor:
    """
    Converts a batch of tokenized sentences to a tensor representing the sentences with encoded characters.

    Parameters
    ----------
    batch : List[List[str]]
        A list of tokenized sentences.

    Returns
    -------
    torch.Tensor
        A tensor of padded character ids.
    """
    instances: List[Instance] = []
    indexer = ELMoTokenCharactersIndexer()
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {'character_ids': indexer})
        instance = Instance({'elmo': field})
        instances.append(instance)
    dataset = Batch(instances)
    vocab = Vocabulary()
    dataset.index_instances(vocab)
    return dataset.as_tensor_dict()['elmo']['character_ids']['elmo_tokens']


class _ElmoCharacterEncoder(torch.nn.Module):
    """
    Compute context insensitive token representation using pretrained biLM.
    """

    def __init__(self, options_file: str, weight_file: str, requires_grad: bool = False) -> None:
        super().__init__()
        with open(cached_path(options_file), 'r') as fin:
            self._options: Dict[str, Any] = json.load(fin)
        self._weight_file: str = weight_file
        self.output_dim: int = self._options['lstm']['projection_dim']
        self.requires_grad: bool = requires_grad
        self._load_weights()
        self._beginning_of_sentence_characters: Tensor = torch.from_numpy(
            numpy.array(ELMoCharacterMapper.beginning_of_sentence_characters) + 1)
        self._end_of_sentence_characters: Tensor = torch.from_numpy(
            numpy.array(ELMoCharacterMapper.end_of_sentence_characters) + 1)

    def get_output_dim(self) -> int:
        return self.output_dim

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        """
        Compute context insensitive token embeddings for ELMo.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch_size, sequence_length, 50) of character ids.

        Returns
        -------
        Dict[str, torch.Tensor]
            - 'token_embedding': (batch_size, sequence_length + 2, embedding_dim)
            - 'mask': (batch_size, sequence_length + 2)
        """
        mask: Tensor = (inputs > 0).sum(dim=-1) > 0
        character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids(
            inputs, mask, self._beginning_of_sentence_characters, self._end_of_sentence_characters)
        max_chars_per_token: int = self._options['char_cnn']['max_characters_per_token']
        character_embedding: Tensor = torch.nn.functional.embedding(
            character_ids_with_bos_eos.view(-1, max_chars_per_token), self._char_embedding_weights)
        cnn_options: Dict[str, Any] = self._options['char_cnn']
        if cnn_options['activation'] == 'tanh':
            activation = torch.tanh
        elif cnn_options['activation'] == 'relu':
            activation = torch.nn.functional.relu
        else:
            raise ConfigurationError('Unknown activation')
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs: List[Tensor] = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, 'char_conv_{}'.format(i))
            convolved: Tensor = conv(character_embedding)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)
        token_embedding: Tensor = torch.cat(convs, dim=-1)
        token_embedding = self._highways(token_embedding)
        token_embedding = self._projection(token_embedding)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()
        return {'mask': mask_with_bos_eos,
                'token_embedding': token_embedding.view(batch_size, sequence_length, -1)}

    def _load_weights(self) -> None:
        self._load_char_embedding()
        self._load_cnn_weights()
        self._load_highway()
        self._load_projection()

    def _load_char_embedding(self) -> None:
        with h5py.File(cached_path(self._weight_file), 'r') as fin:
            char_embed_weights = fin['char_embed'][...]
        weights = numpy.zeros((char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]), dtype='float32')
        weights[1:, :] = char_embed_weights
        self._char_embedding_weights = torch.nn.Parameter(torch.FloatTensor(weights), requires_grad=self.requires_grad)

    def _load_cnn_weights(self) -> None:
        cnn_options: Dict[str, Any] = self._options['char_cnn']
        filters: List[List[int]] = cnn_options['filters']
        char_embed_dim: int = cnn_options['embedding']['dim']
        convolutions: List[torch.nn.Conv1d] = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(in_channels=char_embed_dim, out_channels=num, kernel_size=width, bias=True)
            with h5py.File(cached_path(self._weight_file), 'r') as fin:
                weight = fin['CNN']['W_cnn_{}'.format(i)][...]
                bias = fin['CNN']['b_cnn_{}'.format(i)][...]
            w_reshaped = numpy.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
            if w_reshaped.shape != tuple(conv.weight.data.shape):
                raise ValueError('Invalid weight file')
            conv.weight.data.copy_(torch.FloatTensor(w_reshaped))
            conv.bias.data.copy_(torch.FloatTensor(bias))
            conv.weight.requires_grad = self.requires_grad
            conv.bias.requires_grad = self.requires_grad
            convolutions.append(conv)
            self.add_module('char_conv_{}'.format(i), conv)
        self._convolutions = convolutions

    def _load_highway(self) -> None:
        cnn_options: Dict[str, Any] = self._options['char_cnn']
        filters: List[List[int]] = cnn_options['filters']
        n_filters: int = sum((f[1] for f in filters))
        n_highway: int = cnn_options['n_highway']
        self._highways = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)
        for k in range(n_highway):
            with h5py.File(cached_path(self._weight_file), 'r') as fin:
                w_transform = numpy.transpose(fin['CNN_high_{}'.format(k)]['W_transform'][...])
                w_carry = -1.0 * numpy.transpose(fin['CNN_high_{}'.format(k)]['W_carry'][...])
                weight = numpy.concatenate([w_transform, w_carry], axis=0)
                self._highways._layers[k].weight.data.copy_(torch.FloatTensor(weight))
                self._highways._layers[k].weight.requires_grad = self.requires_grad
                b_transform = fin['CNN_high_{}'.format(k)]['b_transform'][...]
                b_carry = -1.0 * fin['CNN_high_{}'.format(k)]['b_carry'][...]
                bias = numpy.concatenate([b_transform, b_carry], axis=0)
                self._highways._layers[k].bias.data.copy_(torch.FloatTensor(bias))
                self._highways._layers[k].bias.requires_grad = self.requires_grad

    def _load_projection(self) -> None:
        cnn_options: Dict[str, Any] = self._options['char_cnn']
        filters: List[List[int]] = cnn_options['filters']
        n_filters: int = sum((f[1] for f in filters))
        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)
        with h5py.File(cached_path(self._weight_file), 'r') as fin:
            weight = fin['CNN_proj']['W_proj'][...]
            bias = fin['CNN_proj']['b_proj'][...]
            self._projection.weight.data.copy_(torch.FloatTensor(numpy.transpose(weight)))
            self._projection.bias.data.copy_(torch.FloatTensor(bias))
            self._projection.weight.requires_grad = self.requires_grad
            self._projection.bias.requires_grad = self.requires_grad


class _ElmoBiLm(torch.nn.Module):
    """
    Run a pre-trained bidirectional language model.
    """

    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 requires_grad: bool = False,
                 vocab_to_cache: Optional[List[str]] = None) -> None:
        super().__init__()
        self._token_embedder = _ElmoCharacterEncoder(options_file, weight_file, requires_grad=requires_grad)
        self._requires_grad: bool = requires_grad
        if requires_grad and vocab_to_cache:
            logging.warning('You are fine tuning ELMo and caching char CNN word vectors. '
                            'This behaviour is not guaranteed to be well defined, particularly '
                            'if not all of your inputs will occur in the vocabulary cache.')
        self._word_embedding: Optional[torch.nn.Module] = None
        self._bos_embedding: Optional[Tensor] = None
        self._eos_embedding: Optional[Tensor] = None
        if vocab_to_cache:
            logging.info('Caching character cnn layers for words in vocabulary.')
            self.create_cached_cnn_embeddings(vocab_to_cache)
        with open(cached_path(options_file), 'r') as fin:
            options: Dict[str, Any] = json.load(fin)
        if not options['lstm'].get('use_skip_connections'):
            raise ConfigurationError('We only support pretrained biLMs with residual connections')
        self._elmo_lstm = ElmoLstm(input_size=options['lstm']['projection_dim'],
                                   hidden_size=options['lstm']['projection_dim'],
                                   cell_size=options['lstm']['dim'],
                                   num_layers=options['lstm']['n_layers'],
                                   memory_cell_clip_value=options['lstm']['cell_clip'],
                                   state_projection_clip_value=options['lstm']['proj_clip'],
                                   requires_grad=requires_grad)
        self._elmo_lstm.load_weights(weight_file)
        self.num_layers: int = options['lstm']['n_layers'] + 1

    def get_output_dim(self) -> int:
        return 2 * self._token_embedder.get_output_dim()

    def forward(self, inputs: Tensor, word_inputs: Optional[Tensor] = None) -> Dict[str, Union[List[Tensor], Tensor]]:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch_size, timesteps, 50) of character ids.
        word_inputs : Optional[torch.Tensor]
            Optionally, a tensor of shape (batch_size, timesteps) representing word ids for cached vocab.

        Returns
        -------
        Dict[str, Union[List[torch.Tensor], torch.Tensor]]
            - 'activations': list of activations at each layer.
            - 'mask': tensor with sequence mask including sentence boundaries.
        """
        if self._word_embedding is not None and word_inputs is not None:
            try:
                mask_without_bos_eos: Tensor = word_inputs > 0
                embedded_inputs = self._word_embedding(word_inputs)
                type_representation, mask = add_sentence_boundary_token_ids(embedded_inputs,
                                                                            mask_without_bos_eos,
                                                                            self._bos_embedding,
                                                                            self._eos_embedding)
            except (RuntimeError, IndexError):
                token_embedding = self._token_embedder(inputs)
                mask = token_embedding['mask']
                type_representation = token_embedding['token_embedding']
        else:
            token_embedding = self._token_embedder(inputs)
            mask = token_embedding['mask']
            type_representation = token_embedding['token_embedding']
        lstm_outputs: Tensor = self._elmo_lstm(type_representation, mask)
        output_tensors: List[Tensor] = [torch.cat([type_representation, type_representation], dim=-1) * mask.unsqueeze(-1)]
        for layer_activations in torch.chunk(lstm_outputs, lstm_outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))
        return {'activations': output_tensors, 'mask': mask}

    def create_cached_cnn_embeddings(self, tokens: List[str]) -> None:
        """
        Precompute word representations via character convolutions and highway layers.
        
        Parameters
        ----------
        tokens : List[str]
            A list of tokens to precompute representations for.
        """
        tokens = [ELMoCharacterMapper.bos_token, ELMoCharacterMapper.eos_token] + tokens
        timesteps: int = 32
        batch_size: int = 32
        chunked_tokens: Iterator[List[str]] = lazy_groups_of(iter(tokens), timesteps)
        all_embeddings: List[Tensor] = []
        device: int = get_device_of(next(self.parameters()))
        for batch_group in lazy_groups_of(chunked_tokens, batch_size):
            batched_tensor: Tensor = batch_to_ids(list(batch_group))
            if device >= 0:
                batched_tensor = batched_tensor.cuda(device)
            output = self._token_embedder(batched_tensor)
            token_embedding: Tensor = output['token_embedding']
            mask: Tensor = output['mask']
            token_embedding, _ = remove_sentence_boundaries(token_embedding, mask)
            all_embeddings.append(token_embedding.view(-1, token_embedding.size(-1)))
        full_embedding: Tensor = torch.cat(all_embeddings, 0)
        full_embedding = full_embedding[:len(tokens), :]
        embedding: Tensor = full_embedding[2:len(tokens), :]
        vocab_size, embedding_dim = list(embedding.size())
        from allennlp.modules.token_embedders import Embedding
        self._bos_embedding = full_embedding[0, :]
        self._eos_embedding = full_embedding[1, :]
        self._word_embedding = Embedding(num_embeddings=vocab_size,
                                         embedding_dim=embedding_dim,
                                         weight=embedding.data,
                                         trainable=self._requires_grad)