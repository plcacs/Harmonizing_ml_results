import logging
import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import batched_index_select
from transformers import XLNetConfig

logger = logging.getLogger(__name__)


@TokenEmbedder.register('pretrained_transformer')
class PretrainedTransformerEmbedder(TokenEmbedder):
    """
    Uses a pretrained model from `transformers` as a `TokenEmbedder`.

    Registered as a `TokenEmbedder` with name "pretrained_transformer".
    
    # Parameters

    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerIndexer`.
    sub_module: `str`, optional (default = `None`)
        The name of a submodule of the transformer to be used as the embedder.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    eval_mode: `bool`, optional (default = `False`)
        If this is `True`, the model is always set to evaluation mode.
    last_layer_only: `bool`, optional (default = `True`)
        When `True` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings.
    override_weights_file: `Optional[str]`, optional (default = `None`)
        If set, this specifies a file from which to load alternate weights.
    override_weights_strip_prefix: `Optional[str]`, optional (default = `None`)
        If set, strip the given prefix from the state dict when loading it.
    reinit_modules: `Optional[Union[int, Tuple[int, ...], Tuple[str, ...]]]`, optional (default = `None`)
        Allows reinitialization of modules.
    load_weights: `bool`, optional (default = `True`)
        Whether to load the pretrained weights.
    gradient_checkpointing: `Optional[bool]`, optional (default = `None`)
        Enable or disable gradient checkpointing.
    tokenizer_kwargs: `Optional[Dict[str, Any]]`, optional (default = `None`)
        Dictionary with additional arguments for `AutoTokenizer.from_pretrained`.
    transformer_kwargs: `Optional[Dict[str, Any]]`, optional (default = `None`)
        Dictionary with additional arguments for `AutoModel.from_pretrained`.
    """
    authorized_missing_keys = ['position_ids$']

    def __init__(
        self,
        model_name: str,
        *,
        max_length: Optional[int] = None,
        sub_module: Optional[str] = None,
        train_parameters: bool = True,
        eval_mode: bool = False,
        last_layer_only: bool = True,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        reinit_modules: Optional[Union[int, Tuple[int, ...], Tuple[str, ...]]] = None,
        load_weights: bool = True,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        from allennlp.common import cached_transformers
        self.transformer_model = cached_transformers.get(
            model_name,
            True,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
            reinit_modules=reinit_modules,
            load_weights=load_weights,
            **(transformer_kwargs or {})
        )
        if gradient_checkpointing is not None:
            self.transformer_model.config.update({'gradient_checkpointing': gradient_checkpointing})
        self.config = self.transformer_model.config
        if sub_module:
            assert hasattr(self.transformer_model, sub_module)
            self.transformer_model = getattr(self.transformer_model, sub_module)
        self._max_length = max_length
        self.output_dim = self.config.hidden_size
        self._scalar_mix: Optional[ScalarMix] = None
        if not last_layer_only:
            self._scalar_mix = ScalarMix(self.config.num_hidden_layers)
            self.config.output_hidden_states = True
        tokenizer = PretrainedTransformerTokenizer(model_name, tokenizer_kwargs=tokenizer_kwargs)
        try:
            if self.transformer_model.get_input_embeddings().num_embeddings != len(tokenizer.tokenizer):
                self.transformer_model.resize_token_embeddings(len(tokenizer.tokenizer))
        except NotImplementedError:
            logger.warning('Could not resize the token embedding matrix of the transformer model. This model does not support resizing.')
        self._num_added_start_tokens: int = len(tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens: int = len(tokenizer.single_sequence_end_tokens)
        self._num_added_tokens: int = self._num_added_start_tokens + self._num_added_end_tokens
        self.train_parameters = train_parameters
        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False
        self.eval_mode = eval_mode
        if eval_mode:
            self.transformer_model.eval()

    def train(self, mode: bool = True) -> "PretrainedTransformerEmbedder":
        self.training = mode
        for name, module in self.named_children():
            if self.eval_mode and name == 'transformer_model':
                module.eval()
            else:
                module.train(mode)
        return self

    def get_output_dim(self) -> int:
        return self.output_dim

    def _number_of_token_type_embeddings(self) -> int:
        if isinstance(self.config, XLNetConfig):
            return 3
        elif hasattr(self.config, 'type_vocab_size'):
            return self.config.type_vocab_size
        else:
            return 0

    def forward(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        type_ids: Optional[torch.Tensor] = None,
        segment_concat_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        # Parameters

        token_ids: `torch.LongTensor`
            Shape: [batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces].
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Optional tensor of token type ids.
        segment_concat_mask: `Optional[torch.BoolTensor]`
            Optional mask for concatenated segments.

        # Returns

        `torch.Tensor`
            Embeddings of shape [batch_size, num_wordpieces, embedding_size].
        """
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings():
                    raise ValueError('Found type ids too large for the chosen transformer model.')
                assert token_ids.shape == type_ids.shape
        fold_long_sequences: bool = self._max_length is not None and token_ids.size(1) > self._max_length
        if fold_long_sequences:
            batch_size, num_segment_concat_wordpieces = token_ids.size()
            token_ids, segment_concat_mask, type_ids = self._fold_long_sequences(token_ids, segment_concat_mask, type_ids)
        transformer_mask: torch.Tensor = segment_concat_mask if self._max_length is not None else mask
        assert transformer_mask is not None
        parameters: Dict[str, torch.Tensor] = {'input_ids': token_ids, 'attention_mask': transformer_mask.float()}
        if type_ids is not None:
            parameters['token_type_ids'] = type_ids
        transformer_output = self.transformer_model(**parameters)
        if self._scalar_mix is not None:
            hidden_states = transformer_output.hidden_states[1:]
            embeddings = self._scalar_mix(hidden_states)
        else:
            embeddings = transformer_output.last_hidden_state
        if fold_long_sequences:
            embeddings = self._unfold_long_sequences(embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces)
        return embeddings

    def _fold_long_sequences(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        type_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Folds long sequences into multiple segments.

        # Parameters

        token_ids: `torch.LongTensor`
            Shape: [batch_size, num_segment_concat_wordpieces].
        mask: `torch.BoolTensor`
            Mask for the concatenated segments.
        type_ids: `Optional[torch.LongTensor]`
            Optional type ids.

        # Returns:

        A tuple containing:
            - token_ids: Shape [batch_size * num_segments, self._max_length].
            - mask: Shape [batch_size * num_segments, self._max_length].
            - type_ids: Optional tensor with shape [batch_size * num_segments, self._max_length].
        """
        num_segment_concat_wordpieces: int = token_ids.size(1)
        num_segments: int = math.ceil(num_segment_concat_wordpieces / self._max_length)  # type: ignore
        padded_length: int = num_segments * self._max_length
        length_to_pad: int = padded_length - num_segment_concat_wordpieces

        def fold(tensor: torch.Tensor) -> torch.Tensor:
            tensor = F.pad(tensor, [0, length_to_pad], value=0)
            return tensor.reshape(-1, self._max_length)

        return (fold(token_ids), fold(mask), fold(type_ids) if type_ids is not None else None)

    def _unfold_long_sequences(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        num_segment_concat_wordpieces: int,
    ) -> torch.Tensor:
        """
        Unfolds segments to reconstruct the original sequence embeddings.

        # Parameters

        embeddings: `torch.FloatTensor`
            Embeddings with shape [batch_size * num_segments, self._max_length, embedding_size].
        mask: `torch.BoolTensor`
            Mask with shape [batch_size * num_segments, self._max_length].
        batch_size: `int`
            The batch size.
        num_segment_concat_wordpieces: `int`
            Original sequence length.

        # Returns:

        Reconstructed embeddings of shape [batch_size, num_wordpieces, embedding_size].
        """
        def lengths_to_mask(lengths: torch.Tensor, max_len: int, device: torch.device) -> torch.Tensor:
            return torch.arange(max_len, device=device).expand(lengths.size(0), max_len) < lengths.unsqueeze(1)

        device: torch.device = embeddings.device
        num_segments: int = int(embeddings.size(0) / batch_size)
        embedding_size: int = embeddings.size(2)
        num_wordpieces: int = num_segment_concat_wordpieces - (num_segments - 1) * self._num_added_tokens
        embeddings = embeddings.reshape(batch_size, num_segments * self._max_length, embedding_size)
        mask = mask.reshape(batch_size, num_segments * self._max_length)
        seq_lengths: torch.Tensor = mask.sum(-1)
        if not (lengths_to_mask(seq_lengths, mask.size(1), device) == mask).all():
            raise ValueError('Long sequence splitting only supports masks with all 1s preceding all 0s.')
        end_token_indices: torch.Tensor = seq_lengths.unsqueeze(-1) - torch.arange(self._num_added_end_tokens, device=device) - 1
        start_token_embeddings: torch.Tensor = embeddings[:, :self._num_added_start_tokens, :]
        end_token_embeddings: torch.Tensor = batched_index_select(embeddings, end_token_indices)
        embeddings = embeddings.reshape(batch_size, num_segments, self._max_length, embedding_size)
        embeddings = embeddings[:, :, self._num_added_start_tokens:embeddings.size(2) - self._num_added_end_tokens, :]
        embeddings = embeddings.reshape(batch_size, -1, embedding_size)
        num_effective_segments: torch.Tensor = (seq_lengths + self._max_length - 1) // self._max_length
        num_removed_non_end_tokens: torch.Tensor = num_effective_segments * self._num_added_tokens - self._num_added_end_tokens
        end_token_indices = end_token_indices - num_removed_non_end_tokens.unsqueeze(-1)
        assert (end_token_indices >= self._num_added_start_tokens).all()
        embeddings = torch.cat([embeddings, torch.zeros_like(end_token_embeddings)], 1)
        embeddings.scatter_(1, end_token_indices.unsqueeze(-1).expand_as(end_token_embeddings), end_token_embeddings)
        embeddings = torch.cat([start_token_embeddings, embeddings], 1)
        embeddings = embeddings[:, :num_wordpieces, :]
        return embeddings