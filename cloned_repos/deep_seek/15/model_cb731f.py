import copy
import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import cast, Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import torch as pt
from sockeye import __version__
from . import constants as C
from . import data_io
from . import decoder
from . import encoder
from . import layers
from . import transformer
from . import utils
from . import vocab
from .config import Config
from .encoder import FactorConfig
from .layers import LengthRatioConfig
from . import nvs
from sockeye.knn import KNNConfig
try:
    import faiss
    import faiss.contrib.torch_utils
except:
    pass
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig(Config):
    """
    ModelConfig defines model parameters defined at training time which are relevant to model inference.
    Add new model parameters here. If you want backwards compatibility for models trained with code that did not
    contain these parameters, provide a reasonable default under default_values.

    :param config_data: Used training data.
    :param vocab_source_size: Source vocabulary size.
    :param vocab_target_size: Target vocabulary size.
    :param config_embed_source: Embedding config for source.
    :param config_embed_target: Embedding config for target.
    :param config_encoder: Encoder configuration.
    :param config_decoder: Decoder configuration.
    :param config_length_task: Optional length task configuration.
    :param weight_tying_type: Determines which weights get tied.
    :param lhuc: LHUC (Vilar 2018) is applied at some part of the model.
    :param dtype: Data type (string) of model parameters. Default: float32.
    :param neural_vocab_selection: When True the model contains a neural vocab selection model that restricts
                                   the target output vocabulary to speed up inference.
    :param neural_vocab_selection_block_loss: When true the gradients of the NVS models are blocked before the encoder.
    """
    config_data: Any
    vocab_source_size: int
    vocab_target_size: int
    config_embed_source: Any
    config_embed_target: Any
    config_encoder: Any
    config_decoder: Any
    config_length_task: Optional[Any] = None
    weight_tying_type: str = C.WEIGHT_TYING_SRC_TRG_SOFTMAX
    lhuc: bool = False
    dtype: str = C.DTYPE_FP32
    neural_vocab_selection: Optional[str] = None
    neural_vocab_selection_block_loss: bool = False

class SockeyeModel(pt.nn.Module):
    """
    SockeyeModel shares components needed for both training and inference.
    The main components of a Sockeye model are
    1) Source embedding
    2) Target embedding
    3) Encoder
    4) Decoder
    5) Output Layer

    ModelConfig contains parameters and their values that are fixed at training time and must be re-used at inference
    time.

    :param config: Model configuration.
    :param inference_only: Use the model only for inference, enabling optimizations.
    :param clamp_to_dtype: Avoid -inf/inf by clamping outputs to min/max finite values for their dtype.
    """

    def __init__(self, config: ModelConfig, inference_only: bool = False, clamp_to_dtype: bool = False, train_decoder_only: bool = False, forward_pass_cache_size: int = 0) -> None:
        super().__init__()
        self.config = copy.deepcopy(config)
        self.dtype = utils.get_torch_dtype(config.dtype)
        self.clamp_to_dtype = clamp_to_dtype
        logger.info('%s', self.config)
        self.train_decoder_only = train_decoder_only
        self.forward_pass_cache_size = forward_pass_cache_size
        self.embed_and_encode = self._embed_and_encode
        if self.forward_pass_cache_size > 0:
            self.embed_and_encode = self._cache_wrapper(self._embed_and_encode)
        source_embedding, target_embedding, output_weight = self._get_embeddings()
        self.embedding_source = encoder.Embedding(config.config_embed_source, embedding=source_embedding, dtype=self.dtype)
        self.embedding_target = encoder.Embedding(config.config_embed_target, embedding=target_embedding, dtype=self.dtype)
        self.encoder = encoder.get_transformer_encoder(self.config.config_encoder, inference_only=inference_only, dtype=self.dtype, clamp_to_dtype=clamp_to_dtype)
        self.decoder = decoder.get_decoder(self.config.config_decoder, inference_only=inference_only, dtype=self.dtype, clamp_to_dtype=clamp_to_dtype)
        self.nvs = None
        if self.config.neural_vocab_selection:
            self.nvs = nvs.NeuralVocabSelection(model_size=self.config.config_encoder.model_size, vocab_target_size=self.config.vocab_target_size, model_type=self.config.neural_vocab_selection, dtype=self.dtype)
        self.output_layer = layers.OutputLayer(hidden_size=self.decoder.get_num_hidden(), vocab_size=self.config.vocab_target_size, weight=output_weight, dtype=self.dtype)
        self.output_layer_module_cached = self.output_layer
        self.output_layer_script_cached = pt.jit.script(self.output_layer_module_cached)
        self.set_inference_only(inference_only)
        self.factor_output_layers = pt.nn.ModuleList()
        for i, factor_config in enumerate(self.target_factor_configs, 1):
            output_layer = pt.nn.Linear(in_features=self.decoder.get_num_hidden(), out_features=factor_config.vocab_size, bias=True, dtype=self.dtype)
            self.factor_output_layers.append(output_layer)
        self.factor_vocab_size = factor_config.vocab_size if self.target_factor_configs else None
        self.length_ratio = None
        if self.config.config_length_task is not None:
            utils.check_condition(self.config.config_length_task.weight > 0.0, 'Auxiliary length task requested, but its loss weight is zero')
            self.length_ratio = layers.LengthRatio(hidden_size=self.encoder.get_num_hidden(), num_layers=self.config.config_length_task.num_layers, dtype=self.dtype)
        self.traced_embedding_source = None
        self.traced_encoder = None
        self.traced_decode_step = None
        mismatched_dtype_params = [(name, param.dtype) for name, param in self.named_parameters() if param.dtype != self.dtype]
        self.to(self.dtype)
        if mismatched_dtype_params:
            logger.warn("Some parameters were created in a different dtype and then converted to the SockeyeModel's dtype. This works but can cause memory spikes when creating/loading models. To avoid this, pass the SockeyeModel's dtype when instantiating all submodules. Converted parameters:")
            for name, dtype in mismatched_dtype_params:
                logger.warn(f'{name}: {dtype} -> {self.dtype}')
        self.knn = None

    def set_inference_only(self, inference_only: bool) -> None:
        """
        Turn inference_only optimization on or off.
        """
        self.inference_only = inference_only
        self.output_layer = self.output_layer_script_cached if self.inference_only else self.output_layer_module_cached
        self.decoder.set_inference_only(self.inference_only)

    def cast(self, dtype: Union[pt.dtype, str]) -> None:
        dtype = utils.get_torch_dtype(dtype)
        if self.dtype == dtype:
            return
        if dtype in {pt.bfloat16, pt.float16, pt.float32}:
            logger.info(f'Casting SockeyeModel to dtype {dtype}')
            self.to(dtype)
            self.dtype = dtype
        elif dtype == pt.int8:
            logger.info('Dynamic quantization to int8 for (fused) Linear layers')
            quant_mapping = {pt.nn.Linear: pt.nn.quantized.dynamic.Linear}
            pt.quantization.quantize_dynamic(self, {pt.nn.Linear}, dtype=pt.qint8, inplace=self.inference_only, mapping=quant_mapping)
        else:
            raise ValueError(f'Unsupported SockeyeModel dtype: {dtype}')
        self.config.dtype = utils.dtype_to_str(self.dtype)

    def state_structure(self) -> List[Tuple[str, Tuple[int, ...]]]:
        return self.decoder.state_structure()

    def encode(self, inputs: pt.Tensor, valid_length: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
        """
        Encodes the input sequence.

        :param inputs: Source input data. Shape: (batch_size, length, num_source_factors).
        :param valid_length: Optional Tensor of sequence lengths within this batch. Shape: (batch_size, 2)
        :return: Encoder outputs, encoded output lengths, attention mask
        """
        if self.traced_embedding_source is None:
            logger.debug('Tracing embedding_source')
            self.traced_embedding_source = pt.jit.trace(self.embedding_source, inputs)
        source_embed = self.traced_embedding_source(inputs)
        if self.traced_encoder is None:
            logger.debug('Tracing encoder')
            self.traced_encoder = pt.jit.trace(self.encoder, (source_embed, valid_length))
        source_encoded, source_encoded_length, att_mask = self.traced_encoder(source_embed, valid_length)
        return (source_encoded, source_encoded_length, att_mask)

    def encode_and_initialize(self, inputs: pt.Tensor, valid_length: pt.Tensor, constant_length_ratio: float = 0.0) -> Tuple[List[pt.Tensor], pt.Tensor, Optional[pt.Tensor]]:
        """
        Encodes the input sequence and initializes decoder states (and predicted output lengths if available).
        Used for inference/decoding.

        :param inputs: Source input data. Shape: (batch_size, length, num_source_factors).
        :param valid_length: Tensor of sequence lengths within this batch. Shape: (batch_size, 2)
        :param constant_length_ratio: Constant length ratio
        :return: Initial states for the decoder, predicted output length of shape (batch_size,), 0 if not available.
                 Returns the neural vocabulary selection model prediction if enabled, None otherwise.
        """
        source_encoded, source_encoded_lengths, att_mask = self.encode(inputs, valid_length=valid_length)
        predicted_output_length = self.predict_output_length(source_encoded, source_encoded_lengths[:, 0], constant_length_ratio)
        states = self.decoder.init_state_from_encoder(source_encoded, source_encoded_lengths)
        nvs_pred = None
        if self.nvs is not None:
            nvs_pred = pt.sigmoid(self.nvs(source_encoded, source_encoded_lengths, att_mask))
        return (states, predicted_output_length, nvs_pred)

    def _embed_and_encode(self, source: pt.Tensor, source_length: pt.Tensor, target: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, List[pt.Tensor], Optional[pt.Tensor]]:
        """
        Encode the input sequence, embed the target sequence, and initialize the decoder.
        Used for training.

        :param source: Source input data.
        :param source_length: Length of source inputs.
        :param target: Target input data.
        :return: encoder outputs and lengths, target embeddings, decoder initial states, attention mask and neural
                 vocab selection prediction (if present, otherwise None).
        """
        source_embed = self.embedding_source(source)
        target_embed = self.embedding_target(target)
        source_encoded, source_encoded_length, att_mask = self.encoder(source_embed, source_length)
        states = self.decoder.init_state_from_encoder(source_encoded, source_encoded_length, target_embed)
        nvs = None
        if self.nvs is not None:
            source_encoded_for_nvs = source_encoded
            if self.config.neural_vocab_selection_block_loss:
                source_encoded_for_nvs = source_encoded.detach()
            nvs = self.nvs(source_encoded_for_nvs, source_length, att_mask)
        return (source_encoded, source_encoded_length, target_embed, states, nvs)

    def decode_step(self, step_input: pt.Tensor, states: List[pt.Tensor], vocab_slice_ids: Optional[pt.Tensor] = None) -> Tuple[pt.Tensor, Optional[pt.Tensor], List[pt.Tensor], List[pt.Tensor]]:
        """
        One step decoding of the translation model.

        :param step_input: Input to a single decoder step. Shape: (batch_size, num_target_factors).
        :param states: List of previous or initial model states. Shape of state tensors and length of states list
                       determined by self.decoder.state_structure().
        :param vocab_slice_ids: Optional list of vocabulary ids to use
                                for reduced matrix multiplication at the output layer.

        :return: logits, KNN output if present otherwise None, list of new model states, other target factor logits.
        """
        decode_step_inputs = [step_input, states]
        if vocab_slice_ids is not None:
            decode_step_inputs.append(vocab_slice_ids)
        if self.traced_decode_step is None:
            logger.debug('Tracing decode step')
            decode_step_module = _DecodeStep(self.embedding_target, self.decoder, self.output_layer, self.factor_output_layers, self.knn)
            self.traced_decode_step = pt.jit.trace(decode_step_module, decode_step_inputs)
        decode_step_outputs = self.traced_decode_step(*decode_step_inputs)
        step_output, decoder_out, *target_factor_outputs = decode_step_outputs[:self.num_target_factors + 1]
        knn_output = self.knn(decoder_out) if self.knn is not None else None
        new_states = decode_step_outputs[self.num_target_factors + 1:]
        return (step_output, knn_output, new_states, target_factor_outputs)

    def forward(self, source: pt.Tensor, source_length: pt.Tensor, target: pt.Tensor, target_length: pt.Tensor) -> Dict[str, pt.Tensor]:
        with pt.no_grad() if self.train_decoder_only or self.forward_pass_cache_size > 0 else utils.no_context():
            source_encoded, source_encoded_length, target_embed, states, nvs_prediction = self.embed_and_encode(source, source_length, target)
        target = self.decoder.decode_seq(target_embed, states=states)
        forward_output = dict()
        forward_output[C.LOGITS_NAME] = self.output_layer(target, None)
        for i, factor_output_layer in enumerate(self.factor_output_layers, 1):
            forward_output[C.FACTOR_LOGITS_NAME % i] = factor_output_layer(target)
        if self.length_ratio is not None:
            forward_output[C.LENRATIO_NAME] = self.length_ratio(source_encoded, source_encoded_length[:, 0])
        if nvs_prediction is not None:
            forward_output[C.NVS_PRED_NAME] = nvs_prediction
        return forward_output

    def get_decoder_states(self, source: pt.Tensor, source_length: pt.Tensor, target: pt.Tensor, target_length: pt.Tensor) -> pt.Tensor:
        """Same as `forward`, but skip the output layer and return the decoder states."""
        with pt.no_grad() if self.train_decoder_only or self.forward_pass_cache_size > 0 else utils.no_context():
            source_encoded, source_encoded_length, target_embed, states, nvs_prediction = self.embed_and_encode(source, source_length, target)
        decoder_states = self.decoder.decode_seq(target_embed, states=states)
        return decoder_states

    def predict_output_length(self, source_encoded: pt.Tensor, source_encoded_length: pt.Tensor, constant_length_ratio: float = 0.0) -> pt.Tensor:
        if self.length_ratio is not None:
            predicted_length_ratio = self.length_ratio(source_encoded, source_encoded_length)
            predicted_output_length = predicted_length_ratio * source_encoded_length
        elif constant_length_ratio > 0.0:
            predicted_output_length = source_encoded_length * constant_length_ratio
        else:
            predicted_output_length = pt.zeros_like(source_encoded_length)
        return predicted_output_length

    def save_config(self, folder: str) -> None:
        """
        Saves model configuration to <folder>/config

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, C.CONFIG_NAME)
        self.config.save(fname)
        logger.info('Saved model config to "%s"', fname)

    @staticmethod
    def load_config(fname: str) -> ModelConfig:
        """
        Loads model configuration.

        :param fname: Path to load model configuration from.
        :return: Model configuration.
        """
        config = ModelConfig.load(fname)
        logger.info('Loaded model config from "%s"', fname)
        return cast(ModelConfig, config)

    def save_parameters(self, fname: str) -> None:
        """
        Saves model parameters to file. Also see
        See https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

        :param fname: Path to save parameters to.
        """
        self.apply(layers.interleave_kv)
        filtered_state_dict = {name: param for name, param in self.state_dict().items() if 'traced' not in name and 'cached' not in name}
        pt.save(filtered_state_dict, fname)
        self.apply(layers.separate_kv)
        logging.info('Saved params/state_dict to "%s"', fname)

    def load_parameters(self, filename: str, device: pt.device = pt.device('cpu'), allow_missing: bool = False, ignore_extra: bool = False) -> None:
        """
        Loads parameters from file previously saved by `save_parameters`.
        See https://pytorch.org/tutorials/beginner/saving_loading_models.html#