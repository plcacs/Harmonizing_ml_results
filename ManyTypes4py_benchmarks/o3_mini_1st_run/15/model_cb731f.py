#!/usr/bin/env python3
import copy
import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch as pt
from torch import Tensor

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
except Exception:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig(Config):
    """
    ModelConfig defines model parameters defined at training time which are relevant to model inference.
    """
    config_length_task: Any = None
    weight_tying_type: str = C.WEIGHT_TYING_SRC_TRG_SOFTMAX
    lhuc: bool = False
    dtype: str = C.DTYPE_FP32
    neural_vocab_selection: Optional[str] = None
    neural_vocab_selection_block_loss: bool = False


class SockeyeModel(pt.nn.Module):
    """
    SockeyeModel shares components needed for both training and inference.
    """

    def __init__(self, config: ModelConfig, inference_only: bool = False, clamp_to_dtype: bool = False,
                 train_decoder_only: bool = False, forward_pass_cache_size: int = 0) -> None:
        super().__init__()
        self.config: ModelConfig = copy.deepcopy(config)
        self.dtype = utils.get_torch_dtype(config.dtype)
        self.clamp_to_dtype: bool = clamp_to_dtype
        logger.info('%s', self.config)
        self.train_decoder_only: bool = train_decoder_only
        self.forward_pass_cache_size: int = forward_pass_cache_size
        self.embed_and_encode: Callable[..., Any] = self._embed_and_encode
        if self.forward_pass_cache_size > 0:
            self.embed_and_encode = self._cache_wrapper(self._embed_and_encode)
        source_embedding, target_embedding, output_weight = self._get_embeddings()
        self.embedding_source: encoder.Embedding = encoder.Embedding(config.config_embed_source, embedding=source_embedding, dtype=self.dtype)
        self.embedding_target: encoder.Embedding = encoder.Embedding(config.config_embed_target, embedding=target_embedding, dtype=self.dtype)
        self.encoder = encoder.get_transformer_encoder(self.config.config_encoder, inference_only=inference_only, dtype=self.dtype, clamp_to_dtype=clamp_to_dtype)
        self.decoder = decoder.get_decoder(self.config.config_decoder, inference_only=inference_only, dtype=self.dtype, clamp_to_dtype=clamp_to_dtype)
        self.nvs: Optional[nvs.NeuralVocabSelection] = None
        if self.config.neural_vocab_selection:
            self.nvs = nvs.NeuralVocabSelection(model_size=self.config.config_encoder.model_size, vocab_target_size=self.config.vocab_target_size, model_type=self.config.neural_vocab_selection, dtype=self.dtype)
        self.output_layer: layers.OutputLayer = layers.OutputLayer(hidden_size=self.decoder.get_num_hidden(), vocab_size=self.config.vocab_target_size, weight=output_weight, dtype=self.dtype)
        self.output_layer_module_cached: layers.OutputLayer = self.output_layer
        self.output_layer_script_cached: pt.jit.ScriptModule = pt.jit.script(self.output_layer_module_cached)
        self.set_inference_only(inference_only)
        self.factor_output_layers: pt.nn.ModuleList = pt.nn.ModuleList()
        for i, factor_config in enumerate(self.target_factor_configs, 1):
            output_layer = pt.nn.Linear(in_features=self.decoder.get_num_hidden(), out_features=factor_config.vocab_size, bias=True, dtype=self.dtype)
            self.factor_output_layers.append(output_layer)
        self.factor_vocab_size: Optional[int] = self.target_factor_configs[-1].vocab_size if self.target_factor_configs else None
        self.length_ratio: Optional[layers.LengthRatio] = None
        if self.config.config_length_task is not None:
            utils.check_condition(self.config.config_length_task.weight > 0.0, 'Auxiliary length task requested, but its loss weight is zero')
            self.length_ratio = layers.LengthRatio(hidden_size=self.encoder.get_num_hidden(), num_layers=self.config.config_length_task.num_layers, dtype=self.dtype)
        self.traced_embedding_source: Optional[pt.jit.ScriptModule] = None
        self.traced_encoder: Optional[pt.jit.ScriptModule] = None
        self.traced_decode_step: Optional[pt.jit.ScriptModule] = None
        mismatched_dtype_params: List[Tuple[str, Any]] = [(name, param.dtype) for name, param in self.named_parameters() if param.dtype != self.dtype]
        self.to(self.dtype)
        if mismatched_dtype_params:
            logger.warn("Some parameters were created in a different dtype and then converted to the SockeyeModel's dtype. This works but can cause memory spikes when creating/loading models. To avoid this, pass the SockeyeModel's dtype when instantiating all submodules. Converted parameters:")
            for name, dt in mismatched_dtype_params:
                logger.warn(f'{name}: {dt} -> {self.dtype}')
        self.knn: Optional[layers.KNN] = None

    def set_inference_only(self, inference_only: bool) -> None:
        """
        Turn inference_only optimization on or off.
        """
        self.inference_only: bool = inference_only
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

    def state_structure(self) -> Any:
        return self.decoder.state_structure()

    def encode(self, inputs: Tensor, valid_length: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Encodes the input sequence.
        """
        if self.traced_embedding_source is None:
            logger.debug('Tracing embedding_source')
            self.traced_embedding_source = pt.jit.trace(self.embedding_source, inputs)
        source_embed: Tensor = self.traced_embedding_source(inputs)
        if self.traced_encoder is None:
            logger.debug('Tracing encoder')
            self.traced_encoder = pt.jit.trace(self.encoder, (source_embed, valid_length))
        source_encoded, source_encoded_length, att_mask = self.traced_encoder(source_embed, valid_length)
        return (source_encoded, source_encoded_length, att_mask)

    def encode_and_initialize(self, inputs: Tensor, valid_length: Tensor, constant_length_ratio: float = 0.0) -> Tuple[Any, Tensor, Optional[Tensor]]:
        """
        Encodes the input sequence and initializes decoder states.
        """
        source_encoded, source_encoded_lengths, att_mask = self.encode(inputs, valid_length=valid_length)
        predicted_output_length: Tensor = self.predict_output_length(source_encoded, source_encoded_lengths[:, 0], constant_length_ratio)
        states: Any = self.decoder.init_state_from_encoder(source_encoded, source_encoded_lengths)
        nvs_pred: Optional[Tensor] = None
        if self.nvs is not None:
            nvs_pred = pt.sigmoid(self.nvs(source_encoded, source_encoded_lengths, att_mask))
        return (states, predicted_output_length, nvs_pred)

    def _embed_and_encode(self, source: Tensor, source_length: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor, Any, Optional[Tensor]]:
        """
        Encode the input sequence, embed the target sequence, and initialize the decoder.
        """
        source_embed: Tensor = self.embedding_source(source)
        target_embed: Tensor = self.embedding_target(target)
        source_encoded, source_encoded_length, att_mask = self.encoder(source_embed, source_length)
        states: Any = self.decoder.init_state_from_encoder(source_encoded, source_encoded_length, target_embed)
        nvs: Optional[Tensor] = None
        if self.nvs is not None:
            source_encoded_for_nvs: Tensor = source_encoded
            if self.config.neural_vocab_selection_block_loss:
                source_encoded_for_nvs = source_encoded.detach()
            nvs = self.nvs(source_encoded_for_nvs, source_length, att_mask)
        return (source_encoded, source_encoded_length, target_embed, states, nvs)

    def decode_step(self, step_input: Tensor, states: Any, vocab_slice_ids: Optional[List[int]] = None) -> Tuple[Tensor, Optional[Tensor], List[Any], List[Tensor]]:
        """
        One step decoding of the translation model.
        """
        decode_step_inputs: List[Any] = [step_input, states]
        if vocab_slice_ids is not None:
            decode_step_inputs.append(vocab_slice_ids)
        if self.traced_decode_step is None:
            logger.debug('Tracing decode step')
            decode_step_module = _DecodeStep(self.embedding_target, self.decoder, self.output_layer, self.factor_output_layers, self.knn)
            self.traced_decode_step = pt.jit.trace(decode_step_module, decode_step_inputs)
        decode_step_outputs: List[Any] = self.traced_decode_step(*decode_step_inputs)
        step_output: Tensor = decode_step_outputs[:1][0]  # first element is step_output
        decoder_out: Tensor = decode_step_outputs[1]
        target_factor_outputs: List[Tensor] = decode_step_outputs[2:self.num_target_factors + 1]
        knn_output: Optional[Tensor] = self.knn(decoder_out) if self.knn is not None else None
        new_states: List[Any] = decode_step_outputs[self.num_target_factors + 1:]
        return (step_output, knn_output, new_states, target_factor_outputs)

    def forward(self, source: Tensor, source_length: Tensor, target: Tensor, target_length: Tensor) -> Dict[str, Tensor]:
        context_manager = pt.no_grad() if self.train_decoder_only or self.forward_pass_cache_size > 0 else utils.no_context()
        with context_manager:
            source_encoded, source_encoded_length, target_embed, states, nvs_prediction = self.embed_and_encode(source, source_length, target)
        target_decoded: Tensor = self.decoder.decode_seq(target_embed, states=states)
        forward_output: Dict[str, Tensor] = dict()
        forward_output[C.LOGITS_NAME] = self.output_layer(target_decoded, None)
        for i, factor_output_layer in enumerate(self.factor_output_layers, 1):
            forward_output[C.FACTOR_LOGITS_NAME % i] = factor_output_layer(target_decoded)
        if self.length_ratio is not None:
            forward_output[C.LENRATIO_NAME] = self.length_ratio(source_encoded, source_encoded_length[:, 0])
        if nvs_prediction is not None:
            forward_output[C.NVS_PRED_NAME] = nvs_prediction
        return forward_output

    def get_decoder_states(self, source: Tensor, source_length: Tensor, target: Tensor, target_length: Tensor) -> Any:
        """Same as `forward`, but skip the output layer and return the decoder states."""
        context_manager = pt.no_grad() if self.train_decoder_only or self.forward_pass_cache_size > 0 else utils.no_context()
        with context_manager:
            source_encoded, source_encoded_length, target_embed, states, _ = self.embed_and_encode(source, source_length, target)
        decoder_states: Any = self.decoder.decode_seq(target_embed, states=states)
        return decoder_states

    def predict_output_length(self, source_encoded: Tensor, source_encoded_length: Tensor, constant_length_ratio: float = 0.0) -> Tensor:
        if self.length_ratio is not None:
            predicted_length_ratio: Tensor = self.length_ratio(source_encoded, source_encoded_length)
            predicted_output_length: Tensor = predicted_length_ratio * source_encoded_length
        elif constant_length_ratio > 0.0:
            predicted_output_length: Tensor = source_encoded_length * constant_length_ratio
        else:
            predicted_output_length = pt.zeros_like(source_encoded_length)
        return predicted_output_length

    def save_config(self, folder: str) -> None:
        """
        Saves model configuration to <folder>/config
        """
        fname: str = os.path.join(folder, C.CONFIG_NAME)
        self.config.save(fname)
        logger.info('Saved model config to "%s"', fname)

    @staticmethod
    def load_config(fname: str) -> ModelConfig:
        """
        Loads model configuration.
        """
        config: Config = ModelConfig.load(fname)
        logger.info('Loaded model config from "%s"', fname)
        return cast(ModelConfig, config)

    def save_parameters(self, fname: str) -> None:
        """
        Saves model parameters to file.
        """
        self.apply(layers.interleave_kv)
        filtered_state_dict: Dict[str, Any] = {name: param for name, param in self.state_dict().items() if 'traced' not in name and 'cached' not in name}
        pt.save(filtered_state_dict, fname)
        self.apply(layers.separate_kv)
        logging.info('Saved params/state_dict to "%s"', fname)

    def load_parameters(self, filename: str, device: pt.device = pt.device('cpu'), allow_missing: bool = False, ignore_extra: bool = False) -> None:
        """
        Loads parameters from file previously saved by `save_parameters`.
        """
        utils.check_condition(os.path.exists(filename), 'No model parameter file found under %s. This is either not a model directory or the first training checkpoint has not happened yet.' % filename)
        state_dict: Dict[str, Any] = pt.load(filename, weights_only=True, map_location=device)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in unexpected if 'traced' not in key and 'cached' not in key]
        missing = [key for key in missing if 'traced' not in key and 'cached' not in key]
        if not allow_missing:
            utils.check_condition(not missing, f'missing keys: {missing}')
        if not ignore_extra:
            utils.check_condition(not unexpected, f'extra keys: {unexpected}')
        for module in self.modules():
            if hasattr(module, 'kv_interleaved') and isinstance(module.kv_interleaved, bool):
                module.kv_interleaved = True
        if self.training:
            self.apply(layers.separate_kv)
        logger.info('Loaded params from "%s" to "%s"', filename, device)

    def set_parameters(self, new_params: Dict[str, pt.nn.Parameter], allow_missing: bool = True, ignore_extra: bool = False) -> None:
        """
        Update model params with new values from a dictionary.
        """
        model_params: Dict[str, pt.nn.Parameter] = dict(self.named_parameters())
        if not allow_missing:
            for name, _ in model_params.items():
                assert name in new_params.keys(), "Parameter '%s' is missing in new_params dictionary. Set allow_missing=True to ignore missing parameters." % name
        for name in new_params:
            if not ignore_extra and name not in model_params:
                raise ValueError("Parameter '%s' in new_params dictionary is not present in ParameterDict. Set ignore_extra=True to ignore." % name)
            if name in model_params:
                assert model_params[name].size() == new_params[name].size(), "Parameter '%s' has shape '%s' in the model but shape '%s' in the new_params dictionary." % (name, model_params[name].size(), new_params[name].size())
                model_params[name].data[:] = new_params[name].data

    def load_knn_index(self, knn_index_folder: str) -> None:
        """
        Load kNN index from a directory.
        """
        utils.check_import_faiss()
        knn_config: KNNConfig = KNNConfig.load(os.path.join(knn_index_folder, C.KNN_CONFIG_NAME))
        knn_config = cast(KNNConfig, knn_config)
        keys_index = faiss.read_index(os.path.join(knn_index_folder, C.KNN_INDEX_NAME))
        vals = np.memmap(os.path.join(knn_index_folder, C.KNN_WORD_DATA_STORE_NAME), dtype=utils.get_numpy_dtype(knn_config.word_data_type), mode='r', shape=(knn_config.index_size, 1))
        state_store: Optional[np.memmap] = None
        state_store_path = os.path.join(knn_index_folder, C.KNN_STATE_DATA_STORE_NAME)
        if os.path.isfile(state_store_path):
            state_store = np.memmap(state_store_path, dtype=utils.get_numpy_dtype(knn_config.state_data_type), mode='r', shape=(knn_config.index_size, knn_config.dimension))
        self.knn = layers.KNN(keys_index, vals, vocab_size=self.config.vocab_target_size, state_store=state_store)

    @staticmethod
    def save_version(folder: str) -> None:
        """
        Saves version to <folder>/version.
        """
        fname: str = os.path.join(folder, C.VERSION_NAME)
        with open(fname, 'w') as out:
            out.write(__version__)

    def _get_embeddings(self) -> Tuple[pt.nn.Embedding, pt.nn.Embedding, Optional[Tensor]]:
        """
        Returns embeddings for source, target, and output layer.
        """
        share_embed: bool = C.WEIGHT_TYING_SRC in self.config.weight_tying_type and C.WEIGHT_TYING_TRG in self.config.weight_tying_type
        tie_weights: bool = C.WEIGHT_TYING_SOFTMAX in self.config.weight_tying_type
        source_grad_sparse: bool = self.config.config_embed_source.allow_sparse_grad and (not tie_weights)
        source_embedding: pt.nn.Embedding = pt.nn.Embedding(self.config.config_embed_source.vocab_size, self.config.config_embed_source.num_embed, sparse=source_grad_sparse, dtype=self.dtype)
        if share_embed:
            target_embedding: pt.nn.Embedding = source_embedding
        else:
            target_grad_sparse: bool = self.config.config_embed_target.allow_sparse_grad and (not tie_weights)
            target_embedding = pt.nn.Embedding(self.config.config_embed_target.vocab_size, self.config.config_embed_target.num_embed, sparse=target_grad_sparse, dtype=self.dtype)
        output_weight: Optional[Tensor] = target_embedding.weight if tie_weights else None
        return (source_embedding, target_embedding, output_weight)

    @property
    def num_source_factors(self) -> int:
        """ Returns the number of source factors of this model (at least 1). """
        return self.config.config_data.num_source_factors

    @property
    def num_target_factors(self) -> int:
        """ Returns the number of target factors of this model (at least 1). """
        return self.config.config_data.num_target_factors

    @property
    def target_factor_configs(self) -> List[Any]:
        """ Returns the factor configs for target factors. """
        factor_configs: List[Any] = []
        if self.config.config_embed_target.factor_configs:
            factor_configs = self.config.config_embed_target.factor_configs
        return factor_configs

    @property
    def training_max_observed_len_source(self) -> int:
        """ The maximum sequence length on the source side observed during training. """
        return self.config.config_data.data_statistics.max_observed_len_source

    @property
    def training_max_observed_len_target(self) -> int:
        """ The maximum sequence length on the target side observed during training. """
        return self.config.config_data.data_statistics.max_observed_len_target

    @property
    def max_supported_len_source(self) -> int:
        """ The maximum supported source length. """
        return self.config.config_data.max_seq_len_source

    @property
    def max_supported_len_target(self) -> int:
        """ The maximum supported target length. """
        return self.config.config_data.max_seq_len_target

    @property
    def length_ratio_mean(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_mean

    @property
    def length_ratio_std(self) -> float:
        return self.config.config_data.data_statistics.length_ratio_std

    @property
    def output_layer_vocab_size(self) -> int:
        return self.output_layer.vocab_size

    @property
    def eop_id(self) -> int:
        return self.config.config_data.eop_id

    def _cache_wrapper(self, class_func: Callable[..., Any]) -> Callable[..., Any]:
        @lru_cache(maxsize=self.forward_pass_cache_size)
        def cache_func(*args: Any) -> Any:
            return class_func(*args)
        return cache_func


class _DecodeStep(pt.nn.Module):
    """
    Auxiliary module that wraps computation for a single decode step for a SockeyeModel.
    """

    def __init__(self, embedding_target: encoder.Embedding, decoder: Any, output_layer: layers.OutputLayer, factor_output_layers: pt.nn.ModuleList, knn: Optional[layers.KNN] = None) -> None:
        super().__init__()
        self.embedding_target: encoder.Embedding = embedding_target
        self.decoder: Any = decoder
        self.output_layer: pt.jit.ScriptModule = pt.jit.script(output_layer)
        self.factor_output_layers: pt.nn.ModuleList = factor_output_layers
        self.has_target_factors: bool = bool(factor_output_layers)
        self.knn: Optional[layers.KNN] = knn

    def forward(self, step_input: Tensor, states: Any, vocab_slice_ids: Optional[Tensor] = None) -> List[Any]:
        target_embed: Tensor = self.embedding_target(step_input.unsqueeze(1))
        decoder_out, new_states = self.decoder(target_embed, states)
        decoder_out = decoder_out.squeeze(1)
        step_output: Tensor = self.output_layer(decoder_out, vocab_slice_ids)
        outputs: List[Any] = [step_output, decoder_out]
        if self.has_target_factors:
            outputs += [fol(decoder_out) for fol in self.factor_output_layers]
        outputs += new_states
        return outputs


def initialize_parameters(module: pt.nn.Module) -> None:
    """
    Can be applied to a SockeyeModel (via `model.apply(initialize_parameters)`)
    to initialize the parameters of a PyTorch SockeyeModel.
    """
    if isinstance(module, pt.nn.Linear) or isinstance(module, layers.OutputLayer):
        pt.nn.init.xavier_uniform_(module.weight, gain=1)
        if module.bias is not None:
            pt.nn.init.zeros_(module.bias)
    elif isinstance(module, pt.nn.Embedding):
        pt.nn.init.uniform_(module.weight, -0.07, 0.07)
    elif isinstance(module, pt.nn.LayerNorm):
        if module.elementwise_affine:
            pt.nn.init.ones_(module.weight)
            pt.nn.init.zeros_(module.bias)
    elif isinstance(module, layers.LHUC):
        pt.nn.init.uniform_(module.weight, a=0.1)
    elif isinstance(module, layers.PositionalEmbeddings):
        if module.weight_type == C.LEARNED_POSITIONAL_EMBEDDING:
            pt.nn.init.xavier_uniform(module.weight, gain=1.0)


def load_model(model_folder: str, device: pt.device = pt.device('cpu'), dtype: Optional[Union[pt.dtype, str]] = None,
               clamp_to_dtype: bool = False, checkpoint: Optional[str] = None, inference_only: bool = False,
               train_decoder_only: bool = False, allow_missing: bool = False, forward_pass_cache_size: int = 0,
               knn_index: Optional[str] = None) -> Tuple[SockeyeModel, Any, Any]:
    """
    Load a model from model_folder.
    """
    source_vocabs: Any = vocab.load_source_vocabs(model_folder)
    target_vocabs: Any = vocab.load_target_vocabs(model_folder)
    model_version: str = utils.load_version(os.path.join(model_folder, C.VERSION_NAME))
    logger.info('Model version: %s', model_version)
    utils.check_version(model_version)
    model_config: ModelConfig = SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME))
    if checkpoint is None:
        params_fname: str = os.path.join(model_folder, C.PARAMS_BEST_NAME)
    else:
        params_fname = os.path.join(model_folder, C.PARAMS_NAME % checkpoint)
    model: SockeyeModel = SockeyeModel(model_config, inference_only=inference_only, clamp_to_dtype=clamp_to_dtype, train_decoder_only=train_decoder_only, forward_pass_cache_size=forward_pass_cache_size)
    model.load_parameters(filename=params_fname, device=device, allow_missing=allow_missing, ignore_extra=False)
    if knn_index is not None:
        model.load_knn_index(knn_index)
    model.to(device)
    if dtype is None:
        logger.info('Model dtype: %s' % model.dtype)
    else:
        model.cast(dtype)
        logger.info('Model dtype: overridden to %s' % dtype)
    utils.check_condition(model.num_source_factors == len(source_vocabs), "Number of loaded source vocabularies (%d) does not match number of source factors for model '%s' (%d)" % (len(source_vocabs), model_folder, model.num_source_factors))
    utils.check_condition(model.num_target_factors == len(target_vocabs), "Number of loaded target vocabularies (%d) does not match number of target factors for model '%s' (%d)" % (len(target_vocabs), model_folder, model.num_target_factors))
    return (model, source_vocabs, target_vocabs)


def load_models(device: pt.device, model_folders: List[str], checkpoints: Optional[List[Optional[str]]] = None,
                dtype: Optional[Union[pt.dtype, str]] = None, clamp_to_dtype: bool = False, inference_only: bool = False,
                train_decoder_only: bool = False, allow_missing: bool = False, forward_pass_cache_size: int = 0,
                knn_index: Optional[str] = None) -> Tuple[List[SockeyeModel], Any, Any]:
    """
    Loads a list of models for inference.
    """
    logger.info('Loading %d model(s) from %s ...', len(model_folders), model_folders)
    load_time_start: float = time.time()
    models: List[SockeyeModel] = []
    source_vocabs: List[Any] = []
    target_vocabs: List[Any] = []
    if checkpoints is None:
        checkpoints = [None] * len(model_folders)
    else:
        utils.check_condition(len(checkpoints) == len(model_folders), 'Must provide checkpoints for each model')
    for model_folder, checkpoint in zip(model_folders, checkpoints):
        model, src_vcbs, trg_vcbs = load_model(model_folder, device=device, dtype=dtype, clamp_to_dtype=clamp_to_dtype, checkpoint=checkpoint, inference_only=inference_only, train_decoder_only=train_decoder_only, allow_missing=allow_missing, forward_pass_cache_size=forward_pass_cache_size, knn_index=knn_index)
        models.append(model)
        source_vocabs.append(src_vcbs)
        target_vocabs.append(trg_vcbs)
    first_model_vocabs: Any = source_vocabs[0]
    for fi in range(len(first_model_vocabs)):
        utils.check_condition(vocab.are_identical(*[source_vocabs[i][fi] for i in range(len(source_vocabs))]), 'Source vocabulary ids do not match. Factor %d' % fi)
    first_model_vocabs = target_vocabs[0]
    for fi in range(len(first_model_vocabs)):
        utils.check_condition(vocab.are_identical(*[target_vocabs[i][fi] for i in range(len(target_vocabs))]), 'Target vocabulary ids do not match. Factor %d' % fi)
    load_time: float = time.time() - load_time_start
    logger.info('%d model(s) loaded in %.4fs', len(models), load_time)
    return (models, source_vocabs[0], target_vocabs[0])