import copy
import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import cast, Dict, List, Optional, Tuple, Union
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
    config_length_task = None
    weight_tying_type = C.WEIGHT_TYING_SRC_TRG_SOFTMAX
    lhuc = False
    dtype = C.DTYPE_FP32
    neural_vocab_selection = None
    neural_vocab_selection_block_loss = False

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

    def __init__(self, config, inference_only=False, clamp_to_dtype=False, train_decoder_only=False, forward_pass_cache_size=0):
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

    def set_inference_only(self, inference_only):
        """
        Turn inference_only optimization on or off.
        """
        self.inference_only = inference_only
        self.output_layer = self.output_layer_script_cached if self.inference_only else self.output_layer_module_cached
        self.decoder.set_inference_only(self.inference_only)

    def cast(self, dtype):
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

    def state_structure(self):
        return self.decoder.state_structure()

    def encode(self, inputs, valid_length):
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

    def encode_and_initialize(self, inputs, valid_length, constant_length_ratio=0.0):
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

    def _embed_and_encode(self, source, source_length, target):
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

    def decode_step(self, step_input, states, vocab_slice_ids=None):
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

    def forward(self, source, source_length, target, target_length):
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

    def get_decoder_states(self, source, source_length, target, target_length):
        """Same as `forward`, but skip the output layer and return the decoder states."""
        with pt.no_grad() if self.train_decoder_only or self.forward_pass_cache_size > 0 else utils.no_context():
            source_encoded, source_encoded_length, target_embed, states, nvs_prediction = self.embed_and_encode(source, source_length, target)
        decoder_states = self.decoder.decode_seq(target_embed, states=states)
        return decoder_states

    def predict_output_length(self, source_encoded, source_encoded_length, constant_length_ratio=0.0):
        if self.length_ratio is not None:
            predicted_length_ratio = self.length_ratio(source_encoded, source_encoded_length)
            predicted_output_length = predicted_length_ratio * source_encoded_length
        elif constant_length_ratio > 0.0:
            predicted_output_length = source_encoded_length * constant_length_ratio
        else:
            predicted_output_length = pt.zeros_like(source_encoded_length)
        return predicted_output_length

    def save_config(self, folder):
        """
        Saves model configuration to <folder>/config

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, C.CONFIG_NAME)
        self.config.save(fname)
        logger.info('Saved model config to "%s"', fname)

    @staticmethod
    def load_config(fname):
        """
        Loads model configuration.

        :param fname: Path to load model configuration from.
        :return: Model configuration.
        """
        config = ModelConfig.load(fname)
        logger.info('Loaded model config from "%s"', fname)
        return cast(ModelConfig, config)

    def save_parameters(self, fname):
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

    def load_parameters(self, filename, device=pt.device('cpu'), allow_missing=False, ignore_extra=False):
        """
        Loads parameters from file previously saved by `save_parameters`.
        See https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

        :param filename: Path to parameter file
        :param device: Torch device to load parameters to. Default: cpu.
        :param allow_missing: Whether to silently skip loading parameters not represents in the file. Default: False.
        :param ignore_extra: Whether to silently ignore parameters from the file that are not part of this Module.
                             Default: False.
        """
        utils.check_condition(os.path.exists(filename), 'No model parameter file found under %s. This is either not a model directory or the first training checkpoint has not happened yet.' % filename)
        state_dict = pt.load(filename, weights_only=True, map_location=device)
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

    def set_parameters(self, new_params, allow_missing=True, ignore_extra=False):
        """
        Update model params with new values from a dictionary.

        :param new_params: Dictionary containing the new parameters.
        :param allow_missing: Whether to skip setting parameters not represented in the dictionary.
        :param ignore_extra: Whether to ignore parameters from new_params that are not present in this model.
        """
        model_params = dict(self.named_parameters())
        if not allow_missing:
            for name, _ in model_params.items():
                assert name in new_params.keys(), "Parameter '%s' is missing in new_params dictionary. Set allow_missing=True to ignore missing parameters." % name
        for name in new_params:
            if not ignore_extra and name not in model_params:
                raise ValueError("Parameter '%s' in new_params dictionary is not present in ParameterDict. Set ignore_extra=True to ignore." % name)
            if name in model_params:
                assert model_params[name].size() == new_params[name].size(), "Parameter '%s' has shape '%s' in the model but shape '%s' in the new_params dictionary." % (name, model_params[name].size(), new_params[name].size())
                model_params[name].data[:] = new_params[name].data

    def load_knn_index(self, knn_index_folder):
        """
        Load kNN index from a directory.

        :param knn_index_folder: same as `output_dir` from the `sockeye-knn` command,
                                 containing the index and a config file.
        """
        utils.check_import_faiss()
        knn_config = KNNConfig.load(os.path.join(knn_index_folder, C.KNN_CONFIG_NAME))
        knn_config = cast(KNNConfig, knn_config)
        keys_index = faiss.read_index(os.path.join(knn_index_folder, C.KNN_INDEX_NAME))
        vals = np.memmap(os.path.join(knn_index_folder, C.KNN_WORD_DATA_STORE_NAME), dtype=utils.get_numpy_dtype(knn_config.word_data_type), mode='r', shape=(knn_config.index_size, 1))
        state_store = None
        if os.path.isfile(os.path.join(knn_index_folder, C.KNN_STATE_DATA_STORE_NAME)):
            state_store = np.memmap(os.path.join(knn_index_folder, C.KNN_STATE_DATA_STORE_NAME), dtype=utils.get_numpy_dtype(knn_config.state_data_type), mode='r', shape=(knn_config.index_size, knn_config.dimension))
        self.knn = layers.KNN(keys_index, vals, vocab_size=self.config.vocab_target_size, state_store=state_store)

    @staticmethod
    def save_version(folder):
        """
        Saves version to <folder>/version.

        :param folder: Destination folder.
        """
        fname = os.path.join(folder, C.VERSION_NAME)
        with open(fname, 'w') as out:
            out.write(__version__)

    def _get_embeddings(self):
        """
        Returns embeddings for source, target, and output layer. Handles sharing and weight tying.
        """
        share_embed = C.WEIGHT_TYING_SRC in self.config.weight_tying_type and C.WEIGHT_TYING_TRG in self.config.weight_tying_type
        tie_weights = C.WEIGHT_TYING_SOFTMAX in self.config.weight_tying_type
        source_grad_sparse = self.config.config_embed_source.allow_sparse_grad and (not tie_weights)
        source_embedding = pt.nn.Embedding(self.config.config_embed_source.vocab_size, self.config.config_embed_source.num_embed, sparse=source_grad_sparse, dtype=self.dtype)
        if share_embed:
            target_embedding = source_embedding
        else:
            target_grad_sparse = self.config.config_embed_target.allow_sparse_grad and (not tie_weights)
            target_embedding = pt.nn.Embedding(self.config.config_embed_target.vocab_size, self.config.config_embed_target.num_embed, sparse=target_grad_sparse, dtype=self.dtype)
        if tie_weights:
            output_weight = target_embedding.weight
        else:
            output_weight = None
        return (source_embedding, target_embedding, output_weight)

    @property
    def num_source_factors(self):
        """ Returns the number of source factors of this model (at least 1). """
        return self.config.config_data.num_source_factors

    @property
    def num_target_factors(self):
        """ Returns the number of target factors of this model (at least 1). """
        return self.config.config_data.num_target_factors

    @property
    def target_factor_configs(self):
        """ Returns the factor configs for target factors. """
        factor_configs = []
        if self.config.config_embed_target.factor_configs:
            factor_configs = self.config.config_embed_target.factor_configs
        return factor_configs

    @property
    def training_max_observed_len_source(self):
        """ The maximum sequence length on the source side observed during training. This includes the <eos> token. """
        return self.config.config_data.data_statistics.max_observed_len_source

    @property
    def training_max_observed_len_target(self):
        """ The maximum sequence length on the target side observed during training. This includes the <bos> token. """
        return self.config.config_data.data_statistics.max_observed_len_target

    @property
    def max_supported_len_source(self):
        """ The maximum supported source length. This includes the <eos> token. """
        return self.config.config_data.max_seq_len_source

    @property
    def max_supported_len_target(self):
        """ The maximum supported target length. This includes the <bos> token. """
        return self.config.config_data.max_seq_len_target

    @property
    def length_ratio_mean(self):
        return self.config.config_data.data_statistics.length_ratio_mean

    @property
    def length_ratio_std(self):
        return self.config.config_data.data_statistics.length_ratio_std

    @property
    def output_layer_vocab_size(self):
        return self.output_layer.vocab_size

    @property
    def eop_id(self):
        return self.config.config_data.eop_id

    def _cache_wrapper(self, class_func):

        @lru_cache(maxsize=self.forward_pass_cache_size)
        def cache_func(*args):
            return class_func(*args)
        return cache_func

class _DecodeStep(pt.nn.Module):
    """
    Auxiliary module that wraps computation for a single decode step for a SockeyeModel.
    End-to-end traceable. Return values are put into a flat list to avoid return type constraints
    for traced modules.
    """

    def __init__(self, embedding_target, decoder, output_layer, factor_output_layers, knn=None):
        super().__init__()
        self.embedding_target = embedding_target
        self.decoder = decoder
        self.output_layer = pt.jit.script(output_layer)
        self.factor_output_layers = factor_output_layers
        self.has_target_factors = bool(factor_output_layers)
        self.knn = knn

    def forward(self, step_input, states, vocab_slice_ids=None):
        target_embed = self.embedding_target(step_input.unsqueeze(1))
        decoder_out, new_states = self.decoder(target_embed, states)
        decoder_out = decoder_out.squeeze(1)
        step_output = self.output_layer(decoder_out, vocab_slice_ids)
        outputs = [step_output, decoder_out]
        if self.has_target_factors:
            outputs += [fol(decoder_out) for fol in self.factor_output_layers]
        outputs += new_states
        return outputs

def initialize_parameters(module):
    """
    Can be applied to a SockeyeModel (via `model.apply(initialize_parameters)`)
    to initialize the parameters of a PyTorch SockeyeModel.
    For reproducibility, set pt.random.manual_seed.

    This implementation follows the default MXNet initialization scheme:
    - linear/feed-forward weights: Xavier(uniform, avg, magnitude=3.0)
    - biases: 0.0
    - layer norm gamma / weight: 1.0
    - layer norm beta / bias: 0.0
    - embedding parameters: uniform(-0.07, 0.07) [matches MXNet's default initialization]

    MXNet computes the uniform bounds for Xavier initialization as follows:
      sqrt(3 / ((fan_in + fan_out) / 2))
    PyTorch computes the uniform bounds for Xavier initialization as follows:
      (sqrt(2/(fan_in + fan_out)) * gain) * sqrt(3)
      where gain is set to 1.0 by default
    Both are equivalent.
    For some background on the equivalence of mx.init.Xavier and pt.nn.init.xavier_uniform_, see
    https://jamesmccaffrey.wordpress.com/2020/11/20/the-gain-parameter-
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

def load_model(model_folder, device=pt.device('cpu'), dtype=None, clamp_to_dtype=False, checkpoint=None, inference_only=False, train_decoder_only=False, allow_missing=False, forward_pass_cache_size=0, knn_index=None):
    """
    Load a model from model_folder.

    :param model_folder: Model folder.
    :param device: Torch device to load model to.
    :param checkpoint: Checkpoint to use. If none, uses best checkpoint.
    :param dtype: Optional data type (torch.dtype or str) to use. If None, will be inferred from stored model.
    :param clamp_to_dtype: Avoid -inf/inf by clamping outputs to min/max finite values for their dtype.
    :param inference_only: Use the model only for inference, enabling optimizations.
    :param train_decoder_only: Training will only update the decoder. Disable
           autograd for encoder and embeddings to save memory.
    :param allow_missing: Allow missing parameters in the loaded model.
    :param forward_pass_cache_size: If > 0, cache encoder and embedding calculations of forward pass.
    :param knn_index: Optional path to a folder containing a KNN model index.
    :return: List of models, source vocabularies, target vocabularies.
    """
    source_vocabs = vocab.load_source_vocabs(model_folder)
    target_vocabs = vocab.load_target_vocabs(model_folder)
    model_version = utils.load_version(os.path.join(model_folder, C.VERSION_NAME))
    logger.info('Model version: %s', model_version)
    utils.check_version(model_version)
    model_config = SockeyeModel.load_config(os.path.join(model_folder, C.CONFIG_NAME))
    if checkpoint is None:
        params_fname = os.path.join(model_folder, C.PARAMS_BEST_NAME)
    else:
        params_fname = os.path.join(model_folder, C.PARAMS_NAME % checkpoint)
    model = SockeyeModel(model_config, inference_only=inference_only, clamp_to_dtype=clamp_to_dtype, train_decoder_only=train_decoder_only, forward_pass_cache_size=forward_pass_cache_size)
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

def load_models(device, model_folders, checkpoints=None, dtype=None, clamp_to_dtype=False, inference_only=False, train_decoder_only=False, allow_missing=False, forward_pass_cache_size=0, knn_index=None):
    """
    Loads a list of models for inference.

    :param device: PyTorch device.
    :param model_folders: List of model folders to load models from.
    :param checkpoints: List of checkpoints to use for each model in model_folders. Use None to load best checkpoint.
    :param dtype: Optional data type (torch.dtype or str) to use. If None, will be inferred from stored model.
    :param clamp_to_dtype: Avoid -inf/inf by clamping outputs to min/max finite values for their dtype.
    :param inference_only: Use the model only for inference, enabling optimizations.
    :param train_decoder_only: Training will only update the decoder. Disable
           autograd for encoder and embeddings to save memory.
    :param allow_missing: Allow missing parameters in the loaded models.
    :param forward_pass_cache_size: If > 0, cache encoder and embedding calculations of forward pass.
    :param knn_index: Optional path to a folder containing a KNN model index.
    :return: List of models, source vocabulary, target vocabulary, source factor vocabularies.
    """
    logger.info('Loading %d model(s) from %s ...', len(model_folders), model_folders)
    load_time_start = time.time()
    models = []
    source_vocabs = []
    target_vocabs = []
    if checkpoints is None:
        checkpoints = [None] * len(model_folders)
    else:
        utils.check_condition(len(checkpoints) == len(model_folders), 'Must provide checkpoints for each model')
    for model_folder, checkpoint in zip(model_folders, checkpoints):
        model, src_vcbs, trg_vcbs = load_model(model_folder, device=device, dtype=dtype, clamp_to_dtype=clamp_to_dtype, checkpoint=checkpoint, inference_only=inference_only, train_decoder_only=train_decoder_only, allow_missing=allow_missing, forward_pass_cache_size=forward_pass_cache_size, knn_index=knn_index)
        models.append(model)
        source_vocabs.append(src_vcbs)
        target_vocabs.append(trg_vcbs)
    first_model_vocabs = source_vocabs[0]
    for fi in range(len(first_model_vocabs)):
        utils.check_condition(vocab.are_identical(*[source_vocabs[i][fi] for i in range(len(source_vocabs))]), 'Source vocabulary ids do not match. Factor %d' % fi)
    first_model_vocabs = target_vocabs[0]
    for fi in range(len(first_model_vocabs)):
        utils.check_condition(vocab.are_identical(*[target_vocabs[i][fi] for i in range(len(target_vocabs))]), 'Target vocabulary ids do not match. Factor %d' % fi)
    load_time = time.time() - load_time_start
    logger.info('%d model(s) loaded in %.4fs', len(models), load_time)
    return (models, source_vocabs[0], target_vocabs[0])