"""
Simple Training CLI.
"""
from __future__ import annotations
from typing import Any, cast, Callable, Optional, Dict, List, Tuple, Type
import argparse
import gc
import logging
import os
import shutil
import sys
import tempfile

import torch
import torch.distributed
import torch.distributed.elastic.multiprocessing.errors
try:
    import deepspeed
except ImportError:
    deepspeed = None

from . import initial_setup
from . import arguments
from . import checkpoint_decoder
from . import constants as C
from . import convert_deepspeed
from . import data_io
from . import encoder
from . import layers
from . import loss
from . import lr_scheduler
from . import model
from . import optimizers
from . import training
from . import transformer
from . import utils
from . import vocab
from .config import Config
from .log import setup_main_logger
from .utils import check_condition

logger = logging.getLogger(__name__)

def none_if_negative(val: float) -> Optional[float]:
    return None if val < 0 else val

def _list_to_tuple(v: Any) -> Any:
    """Convert v to a tuple if it is a list."""
    if isinstance(v, list):
        return tuple(v)
    return v

def _dict_difference(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> set:
    diffs = set()
    for k, v in dict1.items():
        if k not in dict2 or _list_to_tuple(dict2[k]) != _list_to_tuple(v):
            diffs.add(k)
    return diffs

def check_arg_compatibility(args: argparse.Namespace) -> None:
    """
    Check if some arguments are incompatible with each other.

    :param args: Arguments as returned by argparse.
    """
    check_condition(
        any((
            args.max_samples,
            args.max_updates,
            args.max_seconds,
            args.max_checkpoints,
            args.max_num_epochs,
            args.max_num_checkpoint_not_improved
        )),
        'Please specify at least one stopping criteria: --max-samples --max-updates --max-checkpoints --max-num-epochs --max-num-checkpoint-not-improved'
    )
    n_source_factors = len(args.validation_source_factors)
    if len(args.source_factors_combine) > 1:
        check_condition(
            n_source_factors == len(args.source_factors_combine),
            'The number of combination strategies for source factors does not match the number of source factors.'
        )
    else:
        args.source_factors_combine = args.source_factors_combine * n_source_factors
    if len(args.source_factors_share_embedding) > 1:
        check_condition(
            n_source_factors == len(args.source_factors_share_embedding),
            'The number of vocabulary sharing flags for source factors does not match the number of source factors.'
        )
    else:
        args.source_factors_share_embedding = args.source_factors_share_embedding * n_source_factors
    n_target_factors = len(args.validation_target_factors)
    if len(args.target_factors_combine) > 1:
        check_condition(
            n_target_factors == len(args.target_factors_combine),
            'The number of combination strategies for target factors does not match the number of target factors.'
        )
    else:
        args.target_factors_combine = args.target_factors_combine * n_target_factors
    if len(args.target_factors_share_embedding) > 1:
        check_condition(
            n_target_factors == len(args.target_factors_share_embedding),
            'The number of vocabulary sharing flags for target factors does not match the number of target factors.'
        )
    else:
        args.target_factors_share_embedding = args.target_factors_share_embedding * n_target_factors
    check_condition(
        not (args.amp and args.apex_amp),
        'Use either --amp (safer) or --apex-amp (faster).'
    )
    if args.dtype != C.DTYPE_FP32:
        logger.warning(
            'Specifying a non-float32 dtype to sockeye.train has no effect. '
            'For 16-bit or mixed precision training, use one of the following: '
            '--amp --apex-amp --deepspeed-fp16 --deepspeed-bf16'
        )
    if args.local_rank is not None:
        check_condition(
            not args.amp and (not args.apex_amp),
            'DeepSpeed mode does not support --amp or --apex-amp. Use --deepspeed-fp16 or --deepspeed-bf16.'
        )
        check_condition(
            not (args.learning_rate_scheduler_type == C.LR_SCHEDULER_PLATEAU_REDUCE and (not args.no_reload_on_learning_rate_reduce)),
            'DeepSpeed mode does not support learning rate schedulers that reload checkpoints. '
            'Use a different --learning-rate-scheduler-type or specify --no-reload-on-learning-rate-reduce.'
        )

def check_resume(args: argparse.Namespace, output_folder: str) -> bool:
    """
    Check if we should resume a broken training run.

    :param args: Arguments as returned by argparse.
    :param output_folder: Main output folder for the model.

    :return: Flag signaling if we are resuming training and the directory with
        the training status.
    """
    resume_training = False
    training_state_dir = os.path.join(output_folder, C.TRAINING_STATE_DIRNAME)
    if os.path.exists(output_folder):
        if args.overwrite_output:
            if utils.is_primary_worker():
                logger.info('Removing existing output folder %s.', output_folder)
                shutil.rmtree(output_folder)
                os.makedirs(output_folder)
        elif os.path.exists(training_state_dir):
            old_args = vars(arguments.load_args(os.path.join(output_folder, C.ARGS_STATE_NAME)))
            arg_diffs = _dict_difference(vars(args), old_args) | _dict_difference(old_args, vars(args))
            arg_diffs -= set(C.ARGS_MAY_DIFFER)
            if not arg_diffs:
                resume_training = True
            else:
                logger.error('Mismatch in arguments for training continuation.')
                logger.error('Differing arguments: %s.', ', '.join(arg_diffs))
                sys.exit(1)
        elif os.path.exists(os.path.join(output_folder, C.PARAMS_BEST_NAME)):
            logger.error(
                'Refusing to overwrite model folder %s as it seems to contain a trained model.',
                output_folder
            )
            sys.exit(1)
        else:
            logger.info(
                'The output folder %s already exists, but no training state or parameter file was found. '
                'Will start training from scratch.',
                output_folder
            )
    elif utils.is_primary_worker():
        os.makedirs(output_folder)
    if utils.is_distributed():
        if utils.is_primary_worker():
            os.makedirs(os.path.join(output_folder, C.DIST_SECONDARY_WORKERS_LOGDIR), exist_ok=True)
        torch.distributed.barrier()
    return resume_training

def create_checkpoint_decoder(
    args: argparse.Namespace,
    device: torch.device,
    sockeye_model: model.SockeyeModel,
    source_vocabs: List[vocab.Vocab],
    target_vocabs: List[vocab.Vocab]
) -> Optional[checkpoint_decoder.CheckpointDecoder]:
    """
    Returns a checkpoint decoder or None.

    :param args: Arguments as returned by argparse.
    :param device: Torch device for checkpoint decoder.
    :param sockeye_model: The Sockeye model instance.
    :param source_vocabs: The source vocabs.
    :param target_vocabs: The target vocabs.
    :return: A CheckpointDecoder if --decode-and-evaluate != 0, else None.
    """
    sample_size = args.decode_and_evaluate
    if args.optimized_metric in C.METRICS_REQUIRING_DECODER and sample_size == 0:
        logger.info(
            'You chose %s as the optimized metric, will turn on %s monitoring during training. '
            'To control how many validation sentences are used for calculating bleu use the --decode-and-evaluate argument.',
            args.optimized_metric, args.optimized_metric
        )
        sample_size = -1
    if sample_size == 0:
        return None
    if utils.using_deepspeed():
        logger.info('Turning off checkpoint decoder when using DeepSpeed')
        return None
    cpd = checkpoint_decoder.CheckpointDecoder(
        model_folder=args.output,
        inputs=[args.validation_source] + args.validation_source_factors,
        references=[args.validation_target] + args.validation_target_factors,
        sample_size=sample_size,
        model=sockeye_model,
        source_vocabs=source_vocabs,
        target_vocabs=target_vocabs,
        device=device
    )
    cpd.warmup()
    return cpd

def use_shared_vocab(args: argparse.Namespace) -> bool:
    """
    True if arguments entail a shared source and target vocabulary.

    :param args: Arguments as returned by argparse.
    """
    weight_tying_type = args.weight_tying_type
    shared_vocab = args.shared_vocab
    if C.WEIGHT_TYING_SRC in weight_tying_type and C.WEIGHT_TYING_TRG in weight_tying_type:
        if not shared_vocab:
            logger.info(
                'A shared source/target vocabulary will be used as weight tying '
                'source/target weight tying is enabled'
            )
        shared_vocab = True
    return shared_vocab

def create_data_iters_and_vocabs(
    args: argparse.Namespace,
    max_seq_len_source: int,
    max_seq_len_target: int,
    shared_vocab: bool,
    resume_training: bool,
    output_folder: str
) -> Tuple[
    data_io.DataIterator,
    data_io.DataIterator,
    data_io.DataConfig,
    List[vocab.Vocab],
    List[vocab.Vocab]
]:
    """
    Create the data iterators and the vocabularies.

    :param args: Arguments as returned by argparse.
    :param max_seq_len_source: Source maximum sequence length.
    :param max_seq_len_target: Target maximum sequence length.
    :param shared_vocab: Whether to create a shared vocabulary.
    :param resume_training: Whether to resume training.
    :param output_folder: Output folder.
    :return: The data iterators (train, validation, config_data) as well as the source and target vocabularies.
    """
    num_words_source, num_words_target = args.num_words
    num_words_source = num_words_source if num_words_source > 0 else None
    num_words_target = num_words_target if num_words_target > 0 else None
    word_min_count_source, word_min_count_target = args.word_min_count
    validation_sources = [args.validation_source] + args.validation_source_factors
    validation_sources = [str(os.path.abspath(source)) for source in validation_sources]
    validation_targets = [args.validation_target] + args.validation_target_factors
    validation_targets = [str(os.path.abspath(target)) for target in validation_targets]
    if utils.is_distributed():
        error_msg = (
            'Distributed training requires prepared training data. '
            'Use `python -m sockeye.prepare_data` and specify with %s' % C.TRAINING_ARG_PREPARED_DATA
        )
        check_condition(args.prepared_data is not None, error_msg)
    either_raw_or_prepared_error_msg = (
        'Either specify a raw training corpus with %s and %s or a preprocessed corpus with %s.'
        % (C.TRAINING_ARG_SOURCE, C.TRAINING_ARG_TARGET, C.TRAINING_ARG_PREPARED_DATA)
    )
    if args.prepared_data is not None:
        utils.check_condition(
            args.source is None and args.target is None,
            either_raw_or_prepared_error_msg
        )
        if not resume_training:
            utils.check_condition(
                args.source_vocab is None and args.target_vocab is None,
                'You are using a prepared data folder, which is tied to a vocabulary. '
                'To change it you need to rerun data preparation with a different vocabulary.'
            )
        train_iter, validation_iter, data_config, source_vocabs, target_vocabs = data_io.get_prepared_data_iters(
            prepared_data_dir=args.prepared_data,
            validation_sources=validation_sources,
            validation_targets=validation_targets,
            shared_vocab=shared_vocab,
            batch_size=args.batch_size,
            batch_type=args.batch_type,
            batch_sentences_multiple_of=args.batch_sentences_multiple_of
        )
        if args.transformer_block_prepended_cross_attention:
            check_condition(
                data_config.eop_id != C.INVALID_ID,
                'In order to block cross-attention between decoder and encoded prepended tokens, '
                'please specify the tag indicating the end of prepended text when preparing the data using --end-of-prepending-tag'
            )
        if args.end_of_prepending_tag is not None:
            logger.warning('The end-of-prepending tag specified in the prepared data will be used.')
        check_condition(
            all([
                combine in [C.FACTORS_COMBINE_SUM, C.FACTORS_COMBINE_AVERAGE]
                for combine in args.source_factors_combine
            ]) or len(source_vocabs) == len(args.source_factors_num_embed) + 1,
            'Data was prepared with %d source factors, but only provided %d source factor dimensions.'
            % (len(source_vocabs), len(args.source_factors_num_embed) + 1)
        )
        check_condition(
            all([
                combine in [C.FACTORS_COMBINE_SUM, C.FACTORS_COMBINE_AVERAGE]
                for combine in args.target_factors_combine
            ]) or len(target_vocabs) == len(args.target_factors_num_embed) + 1,
            'Data was prepared with %d target factors, but only provided %d target factor dimensions.'
            % (len(target_vocabs), len(args.target_factors_num_embed) + 1)
        )
        if resume_training:
            model_source_vocabs = vocab.load_source_vocabs(output_folder)
            for i, (v, mv) in enumerate(zip(source_vocabs, model_source_vocabs)):
                utils.check_condition(
                    vocab.are_identical(v, mv),
                    'Prepared data and resumed model source vocab %d do not match.' % i
                )
            model_target_vocabs = vocab.load_target_vocabs(output_folder)
            for i, (v, mv) in enumerate(zip(target_vocabs, model_target_vocabs)):
                utils.check_condition(
                    vocab.are_identical(v, mv),
                    'Prepared data and resumed model target vocab %d do not match.' % i
                )
        check_condition(
            data_config.num_source_factors == len(validation_sources),
            'Training and validation data must have the same number of source factors, but found %d and %d.'
            % (data_config.num_source_factors, len(validation_sources))
        )
        check_condition(
            data_config.num_target_factors == len(validation_targets),
            'Training and validation data must have the same number of target factors, but found %d and %d.'
            % (data_config.num_target_factors, len(validation_targets))
        )
        return (train_iter, validation_iter, data_config, source_vocabs, target_vocabs)
    else:
        utils.check_condition(
            args.prepared_data is None and args.source is not None and (args.target is not None),
            either_raw_or_prepared_error_msg
        )
        if args.transformer_block_prepended_cross_attention:
            check_condition(
                args.end_of_prepending_tag is not None,
                'In order to block cross-attention between decoder and encoded prepended tokens, '
                'please specify the tag indicating the end of prepended text using --end-of-prepending-tag'
            )
        if resume_training:
            source_vocabs = vocab.load_source_vocabs(output_folder)
            target_vocabs = vocab.load_target_vocabs(output_folder)
            data_info = cast(data_io.DataInfo, Config.load(os.path.join(output_folder, C.DATA_INFO)))
            source_vocab_paths = data_info.source_vocabs
            target_vocab_paths = data_info.target_vocabs
        else:
            source_factor_vocab_paths = [
                args.source_factor_vocabs[i] if i < len(args.source_factor_vocabs) else None
                for i in range(len(args.source_factors))
            ]
            source_vocab_paths = [args.source_vocab] + source_factor_vocab_paths
            target_factor_vocab_paths = [
                args.target_factor_vocabs[i] if i < len(args.target_factor_vocabs) else None
                for i in range(len(args.target_factors))
            ]
            target_vocab_paths = [args.target_vocab] + target_factor_vocab_paths
            source_vocabs, target_vocabs = vocab.load_or_create_vocabs(
                shard_source_paths=[[args.source] + args.source_factors],
                shard_target_paths=[[args.target] + args.target_factors],
                source_vocab_paths=source_vocab_paths,
                source_factor_vocab_same_as_source=args.source_factors_share_embedding,
                target_vocab_paths=target_vocab_paths,
                target_factor_vocab_same_as_target=args.target_factors_share_embedding,
                shared_vocab=shared_vocab,
                num_words_source=num_words_source,
                num_words_target=num_words_target,
                word_min_count_source=word_min_count_source,
                word_min_count_target=word_min_count_target,
                pad_to_multiple_of=args.pad_vocab_to_multiple_of
            )
        check_condition(
            all([
                combine in [C.FACTORS_COMBINE_SUM, C.FACTORS_COMBINE_AVERAGE]
                for combine in args.source_factors_combine
            ]) or len(args.source_factors) == len(args.source_factors_num_embed),
            'Number of source factor data (%d) differs from provided source factor dimensions (%d)'
            % (len(args.source_factors), len(args.source_factors_num_embed))
        )
        check_condition(
            all([
                combine in [C.FACTORS_COMBINE_SUM, C.FACTORS_COMBINE_AVERAGE]
                for combine in args.target_factors_combine
            ]) or len(args.target_factors) == len(args.target_factors_num_embed),
            'Number of target factor data (%d) differs from provided source factor dimensions (%d)'
            % (len(args.target_factors), len(args.target_factors_num_embed))
        )
        sources = [args.source] + args.source_factors
        sources = [str(os.path.abspath(s)) for s in sources]
        targets = [args.target] + args.target_factors
        targets = [str(os.path.abspath(t)) for t in targets]
        check_condition(
            len(sources) == len(validation_sources),
            'Training and validation data must have the same number of source factors, but found %d and %d.'
            % (len(source_vocabs), len(validation_sources))
        )
        check_condition(
            len(targets) == len(validation_targets),
            'Training and validation data must have the same number of target factors, but found %d and %d.'
            % (len(source_vocabs), len(validation_sources))
        )
        train_iter, validation_iter, config_data, data_info = data_io.get_training_data_iters(
            sources=sources,
            targets=targets,
            validation_sources=validation_sources,
            validation_targets=validation_targets,
            source_vocabs=source_vocabs,
            target_vocabs=target_vocabs,
            source_vocab_paths=source_vocab_paths,
            target_vocab_paths=target_vocab_paths,
            shared_vocab=shared_vocab,
            batch_size=args.batch_size,
            batch_type=args.batch_type,
            max_seq_len_source=max_seq_len_source,
            max_seq_len_target=max_seq_len_target,
            bucketing=not args.no_bucketing,
            bucket_width=args.bucket_width,
            bucket_scaling=args.bucket_scaling,
            end_of_prepending_tag=args.end_of_prepending_tag,
            batch_sentences_multiple_of=args.batch_sentences_multiple_of
        )
        data_info_fname = os.path.join(output_folder, C.DATA_INFO)
        logger.info("Writing data config to '%s'", data_info_fname)
        data_info.save(data_info_fname)
        return (train_iter, validation_iter, config_data, source_vocabs, target_vocabs)

def create_encoder_config(
    args: argparse.Namespace,
    max_seq_len_source: int,
    max_seq_len_target: int,
    num_embed_source: int
) -> Tuple[transformer.TransformerConfig, int]:
    """
    Create the encoder config.

    :param args: Arguments as returned by argparse.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param num_embed_source: The size of the source embedding.
    :return: The encoder config and the number of hidden units of the encoder.
    """
    encoder_num_layers, _ = args.num_layers
    encoder_transformer_preprocess, _ = args.transformer_preprocess
    encoder_transformer_postprocess, _ = args.transformer_postprocess
    encoder_transformer_model_size, _ = args.transformer_model_size
    total_source_factor_size = 0
    for factor_combine, factor_size in zip(args.source_factors_combine, args.source_factors_num_embed):
        if factor_combine == C.FACTORS_COMBINE_CONCAT:
            total_source_factor_size += factor_size
    if total_source_factor_size > 0:
        logger.info(
            'Encoder transformer-model-size adjusted to account for source factor embeddings: %d -> %d'
            % (encoder_transformer_model_size, num_embed_source + total_source_factor_size)
        )
        encoder_transformer_model_size = num_embed_source + total_source_factor_size
    config_encoder = transformer.TransformerConfig(
        model_size=encoder_transformer_model_size,
        attention_heads=args.transformer_attention_heads[0],
        feed_forward_num_hidden=args.transformer_feed_forward_num_hidden[0],
        act_type=args.transformer_activation_type[0],
        num_layers=encoder_num_layers,
        dropout_attention=args.transformer_dropout_attention[0],
        dropout_act=args.transformer_dropout_act[0],
        dropout_prepost=args.transformer_dropout_prepost[0],
        positional_embedding_type=args.transformer_positional_embedding_type,
        preprocess_sequence=encoder_transformer_preprocess,
        postprocess_sequence=encoder_transformer_postprocess,
        max_seq_len_source=max_seq_len_source,
        max_seq_len_target=max_seq_len_target,
        depth_key_value=encoder_transformer_model_size,
        use_lhuc=(
            args.lhuc is not None and
            (C.LHUC_ENCODER in args.lhuc or C.LHUC_ALL in args.lhuc)
        ),
        decoder_type=args.decoder,
        use_glu=args.transformer_feed_forward_use_glu
    )
    encoder_num_hidden = encoder_transformer_model_size
    return (config_encoder, encoder_num_hidden)

def create_decoder_config(
    args: argparse.Namespace,
    encoder_num_hidden: int,
    max_seq_len_source: int,
    max_seq_len_target: int,
    num_embed_target: int
) -> transformer.TransformerConfig:
    """
    Create the config for the decoder.

    :param args: Arguments as returned by argparse.
    :param encoder_num_hidden: Number of hidden units of the Encoder.
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param num_embed_target: The size of the target embedding.
    :return: The config for the decoder.
    """
    _, decoder_num_layers = args.num_layers
    _, decoder_transformer_preprocess = args.transformer_preprocess
    _, decoder_transformer_postprocess = args.transformer_postprocess
    _, decoder_transformer_model_size = args.transformer_model_size
    total_target_factor_size = 0
    for factor_combine, factor_size in zip(args.target_factors_combine, args.target_factors_num_embed):
        if factor_combine == C.FACTORS_COMBINE_CONCAT:
            total_target_factor_size += factor_size
    if total_target_factor_size > 0:
        logger.info(
            'Decoder transformer-model-size adjusted to account for target factor embeddings: %d -> %d'
            % (decoder_transformer_model_size, num_embed_target + total_target_factor_size)
        )
        decoder_transformer_model_size = num_embed_target + total_target_factor_size
    config_decoder = transformer.TransformerConfig(
        model_size=decoder_transformer_model_size,
        attention_heads=args.transformer_attention_heads[1],
        feed_forward_num_hidden=args.transformer_feed_forward_num_hidden[1],
        act_type=args.transformer_activation_type[1],
        num_layers=decoder_num_layers,
        dropout_attention=args.transformer_dropout_attention[1],
        dropout_act=args.transformer_dropout_act[1],
        dropout_prepost=args.transformer_dropout_prepost[1],
        positional_embedding_type=args.transformer_positional_embedding_type,
        block_prepended_cross_attention=args.transformer_block_prepended_cross_attention,
        preprocess_sequence=decoder_transformer_preprocess,
        postprocess_sequence=decoder_transformer_postprocess,
        max_seq_len_source=max_seq_len_source,
        max_seq_len_target=max_seq_len_target,
        use_lhuc=(
            args.lhuc is not None and
            (C.LHUC_DECODER in args.lhuc or C.LHUC_ALL in args.lhuc)
        ),
        depth_key_value=encoder_num_hidden,
        decoder_type=args.decoder,
        use_glu=args.transformer_feed_forward_use_glu
    )
    return config_decoder

def get_num_embed(args: argparse.Namespace) -> Tuple[int, int]:
    num_embed_source, num_embed_target = args.num_embed
    transformer_model_size_source = args.transformer_model_size[0]
    if not num_embed_source:
        logger.info(
            'Source embedding size was not set it will automatically be adjusted to match the Transformer source model size (%d).',
            transformer_model_size_source
        )
        num_embed_source = transformer_model_size_source
    else:
        check_condition(
            args.transformer_model_size[0] == num_embed_source,
            'Source embedding size must match transformer model size: %s vs. %s'
            % (args.transformer_model_size[0], num_embed_source)
        )
    total_source_factor_size = 0
    for factor_combine, factor_size in zip(args.source_factors_combine, args.source_factors_num_embed):
        if factor_combine == C.FACTORS_COMBINE_CONCAT:
            total_source_factor_size += factor_size
    if total_source_factor_size > 0:
        adjusted_transformer_encoder_model_size = num_embed_source + total_source_factor_size
        check_condition(
            adjusted_transformer_encoder_model_size % 2 == 0 and
            adjusted_transformer_encoder_model_size % args.transformer_attention_heads[0] == 0,
            'Sum of source factor sizes, i.e. num-embed plus source-factors-num-embed, (%d) has to be even and a multiple of encoder attention heads (%d)'
            % (adjusted_transformer_encoder_model_size, args.transformer_attention_heads[0])
        )
    if not num_embed_source:
        num_embed_source = C.DEFAULT_NUM_EMBED
    transformer_model_size_target = args.transformer_model_size[1]
    total_target_factor_size = 0
    for factor_combine, factor_size in zip(args.target_factors_combine, args.target_factors_num_embed):
        if factor_combine == C.FACTORS_COMBINE_CONCAT:
            total_target_factor_size += factor_size
    if not num_embed_target:
        logger.info(
            'Target embedding size was not set it will automatically be adjusted to match the Transformer target model size (%d).',
            transformer_model_size_target
        )
        num_embed_target = transformer_model_size_target
    else:
        check_condition(
            args.transformer_model_size[1] == num_embed_target + total_target_factor_size,
            'Target embedding size must match transformer model size: %s vs. %s'
            % (args.transformer_model_size[1], num_embed_target + total_target_factor_size)
        )
    if total_target_factor_size > 0:
        adjusted_transformer_decoder_model_size = num_embed_target + total_target_factor_size
        check_condition(
            adjusted_transformer_decoder_model_size % 2 == 0 and
            adjusted_transformer_decoder_model_size % args.transformer_attention_heads[0] == 0,
            'Sum of target factor sizes, i.e. num-embed plus target-factors-num-embed, (%d) has to be even and a multiple of encoder attention heads (%d)'
            % (adjusted_transformer_decoder_model_size, args.transformer_attention_heads[0])
        )
        check_condition(
            C.WEIGHT_TYING_SOFTMAX not in args.weight_tying_type,
            "Cannot use weight tying of target input and output embeddings when target factors are defined and to be combined via 'concat'. Use 'sum' instead or disable weight tying"
        )
    if not num_embed_target:
        num_embed_target = C.DEFAULT_NUM_EMBED
    return (num_embed_source, num_embed_target)

def create_model_config(
    args: argparse.Namespace,
    source_vocab_sizes: List[int],
    target_vocab_sizes: List[int],
    max_seq_len_source: int,
    max_seq_len_target: int,
    config_data: data_io.DataConfig
) -> model.ModelConfig:
    """
    Create a ModelConfig from the argument given in the command line.

    :param args: Arguments as returned by argparse.
    :param source_vocab_sizes: The size of the source vocabulary (and source factors).
    :param target_vocab_sizes: The size of the target vocabulary (and target factors).
    :param max_seq_len_source: Maximum source sequence length.
    :param max_seq_len_target: Maximum target sequence length.
    :param config_data: Data config.
    :return: The model configuration.
    """
    num_embed_source, num_embed_target = get_num_embed(args)
    embed_dropout_source, embed_dropout_target = args.embed_dropout
    source_vocab_size, *source_factor_vocab_sizes = source_vocab_sizes
    target_vocab_size, *target_factor_vocab_sizes = target_vocab_sizes
    config_encoder, encoder_num_hidden = create_encoder_config(
        args=args,
        max_seq_len_source=max_seq_len_source,
        max_seq_len_target=max_seq_len_target,
        num_embed_source=num_embed_source
    )
    config_decoder = create_decoder_config(
        args=args,
        encoder_num_hidden=encoder_num_hidden,
        max_seq_len_source=max_seq_len_source,
        max_seq_len_target=max_seq_len_target,
        num_embed_target=num_embed_target
    )
    source_factor_configs: Optional[List[encoder.FactorConfig]] = None
    if len(source_vocab_sizes) > 1:
        source_factors_num_embed = args.source_factors_num_embed
        if not source_factors_num_embed:
            logger.info(
                "Setting all source factor embedding sizes to `num_embed` ('%d')",
                num_embed_source
            )
            source_factors_num_embed = [num_embed_source] * len(source_factor_vocab_sizes)
        else:
            for i, combine in enumerate(args.source_factors_combine):
                if combine in [C.FACTORS_COMBINE_SUM, C.FACTORS_COMBINE_AVERAGE]:
                    logger.info(
                        "Setting embedding size of source factor %d to `num_embed` ('%d') for %s",
                        i + 1,
                        num_embed_source,
                        'summing' if combine == C.FACTORS_COMBINE_SUM else 'averaging'
                    )
                    source_factors_num_embed[i] = num_embed_source
        source_factor_configs = [
            encoder.FactorConfig(size, dim, combine, share)
            for size, dim, combine, share in zip(
                source_factor_vocab_sizes,
                source_factors_num_embed,
                args.source_factors_combine,
                args.source_factors_share_embedding
            )
        ]
    target_factor_configs: Optional[List[encoder.FactorConfig]] = None
    if len(target_vocab_sizes) > 1:
        target_factors_num_embed = args.target_factors_num_embed
        if not target_factors_num_embed:
            logger.info(
                "Setting all target factor embedding sizes to `num_embed` ('%d')",
                num_embed_target
            )
            target_factors_num_embed = [num_embed_target] * len(target_factor_vocab_sizes)
        else:
            for i, combine in enumerate(args.target_factors_combine):
                if combine in [C.FACTORS_COMBINE_SUM, C.FACTORS_COMBINE_AVERAGE]:
                    logger.info(
                        "Setting embedding size of target factor %d to `num_embed` ('%d') for %s",
                        i + 1,
                        num_embed_target,
                        'summing' if combine == C.FACTORS_COMBINE_SUM else 'averaging'
                    )
                    target_factors_num_embed[i] = num_embed_target
        target_factor_configs = [
            encoder.FactorConfig(size, dim, combine, share)
            for size, dim, combine, share in zip(
                target_factor_vocab_sizes,
                target_factors_num_embed,
                args.target_factors_combine,
                args.target_factors_share_embedding
            )
        ]
    config_embed_source = encoder.EmbeddingConfig(
        vocab_size=source_vocab_size,
        num_embed=num_embed_source,
        dropout=embed_dropout_source,
        factor_configs=source_factor_configs,
        allow_sparse_grad=False
    )
    config_embed_target = encoder.EmbeddingConfig(
        vocab_size=target_vocab_size,
        num_embed=num_embed_target,
        dropout=embed_dropout_target,
        factor_configs=target_factor_configs,
        allow_sparse_grad=False
    )
    config_length_task: Optional[layers.LengthRatioConfig] = None
    if args.length_task is not None:
        config_length_task = layers.LengthRatioConfig(
            num_layers=args.length_task_layers,
            weight=args.length_task_weight
        )
    model_config = model.ModelConfig(
        config_data=config_data,
        vocab_source_size=source_vocab_size,
        vocab_target_size=target_vocab_size,
        config_embed_source=config_embed_source,
        config_embed_target=config_embed_target,
        config_encoder=config_encoder,
        config_decoder=config_decoder,
        config_length_task=config_length_task,
        weight_tying_type=args.weight_tying_type,
        neural_vocab_selection=args.neural_vocab_selection,
        neural_vocab_selection_block_loss=args.neural_vocab_selection_block_loss,
        lhuc=(args.lhuc is not None),
        dtype=C.DTYPE_FP32
    )
    return model_config

def create_losses(
    args: argparse.Namespace,
    all_num_classes: List[int]
) -> List[loss.LossFunction]:
    if len(args.target_factors_weight) != len(all_num_classes) - 1:
        check_condition(
            len(args.target_factors_weight) == 1,
            'Must provide the same number of target factor weights as secondary target factors, or one.'
        )
        factor_weights = args.target_factors_weight * (len(all_num_classes) - 1)
    else:
        factor_weights = args.target_factors_weight
    loss_weights = [1.0] + factor_weights
    losses: List[loss.LossFunction] = []
    for i, (num_classes, weight) in enumerate(zip(all_num_classes, loss_weights)):
        name = C.CROSS_ENTROPY
        metric_prefix = '' if i == 0 else f'f{i}-'
        output_name = C.LOGITS_NAME if i == 0 else C.FACTOR_LOGITS_NAME % i
        label_name = C.TARGET_LABEL_NAME if i == 0 else C.TARGET_FACTOR_LABEL_NAME % i
        label_smoothing = args.label_smoothing if i == 0 else 0.0
        losses.append(loss.CrossEntropyLoss(
            name=name,
            weight=weight,
            label_smoothing=label_smoothing,
            dtype=C.DTYPE_FP32,
            output_name=output_name,
            label_name=label_name,
            metric_prefix=metric_prefix,
            label_smoothing_impl=args.label_smoothing_impl
        ))
    if args.length_task is not None:
        weight = args.length_task_weight
        if args.length_task == C.LENGTH_TASK_RATIO:
            losses.append(loss.MSELoss(
                name=f'{C.LENRATIO_NAME}_{C.LINK_NORMAL}',
                weight=weight,
                output_name=C.LENRATIO_NAME,
                label_name=C.LENRATIO_LABEL_NAME
            ))
        else:
            losses.append(loss.PoissonLoss(
                name=f'{C.LENRATIO_NAME}_{C.LINK_POISSON}',
                weight=weight,
                output_name=C.LENRATIO_NAME,
                label_name=C.LENRATIO_LABEL_NAME
            ))
    if args.neural_vocab_selection:
        bow_loss = loss.BinaryCrossEntropyBowLoss(
            name='bow_ce',
            output_name=C.NVS_PRED_NAME,
            weight=args.bow_task_weight,
            pos_weight=args.bow_task_pos_weight,
            num_labels=all_num_classes[0],
            label_name=C.TARGET_LABEL_NAME,
            metric_prefix='bow'
        )
        losses.append(bow_loss)
    return losses

def create_optimizer_config(args: argparse.Namespace) -> optimizers.OptimizerConfig:
    """
    Returns an OptimizerConfig.

    :param args: Arguments as returned by argparse.

    :return: The config dataclass specifying the optimizer and related settings.
    """
    gradient_clipping_threshold = none_if_negative(args.gradient_clipping_threshold)
    if gradient_clipping_threshold is None:
        logger.info(
            'Gradient clipping threshold set to negative value. Will not perform gradient clipping.'
        )
        gradient_clipping_type = C.GRADIENT_CLIPPING_TYPE_NONE
    else:
        gradient_clipping_type = args.gradient_clipping_type
    config = optimizers.OptimizerConfig(
        name=args.optimizer,
        running_on_gpu=not args.use_cpu,
        lr=args.initial_learning_rate,
        betas=args.optimizer_betas,
        eps=args.optimizer_eps,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        gradient_clipping_type=gradient_clipping_type,
        gradient_clipping_threshold=gradient_clipping_threshold
    )
    num_workers = 1 if not utils.is_distributed() else torch.distributed.get_world_size()
    effective_batch_size = args.batch_size * args.update_interval * num_workers
    logger.info(config)
    logger.info(
        f'Gradient accumulation over {args.update_interval} batch(es) by {num_workers} worker(s). '
        f'Effective batch size: {effective_batch_size}'
    )
    return config

def create_deepspeed_config(
    args: argparse.Namespace,
    optimizer_config: optimizers.OptimizerConfig,
    optimizer_class: Type,
    optimizer_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generates a DeepSpeed config dictionary from training arguments. See:
    https://www.deepspeed.ai/docs/config-json/

    :param args: Arguments as returned by argparse.
    :param optimizer_config: Optimizer config.
    :param optimizer_class: Optimizer class.
    :param optimizer_kwargs: Optimizer kwargs.

    :return: Dictionary of config options that can be used to initialize the
             DeepSpeed engine.
    """
    utils.check_condition(
        utils.using_deepspeed(),
        'Initialize DeepSpeed before generating the config'
    )
    ds_config: Dict[str, Any] = {
        'train_micro_batch_size_per_gpu': args.batch_size,
        'gradient_accumulation_steps': args.update_interval,
        'optimizer': {
            'type': optimizer_class.__name__,
            'params': optimizer_kwargs
        },
        'zero_optimization': {'stage': 1},
        'steps_per_print': args.update_interval * args.checkpoint_interval
    }
    if args.deepspeed_fp16:
        utils.update_dict(ds_config, {
            'fp16': {
                'enabled': True,
                'initial_scale_power': 18
            }
        })
    if args.deepspeed_bf16:
        utils.update_dict(ds_config, {
            'bf16': {
                'enabled': True
            }
        })
    if optimizer_config.gradient_clipping_type != C.GRADIENT_CLIPPING_TYPE_NONE:
        utils.update_dict(ds_config, {'gradient_clipping': optimizer_config.gradient_clipping_threshold})
    return ds_config

def unset_requires_grad_for_fixed_params(
    config: model.ModelConfig,
    params: Dict[str, torch.nn.Parameter],
    fixed_param_names: List[str],
    fixed_param_strategy: Optional[str] = None
) -> None:
    utils.check_condition(
        not config.lhuc or fixed_param_strategy is None,
        'LHUC fixes all other parameters and is thus not compatible with other fixing strategies.'
    )
    if config.lhuc:
        fixed_param_names += [name for name in params if not name.endswith('lhuc.weight')]
        logger.info('LHUC enabled, fixing all non-LHUC parameters')
    elif fixed_param_strategy is not None:
        fixed_param_names += fixed_param_names_from_strategy(config, params, fixed_param_strategy)
        logger.info("Fixed param strategy: '%s'", fixed_param_strategy)
    for name in fixed_param_names:
        if name not in params:
            logger.warning(
                "Fixed parameter name '%s' not part of model parameters, ignoring", name
            )
            continue
        params[name].requires_grad = False

def fixed_param_names_from_strategy(
    config: model.ModelConfig,
    params: Dict[str, torch.nn.Parameter],
    strategy: str
) -> List[str]:
    """
    Generate a fixed parameter list given a list of all parameter names and
    a strategy.
    """
    num_encoder_layers = config.config_encoder.num_layers
    num_decoder_layers = config.config_decoder.num_layers

    def is_fixed(name: str) -> bool:
        if strategy == C.FIXED_PARAM_STRATEGY_ALL_EXCEPT_DECODER:
            return not name.startswith(C.DECODER_PREFIX)
        if strategy == C.FIXED_PARAM_STRATEGY_ALL_EXCEPT_OUTER_LAYERS:
            first_encoder_prefix = f'{C.ENCODER_PREFIX}.layers.{0}'
            last_encoder_prefix = f'{C.ENCODER_PREFIX}.layers.{num_encoder_layers - 1}'
            first_decoder_prefix = f'{C.DECODER_PREFIX}.layers.{0}'
            last_decoder_prefix = f'{C.DECODER_PREFIX}.layers.{num_decoder_layers - 1}'
            return not (
                name.startswith(first_encoder_prefix) or
                name.startswith(last_encoder_prefix) or
                name.startswith(first_decoder_prefix) or
                name.startswith(last_decoder_prefix)
            )
        if strategy == C.FIXED_PARAM_STRATEGY_ALL_EXCEPT_EMBEDDINGS:
            return not (
                name.startswith(C.SOURCE_EMBEDDING_PREFIX) or
                name.startswith(C.TARGET_EMBEDDING_PREFIX)
            )
        if strategy == C.FIXED_PARAM_STRATEGY_ALL_EXCEPT_OUTPUT_PROJ:
            return not name.startswith(C.DEFAULT_OUTPUT_LAYER_PREFIX) and (
                not name.startswith(C.NVS_LAYER_PREFIX)
            )
        if strategy == C.FIXED_PARAM_STRATEGY_ALL_EXCEPT_FEED_FORWARD:
            return not (
                name.endswith('ff.ff1.bias') or
                name.endswith('ff.ff1.weight') or
                name.endswith('ff.ff2.bias') or
                name.endswith('ff.ff2.weight')
            )
        if strategy == C.FIXED_PARAM_STRATEGY_ENCODER_AND_SOURCE_EMBEDDINGS:
            return name.startswith(C.ENCODER_PREFIX) or name.startswith(C.SOURCE_EMBEDDING_PREFIX)
        if strategy == C.FIXED_PARAM_STRATEGY_ENCODER_HALF_AND_SOURCE_EMBEDDINGS:
            if name.startswith(C.ENCODER_PREFIX):
                for i in range(num_encoder_layers // 2):
                    if name.startswith(f'{C.ENCODER_PREFIX}.layers.{i}'):
                        return True
            return name.startswith(C.SOURCE_EMBEDDING_PREFIX)
        raise ValueError('Unknown fixed parameter strategy: %s' % strategy)
    
    return [name for name in params if is_fixed(name)]

def main() -> None:
    params = arguments.ConfigArgumentParser(description='Train Sockeye sequence-to-sequence models.')
    arguments.add_train_cli_args(params)
    args = params.parse_args()
    train(args)

@torch.distributed.elastic.multiprocessing.errors.record
def train(
    args: argparse.Namespace,
    custom_metrics_logger: Optional[Callable[[Any, Optional[int]], None]] = None,
    checkpoint_callback: Optional[Callable[[int], None]] = None
) -> training.TrainingState:
    """
    :param custom_metrics_logger: Optional custom metrics logging function. If supplied, takes care of metrics produced
                                  during training in a custom way. It should accept a list or a dictionary of
                                  (metric name, metric value) pairs, and an optional global_step/checkpoint parameter.
    :param checkpoint_callback: An optional callback function (int -> None). The function will be called
                                each time a checkpoint has been reached
    """
    if args.local_rank is not None:
        utils.init_deepspeed()
        check_condition(
            args.local_rank == utils.get_local_rank(),
            f'Mismatch between local rank argument and environment variable: {args.local_rank} != {utils.get_local_rank()}'
        )
    elif args.dist:
        torch.distributed.init_process_group(
            torch.distributed.Backend.GLOO if args.use_cpu else torch.distributed.Backend.NCCL
        )
    if args.dry_run:
        temp_dir = tempfile.TemporaryDirectory()
        args.output = temp_dir.name
        args.max_updates = 0
    check_arg_compatibility(args)
    output_folder = os.path.abspath(args.output)
    resume_training = check_resume(args, output_folder)
    logfile: str = os.path.join(output_folder, C.LOG_NAME)
    console_level: Optional[int] = None
    if not utils.is_primary_worker():
        logfile = os.path.join(output_folder, C.DIST_SECONDARY_WORKERS_LOGDIR, f'{torch.distributed.get_rank()}.{C.LOG_NAME}')
        if args.quiet_secondary_workers:
            args.quiet = True
        console_level = args.loglevel_secondary_workers
    setup_main_logger(
        file_logging=not args.no_logfile,
        console=not args.quiet,
        path=logfile,
        level=args.loglevel,
        console_level=console_level
    )
    utils.log_basic_info(args)
    if utils.is_primary_worker():
        arguments.save_args(args, os.path.join(output_folder, C.ARGS_STATE_NAME))
    max_seq_len_source, max_seq_len_target = args.max_seq_len
    max_seq_len_source = max_seq_len_source + C.SPACE_FOR_XOS
    max_seq_len_target = max_seq_len_target + C.SPACE_FOR_XOS
    logger.info(
        'Adjusting maximum length to reserve space for a BOS/EOS marker. New maximum length: (%d, %d)',
        max_seq_len_source, max_seq_len_target
    )
    device = utils.init_device(args)
    logger.info(f'Training Device: {device}')
    utils.seed_rngs(args.seed)
    train_iter, eval_iter, config_data, source_vocabs, target_vocabs = create_data_iters_and_vocabs(
        args=args,
        max_seq_len_source=max_seq_len_source,
        max_seq_len_target=max_seq_len_target,
        shared_vocab=use_shared_vocab(args),
        resume_training=resume_training,
        output_folder=output_folder
    )
    if max_seq_len_source != config_data.max_seq_len_source:
        logger.info(
            'Maximum source length determined by prepared data. Using %d instead of %d',
            config_data.max_seq_len_source, max_seq_len_source
        )
        max_seq_len_source = config_data.max_seq_len_source
    if max_seq_len_target != config_data.max_seq_len_target:
        logger.info(
            'Maximum target length determined by prepared data. Using %d instead of %d',
            config_data.max_seq_len_target, max_seq_len_target
        )
        max_seq_len_target = config_data.max_seq_len_target
    if utils.is_primary_worker() and (not resume_training):
        vocab.save_source_vocabs(source_vocabs, output_folder)
        vocab.save_target_vocabs(target_vocabs, output_folder)
    source_vocab_sizes = [len(v) for v in source_vocabs]
    target_vocab_sizes = [len(v) for v in target_vocabs]
    logger.info(
        'Vocabulary sizes: source=[%s] target=[%s]',
        '|'.join([str(size) for size in source_vocab_sizes]),
        '|'.join([str(size) for size in target_vocab_sizes])
    )
    model_config = create_model_config(
        args=args,
        source_vocab_sizes=source_vocab_sizes,
        target_vocab_sizes=target_vocab_sizes,
        max_seq_len_source=max_seq_len_source,
        max_seq_len_target=max_seq_len_target,
        config_data=config_data
    )
    trainer_config = training.TrainerConfig(
        output_dir=args.output,
        early_stopping_metric=args.optimized_metric,
        max_params_files_to_keep=args.keep_last_params,
        keep_initializations=args.keep_initializations,
        max_params_files_to_cache=args.cache_last_best_params,
        cache_strategy=args.cache_strategy,
        cache_metric=args.cache_metric,
        checkpoint_interval=args.checkpoint_interval,
        max_num_checkpoint_not_improved=args.max_num_checkpoint_not_improved,
        checkpoint_improvement_threshold=args.checkpoint_improvement_threshold,
        max_checkpoints=args.max_checkpoints,
        min_samples=args.min_samples,
        max_samples=args.max_samples,
        min_updates=args.min_updates,
        max_updates=args.max_updates,
        min_epochs=args.min_num_epochs,
        max_epochs=args.max_num_epochs,
        max_seconds=args.max_seconds,
        update_interval=args.update_interval,
        stop_training_on_decoder_failure=args.stop_training_on_decoder_failure,
        no_reload_on_learning_rate_reduce=args.no_reload_on_learning_rate_reduce
    )
    if trainer_config.min_epochs is not None and trainer_config.max_epochs is not None:
        check_condition(
            trainer_config.min_epochs <= trainer_config.max_epochs,
            'Minimum number of epochs must be smaller than maximum number of epochs'
        )
    optimizer_config = create_optimizer_config(args)
    sockeye_model = model.SockeyeModel(
        model_config,
        clamp_to_dtype=args.clamp_to_dtype,
        train_decoder_only=(args.fixed_param_strategy == C.FIXED_PARAM_STRATEGY_ALL_EXCEPT_DECODER)
    )
    if not utils.using_deepspeed():
        sockeye_model.to(device)
    sockeye_model.apply(model.initialize_parameters)
    if args.params is not None:
        sockeye_model.load_parameters(
            filename=args.params,
            device=device,
            allow_missing=args.allow_missing_params or model_config.lhuc,
            ignore_extra=args.ignore_extra_params
        )
    unset_requires_grad_for_fixed_params(
        config=model_config,
        params=dict(sockeye_model.named_parameters()),
        fixed_param_names=args.fixed_param_names,
        fixed_param_strategy=args.fixed_param_strategy
    )
    utils.log_parameters(sockeye_model)
    losses = create_losses(args, all_num_classes=target_vocab_sizes)
    optimizer_class, optimizer_kwargs, zero_grad_kwargs = optimizers.get_optimizer(optimizer_config)
    optimizer: Optional[torch.optim.Optimizer] = (
        optimizer_class(sockeye_model.parameters(), **optimizer_kwargs)
        if not utils.using_deepspeed() else None
    )
    lr_scheduler_class, lr_scheduler_kwargs = lr_scheduler.get_lr_scheduler(
        args.learning_rate_scheduler_type,
        args.initial_learning_rate,
        args.learning_rate_reduce_factor,
        args.learning_rate_reduce_num_not_improved,
        args.learning_rate_warmup,
        args.max_updates
    )
    _lr_scheduler: Optional[lr_scheduler.LRScheduler] = (
        lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
        if lr_scheduler_class is not None else None
    )
    ds_config: Optional[Dict[str, Any]] = (
        create_deepspeed_config(args, optimizer_config, optimizer_class, optimizer_kwargs)
        if utils.using_deepspeed() else None
    )
    training_model: torch.nn.Module = sockeye_model
    if args.apex_amp:
        try:
            import apex.amp
        except ImportError:
            logger.error(
                'Cannot import NVIDIA Apex AMP. Please install Apex: https://github.com/NVIDIA/apex'
            )
            sys.exit(1)
        training_model, optimizer = apex.amp.initialize(training_model, optimizer, opt_level='O2')
    if utils.using_deepspeed():
        logger.info('Skipping SockeyeModel trace when using DeepSpeed')
    else:
        logger.info('Tracing SockeyeModel on a validation batch')
        batch = eval_iter.next().load(device=device)
        with torch.cuda.amp.autocast(cache_enabled=False) if args.amp else utils.no_context():
            training_model = torch.jit.trace(
                training_model,
                (batch.source, batch.source_length, batch.target, batch.target_length),
                strict=False
            )
        eval_iter.reset()
    if utils.is_distributed() and (not utils.using_deepspeed()):
        training_model = torch.nn.parallel.DistributedDataParallel(
            training_model,
            device_ids=None if args.use_cpu else [device],
            output_device=None if args.use_cpu else device
        )
    model_object = training.ModelWithLoss(
        model=training_model,
        losses=losses
    )
    if utils.using_deepspeed():
        if deepspeed is None:
            raise ImportError('DeepSpeed is not installed.')
        model_object, optimizer, _, _lr_scheduler = deepspeed.initialize(
            model=model_object,
            model_parameters=sockeye_model.parameters(),
            lr_scheduler=_lr_scheduler,
            config=ds_config
        )
        _lr_scheduler.step()
    trainer = training.EarlyStoppingTrainer(
        config=trainer_config,
        optimizer_config=optimizer_config,
        sockeye_model=sockeye_model,
        model_object=model_object,
        optimizer=optimizer,
        lr_scheduler=_lr_scheduler,
        zero_grad_kwargs=zero_grad_kwargs,
        loss_functions=losses,
        device=device,
        using_amp=args.amp,
        using_apex_amp=args.apex_amp,
        custom_metrics_logger=custom_metrics_logger,
        checkpoint_callback=checkpoint_callback
    )
    checkpoint_decoder_instance: Optional[checkpoint_decoder.CheckpointDecoder] = None
    if utils.is_primary_worker():
        checkpoint_decoder_instance = create_checkpoint_decoder(
            args,
            device,
            sockeye_model,
            source_vocabs,
            target_vocabs
        )
    torch.cuda.empty_cache()
    gc.collect()
    training_state = trainer.fit(
        train_iter=train_iter,
        validation_iter=eval_iter,
        checkpoint_decoder=checkpoint_decoder_instance
    )
    if utils.using_deepspeed() and utils.is_primary_worker():
        del model_object
        del sockeye_model
        torch.cuda.empty_cache()
        gc.collect()
        convert_deepspeed.convert_model_checkpoints(
            trainer_config.output_dir,
            keep_deepspeed=False
        )
    return training_state

if __name__ == '__main__':
    main()
