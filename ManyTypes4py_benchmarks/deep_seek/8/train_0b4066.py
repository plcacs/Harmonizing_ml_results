"""
Simple Training CLI.
"""
from . import initial_setup
initial_setup.handle_env_cli_arg()
import argparse
import gc
import logging
import os
import shutil
import sys
import tempfile
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast
import torch
import torch.distributed
import torch.distributed.elastic.multiprocessing.errors
try:
    import deepspeed
except ImportError:
    pass
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

def none_if_negative(val: int) -> Optional[int]:
    return None if val < 0 else val

def _list_to_tuple(v: Union[List, Tuple]) -> Tuple:
    """Convert v to a tuple if it is a list."""
    if isinstance(v, list):
        return tuple(v)
    return v

def _dict_difference(dict1: Dict, dict2: Dict) -> Set:
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
    check_condition(any((args.max_samples, args.max_updates, args.max_seconds, args.max_checkpoints, args.max_num_epochs, args.max_num_checkpoint_not_improved)), 'Please specify at least one stopping criteria: --max-samples --max-updates --max-checkpoints --max-num-epochs --max-num-checkpoint-not-improved')
    n_source_factors = len(args.validation_source_factors)
    if len(args.source_factors_combine) > 1:
        check_condition(n_source_factors == len(args.source_factors_combine), 'The number of combination strategies for source factors does not match the number of source factors.')
    else:
        args.source_factors_combine = args.source_factors_combine * n_source_factors
    if len(args.source_factors_share_embedding) > 1:
        check_condition(n_source_factors == len(args.source_factors_share_embedding), 'The number of vocabulary sharing flags for source factors does not match the number of source factors.')
    else:
        args.source_factors_share_embedding = args.source_factors_share_embedding * n_source_factors
    n_target_factors = len(args.validation_target_factors)
    if len(args.target_factors_combine) > 1:
        check_condition(n_target_factors == len(args.target_factors_combine), 'The number of combination strategies for target factors does not match the number of target factors.')
    else:
        args.target_factors_combine = args.target_factors_combine * n_target_factors
    if len(args.target_factors_share_embedding) > 1:
        check_condition(n_target_factors == len(args.target_factors_share_embedding), 'The number of vocabulary sharing flags for target factors does not match the number of target factors.')
    else:
        args.target_factors_share_embedding = args.target_factors_share_embedding * n_target_factors
    check_condition(not (args.amp and args.apex_amp), 'Use either --amp (safer) or --apex-amp (faster).')
    if args.dtype != C.DTYPE_FP32:
        logger.warning('Specifying a non-float32 dtype to sockeye.train has no effect. For 16-bit or mixed precision training, use one of the following: --amp --apex-amp --deepspeed-fp16 --deepspeed-bf16')
    if args.local_rank is not None:
        check_condition(not args.amp and (not args.apex_amp), 'DeepSpeed mode does not support --amp or --apex-amp. Use --deepspeed-fp16 or --deepspeed-bf16.')
        check_condition(not (args.learning_rate_scheduler_type == C.LR_SCHEDULER_PLATEAU_REDUCE and (not args.no_reload_on_learning_rate_reduce)), 'DeepSpeed mode does not support learning rate schedulers that reload checkpoints. Use a different --learning-rate-scheduler-type or specify --no-reload-on-learning-rate-reduce.')

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
            logger.error('Refusing to overwrite model folder %s as it seems to contain a trained model.', output_folder)
            sys.exit(1)
        else:
            logger.info('The output folder %s already exists, but no training state or parameter file was found. Will start training from scratch.', output_folder)
    elif utils.is_primary_worker():
        os.makedirs(output_folder)
    if utils.is_distributed():
        if utils.is_primary_worker():
            os.makedirs(os.path.join(output_folder, C.DIST_SECONDARY_WORKERS_LOGDIR), exist_ok=True)
        torch.distributed.barrier()
    return resume_training

def create_checkpoint_decoder(args: argparse.Namespace, device: torch.device, sockeye_model: model.SockeyeModel, source_vocabs: List[vocab.Vocab], target_vocabs: List[vocab.Vocab]) -> Optional[checkpoint_decoder.CheckpointDecoder]:
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
        logger.info('You chose %s as the optimized metric, will turn on %s monitoring during training. To control how many validation sentences are used for calculating bleu use the --decode-and-evaluate argument.', args.optimized_metric, args.optimized_metric)
        sample_size = -1
    if sample_size == 0:
        return None
    if utils.using_deepspeed():
        logger.info('Turning off checkpoint decoder when using DeepSpeed')
        return None
    cpd = checkpoint_decoder.CheckpointDecoder(model_folder=args.output, inputs=[args.validation_source] + args.validation_source_factors, references=[args.validation_target] + args.validation_target_factors, sample_size=sample_size, model=sockeye_model, source_vocabs=source_vocabs, target_vocabs=target_vocabs, device=device)
    cpd.warmup()
    return cpd

def use_shared_vocab(args: argparse.Namespace) -> bool:
    """
    True if arguments entail a shared source and target vocabulary.

    :param: args: Arguments as returned by argparse.
    """
    weight_tying_type = args.weight_tying_type
    shared_vocab = args.shared_vocab
    if C.WEIGHT_TYING_SRC in weight_tying_type and C.WEIGHT_TYING_TRG in weight_tying_type:
        if not shared_vocab:
            logger.info('A shared source/target vocabulary will be used as weight tying source/target weight tying is enabled')
        shared_vocab = True
    return shared_vocab

def create_data_iters_and_vocabs(args: argparse.Namespace, max_seq_len_source: int, max_seq_len_target: int, shared_vocab: bool, resume_training: bool, output_folder: str) -> Tuple[data_io.BaseDataIter, data_io.BaseDataIter, data_io.DataConfig, List[vocab.Vocab], List[vocab.Vocab]]:
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
        error_msg = 'Distributed training requires prepared training data. Use `python -m sockeye.prepare_data` and specify with %s' % C.TRAINING_ARG_PREPARED_DATA
        check_condition(args.prepared_data is not None, error_msg)
    either_raw_or_prepared_error_msg = 'Either specify a raw training corpus with %s and %s or a preprocessed corpus with %s.' % (C.TRAINING_ARG_SOURCE, C.TRAINING_ARG_TARGET, C.TRAINING_ARG_PREPARED_DATA)
    if args.prepared_data is not None:
        utils.check_condition(args.source is None and args.target is None, either_raw_or_prepared_error_msg)
        if not resume_training:
            utils.check_condition(args.source_vocab is None and args.target_vocab is None, 'You are using a prepared data folder, which is tied to a vocabulary. To change it you need to rerun data preparation with a different vocabulary.')
        train_iter, validation_iter, data_config, source_vocabs, target_vocabs = data_io.get_prepared_data_iters(prepared_data_dir=args.prepared_data, validation_sources=validation_sources, validation_targets=validation_targets, shared_vocab=shared_vocab, batch_size=args.batch_size, batch_type=args.batch_type, batch_sentences_multiple_of=args.batch_sentences_multiple_of)
        if args.transformer_block_prepended_cross_attention:
            check_condition(data_config.eop_id != C.INVALID_ID, 'In order to block cross-attention between decoder and encoded prepended tokens, please specify the tag indicating the end of prepended text when preparing the data using --end-of-prepending-tag')
        if args.end_of_prepending_tag is not None:
            logger.warning('The end-of-prepending tag specified in the prepared data will be used.')
        check_condition(all([combine in [C.FACTORS_COMBINE_SUM, C.FACTORS_COMBINE_AVERAGE] for combine in args.source_factors_combine]) or len(source_vocabs) == len(args.source_factors_num_embed) + 1, 'Data was prepared with %d source factors, but only provided %d source factor dimensions.' % (len(source_vocabs), len(args.source_factors_num_embed) + 1))
        check_condition(all([combine in [C.FACTORS_COMBINE_SUM, C.FACTORS_COMBINE_AVERAGE] for combine in args.target_factors_combine]) or len(target_vocabs) == len(args.target_factors_num_embed) + 1, 'Data was prepared with %d target factors, but only provided %d target factor dimensions.' % (len(target_vocabs), len(args.target_factors_num_embed) + 1))
        if resume_training:
            model_source_vocabs = vocab.load_source_vocabs(output_folder)
            for i, (v, mv) in enumerate(zip(source_vocabs, model_source_vocabs)):
                utils.check_condition(vocab.are_identical(v, mv), 'Prepared data and resumed model source vocab %d do not match.' % i)
            model_target_vocabs = vocab.load_target_vocabs(output_folder)
            for i, (v, mv) in enumerate(zip(target_vocabs, model_target_vocabs)):
                utils.check_condition(vocab.are_identical(v, mv), 'Prepared data and resumed model target vocab %d do not match.' % i)
        check_condition(data_config.num_source_factors == len(validation_sources), 'Training and validation data must have the same number of source factors, but found %d and %d.' % (data_config.num_source_factors, len(validation_sources)))
        check_condition(data_config.num_target_factors == len(validation_targets), 'Training and validation data must have the same number of target factors, but found %d and %d.' % (data_config.num_target_factors, len(validation_targets)))
        return (train_iter, validation_iter, data_config, source_vocabs, target_vocabs)
    else:
        utils.check_condition(args.prepared_data is None and args.source is not None and (args.target is not None), either_raw_or_prepared_error_msg)
        if args.transformer_block_prepended_cross_attention:
            check_condition(args.end_of_prepending_tag is not None, 'In order to block cross-attention between decoder and encoded prepended tokens, please specify the tag indicating the end of prepended text using --end-of-prepending-tag')
        if resume_training:
            source_vocabs = vocab.load_source_vocabs(output_folder)
            target_vocabs = vocab.load_target_vocabs(output_folder)
            data_info = cast(data_io.DataInfo, Config.load(os.path.join(output_folder, C.DATA_INFO)))
            source_vocab_paths = data_info.source_vocabs
            target_vocab_paths = data_info.target_vocabs
        else:
            source_factor_vocab_paths = [args.source_factor_vocabs[i] if i < len(args.source_factor_vocabs) else None for i in range(len(args.source_factors))]
            source_vocab_paths = [args.source_vocab] + source_factor_vocab_paths
            target_factor_vocab_paths = [args.target_factor_vocabs[i] if i < len(args.target_factor_vocabs) else None for i in range(len(args.target_factors))]
            target_vocab_paths = [args.target_vocab] + target_factor_vocab_paths
            source_vocabs, target_vocabs = vocab.load_or_create_vocabs(shard_source_paths=[[args.source] + args.source_factors], shard_target_paths=[[args.target] + args.target_factors], source_vocab_paths=source_vocab_paths, source_factor_vocab_same_as_source=args.source_factors_share_embedding, target_vocab_paths=target_vocab_paths, target_factor_vocab_same_as_target=args.target_factors_share_embedding, shared_vocab=shared_vocab, num_words_source=num_words_source, num_words_target=num_words_target, word_min_count_source=word_min_count_source, word_min_count_target=word_min_count_target, pad_to_multiple_of=args.pad_vocab_to_multiple_of)
        check_condition(all([combine in [C.FACTORS_COMBINE_SUM, C.FACTORS_COMBINE_AVERAGE] for combine in args.source_factors_combine]) or len(args.source_factors) == len(args.source_factors_num_embed), 'Number of source factor data (%d) differs from provided source factor dimensions (%d)' % (len(args.source_factors), len(args.source_factors_num_embed)))
        check_condition(all([combine in [C.FACTORS_COMBINE_SUM, C.FACTORS_COMBINE_AVERAGE] for combine in args.target_factors_combine]) or len(args.target_factors) == len(args.target_factors_num_embed), 'Number of target factor data (%d) differs from provided source factor dimensions (%d)' % (len(args.target_factors), len(args.target_factors_num_embed)))
        sources = [args.source] + args.source_factors
        sources = [str(os.path.abspath(s)) for s in sources]
        targets = [args.target] +