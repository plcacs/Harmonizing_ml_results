import argparse
import logging
import sys
import time
from contextlib import ExitStack
from typing import Dict, Generator, List, Optional, Union
from sockeye.lexicon import load_restrict_lexicon, RestrictLexicon
from sockeye.log import setup_main_logger
from sockeye.model import load_models
from sockeye.output_handler import get_output_handler, OutputHandler
from sockeye.utils import log_basic_info, check_condition, grouper, smart_open, seed_rngs
from . import arguments
from . import constants as C
from . import inference
from . import utils

logger: logging.Logger = logging.getLogger(__name__)

def main() -> None:
    params: argparse.ArgumentParser = arguments.ConfigArgumentParser(description='Translate CLI')
    arguments.add_translate_cli_args(params)
    args: argparse.Namespace = params.parse_args()
    run_translate(args)

def run_translate(args: argparse.Namespace) -> None:
    seed_rngs(args.seed if args.seed is not None else int(time.time()))
    if args.output is not None:
        setup_main_logger(console=not args.quiet, file_logging=not args.no_logfile, path='%s.%s' % (args.output, C.LOG_NAME), level=args.loglevel)
    else:
        setup_main_logger(file_logging=False, level=args.loglevel)
    log_basic_info(args)
    if args.nbest_size > 1:
        if args.output_type != C.OUTPUT_HANDLER_JSON:
            logger.warning("For nbest translation, you must specify `--output-type '%s'; overriding your setting of '%s'.", C.OUTPUT_HANDLER_JSON, args.output_type)
            args.output_type = C.OUTPUT_HANDLER_JSON
    output_handler: OutputHandler = get_output_handler(args.output_type, args.output)
    device = utils.init_device(args)
    logger.info(f'Translate Device: {device}')
    models, source_vocabs, target_vocabs = load_models(device=device, model_folders=args.models, checkpoints=args.checkpoints, dtype=args.dtype, clamp_to_dtype=args.clamp_to_dtype, inference_only=True, knn_index=args.knn_index)
    restrict_lexicon: Optional[Union[RestrictLexicon, Dict[str, RestrictLexicon]]] = None
    if args.restrict_lexicon is not None:
        logger.info(str(args.restrict_lexicon))
        if len(args.restrict_lexicon) == 1:
            restrict_lexicon = load_restrict_lexicon(args.restrict_lexicon[0][1], source_vocabs[0], target_vocabs[0], k=args.restrict_lexicon_topk)
            logger.info(f'Loaded a single lexicon ({args.restrict_lexicon[0][0]}) that will be applied to all inputs.')
        else:
            check_condition(args.json_input, 'JSON input is required when using multiple lexicons for vocabulary restriction')
            restrict_lexicon = dict()
            for key, path in args.restrict_lexicon:
                lexicon = load_restrict_lexicon(path, source_vocabs[0], target_vocabs[0], k=args.restrict_lexicon_topk)
                restrict_lexicon[key] = lexicon
    brevity_penalty_weight: float = args.brevity_penalty_weight
    if args.brevity_penalty_type == C.BREVITY_PENALTY_CONSTANT:
        if args.brevity_penalty_constant_length_ratio > 0.0:
            constant_length_ratio: float = args.brevity_penalty_constant_length_ratio
        else:
            constant_length_ratio = sum((model.length_ratio_mean for model in models)) / len(models)
            logger.info('Using average of constant length ratios saved in the model configs: %f', constant_length_ratio)
    elif args.brevity_penalty_type == C.BREVITY_PENALTY_LEARNED:
        constant_length_ratio: float = -1.0
    elif args.brevity_penalty_type == C.BREVITY_PENALTY_NONE:
        brevity_penalty_weight = 0.0
        constant_length_ratio = -1.0
    else:
        raise ValueError('Unknown brevity penalty type %s' % args.brevity_penalty_type)
    for model in models:
        model.eval()
    scorer = inference.CandidateScorer(length_penalty_alpha=args.length_penalty_alpha, length_penalty_beta=args.length_penalty_beta, brevity_penalty_weight=brevity_penalty_weight)
    scorer.to(models[0].dtype)
    translator = inference.Translator(device=device, ensemble_mode=args.ensemble_mode, scorer=scorer, batch_size=args.batch_size, beam_size=args.beam_size, beam_search_stop=args.beam_search_stop, nbest_size=args.nbest_size, models=models, source_vocabs=source_vocabs, target_vocabs=target_vocabs, restrict_lexicon=restrict_lexicon, strip_unknown_words=args.strip_unknown_words, sample=args.sample, output_scores=output_handler.reports_score(), constant_length_ratio=constant_length_ratio, knn_lambda=args.knn_lambda, max_output_length_num_stds=args.max_output_length_num_stds, max_input_length=args.max_input_length, max_output_length=args.max_output_length, prevent_unk=args.prevent_unk, greedy=args.greedy, skip_nvs=args.skip_nvs, nvs_thresh=args.nvs_thresh)
    read_and_translate(translator=translator, output_handler=output_handler, chunk_size=args.chunk_size, input_file=args.input, input_factors=args.input_factors, input_is_json=args.json_input)

def make_inputs(input_file: Optional[str], translator: inference.Translator, input_is_json: bool, input_factors: Optional[List[str]] = None) -> Generator:
    ...

def read_and_translate(translator: inference.Translator, output_handler: OutputHandler, chunk_size: Optional[int], input_file: Optional[str] = None, input_factors: Optional[List[str]] = None, input_is_json: bool = False) -> None:
    ...

def translate(output_handler: OutputHandler, trans_inputs: List, translator: inference.Translator) -> float:
    ...

if __name__ == '__main__':
    main()
