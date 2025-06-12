import logging
import os
import platform
import sys
from tempfile import TemporaryDirectory
from typing import List
from unittest.mock import patch
import pytest
import torch as pt
import sockeye.average
import sockeye.checkpoint_decoder
import sockeye.evaluate
from sockeye import constants as C
from sockeye.config import Config
from sockeye.model import load_model
from sockeye.test_utils import run_train_translate, tmp_digits_dataset
from test.common import check_train_translate
logger = logging.getLogger(__name__)
_TRAIN_LINE_COUNT = 20
_TRAIN_LINE_COUNT_EMPTY = 1
_DEV_LINE_COUNT = 5
_TEST_LINE_COUNT = 5
_TEST_LINE_COUNT_EMPTY = 2
_LINE_MAX_LENGTH = 9
_TEST_MAX_LENGTH = 20
_EOP_TAG = '<EOP>'
ENCODER_DECODER_SETTINGS_TEMPLATE = [('--encoder transformer --decoder {decoder} --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8 --transformer-feed-forward-num-hidden 16 --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr --weight-tying-type src_trg_softmax --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 2 --checkpoint-interval 20 --optimizer adam --initial-learning-rate 0.01 --learning-rate-scheduler none', '--beam-size 2 --nbest-size 2', False, 0, 0), ('--encoder transformer --decoder {decoder} --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8 --transformer-feed-forward-num-hidden 16 --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr --weight-tying-type src_trg --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 0 --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --neural-vocab-selection logit_max --bow-task-weight 2', '--beam-size 2 --nbest-size 2', False, 0, 0), (f'--encoder transformer --decoder {{decoder}} --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8 --transformer-feed-forward-num-hidden 16 --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr --weight-tying-type src_trg --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 2 --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --end-of-prepending-tag {_EOP_TAG} --transformer-block-prepended-cross-attention', '--beam-size 2 --nbest-size 2', False, 0, 0), ('--encoder transformer --decoder {decoder} --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8 --transformer-feed-forward-num-hidden 16 --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr --weight-tying-type src_trg --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 2 --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01', '--beam-size 1 --greedy', True, 0, 0), ('--encoder transformer --decoder {decoder} --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8 --transformer-feed-forward-num-hidden 16 --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr --weight-tying-type trg_softmax --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 2 --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --source-factors-combine sum concat average --source-factors-share-embedding true false true --source-factors-num-embed 8 2 8 --target-factors-combine sum --target-factors-share-embedding false --target-factors-num-embed 8', '--beam-size 2 --beam-search-stop first', True, 3, 1), ('--encoder transformer --decoder transformer --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8 --transformer-feed-forward-num-hidden 16 --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr --weight-tying-type src_trg_softmax --batch-size 2 --max-updates 2 --batch-type sentence  --decode-and-evaluate 2 --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --lhuc all', '--beam-size 2', False, 0, 0), ('--encoder transformer --decoder {decoder} --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8 --transformer-feed-forward-num-hidden 16 --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr --weight-tying-type src_trg_softmax --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 2 --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --length-task ratio --length-task-weight 1.0 --length-task-layers 1', '--beam-size 2 --brevity-penalty-type learned --brevity-penalty-weight 1.0', True, 0, 0), ('--encoder transformer --decoder {decoder} --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8 --transformer-feed-forward-num-hidden 16 --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr --weight-tying-type src_trg_softmax --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 2 --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --length-task length --length-task-weight 1.0 --length-task-layers 1', '--beam-size 2 --brevity-penalty-type constant --brevity-penalty-weight 2.0 --brevity-penalty-constant-length-ratio 1.5', False, 0, 0), ('--encoder transformer --decoder {decoder} --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8 --transformer-feed-forward-num-hidden 16 --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr --weight-tying-type src_trg_softmax --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 2 --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --clamp-to-dtype', '--beam-size 2 --clamp-to-dtype', False, 0, 0), ('--encoder transformer --decoder {decoder} --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 8 --num-embed 8 --transformer-feed-forward-num-hidden 16 --transformer-dropout-prepost 0.1 --transformer-preprocess n --transformer-postprocess dr --weight-tying-type src_trg_softmax --batch-size 2 --max-updates 2 --batch-type sentence --decode-and-evaluate 2 --checkpoint-interval 2 --optimizer adam --initial-learning-rate 0.01 --fixed-param-strategy ' + C.FIXED_PARAM_STRATEGY_ALL_EXCEPT_DECODER, '--beam-size 2 --dtype bfloat16', False, 0, 0)]
TEST_CASES = [(train_params.format(decoder=decoder), *other_params) for decoder in C.DECODERS for train_params, *other_params in ENCODER_DECODER_SETTINGS_TEMPLATE]

@pytest.mark.parametrize('train_params, translate_params, use_prepared_data,n_source_factors, n_target_factors', TEST_CASES)
def test_seq_copy(train_params, translate_params, use_prepared_data, n_source_factors, n_target_factors):
    """
    Task: copy short sequences of digits
    """
    source_text_prefix_token = _EOP_TAG if '--end-of-prepending-tag' in train_params else ''
    with tmp_digits_dataset(prefix='test_seq_copy', train_line_count=_TRAIN_LINE_COUNT, train_line_count_empty=_TRAIN_LINE_COUNT_EMPTY, train_max_length=_LINE_MAX_LENGTH, dev_line_count=_DEV_LINE_COUNT, dev_max_length=_LINE_MAX_LENGTH, test_line_count=_TEST_LINE_COUNT, test_line_count_empty=_TEST_LINE_COUNT_EMPTY, test_max_length=_TEST_MAX_LENGTH, sort_target=False, with_n_source_factors=n_source_factors, with_n_target_factors=n_target_factors, source_text_prefix_token=source_text_prefix_token) as data:
        check_train_translate(train_params=train_params, translate_params=translate_params, data=data, use_prepared_data=use_prepared_data, max_seq_len=_LINE_MAX_LENGTH, compare_output=False)
TINY_TEST_MODEL = [(' --num-layers 2 --transformer-attention-heads 2 --transformer-model-size 4 --num-embed 4 --transformer-feed-forward-num-hidden 4 --weight-tying-type src_trg_softmax --batch-size 2 --batch-type sentence --max-updates 4 --decode-and-evaluate 2 --checkpoint-interval 4', '--beam-size 1')]

@pytest.mark.parametrize('train_params, translate_params', TINY_TEST_MODEL)
def test_other_clis(train_params, translate_params):
    """
    Task: test CLIs and core features other than train & translate.
    """
    with tmp_digits_dataset(prefix='test_other_clis', train_line_count=_TRAIN_LINE_COUNT, train_line_count_empty=_TRAIN_LINE_COUNT_EMPTY, train_max_length=_LINE_MAX_LENGTH, dev_line_count=_DEV_LINE_COUNT, dev_max_length=_LINE_MAX_LENGTH, test_line_count=_TEST_LINE_COUNT, test_line_count_empty=0, test_max_length=_TEST_MAX_LENGTH) as data:
        data = run_train_translate(train_params=train_params, translate_params=translate_params, data=data, max_seq_len=_LINE_MAX_LENGTH)
        _test_checkpoint_decoder(data['dev_source'], data['dev_target'], data['model'])
        _test_parameter_averaging(data['model'])
        _test_evaluate_cli(data['test_outputs'], data['test_target'])

def _test_evaluate_cli(test_outputs, test_target_path):
    """
    Runs sockeye-evaluate CLI with translations and a reference file.
    """
    with TemporaryDirectory(prefix='test_evaluate') as work_dir:
        out_path = os.path.join(work_dir, 'hypotheses')
        with open(out_path, 'w') as fd:
            for output in test_outputs:
                print(output['translation'], file=fd)
        eval_params = '{} --hypotheses {hypotheses} --references {references} --metrics {metrics}'.format(sockeye.evaluate.__file__, hypotheses=out_path, references=test_target_path, metrics='bleu chrf rouge1 ter')
        with patch.object(sys, 'argv', eval_params.split()):
            sockeye.evaluate.main()

def _test_parameter_averaging(model_path):
    """
    Runs parameter averaging with all available strategies
    """
    for strategy in C.AVERAGE_CHOICES:
        points = sockeye.average.find_checkpoints(model_path=model_path, size=4, strategy=strategy, metric=C.PERPLEXITY)
        assert len(points) > 0
        averaged_params = sockeye.average.average(points)
        assert averaged_params

def _test_checkpoint_decoder(dev_source_path, dev_target_path, model_path):
    """
    Runs checkpoint decoder on 10% of the dev data and checks whether metric
    keys are present in the result dict. Also checks that we can reload model
    parameters after running the checkpoint decoder (case when using the
    plateau-reduce scheduler).
    """
    with open(dev_source_path) as dev_fd:
        num_dev_sent = sum((1 for _ in dev_fd))
    sample_size = min(1, int(num_dev_sent * 0.1))
    model, source_vocabs, target_vocabs = load_model(model_folder=model_path, device=pt.device('cpu'))
    cp_decoder = sockeye.checkpoint_decoder.CheckpointDecoder(device=pt.device('cpu'), inputs=[dev_source_path], references=[dev_target_path], source_vocabs=source_vocabs, target_vocabs=target_vocabs, model=model, model_folder=model_path, sample_size=sample_size, batch_size=2, beam_size=2)
    cp_metrics = cp_decoder.decode_and_evaluate()
    logger.info('Checkpoint decoder metrics: %s', cp_metrics)
    assert 'bleu' in cp_metrics
    assert 'chrf' in cp_metrics
    assert 'decode-walltime' in cp_metrics
    model.load_parameters(os.path.join(model_path, C.PARAMS_BEST_NAME), device=pt.device('cpu'))