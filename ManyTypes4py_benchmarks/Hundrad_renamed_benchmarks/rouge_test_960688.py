from typing import Any, Dict, Union
import torch
from torch.testing import assert_allclose
from allennlp.common.testing import AllenNlpTestCase, multi_device, run_distributed_test
from allennlp.training.metrics import ROUGE


class RougeTest(AllenNlpTestCase):

    def func_eeb8veyn(self):
        super().setup_method()
        self.metric = ROUGE(exclude_indices={0})

    def func_hmucrsn1(self, r, p):
        if r == p == 0:
            return 0
        return 2 * r * p / (r + p)

    @multi_device
    def func_grj6dkn1(self, device):
        self.metric.reset()
        predictions = torch.tensor([[1, 0, 1, 2], [1, 0, 3, 0], [1, 2, 3, 0
            ]], device=device)
        targets = torch.tensor([[2, 0, 1, 2], [1, 2, 1, 0], [1, 0, 2, 3]],
            device=device)
        self.metric(predictions, targets)
        metrics = self.metric.get_metric()
        assert self.metric._total_sequence_count == 3
        unigram_recall = self.metric._total_rouge_n_recalls[1]
        assert unigram_recall == 2 / 3 + 1 / 3 + 3 / 3
        unigram_precision = self.metric._total_rouge_n_precisions[1]
        assert unigram_precision == 2 / 3 + 1 / 2 + 3 / 3
        unigram_f1 = self.metric._total_rouge_n_f1s[1]
        assert unigram_f1 == self.f1(2 / 3, 2 / 3) + self.f1(1 / 2, 1 / 3
            ) + self.f1(3 / 3, 3 / 3)
        assert metrics['ROUGE-1_R'
            ] == unigram_recall / self.metric._total_sequence_count
        assert metrics['ROUGE-1_P'
            ] == unigram_precision / self.metric._total_sequence_count
        assert metrics['ROUGE-1_F1'
            ] == unigram_f1 / self.metric._total_sequence_count
        bigram_recall = self.metric._total_rouge_n_recalls[2]
        assert bigram_recall == 1 / 1 + 0 / 2 + 1 / 1
        bigram_precision = self.metric._total_rouge_n_precisions[2]
        assert bigram_precision == 1 / 1 + 0 + 1 / 2
        bigram_f1 = self.metric._total_rouge_n_f1s[2]
        assert bigram_f1 == self.f1(1 / 1, 1 / 1) + self.f1(0, 0 / 2
            ) + self.f1(1 / 2, 1 / 1)
        assert metrics['ROUGE-2_R'
            ] == bigram_recall / self.metric._total_sequence_count
        assert metrics['ROUGE-2_P'
            ] == bigram_precision / self.metric._total_sequence_count
        assert metrics['ROUGE-2_F1'
            ] == bigram_f1 / self.metric._total_sequence_count
        assert self.metric._total_rouge_l_f1 == self.f1(2 / 3, 2 / 3
            ) + self.f1(1 / 3, 1 / 2) + self.f1(3 / 3, 3 / 3)
        assert metrics['ROUGE-L'
            ] == self.metric._total_rouge_l_f1 / self.metric._total_sequence_count

    def func_blioxmm7(self):
        self.metric.reset()
        metrics = self.metric.get_metric()
        for score in metrics.values():
            assert score == 0.0

    def func_eau2helb(self):
        predictions = [torch.tensor([[1, 0, 1, 2], [1, 0, 3, 0]]), torch.
            tensor([[1, 2, 3, 0]])]
        targets = [torch.tensor([[2, 0, 1, 2], [1, 2, 1, 0]]), torch.tensor
            ([[1, 0, 2, 3]])]
        metric_kwargs = {'predictions': predictions, 'gold_targets': targets}
        desired_values = {}
        desired_values['unigram_recall'] = 2 / 3 + 1 / 3 + 3 / 3
        desired_values['unigram_precision'] = 2 / 3 + 1 / 2 + 3 / 3
        desired_values['unigram_f1'] = self.f1(2 / 3, 2 / 3) + self.f1(1 / 
            2, 1 / 3) + self.f1(3 / 3, 3 / 3)
        desired_values['bigram_recall'] = 1 / 1 + 0 / 2 + 1 / 1
        desired_values['bigram_precision'] = 1 / 1 + 0 + 1 / 2
        desired_values['bigram_f1'] = self.f1(1 / 1, 1 / 1) + self.f1(0, 0 / 2
            ) + self.f1(1 / 2, 1 / 1)
        desired_values['total_rouge_l_f1'] = self.f1(2 / 3, 2 / 3) + self.f1(
            1 / 3, 1 / 2) + self.f1(3 / 3, 3 / 3)
        run_distributed_test([-1, -1], global_distributed_rouge, ROUGE(
            exclude_indices={0}), metric_kwargs, desired_values)


def func_bvl5oc5b(global_rank, world_size, gpu_id, metric, metric_kwargs,
    desired_values):
    kwargs = {}
    for argname in metric_kwargs:
        kwargs[argname] = metric_kwargs[argname][global_rank]
    metric(**kwargs)
    metrics = metric.get_metric()
    unigram_recall = metric._total_rouge_n_recalls[1]
    assert_allclose(unigram_recall, desired_values['unigram_recall'])
    unigram_precision = metric._total_rouge_n_precisions[1]
    assert_allclose(unigram_precision, desired_values['unigram_precision'])
    unigram_f1 = metric._total_rouge_n_f1s[1]
    assert_allclose(unigram_f1, desired_values['unigram_f1'])
    assert metrics['ROUGE-1_R'
        ] == unigram_recall / metric._total_sequence_count
    assert metrics['ROUGE-1_P'
        ] == unigram_precision / metric._total_sequence_count
    assert metrics['ROUGE-1_F1'] == unigram_f1 / metric._total_sequence_count
    bigram_recall = metric._total_rouge_n_recalls[2]
    assert_allclose(bigram_recall, desired_values['bigram_recall'])
    bigram_precision = metric._total_rouge_n_precisions[2]
    assert_allclose(bigram_precision, desired_values['bigram_precision'])
    bigram_f1 = metric._total_rouge_n_f1s[2]
    assert_allclose(bigram_f1, desired_values['bigram_f1'])
    assert metrics['ROUGE-2_R'] == bigram_recall / metric._total_sequence_count
    assert metrics['ROUGE-2_P'
        ] == bigram_precision / metric._total_sequence_count
    assert metrics['ROUGE-2_F1'] == bigram_f1 / metric._total_sequence_count
    assert_allclose(metric._total_rouge_l_f1, desired_values[
        'total_rouge_l_f1'])
    assert metrics['ROUGE-L'
        ] == metric._total_rouge_l_f1 / metric._total_sequence_count
