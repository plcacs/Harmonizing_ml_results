import json
import random
from typing import NamedTuple, Any, Union, Callable, Dict, List
import numpy
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import torch
import pytest
from flaky import flaky
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase, run_distributed_test
from allennlp.common.util import sanitize
from allennlp.data import Token, Vocabulary
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer, TokenCharactersIndexer, SingleIdTokenIndexer
from allennlp.nn import util
from allennlp.nn.parallel import ShardedModuleMixin
from allennlp.models import load_archive

class TestNnUtil(AllenNlpTestCase):
    def test_get_sequence_lengths_from_binary_mask(self) -> None:
        binary_mask: torch.Tensor = torch.tensor([[True, True, True, False, False, False], [True, True, False, False, False, False], [True, True, True, True, True, True], [True, False, False, False, False, False]])
        lengths: torch.Tensor = util.get_lengths_from_binary_sequence_mask(binary_mask)
        numpy.testing.assert_array_equal(lengths.numpy(), numpy.array([3, 2, 6, 1]))

    def test_get_mask_from_sequence_lengths(self) -> None:
        sequence_lengths: torch.LongTensor = torch.LongTensor([4, 3, 1, 4, 2])
        mask: torch.Tensor = util.get_mask_from_sequence_lengths(sequence_lengths, 5).data.numpy()
        assert_almost_equal(mask, [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])

    def test_get_sequence_lengths_converts_to_long_tensor_and_avoids_variable_overflow(self) -> None:
        binary_mask: torch.Tensor = torch.ones(2, 260).bool()
        lengths: torch.Tensor = util.get_lengths_from_binary_sequence_mask(binary_mask)
        numpy.testing.assert_array_equal(lengths.data.numpy(), numpy.array([260, 260]))

    def test_clamp_tensor(self) -> None:
        i: torch.LongTensor = torch.LongTensor([[0, 1, 1, 0], [2, 0, 2, 2]])
        v: torch.FloatTensor = torch.FloatTensor([3, 4, -5, 3])
        tensor: torch.sparse.FloatTensor = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
        clamped_tensor: torch.Tensor = util.clamp_tensor(tensor, minimum=-3, maximum=3).to_dense()
        assert_almost_equal(clamped_tensor, [[0, 0, 3], [3, 0, -3]])
        i: torch.LongTensor = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
        v: torch.FloatTensor = torch.FloatTensor([3, 4, -5])
        tensor: torch.sparse.FloatTensor = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
        clamped_tensor: torch.Tensor = util.clamp_tensor(tensor, minimum=-3, maximum=3).to_dense()
        assert_almost_equal(clamped_tensor, [[0, 0, 3], [3, 0, -3]])
        tensor: torch.Tensor = torch.tensor([[5, -4, 3], [-3, 0, -30]])
        clamped_tensor: torch.Tensor = util.clamp_tensor(tensor, minimum=-3, maximum=3)
        assert_almost_equal(clamped_tensor, [[3, -3, 3], [-3, 0, -3]])

    def test_sort_tensor_by_length(self) -> None:
        tensor: torch.Tensor = torch.rand([5, 7, 9])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 1:, :] = 0
        tensor[3, 5:, :] = 0
        sequence_lengths: torch.LongTensor = torch.LongTensor([3, 4, 1, 5, 7])
        sorted_tensor, sorted_lengths, reverse_indices, _ = util.sort_batch_by_length(tensor, sequence_lengths)
        numpy.testing.assert_array_equal(sorted_tensor[1, 5:, :].data.numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[2, 4:, :].data.numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[3, 3:, :].data.numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[4, 1:, :].data.numpy(), 0.0)
        assert sorted_lengths.data.equal(torch.LongTensor([7, 5, 4, 3, 1]))
        assert sorted_tensor.index_select(0, reverse_indices).data.equal(tensor.data)

    def test_get_final_encoder_states(self) -> None:
        encoder_outputs: torch.Tensor = torch.Tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
        mask: torch.Tensor = torch.tensor([[True, True, True], [True, True, False]])
        final_states: torch.Tensor = util.get_final_encoder_states(encoder_outputs, mask, bidirectional=False)
        assert_almost_equal(final_states.data.numpy(), [[9, 10, 11, 12], [17, 18, 19, 20]])
        final_states: torch.Tensor = util.get_final_encoder_states(encoder_outputs, mask, bidirectional=True)
        assert_almost_equal(final_states.data.numpy(), [[9, 10, 3, 4], [17, 18, 15, 16]])

    def test_masked_softmax_no_mask(self) -> None:
        vector_1d: torch.FloatTensor = torch.FloatTensor([[1.0, 2.0, 3.0]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, None).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.090031, 0.244728, 0.665241]]))
        assert_almost_equal(1.0, numpy.sum(vector_1d_softmaxed), decimal=6)
        vector_1d: torch.FloatTensor = torch.FloatTensor([[1.0, 2.0, 5.0]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, None).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.017148, 0.046613, 0.93624]]))
        vector_zero: torch.FloatTensor = torch.FloatTensor([[0.0, 0.0, 0.0]])
        vector_zero_softmaxed: torch.Tensor = util.masked_softmax(vector_zero, None).data.numpy()
        assert_array_almost_equal(vector_zero_softmaxed, numpy.array([[0.33333334, 0.33333334, 0.33333334]]))
        matrix: torch.FloatTensor = torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, None).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.01714783, 0.04661262, 0.93623955], [0.09003057, 0.24472847, 0.66524096]]))
        matrix: torch.FloatTensor = torch.FloatTensor([[1.0, 2.0, 5.0], [0.0, 0.0, 0.0]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, None).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.01714783, 0.04661262, 0.93623955], [0.33333334, 0.33333334, 0.33333334]]))

    def test_masked_softmax_masked(self) -> None:
        vector_1d: torch.FloatTensor = torch.FloatTensor([[1.0, 2.0, 5.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, False, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.01798621, 0.0, 0.98201382]]))
        vector_1d: torch.FloatTensor = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, False, True, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.01321289, 0.0, 0.26538793, 0.72139918]]))
        vector_1d: torch.FloatTensor = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0, 0, 0, 1]]))
        vector_1d: torch.FloatTensor = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d: torch.FloatTensor = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d: torch.FloatTensor = torch.FloatTensor([[1.0, 1.0, 100000.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, True, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.5, 0.5, 0]]))
        matrix: torch.FloatTensor = torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, True]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.01798621, 0.0, 0.98201382], [0.090031, 0.244728, 0.665241]]))
        matrix: torch.FloatTensor = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, True]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.5, 0.0, 0.5], [0.090031, 0.244728, 0.665241]]))
        matrix: torch.FloatTensor = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [False, False, False]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.5, 0.0, 0.5], [0.33333333, 0.33333333, 0.33333333]]))
        matrix: torch.FloatTensor = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[False, False, False], [True, False, True]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.33333333, 0.33333333, 0.33333333], [0.11920292, 0.0, 0.88079708]]))

    def test_masked_softmax_memory_efficient_masked(self) -> None:
        vector_1d: torch.FloatTensor = torch.FloatTensor([[1.0, 2.0, 5.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, False, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.01798621, 0.0, 0.98201382]]))
        vector_1d: torch.FloatTensor = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, False, True, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.01321289, 0.0, 0.26538793, 0.72139918]]))
        vector_1d: torch.FloatTensor = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0, 0, 0, 1]]))
        vector_1d: torch.FloatTensor = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.25, 0.25, 0.25, 0.25]]))
        vector_1d: torch.FloatTensor = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.25, 0.25, 0.25, 0.25]]))
        vector_1d: torch.FloatTensor = torch.FloatTensor([[1.0, 1.0, 100000.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, True, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.5, 0.5, 0]]))
        matrix: torch.FloatTensor = torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, True]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask, memory_efficient=True).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.01798621, 0.0, 0.98201382], [0.090031, 0.244728, 0.665241]]))
        matrix: torch.FloatTensor = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, True]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask, memory_efficient=True).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.5, 0.0, 0.5], [0.090031, 0.244728, 0.665241]]))
        matrix: torch.FloatTensor = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [False, False, False]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask, memory_efficient=True).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.5, 0.0, 0.5], [0.33333333, 0.33333333, 0.33333333]]))
        matrix: torch.FloatTensor = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[False, False, False], [True, False, True]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask, memory_efficient=True).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.33333333, 0.33333333, 0.33333333], [0.11920292, 0.0, 0.88079708]]))

    def test_masked_log_softmax_masked(self) -> None:
        vector_1d: torch.FloatTensor = torch.FloatTensor([[1.0, 2.0, 5.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, False, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_log_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(numpy.exp(vector_1d_softmaxed), numpy.array([[0.01798621, 0.0, 0.98201382]]))
        vector_1d: torch.FloatTensor = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, False, True, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_log_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(numpy.exp(vector_1d_softmaxed), numpy.array([[0.01321289, 0.0, 0.26538793, 0.72139918]]))
        vector_1d: torch.FloatTensor = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_log_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(numpy.exp(vector_1d_softmaxed), numpy.array([[0.0, 0.0, 0.0, 1.0]]))

    def test_masked_max(self) -> None:
        vector_1d: torch.FloatTensor = torch.FloatTensor([1.0, 12.0, 5.0])
        mask_1d: torch.Tensor = torch.tensor([True, False, True])
        vector_1d_maxed: torch.Tensor = util.masked_max(vector_1d, mask_1d, dim=0).data.numpy()
        assert_array_almost_equal(vector_1d_maxed, 5.0)
        vector_1d: torch.FloatTensor = torch.FloatTensor([1.0, 12.0, 5.0])
        mask_1d: torch.Tensor = torch.tensor([False, False, False])
        vector_1d_maxed: torch.Tensor = util.masked_max(vector_1d, mask_1d, dim=0).data.numpy()
        assert not numpy.isnan(vector_1d_maxed).any()
        matrix: torch.FloatTensor = torch.FloatTensor([[1.0, 12.0, 5.0], [-1.0, -2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, False]])
        matrix_maxed: torch.Tensor = util.masked_max(matrix, mask, dim=-1).data.numpy()
        assert_array_almost_equal(matrix_maxed, numpy.array([5.0, -1.0]))
        matrix: torch.FloatTensor = torch.FloatTensor([[1.0, 12.0, 5.0], [-1.0, -2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, False]])
        matrix_maxed: torch.Tensor = util.masked_max(matrix, mask, dim=-1, keepdim=True).data.numpy()
        assert_array_almost_equal(matrix_maxed, numpy.array([[5.0], [-1.0]]))
        matrix: torch.FloatTensor = torch.FloatTensor([[[1.0, 2.0], [12.0, 3.0], [5.0, -1.0]], [[-1.0, -3.0], [-2.0, -0.5], [3.0, 8.0]]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, False]]).unsqueeze(-1)
        matrix_maxed: torch.Tensor = util.masked_max(matrix, mask, dim=1).data.numpy()
        assert_array_almost_equal(matrix_maxed, numpy.array([[5.0, 2.0], [-1.0, -0.5]]))

    def test_masked_mean(self) -> None:
        vector_1d: torch.FloatTensor = torch.FloatTensor([1.0, 12.0, 5.0])
        mask_1d: torch.Tensor = torch.tensor([True, False, True])
        vector_1d_mean: torch.Tensor = util.masked_mean(vector_1d, mask_1d, dim=0).data.numpy()
        assert_array_almost_equal(vector_1d_mean, 3.0)
        vector_1d: torch.FloatTensor = torch.FloatTensor([1.0, 12.0, 5.0])
        mask_1d: torch.Tensor = torch.tensor([False, False, False])
        vector_1d_mean: torch.Tensor = util.masked_mean(vector_1d, mask_1d, dim=0).data.numpy()
        assert not numpy.isnan(vector_1d_mean).any()
        matrix: torch.FloatTensor = torch.FloatTensor([[1.0, 12.0, 5.0], [-1.0, -2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, False]])
        matrix_mean: torch.Tensor = util.masked_mean(matrix, mask, dim=-1).data.numpy()
        assert_array_almost_equal(matrix_mean, numpy.array([3.0, -1.5]))
        matrix: torch.FloatTensor = torch.FloatTensor([[1.0, 12.0, 5.0], [-1.0, -2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, False]])
        matrix_mean: torch.Tensor = util.masked_mean(matrix, mask, dim=-1, keepdim=True).data.numpy()
        assert_array_almost_equal(matrix_mean, numpy.array([[3.0], [-1.5]]))
        matrix: torch.FloatTensor = torch.FloatTensor([[[1.0, 2.0], [12.0, 3.0], [5.0, -1.0]], [[-1.0, -3.0], [-2.0, -0.5], [3.0, 8.0]]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, False]]).unsqueeze(-1)
        matrix_mean: torch.Tensor = util.masked_mean(matrix, mask, dim=1).data.numpy()
        assert_array_almost_equal(matrix_mean, numpy.array([[3.0, 0.5], [-1.5, -1.75]]))

    def test_masked_flip(self) -> None:
        tensor: torch.FloatTensor = torch.FloatTensor([[[6, 6, 6], [1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4], [5, 5, 5]]])
        solution: List[List[List[int]]] = [[[6, 6, 6], [0, 0, 0]], [[4, 4, 4], [3, 3, 3]]]
        response: torch.Tensor = util.masked_flip(tensor, [1, 2])
        assert_almost_equal(response, solution)
        tensor: torch.FloatTensor = torch.FloatTensor([[[6, 6, 6], [1, 1, 1], [2, 2, 2], [0, 0, 0]], [[3, 3, 3], [4, 4, 4], [5, 5, 5], [1, 2, 3]]])
        solution: List[List[List[int]]] = [[[2, 2, 2], [1, 1, 1], [6, 6, 6], [0, 0, 0]], [[1, 2, 3], [5, 5, 5], [4, 4, 4], [3, 3, 3]]]
        response: torch.Tensor = util.masked_flip(tensor, [3, 4])
        assert_almost_equal(response, solution)
        tensor: torch.FloatTensor = torch.FloatTensor([[[6, 6, 6], [1, 1, 1], [2, 2, 2], [0, 0, 0]], [[3, 3, 3], [4, 4, 4], [5, 5, 5], [1, 2, 3]], [[1, 1, 1], [2, 2, 2], [0, 0, 0], [0, 0, 0]]])
        solution: List[List[List[int]]] = [[[2, 2, 2], [1, 1, 1], [6, 6, 6], [0, 0, 0]], [[1, 2, 3], [5, 5, 5], [4, 4, 4], [3, 3, 3]], [[2, 2, 2], [1, 1, 1], [0, 0, 0], [0, 0, 0]]]
        response: torch.Tensor = util.masked_flip(tensor, [3, 4, 2])
        assert_almost_equal(response, solution)

    def test_get_text_field_mask_returns_a_correct_mask(self) -> None:
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]] = {'indexer_name': {'tokens': torch.LongTensor([[3, 4, 5, 0, 0], [1, 2, 0, 0, 0]]), 'token_characters': torch.LongTensor([[[1, 2], [3, 0], [2, 0], [0, 0], [0, 0]], [[5, 0], [4, 6], [0, 0], [0, 0], [0, 0]]])}}
        assert_almost_equal(util.get_text_field_mask(text_field_tensors).long().numpy(), [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])

    def test_get_text_field_mask_returns_a_correct_mask_custom_padding_id(self) -> None:
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]] = {'indexer_name': {'tokens': torch.LongTensor([[3, 4, 5, 9, 9], [1, 2, 9, 9, 9]]), 'token_characters': torch.LongTensor([[[1, 2], [3, 9], [2, 9], [9, 9], [9, 9]], [[5, 9], [4, 6], [9, 9], [9, 9], [9, 9]]])}}
        assert_almost_equal(util.get_text_field_mask(text_field_tensors, padding_id=9).long().numpy(), [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])

    def test_get_text_field_mask_returns_a_correct_mask_character_only_input(self) -> None:
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]] = {'indexer_name': {'token_characters': torch.LongTensor([[[1, 2, 3], [3, 0, 1], [2, 1, 0], [0, 0, 0]], [[5, 5, 5], [4, 6, 0], [0, 0, 0], [0, 0, 0]]])}}
        assert_almost_equal(util.get_text_field_mask(text_field_tensors).long().numpy(), [[1, 1, 1, 0], [1, 1, 0, 0]])

    def test_get_text_field_mask_returns_a_correct_mask_character_only_input_custom_padding_id(self) -> None:
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]] = {'indexer_name': {'token_characters': torch.LongTensor([[[1, 2, 3], [3, 9, 1], [2, 1, 9], [9, 9, 9]], [[5, 5, 5], [4, 6, 9], [9, 9, 9], [9, 9, 9]]])}}
        assert_almost_equal(util.get_text_field_mask(text_field_tensors, padding_id=9).long().numpy(), [[1, 1, 1, 0], [1, 1, 0, 0]])

    def test_get_text_field_mask_returns_a_correct_mask_list_field(self) -> None:
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]] = {'indexer_name': {'list_tokens': torch.LongTensor([[[1, 2], [3, 0], [2, 0], [0, 0], [0, 0]], [[5, 0], [4, 6], [0, 0], [0, 0], [0, 0]]])}}
        actual_mask: torch.Tensor = util.get_text_field_mask(text_field_tensors, num_wrapping_dims=1).long().numpy()
        expected_mask: torch.Tensor = (text_field_tensors['indexer_name']['list_tokens'].numpy() > 0).astype('int32')
        assert_almost_equal(actual_mask, expected_mask)

    def test_get_text_field_mask_returns_mask_key(self) -> None:
        text_field_tensors: Dict[str, Dict[str, torch.Tensor]] = {'indexer_name': {'tokens': torch.LongTensor([[3, 4, 5, 0, 0], [1, 2, 0, 0, 0]]), 'mask': torch.tensor([[False, False, True]])}}
        assert_almost_equal(util.get_text_field_mask(text_field_tensors).long().numpy(), [[0, 0, 1]])

    def test_weighted_sum_works_on_simple_input(self) -> None:
        batch_size: int = 1
        sentence_length: int = 5
        embedding_dim: int = 4
        sentence_array: numpy.ndarray = numpy.random.rand(batch_size, sentence_length, embedding_dim)
        sentence_tensor: torch.Tensor = torch.from_numpy(sentence_array).float()
        attention_tensor: torch.Tensor = torch.FloatTensor([[0.3, 0.4, 0.1, 0, 1.2]])
        aggregated_array: torch.Tensor = util.weighted_sum(sentence_tensor, attention_tensor).data.numpy()
        assert aggregated_array.shape == (batch_size, embedding_dim)
        expected_array: float = 0.3 * sentence_array[0, 0] + 0.4 * sentence_array[0, 1] + 0.1 * sentence_array[0, 2] + 0.0 * sentence_array[0, 3] + 1.2 * sentence_array[0, 4]
        numpy.testing.assert_almost_equal(aggregated_array, [expected_array], decimal=5)

    def test_weighted_sum_handles_higher_order_input(self) -> None:
        batch_size: int = 1
        length_1: int = 5
        length_2: int = 6
        length_3: int = 2
        embedding_dim: int = 4
        sentence_array: numpy.ndarray = numpy.random.rand(batch_size, length_1, length_2, length_3, embedding_dim)
        attention_array: numpy.ndarray = numpy.random.rand(batch_size, length_1, length_2, length_3)
        sentence_tensor: torch.Tensor = torch.from_numpy(sentence_array).float()
        attention_tensor: torch.Tensor = torch.from_numpy(attention_array).float()
        aggregated_array: torch.Tensor = util.weighted_sum(sentence_tensor, attention_tensor).data.numpy()
        assert aggregated_array.shape == (batch_size, length_1, length_2, embedding_dim)
        expected_array: float = attention_array[0, 3, 2, 0] * sentence_array[0, 3, 2, 0] + attention_array[0, 3, 2, 1] * sentence_array[0, 3, 2, 1]
        numpy.testing.assert_almost_equal(aggregated_array[0, 3, 2], expected_array, decimal=5)

    def test_weighted_sum_handles_uneven_higher_order_input(self) -> None:
        batch_size: int = 1
        length_1: int = 5
        length_2: int = 6
        length_3: int = 2
        embedding_dim: int = 4
        sentence_array: numpy.ndarray = numpy.random.rand(batch_size, length_3, embedding_dim)
        attention_array: numpy.ndarray = numpy.random.rand(batch_size, length_1, length_2, length_3)
        sentence_tensor: torch.Tensor = torch.from_numpy(sentence_array).float()
        attention_tensor: torch.Tensor = torch.from_numpy(attention_array).float()
        aggregated_array: torch.Tensor = util.weighted_sum(sentence_tensor, attention_tensor).data.numpy()
        assert aggregated_array.shape == (batch_size, length_1, length_2, embedding_dim)
        for i in range(length_1):
            for j in range(length_2):
                expected_array: float = attention_array[0, i, j, 0] * sentence_array[0, 0] + attention_array[0, i, j, 1] * sentence_array[0, 1]
                numpy.testing.assert_almost_equal(aggregated_array[0, i, j], expected_array, decimal=5)

    def test_weighted_sum_handles_3d_attention_with_3d_matrix(self) -> None:
        batch_size: int = 1
        length_1: int = 5
        length_2: int = 2
        embedding_dim: int = 4
        sentence_array: numpy.ndarray = numpy.random.rand(batch_size, length_2, embedding_dim)
        attention_array: numpy.ndarray = numpy.random.rand(batch_size, length_1, length_2)
        sentence_tensor: torch.Tensor = torch.from_numpy(sentence_array).float()
        attention_tensor: torch.Tensor = torch.from_numpy(attention_array).float()
        aggregated_array: torch.Tensor = util.weighted_sum(sentence_tensor, attention_tensor).data.numpy()
        assert aggregated_array.shape == (batch_size, length_1, embedding_dim)
        for i in range(length_1):
            expected_array: float = attention_array[0, i, 0] * sentence_array[0, 0] + attention_array[0, i, 1] * sentence_array[0, 1]
            numpy.testing.assert_almost_equal(aggregated_array[0, i], expected_array, decimal=5)

    def test_viterbi_decode(self) -> None:
        sequence_logits: torch.Tensor = torch.nn.functional.softmax(torch.rand([5, 9]), dim=-1)
        transition_matrix: torch.Tensor = torch.zeros([9, 9])
        indices, _ = util.viterbi_decode(sequence_logits.data, transition_matrix)
        _, argmax_indices = torch.max(sequence_logits, 1)
        assert indices == argmax_indices.data.squeeze().tolist()
        sequence_logits: torch.Tensor = torch.nn.functional.softmax(torch.rand([5, 9]), dim=-1)
        transition_matrix: torch.Tensor = torch.zeros([9, 9])
        allowed_start_transitions: torch.Tensor = torch.zeros([9])
        allowed_start_transitions[:8] = float('-inf')
        allowed_end_transitions: torch.Tensor = torch.zeros([9])
        allowed_end_transitions[1:] = float('-inf')
        indices, _ = util.viterbi_decode(sequence_logits.data, transition_matrix, allowed_end_transitions=allowed_end_transitions, allowed_start_transitions=allowed_start_transitions)
        assert indices[0] == 8
        assert indices[-1] == 0
        sequence_logits: torch.Tensor = torch.FloatTensor([[0, 0, 0, 3, 5], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4]])
        transition_matrix: torch.Tensor = torch.zeros([5, 5])
        for i in range(5):
            transition_matrix[i, i] = float('-inf')
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix)
        assert indices == [4, 3, 4, 3, 4, 3]
        sequence_logits: torch.Tensor = torch.FloatTensor([[0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4]])
        transition_matrix: torch.Tensor = torch.zeros([5, 5])
        transition_matrix[4, 4] = -10
        transition_matrix[4, 3] = -10
        transition_matrix[3, 4] = -10
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix)
        assert indices == [3, 3, 3, 3, 3, 3]
        sequence_logits: torch.Tensor = torch.FloatTensor([[1, 0, 0, 4], [1, 0, 6, 2], [0, 3, 0, 4]])
        transition_matrix: torch.Tensor = torch.zeros([4, 4])
        transition_matrix[0, 0] = 1
        transition_matrix[2, 1] = 5
        indices, value = util.viterbi_decode(sequence_logits, transition_matrix)
        assert indices == [3, 2, 1]
        assert value.numpy() == 18
        sequence_logits: torch.Tensor = torch.FloatTensor([[0, 0, 0, 7, 7], [0, 0, 0, 7, 7], [0, 0, 0, 7, 7], [0, 0, 0, 7, 7], [0, 0, 0, 7, 7], [0, 0, 0, 7, 7]])
        transition_matrix: torch.Tensor = torch.zeros([5, 5])
        transition_matrix[4, 4] = -10
        transition_matrix[4, 3] = -2
        transition_matrix[3, 4] = -2
        observations: List[int] = [2, -1, -1, 0, 4, -1]
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix, observations)
        assert indices == [2, 3, 3, 0, 4, 3]

    def test_viterbi_decode_top_k(self) -> None:
        sequence_logits: torch.Tensor = torch.autograd.Variable(torch.rand([5, 9]))
        transition_matrix: torch.Tensor = torch.zeros([9, 9])
        indices, _ = util.viterbi_decode(sequence_logits.data, transition_matrix, top_k=5)
        _, argmax_indices = torch.max(sequence_logits, 1)
        assert indices[0] == argmax_indices.data.squeeze().tolist()
        sequence_logits: torch.Tensor = torch.FloatTensor([[0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4]])
        transition_matrix: torch.Tensor = torch.zeros([5, 5])
        for i in range(5):
            transition_matrix[i, i] = float('-inf')
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix, top_k=5)
        assert indices[0] == [3, 4, 3, 4, 3, 4]
        sequence_logits: torch.Tensor = torch.FloatTensor([[0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 0]])
        transition_matrix: torch.Tensor = torch.zeros([5, 5])
        transition_matrix[4, 4] = -10
        transition_matrix[4, 3] = -10
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix, top_k=5)
        assert indices[0] == [3, 3, 3, 3, 3, 3]
        sequence_logits: torch.Tensor = torch.FloatTensor([[1, 0, 0, 4], [1, 0, 6, 2], [0, 3, 0, 4]])
        transition_matrix: torch.Tensor = torch.zeros([4, 4])
        transition_matrix[0, 0] = 1
        transition_matrix[2, 1] = 5
        indices, value = util.viterbi_decode(sequence_logits, transition_matrix, top_k=5)
        assert indices[0] == [3, 2, 1]
        assert value[0] == 18

        def _brute_decode(tag_sequence: List[List[int]], transition_matrix: torch.Tensor, top_k: int = 5) -> Tuple[List[List[int]], List[float]]:
            """
            Top-k decoder that uses brute search instead of the Viterbi Decode dynamic programing algorithm
            """
            sequences: List[List[int]] = [[]]
            for i in range(len(tag_sequence)):
                new_sequences: List[List[int]] = []
                for j in range(len(tag_sequence[i])):
                    for sequence in sequences:
                        new_sequences.append(sequence[:] + [j])
                sequences = new_sequences
            scored_sequences: List[Tuple[float, List[int]]] = []
            for sequence in sequences:
                emission_score: float = sum((tag_sequence[i][j] for i, j in enumerate(sequence)))
                transition_score: float = sum((transition_matrix[sequence[i - 1], sequence[i]] for i in range(1, len(sequence))))
                score: float = emission_score + transition_score
                scored_sequences.append((score, sequence))
            top_k_sequences: List[Tuple[float, List[int]]] = sorted(scored_sequences, key=lambda r: r[0], reverse=True)[:top_k]
            scores, paths: List[float], List[List[int]] = zip(*top_k_sequences)
            return (paths, scores)

        for i in range(100):
            num_tags: int = random.randint(1, 5)
            seq_len: int = random.randint(1, 5)
            k: int = random.randint(1, 5)
            sequence_logits: torch.Tensor = torch.rand([seq_len, num_tags])
            transition_matrix: torch.Tensor = torch.rand([num_tags, num_tags])
            viterbi_paths_v1, viterbi_scores_v1 = util.viterbi_decode(sequence_logits, transition_matrix, top_k=k)
            viterbi_path_brute, viterbi_score_brute = _brute_decode(sequence_logits, transition_matrix, top_k=k)
            numpy.testing.assert_almost_equal(list(viterbi_score_brute), viterbi_scores_v1.tolist(), decimal=3)
            numpy.testing.assert_equal(sanitize(viterbi_paths_v1), viterbi_path_brute)

    def test_sequence_cross_entropy_with_logits_masks_loss_correctly(self) -> None:
        tensor: torch.Tensor = torch.rand([5, 7, 4])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, :, :] = 0
        weights: torch.Tensor = (tensor != 0.0)[:, :, 0].long().squeeze(-1)
        tensor2: torch.Tensor = tensor.clone()
        tensor2[0, 3:, :] = 2
        tensor2[1, 4:, :] = 13
        tensor2[2, 2:, :] = 234
        tensor2[3, :, :] = 65
        targets: torch.LongTensor = torch.LongTensor(numpy.random.randint(0, 3, [5, 7]))
        targets *= weights
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights)
        loss2: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor2, targets, weights)
        assert loss.data.numpy() == loss2.data.numpy()

    def test_sequence_cross_entropy_with_logits_smooths_labels_correctly(self) -> None:
        tensor: torch.Tensor = torch.rand([1, 3, 4])
        targets: torch.LongTensor = torch.LongTensor(numpy.random.randint(0, 3, [1, 3]))
        weights: torch.Tensor = torch.ones([2, 3])
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights, label_smoothing=0.1)
        correct_loss: float = 0.0
        for prediction, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            prediction: torch.Tensor = torch.nn.functional.log_softmax(prediction, dim=-1)
            correct_loss += prediction[label] * 0.9
            correct_loss += prediction.sum() * 0.1 / 4
        correct_loss = -correct_loss / 3
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())

    def test_sequence_cross_entropy_with_logits_averages_batch_correctly(self) -> None:
        tensor: torch.Tensor = torch.rand([5, 7, 4])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, :, :] = 0
        weights: torch.Tensor = (tensor != 0.0)[:, :, 0].long().squeeze(-1)
        targets: torch.LongTensor = torch.LongTensor(numpy.random.randint(0, 3, [5, 7]))
        targets *= weights
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights)
        vector_loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights, average=None)
        assert loss.data.numpy() == vector_loss.sum().item() / 4

    @flaky(max_runs=3, min_passes=1)
    def test_sequence_cross_entropy_with_logits_averages_token_correctly(self) -> None:
        tensor: torch.Tensor = torch.rand([5, 7, 4])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, :, :] = 0
        weights: torch.Tensor = (tensor != 0.0)[:, :, 0].long().squeeze(-1)
        targets: torch.LongTensor = torch.LongTensor(numpy.random.randint(0, 3, [5, 7]))
        targets *= weights
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights, average='token')
        vector_loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights, average=None)
        total_token_loss: torch.Tensor = (vector_loss * weights.float().sum(dim=-1)).sum()
        average_token_loss: torch.Tensor = (total_token_loss / weights.float().sum()).detach()
        assert_almost_equal(loss.detach().item(), average_token_loss.item(), decimal=5)

    def test_sequence_cross_entropy_with_logits_gamma_correctly(self) -> None:
        batch: int = 1
        length: int = 3
        classes: int = 4
        gamma: float = abs(numpy.random.randn())
        tensor: torch.Tensor = torch.rand([batch, length, classes])
        targets: torch.LongTensor = torch.LongTensor(numpy.random.randint(0, classes, [batch, length]))
        weights: torch.Tensor = torch.ones([batch, length])
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights, gamma=gamma)
        correct_loss: float = 0.0
        for logit, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            p: torch.Tensor = torch.nn.functional.softmax(logit, dim=-1)
            pt: float = p[label]
            ft: float = (1 - pt) ** gamma
            correct_loss += -pt.log() * ft
        correct_loss = correct_loss / length
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())

    def test_sequence_cross_entropy_with_logits_alpha_float_correctly(self) -> None:
        batch: int = 1
        length: int = 3
        classes: int = 2
        alpha: float = numpy.random.rand() if numpy.random.rand() > 0.5 else 1.0 - numpy.random.rand()
        tensor: torch.Tensor = torch.rand([batch, length, classes])
        targets: torch.LongTensor = torch.LongTensor(numpy.random.randint(0, classes, [batch, length]))
        weights: torch.Tensor = torch.ones([batch, length])
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights, alpha=alpha)
        correct_loss: float = 0.0
        for logit, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            logp: torch.Tensor = torch.nn.functional.log_softmax(logit, dim=-1)
            logpt: float = logp[label]
            if label:
                at: float = alpha
            else:
                at: float = 1 - alpha
            correct_loss += -logpt * at
        correct_loss = correct_loss / length
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())

    def test_sequence_cross_entropy_with_logits_alpha_single_float_correctly(self) -> None:
        batch: int = 1
        length: int = 3
        classes: int = 2
        alpha: float = numpy.random.rand() if numpy.random.rand() > 0.5 else 1.0 - numpy.random.rand()
        alpha: torch.Tensor = torch.tensor(alpha)
        tensor: torch.Tensor = torch.rand([batch, length, classes])
        targets: torch.LongTensor = torch.LongTensor(numpy.random.randint(0, classes, [batch, length]))
        weights: torch.Tensor = torch.ones([batch, length])
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights, alpha=alpha)
        correct_loss: float = 0.0
        for logit, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            logp: torch.Tensor = torch.nn.functional.log_softmax(logit, dim=-1)
            logpt: float = logp[label]
            if label:
                at: float = alpha
            else:
                at: float = 1 - alpha
            correct_loss += -logpt * at
        correct_loss = correct_loss / length
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())

    def test_sequence_cross_entropy_with_logits_alpha_list_correctly(self) -> None:
        batch: int = 1
        length: int = 3
        classes: int = 4
        alpha: Union[float, List[float]] = abs(numpy.random.randn(classes))
        tensor: torch.Tensor = torch.rand([batch, length, classes])
        targets: torch.LongTensor = torch.LongTensor(numpy.random.randint(0, classes, [batch, length]))
        weights: torch.Tensor = torch.ones([batch, length])
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights, alpha=alpha)
        correct_loss: float = 0.0
        for logit, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            logp: torch.Tensor = torch.nn.functional.log_softmax(logit, dim=-1)
            logpt: float = logp[label]
            at: float = alpha[label]
            correct_loss += -logpt * at
        correct_loss = correct_loss / length
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())

    def test_replace_masked_values_replaces_masked_values_with_finite_value(self) -> None:
        tensor: torch.Tensor = torch.FloatTensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]])
        mask: torch.Tensor = torch.tensor([[True, True, False]])
        replaced: torch.Tensor = util.replace_masked_values(tensor, mask.unsqueeze(-1), 2).data.numpy()
        assert_almost_equal(replaced, [[[1, 2, 3, 4], [5, 6, 7, 8], [2, 2, 2, 2]]])

    def test_logsumexp(self) -> None:
        tensor: torch.Tensor = torch.FloatTensor([[0.4, 0.1, 0.2]])
        log_tensor: torch.Tensor = tensor.log()
        log_summed: torch.Tensor = util.logsumexp(log_tensor, dim=-1, keepdim=False)
        assert_almost_equal(log_summed.exp().data.numpy(), [0.7])
        log_summed: torch.Tensor = util.logsumexp(log_tensor, dim=-1, keepdim=True)
        assert_almost_equal(log_summed.exp().data.numpy(), [[0.7]])
        tensor: torch.Tensor = torch.FloatTensor([[float('-inf'), 20.0]])
        assert_almost_equal(util.logsumexp(tensor).data.numpy(), [20.0])
        tensor: torch.Tensor = torch.FloatTensor([[-200.0, 20.0]])
        assert_almost_equal(util.logsumexp(tensor).data.numpy(), [20.0])
        tensor: torch.Tensor = torch.FloatTensor([[20.0, 20.0], [-200.0, 200.0]])
        assert_almost_equal(util.logsumexp(tensor, dim=0).data.numpy(), [20.0, 200.0])

    def test_flatten_and_batch_shift_indices(self) -> None:
        indices: numpy.ndarray = numpy.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 9, 9, 9]], [[2, 1, 0, 7], [7, 7, 2, 3], [0, 0, 4, 2]]])
        indices: torch.Tensor = torch.tensor(indices, dtype=torch.long)
        shifted_indices: torch.Tensor = util.flatten_and_batch_shift_indices(indices, 10)
        numpy.testing.assert_array_equal(shifted_indices.data.numpy(), numpy.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 12, 11, 10, 17, 17, 17, 12, 13, 10, 10, 14, 12]))

    def test_batched_index_select(self) -> None:
        indices: numpy.ndarray = numpy.array([[1, 2], [3, 4]])
        targets: torch.Tensor = torch.ones([2, 10, 3]).cumsum(1) - 1
        targets[1, :, :] *= 2
        indices: torch.Tensor = torch.tensor(indices, dtype=torch.long)
        selected: torch.Tensor = util.batched_index_select(targets, indices)
        assert list(selected.size()) == [2, 2, 2, 3]
        ones: numpy.ndarray = numpy.ones([3])
        numpy.testing.assert_array_equal(selected[0, 0, 0, :].data.numpy(), ones)
        numpy.testing.assert_array_equal(selected[0, 0, 1, :].data.numpy(), ones * 2)
        numpy.testing.assert_array_equal(selected[0, 1, 0, :].data.numpy(), ones * 3)
        numpy.testing.assert_array_equal(selected[0, 1, 1, :].data.numpy(), ones * 4)
        numpy.testing.assert_array_equal(selected[1, 0, 0, :].data.numpy(), ones * 2)
        numpy.testing.assert_array_equal(selected[1, 0, 1, :].data.numpy(), ones * 4)
        numpy.testing.assert_array_equal(selected[1, 1, 0, :].data.numpy(), ones * 6)
        numpy.testing.assert_array_equal(selected[1, 1, 1, :].data.numpy(), ones * 8)
        with pytest.raises(ConfigurationError):
            util.batched_index_select(targets, torch.ones([3, 4, 5]))
        indices: numpy.ndarray = numpy.array([[[1, 11], [3, 4]], [[5, 6], [7, 8]]])
        indices: torch.Tensor = torch.tensor(indices, dtype=torch.long)
        with pytest.raises(ConfigurationError):
            util.batched_index_select(targets, indices)
        indices: numpy.ndarray = numpy.array([[[1, -1], [3, 4]], [[5, 6], [7, 8]]])
        indices: torch.Tensor = torch.tensor(indices, dtype=torch.long)
        with pytest.raises(ConfigurationError):
            util.batched_index_select(targets, indices)

    def test_masked_index_fill(self) -> None:
        targets: torch.Tensor = torch.zeros([3, 5])
        indices: torch.Tensor = torch.tensor([[4, 2, 3, -1], [0, 1, -1, -1], [1, 3, -1, -1]])
        mask: torch.Tensor = indices >= 0
        filled: torch.Tensor = util.masked_index_fill(targets, indices, mask)
        numpy.testing.assert_array_equal(filled, [[0, 0, 1, 1, 1], [1, 1, 0, 0, 0], [0, 1, 0, 1, 0]])

    def test_masked_index_replace(self) -> None:
        targets: torch.Tensor = torch.zeros([3, 5, 2])
        indices: torch.Tensor = torch.tensor([[4, 2, 3, -1], [0, 1, -1, -1], [3, 1, -1, -1]])
        replace_with: torch.Tensor = torch.arange(indices.numel()).float().reshape(indices.shape).unsqueeze(-1).expand(indices.shape + (2,))
        mask: torch.Tensor = indices >= 0
        replaced: torch.Tensor = util.masked_index_replace(targets, indices, mask, replace_with)
        numpy.testing.assert_array_equal(replaced, [[[0, 0], [0, 0], [1, 1], [2, 2], [0, 0]], [[4, 4], [5, 5], [0, 0], [0, 0], [0, 0]], [[0, 0], [9, 9], [0, 0], [8, 8], [0, 0]]])

    def test_batched_span_select(self) -> None:
        targets: torch.Tensor = torch.ones([3, 12, 2]).cumsum(1) - 1
        spans: torch.LongTensor = torch.LongTensor([[[0, 0], [1, 2], [5, 8], [10, 10]], [[i, i] for i in range(3, -1, -1)], [[0, 3], [1, 4], [2, 5], [10, 11]]])
        selected, mask = util.batched_span_select(targets, spans)
        selected: torch.Tensor = torch.where(mask.unsqueeze(-1), selected, torch.empty_like(selected).fill_(-1))
        numpy.testing.assert_array_equal(selected, [[[[0, 0], [-1, -1], [-1, -1], [-1, -1]], [[1, 1], [2, 2], [-1, -1], [-1, -1]], [[5, 5], [6, 6], [7, 7], [8, 8]], [[10, 10], [-1, -1], [-1, -1], [-1, -1]]], [[[i, i], [-1, -1], [-1, -1], [-1, -1]] for i in range(3, -1, -1)], [[[0, 0], [1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3], [4, 4]], [[2, 2], [3, 3], [4, 4], [5, 5]], [[10, 10], [11, 11], [-1, -1], [-1, -1]]]])

    def test_flattened_index_select(self) -> None:
        indices: numpy.ndarray = numpy.array([[1, 2], [3, 4]])
        targets: torch.Tensor = torch.ones([2, 6, 3]).cumsum(1) - 1
        targets[1, :, :] *= 2
        indices: torch.Tensor = torch.tensor(indices, dtype=torch.long)
        selected: torch.Tensor = util.flattened_index_select(targets, indices)
        assert list(selected.size()) == [2, 2, 2, 3]
        ones: numpy.ndarray = numpy.ones([3])
        numpy.testing.assert_array_equal(selected[0, 0, 0, :].data.numpy(), ones)
        numpy.testing.assert_array_equal(selected[0, 0, 1, :].data.numpy(), ones * 2)
        numpy.testing.assert_array_equal(selected[0, 1, 0, :].data.numpy(), ones * 3)
        numpy.testing.assert_array_equal(selected[0, 1, 1, :].data.numpy(), ones * 4)
        numpy.testing.assert_array_equal(selected[1, 0, 0, :].data.numpy(), ones * 2)
        numpy.testing.assert_array_equal(selected[1, 0, 1, :].data.numpy(), ones * 4)
        numpy.testing.assert_array_equal(selected[1, 1, 0, :].data.numpy(), ones * 6)
        numpy.testing.assert_array_equal(selected[1, 1, 1, :].data.numpy(), ones * 8)
        with pytest.raises(ConfigurationError):
            util.flattened_index_select(targets, torch.ones([3, 4, 5]))

    def test_bucket_values(self) -> None:
        indices: torch.LongTensor = torch.LongTensor([1, 2, 7, 1, 56, 900])
        bucketed_distances: torch.Tensor = util.bucket_values(indices)
        numpy.testing.assert_array_equal(bucketed_distances.numpy(), numpy.array([1, 2, 5, 1, 8, 9]))

    def test_add_sentence_boundary_token_ids_handles_2D_input(self) -> None:
        tensor: torch.Tensor = torch.from_numpy(numpy.array([[1, 2, 3], [4, 5, 0]]))
        mask: torch.Tensor = tensor > 0
        bos: int = 9
        eos: int = 10
        new_tensor, new_mask = util.add_sentence_boundary_token_ids(tensor, mask, bos, eos)
        expected_new_tensor: numpy.ndarray = numpy.array([[9, 1, 2, 3, 10], [9, 4, 5, 10, 0]])
        assert (new_tensor.data.numpy() == expected_new_tensor).all()
        assert (new_mask.data.numpy() == (expected_new_tensor > 0)).all()

    def test_add_sentence_boundary_token_ids_handles_3D_input(self) -> None:
        tensor: torch.Tensor = torch.from_numpy(numpy.array([[[1, 2, 3, 4], [5, 5, 5, 5], [6, 8, 1, 2]], [[4, 3, 2, 1], [8, 7, 6, 5], [0, 0, 0, 0]]]))
        mask: torch.Tensor = (tensor > 0).sum(dim=-1) > 0
        bos: torch.Tensor = torch.from_numpy(numpy.array([9, 9, 9, 9]))
        eos: torch.Tensor = torch.from_numpy(numpy.array([10, 10, 10, 10]))
        new_tensor, new_mask = util.add_sentence_boundary_token_ids(tensor, mask, bos, eos)
        expected_new_tensor: numpy.ndarray = numpy.array([[[9, 9, 9, 9], [1, 2, 3, 4], [5, 5, 5, 5], [6, 8, 1, 2], [10, 10, 10, 10]], [[9, 9, 9, 9], [4, 3, 2, 1], [8, 7, 6, 5], [10, 10, 10, 10], [0, 0, 0, 0]]])
        assert (new_tensor.data.numpy() == expected_new_tensor).all()
        assert (new_mask.data.numpy() == ((expected_new_tensor > 0).sum(axis=-1) > 0)).all()

    def test_remove_sentence_boundaries(self) -> None:
        tensor: torch.Tensor = torch.from_numpy(numpy.random.rand(3, 5, 7))
        mask: torch.Tensor = torch.from_numpy(numpy.array([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])).bool()
        new_tensor, new_mask = util.remove_sentence_boundaries(tensor, mask)
        expected_new_tensor: torch.Tensor = torch.zeros(3, 3, 7)
        expected_new_tensor[1, 0:3, :] = tensor[1, 1:4, :]
        expected_new_tensor[2, 0:2, :] = tensor[2, 1:3, :]
        assert_array_almost_equal(new_tensor.data.numpy(), expected_new_tensor.data.numpy())
        expected_new_mask: torch.Tensor = torch.from_numpy(numpy.array([[0, 0, 0], [1, 1, 1], [1, 1, 0]])).bool()
        assert (new_mask.data.numpy() == expected_new_mask.data.numpy()).all()

    def test_add_positional_features(self) -> None:
        tensor2tensor_result: numpy.ndarray = numpy.asarray([[0.0, 0.0, 1.0, 1.0], [0.841470957, 9.99999902e-05, 0.540302277, 1.0], [0.909297407, 0.00019999998, -0.416146845, 1.0]])
        tensor: torch.Tensor = torch.zeros([2, 3, 4])
        result: torch.Tensor = util.add_positional_features(tensor, min_timescale=1.0, max_timescale=10000.0)
        numpy.testing.assert_almost_equal(result[0].detach().cpu().numpy(), tensor2tensor_result)
        numpy.testing.assert_almost_equal(result[1].detach().cpu().numpy(), tensor2tensor_result)
        tensor2tensor_result: numpy.ndarray = numpy.asarray([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [0.841470957, 0.00999983307, 9.99999902e-05, 0.540302277, 0.999949992, 1.0, 0.0], [0.909297407, 0.0199986659, 0.00019999998, -0.416146815, 0.999800026, 1.0, 0.0]])
        tensor: torch.Tensor = torch.zeros([2, 3, 7])
        result: torch.Tensor = util.add_positional_features(tensor, min_timescale=1.0, max_timescale=10000.0)
        numpy.testing.assert_almost_equal(result[0].detach().cpu().numpy(), tensor2tensor_result)
        numpy.testing.assert_almost_equal(result[1].detach().cpu().numpy(), tensor2tensor_result)

    def test_combine_tensors_and_multiply(self) -> None:
        tensors: List[torch.Tensor] = [torch.Tensor([[[2, 3]]]), torch.Tensor([[[5, 5]]])]
        weight: torch.Tensor = torch.Tensor([4, 5])
        combination: str = 'x'
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight), [[8 + 15]])
        combination: str = 'y'
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight), [[20 + 25]])
        combination: str = 'x,y'
        weight2: torch.Tensor = torch.Tensor([4, 5, 4, 5])
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight2), [[8 + 20 + 15 + 25]])
        combination: str = 'x-y'
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight), [[-3 * 4 + -2 * 5]])
        combination: str = 'y-x'
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight), [[3 * 4 + 2 * 5]])
        combination: str = 'y+x'
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight), [[7 * 4 + 8 * 5]])
        combination: str = 'y*x'
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight), [[10 * 4 + 15 * 5]])
        combination: str = 'y/x'
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight), [[5 / 2 * 4 + 5 / 3 * 5]], decimal=4)
        combination: str = 'x/y'
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight), [[2 / 5 * 4 + 3 / 5 * 5]], decimal=4)
        with pytest.raises(ConfigurationError):
            util.combine_tensors_and_multiply('x+y+y', tensors, weight)
        with pytest.raises(ConfigurationError):
            util.combine_tensors_and_multiply('x%y', tensors, weight)

    def test_combine_tensors_and_multiply_with_same_batch_size_and_embedding_dim(self) -> None:
        tensors: List[torch.Tensor] = [torch.Tensor([[[5, 5], [4, 4]], [[2, 3], [1, 1]]])]
        weight: torch.Tensor = torch.Tensor([4, 5])
        combination: str = 'x'
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight), [[20 + 25, 16 + 20], [8 + 15, 4 + 5]])
        tensors: List[torch.Tensor] = [torch.Tensor([[[5, 5], [2, 2]], [[4, 4], [3, 3]]]), torch.Tensor([[[2, 3]], [[1, 1]]])]
        weight: torch.Tensor = torch.Tensor([4, 5])
        combination: str = 'x*y'
        assert_almost_equal(util.combine_tensors_and_multiply(combination, tensors, weight), [[5 * 2 * 4 + 5 * 3 * 5, 2 * 2 * 4 + 2 * 3 * 5], [4 * 1 * 4 + 4 * 1 * 5, 3 * 1 * 4 + 3 * 1 * 5]])

    def test_combine_tensors_and_multiply_with_batch_size_one(self) -> None:
        seq_len_1: int = 10
        seq_len_2: int = 5
        embedding_dim: int = 8
        combination: str = 'x,y,x*y'
        t1: torch.Tensor = torch.randn(1, seq_len_1, embedding_dim)
        t2: torch.Tensor = torch.randn(1, seq_len_2, embedding_dim)
        combined_dim: int = util.get_combined_dim(combination, [embedding_dim, embedding_dim])
        weight: torch.Tensor = torch.Tensor(combined_dim)
        result: torch.Tensor = util.combine_tensors_and_multiply(combination, [t1.unsqueeze(2), t2.unsqueeze(1)], weight)
        assert_almost_equal(result.size(), [1, seq_len_1, seq_len_2])

    def test_combine_tensors_and_multiply_with_batch_size_one_and_seq_len_one(self) -> None:
        seq_len_1: int = 10
        seq_len_2: int = 1
        embedding_dim: int = 8
        combination: str = 'x,y,x*y'
        t1: torch.Tensor = torch.randn(1, seq_len_1, embedding_dim)
        t2: torch.Tensor = torch.randn(1, seq_len_2, embedding_dim)
        combined_dim: int = util.get_combined_dim(combination, [embedding_dim, embedding_dim])
        weight: torch.Tensor = torch.Tensor(combined_dim)
        result: torch.Tensor = util.combine_tensors_and_multiply(combination, [t1.unsqueeze(2), t2.unsqueeze(1)], weight)
        assert_almost_equal(result.size(), [1, seq_len_1, seq_len_2])

    def test_combine_initial_dims(self) -> None:
        tensor: torch.Tensor = torch.randn(4, 10, 20, 17, 5)
        tensor2d: torch.Tensor = util.combine_initial_dims(tensor)
        assert list(tensor2d.size()) == [4 * 10 * 20 * 17, 5]

    def test_uncombine_initial_dims(self) -> None:
        embedding2d: torch.Tensor = torch.randn(4 * 10 * 20 * 17 * 5, 12)
        embedding: torch.Tensor = util.uncombine_initial_dims(embedding2d, torch.Size((4, 10, 20, 17, 5)))
        assert list(embedding.size()) == [4, 10, 20, 17, 5, 12]

    def test_inspect_model_parameters(self) -> None:
        model_archive: str = str(self.FIXTURES_ROOT / 'basic_classifier' / 'serialization' / 'model.tar.gz')
        parameters_inspection: str = str(self.FIXTURES_ROOT / 'basic_classifier' / 'parameters_inspection.json')
        model: torch.nn.Module = load_archive(model_archive).model
        with open(parameters_inspection) as file:
            parameters_inspection_dict: Dict[str, Any] = json.load(file)
        assert parameters_inspection_dict == util.inspect_parameters(model)

    def test_move_to_device(self) -> None:
        class FakeTensor(torch.Tensor):
            def __init__(self):
                self._device: torch.device = None

            def to(self, device: torch.device, **kwargs):
                self._device = device
                return self

        class A(NamedTuple):
            pass
        structured_obj: Dict[str, Any] = {'a': [A(1, FakeTensor()), A(2, FakeTensor())], 'b': FakeTensor(), 'c': (1, FakeTensor())}
        new_device: torch.device = torch.device(4)
        moved_obj: Dict[str, Any] = util.move_to_device(structured_obj, new_device)
        assert moved_obj['a'][0].a == 1
        assert moved_obj['a'][0].b._device == new_device
        assert moved_obj['a'][1].b._device == new_device
        assert moved_obj['b']._device == new_device
        assert moved_obj['c'][0] == 1
        assert moved_obj['c'][1]._device == new_device

    def test_extend_layer(self) -> None:
        lin_layer: torch.nn.Linear = torch.nn.Linear(10, 5)
        new_dim: int = 8
        old_weights: torch.Tensor = lin_layer.weight.data.clone()
        old_bias: torch.Tensor = lin_layer.bias.data.clone()
        util.extend_layer(lin_layer, new_dim)
        assert lin_layer.weight.data.shape == (8, 10)
        assert lin_layer.bias.data.shape == (8,)
        assert (lin_layer.weight.data[:5] == old_weights).all()
        assert (lin_layer.bias.data[:5] == old_bias).all()
        assert lin_layer.out_features == new_dim

    def test_masked_topk_selects_top_scored_items_and_respects_masking(self) -> None:
        items: torch.Tensor = torch.randn([3, 4, 5]).clamp(min=0.0, max=1.0)
        items[0, :2, :] = 1
        items[1, 2:, :] = 1
        items[2, 2:, :] = 1
        scores: torch.Tensor = items.sum(-1)
        mask: torch.Tensor = torch.ones([3, 4]).bool()
        mask[1, 0] = 0
        mask[1, 3] = 0
        pruned_scores, pruned_mask, pruned_indices = util.masked_topk(scores, mask, 2)
        numpy.testing.assert_array_equal(pruned_indices.data.numpy(), numpy.array([[0, 1], [1, 2], [2, 3]]))
        numpy.testing.assert_array_equal(pruned_mask.data.numpy(), numpy.ones([3, 2]))
        correct_scores: torch.Tensor = util.batched_index_select(scores.unsqueeze(-1), pruned_indices).squeeze(-1)
        self.assert_array_equal_with_mask(correct_scores, pruned_scores, pruned_mask)

    def test_masked_topk_works_for_completely_masked_rows(self) -> None:
        items: torch.Tensor = torch.randn([3, 4, 5]).clamp(min=0.0, max=1.0)
        items[0, :2, :] = 1
        items[1, 2:, :] = 1
        items[2, 2:, :] = 1
        scores: torch.Tensor = items.sum(-1)
        mask: torch.Tensor = torch.ones([3, 4]).bool()
        mask[1, 0] = 0
        mask[1, 3] = 0
        mask[2, :] = 0
        pruned_scores, pruned_mask, pruned_indices = util.masked_topk(scores, mask, 2)
        numpy.testing.assert_array_equal(pruned_indices[:2].data.numpy(), numpy.array([[0, 1], [1, 2]]))
        numpy.testing.assert_array_equal(pruned_mask.data.numpy(), numpy.array([[1, 1], [1, 1], [0, 0]]))
        correct_scores: torch.Tensor = util.batched_index_select(scores.unsqueeze(-1), pruned_indices).squeeze(-1)
        self.assert_array_equal_with_mask(correct_scores, pruned_scores, pruned_mask)

    def test_masked_topk_selects_top_scored_items_and_respects_masking_different_num_items(self) -> None:
        items: torch.Tensor = torch.randn([3, 4, 5]).clamp(min=0.0, max=1.0)
        items[0, 0, :] = 1.5
        items[0, 1, :] = 2
        items[0, 3, :] = 1
        items[1, 1:3, :] = 1
        items[2, 0, :] = 1
        items[2, 1, :] = 2
        items[2, 2, :] = 1.5
        scores: torch.Tensor = items.sum(-1)
        mask: torch.Tensor = torch.ones([3, 4]).bool()
        mask[1, 3] = 0
        k: torch.Tensor = torch.tensor([3, 2, 1], dtype=torch.long)
        pruned_scores, pruned_mask, pruned_indices = util.masked_topk(scores, mask, k)
        numpy.testing.assert_array_equal(pruned_indices.data.numpy(), numpy.array([[0, 1, 3], [1, 2, 2], [1, 2, 2]]))
        numpy.testing.assert_array_equal(pruned_mask.data.numpy(), numpy.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]))
        correct_scores: torch.Tensor = util.batched_index_select(scores.unsqueeze(-1), pruned_indices).squeeze(-1)
        self.assert_array_equal_with_mask(correct_scores, pruned_scores, pruned_mask)

    def test_masked_topk_works_for_row_with_no_items_requested(self) -> None:
        items: torch.Tensor = torch.randn([3, 4, 5]).clamp(min=0.0, max=1.0)
        items[0, :3, :] = 1
        items[1, 2:, :] = 1
        items[2, 2:, :] = 1
        scores: torch.Tensor = items.sum(-1)
        mask: torch.Tensor = torch.ones([3, 4]).bool()
        mask[1, 0] = 0
        mask[1, 3] = 0
        k: torch.Tensor = torch.tensor([3, 2, 0], dtype=torch.long)
        pruned_scores, pruned_mask, pruned_indices = util.masked_topk(scores, mask, k)
        numpy.testing.assert_array_equal(pruned_indices.data.numpy(), numpy.array([[0, 1, 2], [1, 2, 2], [3, 3, 3]]))
        numpy.testing.assert_array_equal(pruned_mask.data.numpy(), numpy.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]]))
        correct_scores: torch.Tensor = util.batched_index_select(scores.unsqueeze(-1), pruned_indices).squeeze(-1)
        self.assert_array_equal_with_mask(correct_scores, pruned_scores, pruned_mask)

    def test_masked_topk_works_for_multiple_dimensions(self) -> None:
        items: torch.Tensor = torch.FloatTensor([[[4, 2, 9, 9, 7], [-4, -2, -9, -9, -7]], [[5, 4, 1, 8, 8], [9, 1, 7, 4, 1]], [[9, 8, 9, 6, 0], [2, 2, 2, 2, 2]]]).unsqueeze(-1).expand(3, 2, 5, 4)
        mask: torch.Tensor = torch.tensor([[[False, False, False, False, False], [True, True, True, True, True]], [[True, True, True, True, False], [False, True, True, True, True]], [[True, False, True, True, True], [False, True, False, True, True]]]).unsqueeze(-1).expand(3, 2, 5, 4)
        k: torch.Tensor = torch.ones(3, 5, 4, dtype=torch.long)
        k[1, 3, :] = 2
        target_items: torch.Tensor = torch.FloatTensor([[[[-4, -2, -9, -9, -7], [0, 0, 0, 0, 0]], [[5, 4, 7, 8, 1], [0, 0, 0, 4, 0]], [[9, 2, 9, 6, 2], [0, 0, 0, 0, 0]]]).unsqueeze(-1).expand(3, 2, 5, 4)
        target_mask: torch.Tensor = torch.ones(3, 2, 5, 4, dtype=torch.bool)
        target_mask[:, 1, :, :] = 0
        target_mask[1, 1, 3, :] = 1
        target_indices: torch.LongTensor = torch.LongTensor([[[1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], [[0, 0, 1, 0, 1], [0, 0, 0, 1, 0]], [[0, 1, 0, 0, 1], [0, 0, 0, 0, 0]]]).unsqueeze(-1).expand(3, 2, 5, 4)
        pruned_items, pruned_mask, pruned_indices = util.masked_topk(items, mask, k, dim=1)
        numpy.testing.assert_array_equal(pruned_mask.data.numpy(), target_mask.data.numpy())
        self.assert_array_equal_with_mask(pruned_items, target_items, pruned_mask)
        self.assert_array_equal_with_mask(pruned_indices, target_indices, pruned_mask)

    def assert_array_equal_with_mask(self, a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> None:
        numpy.testing.assert_array_equal((a * mask).data.numpy(), (b * mask).data.numpy())

    def test_tensors_equal(self) -> None:
        assert util.tensors_equal(torch.tensor([1]), torch.tensor([1]))
        assert not util.tensors_equal(torch.tensor([1]), torch.tensor([2]))
        assert util.tensors_equal(torch.tensor([True]), torch.tensor([True]))
        assert util.tensors_equal(torch.tensor([1]), torch.tensor([1.0]))
        assert util.tensors_equal(torch.tensor([1]), torch.tensor([True]))
        assert util.tensors_equal([torch.tensor([1])], [torch.tensor([1])])
        assert not util.tensors_equal([torch.tensor([1])], [torch.tensor([2])])
        assert util.tensors_equal({'key': torch.tensor([1])}, {'key': torch.tensor([1])})

    def test_info_value_of_dtype(self) -> None:
        with pytest.raises(TypeError):
            util.info_value_of_dtype(torch.bool)
        assert util.min_value_of_dtype(torch.half) == -65504.0
        assert util.max_value_of_dtype(torch.half) == 65504.0
        assert util.tiny_value_of_dtype(torch.half) == 0.0001
        assert util.min_value_of_dtype(torch.float) == -3.4028234663852886e+38
        assert util.max_value_of_dtype(torch.float) == 3.4028234663852886e+38
        assert util.tiny_value_of_dtype(torch.float) == 1e-13
        assert util.min_value_of_dtype(torch.uint8) == 0
        assert util.max_value_of_dtype(torch.uint8) == 255
        assert util.min_value_of_dtype(torch.long) == -9223372036854775808
        assert util.max_value_of_dtype(torch.long) == 9223372036854775807

    def test_get_token_ids_from_text_field_tensors(self) -> None:
        string_tokens: List[str] = ['This', 'is', 'a', 'test']
        tokens: List[Token] = [Token(x) for x in string_tokens]
        vocab: Vocabulary = Vocabulary()
        vocab.add_tokens_to_namespace(string_tokens, 'tokens')
        vocab.add_tokens_to_namespace(set([char for token in string_tokens for char in token]), 'token_characters')
        elmo_indexer: TokenCharactersIndexer = ELMoTokenCharactersIndexer()
        token_chars_indexer: TokenCharactersIndexer = TokenCharactersIndexer()
        single_id_indexer: SingleIdTokenIndexer = SingleIdTokenIndexer()
        indexers: Dict[str, TokenCharactersIndexer] = {'elmo': elmo_indexer, 'chars': token_chars_indexer, 'tokens': single_id_indexer}
        text_field: TextField = TextField(tokens, {'tokens': single_id_indexer})
        text_field.index(vocab)
        tensors: Dict[str, Dict[str, torch.Tensor]] = text_field.as_tensor(text_field.get_padding_lengths())
        expected_token_ids: torch.Tensor = tensors['tokens']['tokens']
        text_field: TextField = TextField(tokens, indexers)
        text_field.index(vocab)
        tensors: Dict[str, Dict[str, torch.Tensor]] = text_field.as_tensor(text_field.get_padding_lengths())
        token_ids: torch.Tensor = util.get_token_ids_from_text_field_tensors(tensors)
        assert (token_ids == expected_token_ids).all()

    def test_dist_reduce_sum(self) -> None:
        value: Union[int, torch.Tensor] = 23
        ret_value: Union[int, torch.Tensor] = util.dist_reduce_sum(value)
        assert ret_value == 23
        value: torch.Tensor = torch.Tensor([1, 2, 3])
        ret_value: torch.Tensor = util.dist_reduce_sum(value)
        assert (ret_value == value).all().item()
        func_kwargs: Dict[str, List[torch.Tensor]] = {'value': [torch.Tensor([1, 2, 3]), torch.Tensor([4, 5, 6])]}
        desired_values: torch.Tensor = torch.Tensor([5, 7, 9])
        run_distributed_test([-1, -1], global_distributed_func, function=util.dist_reduce_sum, func_kwargs=func_kwargs, desired_values=desired_values)

def global_distributed_func(global_rank: int, world_size: int, gpu_id: int, function: Callable, func_kwargs: Dict[str, List[torch.Tensor]], desired_values: torch.Tensor) -> None:
    kwargs: Dict[str, torch.Tensor] = {}
    for argname in func_kwargs:
        kwargs[argname] = func_kwargs[argname][global_rank]
    output: torch.Tensor = function(**kwargs)
    assert (output == desired_values).all().item()

class DistributedFixtureModel(torch.nn.Module):
    """
    Fake model for testing `load_state_dict_distributed()`.
    """

    def __init__(self):
        super().__init__()
        self.direct_param: torch.nn.Parameter = torch.nn.Parameter(torch.randn(3, 5))
        self.register_buffer('direct_buffer', torch.randn(2, 2))
        self.custom_submodule: DistributedFixtureSubmodule = DistributedFixtureSubmodule()
        self.custom_sharded_submodule: ShardedDistributedFixtureSubmodule = ShardedDistributedFixtureSubmodule()
        self.linear_submodule: torch.nn.Linear = torch.nn.Linear(3, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class DistributedFixtureSubmodule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.direct_param: torch.nn.Parameter = torch.nn.Parameter(torch.randn(3, 5))
        self.register_buffer('direct_buffer', torch.randn(2, 2))
        self.linear_submodule: torch.nn.Linear = torch.nn.Linear(3, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class ShardedDistributedFixtureSubmodule(DistributedFixtureSubmodule, ShardedModuleMixin):

    def get_original_module(self) -> DistributedFixtureSubmodule:
        return self

def _dist_load_ok(global_rank: int, world_size: int, gpu_id: int) -> None:
    model: DistributedFixtureModel = DistributedFixtureModel()
    state_dict: Dict[str, torch.Tensor] = None if global_rank != 0 else model.state_dict()
    missing_keys, unexpected_keys = util.load_state_dict_distributed(model, state_dict)
    assert not missing_keys
    assert not unexpected_keys

def _dist_load_with_errors(global_rank: int, world_size: int, gpu_id: int) -> None:
    model: DistributedFixtureModel = DistributedFixtureModel()
    state_dict: Dict[str, torch.Tensor] = None if global_rank != 0 else model.state_dict()
    _missing_keys = ['direct_buffer', 'custom_submodule.linear_submodule.bias', 'custom_submodule.direct_param', 'custom_sharded_submodule.linear_submodule.bias', 'custom_sharded_submodule.direct_buffer']
    _unexpected_keys = ['not_a_parameter', 'custom_submodule.not_a_parameter', 'custom_submodule.linear.not_a_parameter', 'custom_sharded_submodule.not_a_parameter', 'custom_sharded_submodule.linear.not_a_parameter', 'not_even_submodule.not_a_parameter']
    if state_dict is not None:
        for key in _missing_keys:
            del state_dict[key]
        for key in _unexpected_keys:
            state_dict[key] = torch.randn(2, 2)
    missing_keys, unexpected_keys = util.load_state_dict_distributed(model, state_dict, strict=False)
    if global_rank == 0:
        assert set(missing_keys) == set(_missing_keys)
        assert set(unexpected_keys) == set(_unexpected_keys)

@pytest.mark.parametrize('test_func', [_dist_load_ok, _dist_load_with_errors])
def test_load_state_dict_distributed(test_func: Callable) -> None:
    run_distributed_test([-1, -1], func=test_func)
