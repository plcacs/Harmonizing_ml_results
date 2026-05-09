import json
import random
from typing import NamedTuple, Any, Union, Callable, Dict, List, Optional, Tuple
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
        sequence_lengths: torch.Tensor = torch.LongTensor([4, 3, 1, 4, 2])
        mask: numpy.ndarray = util.get_mask_from_sequence_lengths(sequence_lengths, 5).data.numpy()
        assert_almost_equal(mask, [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])

    def test_get_sequence_lengths_converts_to_long_tensor_and_avoids_variable_overflow(self) -> None:
        binary_mask: torch.Tensor = torch.ones(2, 260).bool()
        lengths: torch.Tensor = util.get_lengths_from_binary_sequence_mask(binary_mask)
        numpy.testing.assert_array_equal(lengths.data.numpy(), numpy.array([260, 260]))

    def test_clamp_tensor(self) -> None:
        i: torch.Tensor = torch.LongTensor([[0, 1, 1, 0], [2, 0, 2, 2]])
        v: torch.Tensor = torch.FloatTensor([3, 4, -5, 3])
        tensor: torch.Tensor = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
        clamped_tensor: torch.Tensor = util.clamp_tensor(tensor, minimum=-3, maximum=3).to_dense()
        assert_almost_equal(clamped_tensor, [[0, 0, 3], [3, 0, -3]])
        i: torch.Tensor = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
        v: torch.Tensor = torch.FloatTensor([3, 4, -5])
        tensor: torch.Tensor = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
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
        sequence_lengths: torch.Tensor = torch.LongTensor([3, 4, 1, 5, 7])
        sorted_tensor: torch.Tensor
        sorted_lengths: torch.Tensor
        reverse_indices: torch.Tensor
        _ : torch.Tensor
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
        vector_1d: torch.Tensor = torch.FloatTensor([[1.0, 2.0, 3.0]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, None).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.090031, 0.244728, 0.665241]]))
        assert_almost_equal(1.0, numpy.sum(vector_1d_softmaxed), decimal=6)
        vector_1d: torch.Tensor = torch.FloatTensor([[1.0, 2.0, 5.0]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, None).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.017148, 0.046613, 0.93624]]))
        vector_zero: torch.Tensor = torch.FloatTensor([[0.0, 0.0, 0.0]])
        vector_zero_softmaxed: torch.Tensor = util.masked_softmax(vector_zero, None).data.numpy()
        assert_array_almost_equal(vector_zero_softmaxed, numpy.array([[0.33333334, 0.33333334, 0.33333334]]))
        matrix: torch.Tensor = torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, None).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.01714783, 0.04661262, 0.93623955], [0.09003057, 0.24472847, 0.66524096]]))
        matrix: torch.Tensor = torch.FloatTensor([[1.0, 2.0, 5.0], [0.0, 0.0, 0.0]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, None).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.01714783, 0.04661262, 0.93623955], [0.33333334, 0.33333334, 0.33333334]]))

    def test_masked_softmax_masked(self) -> None:
        vector_1d: torch.Tensor = torch.FloatTensor([[1.0, 2.0, 5.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, False, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.01798621, 0.0, 0.98201382]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, False, True, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.01321289, 0.0, 0.26538793, 0.72139918]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0, 0, 0, 1]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[1.0, 1.0, 100000.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, True, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.5, 0.5, 0]]))
        matrix: torch.Tensor = torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, True]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.01798621, 0.0, 0.98201382], [0.090031, 0.244728, 0.665241]]))
        matrix: torch.Tensor = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, True]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.5, 0.0, 0.5], [0.090031, 0.244728, 0.665241]]))
        matrix: torch.Tensor = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [False, False, False]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]))
        matrix: torch.Tensor = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[False, False, False], [True, False, True]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.0, 0.0, 0.0], [0.11920292, 0.0, 0.88079708]]))

    def test_masked_softmax_memory_efficient_masked(self) -> None:
        vector_1d: torch.Tensor = torch.FloatTensor([[1.0, 2.0, 5.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, False, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.01798621, 0.0, 0.98201382]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, False, True, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.01321289, 0.0, 0.26538793, 0.72139918]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0, 0, 0, 1]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.25, 0.25, 0.25, 0.25]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.25, 0.25, 0.25, 0.25]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[1.0, 1.0, 100000.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, True, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.5, 0.5, 0]]))
        matrix: torch.Tensor = torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, True]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask, memory_efficient=True).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.01798621, 0.0, 0.98201382], [0.090031, 0.244728, 0.665241]]))
        matrix: torch.Tensor = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, True]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask, memory_efficient=True).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.5, 0.0, 0.5], [0.090031, 0.244728, 0.665241]]))
        matrix: torch.Tensor = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [False, False, False]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask, memory_efficient=True).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.5, 0.0, 0.5], [0.33333333, 0.33333333, 0.33333333]]))
        matrix: torch.Tensor = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[False, False, False], [True, False, True]])
        masked_matrix_softmaxed: torch.Tensor = util.masked_softmax(matrix, mask, memory_efficient=True).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.33333333, 0.33333333, 0.33333333], [0.11920292, 0.0, 0.88079708]]))

    def test_masked_log_softmax_masked(self) -> None:
        vector_1d: torch.Tensor = torch.FloatTensor([[1.0, 2.0, 5.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, False, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_log_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(numpy.exp(vector_1d_softmaxed), numpy.array([[0.01798621, 0.0, 0.98201382]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d: torch.Tensor = torch.tensor([[True, False, True, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_log_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(numpy.exp(vector_1d_softmaxed), numpy.array([[0.01321289, 0.0, 0.26538793, 0.72139918]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, True]])
        vector_1d_softmaxed: torch.Tensor = util.masked_log_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(numpy.exp(vector_1d_softmaxed), numpy.array([[0.0, 0.0, 0.0, 1.0]]))
        vector_1d: torch.Tensor = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d: torch.Tensor = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed: torch.Tensor = util.masked_log_softmax(vector_1d, mask_1d).data.numpy()
        assert not numpy.isnan(vector_1d_softmaxed).any()

    def test_masked_max(self) -> None:
        vector_1d: torch.Tensor = torch.FloatTensor([1.0, 12.0, 5.0])
        mask_1d: torch.Tensor = torch.tensor([True, False, True])
        vector_1d_maxed: float = util.masked_max(vector_1d, mask_1d, dim=0).data.numpy()
        assert_array_almost_equal(vector_1d_maxed, 5.0)
        vector_1d: torch.Tensor = torch.FloatTensor([1.0, 12.0, 5.0])
        mask_1d: torch.Tensor = torch.tensor([False, False, False])
        vector_1d_maxed: float = util.masked_max(vector_1d, mask_1d, dim=0).data.numpy()
        assert not numpy.isnan(vector_1d_maxed).any()
        matrix: torch.Tensor = torch.FloatTensor([[1.0, 12.0, 5.0], [-1.0, -2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, False]])
        matrix_maxed: torch.Tensor = util.masked_max(matrix, mask, dim=-1).data.numpy()
        assert_array_almost_equal(matrix_maxed, numpy.array([5.0, -1.0]))
        matrix: torch.Tensor = torch.FloatTensor([[1.0, 12.0, 5.0], [-1.0, -2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, False]])
        matrix_maxed: torch.Tensor = util.masked_max(matrix, mask, dim=-1, keepdim=True).data.numpy()
        assert_array_almost_equal(matrix_maxed, numpy.array([[5.0], [-1.0]]))
        matrix: torch.Tensor = torch.FloatTensor([[[1.0, 2.0], [12.0, 3.0], [5.0, -1.0]], [[-1.0, -3.0], [-2.0, -0.5], [3.0, 8.0]]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, False]]).unsqueeze(-1)
        matrix_maxed: torch.Tensor = util.masked_max(matrix, mask, dim=1).data.numpy()
        assert_array_almost_equal(matrix_maxed, numpy.array([[5.0, 2.0], [-1.0, -0.5]]))

    def test_masked_mean(self) -> None:
        vector_1d: torch.Tensor = torch.FloatTensor([1.0, 12.0, 5.0])
        mask_1d: torch.Tensor = torch.tensor([True, False, True])
        vector_1d_mean: float = util.masked_mean(vector_1d, mask_1d, dim=0).data.numpy()
        assert_array_almost_equal(vector_1d_mean, 3.0)
        vector_1d: torch.Tensor = torch.FloatTensor([1.0, 12.0, 5.0])
        mask_1d: torch.Tensor = torch.tensor([False, False, False])
        vector_1d_mean: float = util.masked_mean(vector_1d, mask_1d, dim=0).data.numpy()
        assert not numpy.isnan(vector_1d_mean).any()
        matrix: torch.Tensor = torch.FloatTensor([[1.0, 12.0, 5.0], [-1.0, -2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, False]])
        matrix_mean: torch.Tensor = util.masked_mean(matrix, mask, dim=-1).data.numpy()
        assert_array_almost_equal(matrix_mean, numpy.array([3.0, -1.5]))
        matrix: torch.Tensor = torch.FloatTensor([[1.0, 12.0, 5.0], [-1.0, -2.0, 3.0]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, False]])
        matrix_mean: torch.Tensor = util.masked_mean(matrix, mask, dim=-1, keepdim=True).data.numpy()
        assert_array_almost_equal(matrix_mean, numpy.array([[3.0], [-1.5]]))
        matrix: torch.Tensor = torch.FloatTensor([[[1.0, 2.0], [12.0, 3.0], [5.0, -1.0]], [[-1.0, -3.0], [-2.0, -0.5], [3.0, 8.0]]])
        mask: torch.Tensor = torch.tensor([[True, False, True], [True, True, False]]).unsqueeze(-1)
        matrix_mean: torch.Tensor = util.masked_mean(matrix, mask, dim=1).data.numpy()
        assert_array_almost_equal(matrix_mean, numpy.array([[3.0, 0.5], [-1.5, -1.75]]))

    def test_masked_flip(self) -> None:
        tensor: torch.Tensor = torch.FloatTensor([[[6, 6, 6], [1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4], [5, 5, 5]]])
        solution: List[List[List[int]]] = [[[6, 6, 6], [0, 0, 0]], [[4, 4, 4], [3, 3, 3]]]
        response: torch.Tensor = util.masked_flip(tensor, [1, 2])
        assert_almost_equal(response, solution)
        tensor: torch.Tensor = torch.FloatTensor([[[6, 6, 6], [1, 1, 1], [2, 2, 2], [0, 0, 0]], [[3, 3, 3], [4, 4, 4], [5, 5, 5], [1, 2, 3]]])
        solution: List[List[List[int]]] = [[[2, 2, 2], [1, 1, 1], [6, 6, 6], [0, 0, 0]], [[1, 2, 3], [5, 5, 5], [4, 4, 4], [3, 3, 3]]]
        response: torch.Tensor = util.masked_flip(tensor, [3, 4])
        assert_almost_equal(response, solution)
        tensor: torch.Tensor = torch.FloatTensor([[[6, 6, 6], [1, 1, 1], [2, 2, 2], [0, 0, 0]], [[3, 3, 3], [4, 4, 4], [5, 5, 5], [1, 2, 3]], [[1, 1, 1], [2, 2, 2], [0, 0, 0], [0, 0, 0]]])
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
        expected_mask: numpy.ndarray = (text_field_tensors['indexer_name']['list_tokens'].numpy() > 0).astype('int32')
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
        expected_array: numpy.ndarray = 0.3 * sentence_array[0, 0] + 0.4 * sentence_array[0, 1] + 0.1 * sentence_array[0, 2] + 0.0 * sentence_array[0, 3] + 1.2 * sentence_array[0, 4]
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
        expected_array: numpy.ndarray = attention_array[0, 3, 2, 0] * sentence_array[0, 3, 2, 0] + attention_array[0, 3, 2, 1] * sentence_array[0, 3, 2, 1]
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
                expected_array: numpy.ndarray = attention_array[0, i, j, 0] * sentence_array[0, 0] + attention_array[0, i, j, 1] * sentence_array[0, 1]
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
            expected_array: numpy.ndarray = attention_array[0, i, 0] * sentence_array[0, 0] + attention_array[0, i, 1] * sentence_array[0, 1]
            numpy.testing.assert_almost_equal(aggregated_array[0, i], expected_array, decimal=5)

    def test_viterbi_decode(self) -> None:
        sequence_logits: torch.Tensor = torch.nn.functional.softmax(torch.rand([5, 9]), dim=-1)
        transition_matrix: torch.Tensor = torch.zeros([9, 9])
        indices: List[int]
        _ : torch.Tensor
        indices, _ = util.viterbi_decode(sequence_logits.data, transition_matrix)
        _, argmax_indices: torch.Tensor = torch.max(sequence_logits, 1)
        assert indices == argmax_indices.data.squeeze().tolist()
        sequence_logits: torch.Tensor = torch.nn.functional.softmax(torch.rand([5, 9]), dim=-1)
        transition_matrix: torch.Tensor = torch.zeros([9, 9])
        allowed_start_transitions: torch.Tensor = torch.zeros([9])
        allowed_start_transitions[:8] = float('-inf')
        allowed_end_transitions: torch.Tensor = torch.zeros([9])
        allowed_end_transitions[1:] = float('-inf')
        indices: List[int]
        _ : torch.Tensor
        indices, _ = util.viterbi_decode(sequence_logits.data, transition_matrix, allowed_end_transitions=allowed_end_transitions, allowed_start_transitions=allowed_start_transitions)
        assert indices[0] == 8
        assert indices[-1] == 0
        sequence_logits: torch.Tensor = torch.FloatTensor([[0, 0, 0, 3, 5], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4]])
        transition_matrix: torch.Tensor = torch.zeros([5, 5])
        for i in range(5):
            transition_matrix[i, i] = float('-inf')
        indices: List[int]
        _ : torch.Tensor
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix)
        assert indices == [4, 3, 4, 3, 4, 3]
        sequence_logits: torch.Tensor = torch.FloatTensor([[0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4]])
        transition_matrix: torch.Tensor = torch.zeros([5, 5])
        transition_matrix[4, 4] = -10
        transition_matrix[4, 3] = -10
        transition_matrix[3, 4] = -10
        indices: List[int]
        _ : torch.Tensor
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix)
        assert indices == [3, 3, 3, 3, 3, 3]
        sequence_logits: torch.Tensor = torch.FloatTensor([[1, 0, 0, 4], [1, 0, 6, 2], [0, 3, 0, 4]])
        transition_matrix: torch.Tensor = torch.zeros([4, 4])
        transition_matrix[0, 0] = 1
        transition_matrix[2, 1] = 5
        indices: List[int]
        _ : torch.Tensor
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix)
        assert indices == [3, 2, 1]
        sequence_logits: torch.Tensor = torch.FloatTensor([[0, 0, 0, 7, 7], [0, 0, 0, 7, 7], [0, 0, 0, 7, 7], [0, 0, 0, 7, 7], [0, 0, 0, 7, 7], [0, 0, 0, 7, 7]])
        transition_matrix: torch.Tensor = torch.zeros([5, 5])
        transition_matrix[4, 4] = -10
        transition_matrix[4, 3] = -2
        transition_matrix[3, 4] = -2
        observations: List[int] = [2, -1, -1, 0, 4, -1]
        indices: List[int]
        _ : torch.Tensor
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix, observations)
        assert indices == [2, 3, 3, 0, 4, 3]

    def test_viterbi_decode_top_k(self) -> None:
        sequence_logits: torch.Tensor = torch.autograd.Variable(torch.rand([5, 9]))
        transition_matrix: torch.Tensor = torch.zeros([9, 9])
        indices: List[List[int]]
        _ : torch.Tensor
        indices, _ = util.viterbi_decode(sequence_logits.data, transition_matrix, top_k=5)
        _, argmax_indices: torch.Tensor = torch.max(sequence_logits, 1)
        assert indices[0] == argmax_indices.data.squeeze().tolist()
        sequence_logits: torch.Tensor = torch.FloatTensor([[0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4], [0, 0, 0, 3, 4]])
        transition_matrix: torch.Tensor = torch.zeros([5, 5])
        for i in range(5):
            transition_matrix[i, i] = float('-inf')
        indices: List[List[int]]
        _ : torch.Tensor
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix, top_k=5)
        assert indices[0] == [3, 4, 3, 4, 3, 4]
        sequence_logits: torch.Tensor = torch.FloatTensor([[0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 4], [0, 0, 0, 4, 0]])
        transition_matrix: torch.Tensor = torch.zeros([5, 5])
        transition_matrix[4, 4] = -10
        transition_matrix[4, 3] = -2
        transition_matrix[3, 4] = -2
        indices: List[List[int]]
        _ : torch.Tensor
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix, top_k=5)
        assert indices[0] == [3, 3, 3, 3, 3, 3]
        sequence_logits: torch.Tensor = torch.FloatTensor([[1, 0, 0, 4], [1, 0, 6, 2], [0, 3, 0, 4]])
        transition_matrix: torch.Tensor = torch.zeros([4, 4])
        transition_matrix[0, 0] = 1
        transition_matrix[2, 1] = 5
        indices: List[List[int]]
        _ : torch.Tensor
        indices, _ = util.viterbi_decode(sequence_logits, transition_matrix, top_k=5)
        assert indices[0] == [3, 2, 1]
        assert _[0] == 18

        def _brute_decode(tag_sequence: torch.Tensor, transition_matrix: torch.Tensor, top_k: int = 5) -> Tuple[List[List[int]], List[float]]:
            sequences = [[]]
            for i in range(len(tag_sequence)):
                new_sequences = []
                for j in range(len(tag_sequence[i])):
                    for sequence in sequences:
                        new_sequences.append(sequence[:] + [j])
                sequences = new_sequences
            scored_sequences = []
            for sequence in sequences:
                emission_score: float = sum((tag_sequence[i, j] for i, j in enumerate(sequence)))
                transition_score: float = sum((transition_matrix[sequence[i - 1], sequence[i]] for i in range(1, len(sequence))))
                score: float = emission_score + transition_score
                scored_sequences.append((score, sequence))
            top_k_sequences = sorted(scored_sequences, key=lambda r: r[0], reverse=True)[:top_k]
            scores: List[float]
            paths: List[List[int]]
            scores, paths = zip(*top_k_sequences)
            return (paths, scores)
        for _ in range(100):
            num_tags: int = random.randint(1, 5)
            seq_len: int = random.randint(1, 5)
            k: int = random.randint(1, 5)
            sequence_logits: torch.Tensor = torch.rand([seq_len, num_tags])
            transition_matrix: torch.Tensor = torch.rand([num_tags, num_tags])
            viterbi_paths_v1: List[List[int]]
            viterbi_scores_v1: List[float]
            viterbi_paths_v1, viterbi_scores_v1 = util.viterbi_decode(sequence_logits, transition_matrix, top_k=k)
            viterbi_path_brute: List[List[int]]
            viterbi_score_brute: List[float]
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
        targets: torch.Tensor = torch.LongTensor(numpy.random.randint(0, 3, [5, 7]))
        targets *= weights
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights)
        loss2: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor2, targets, weights)
        assert loss.data.numpy() == loss2.data.numpy()

    def test_sequence_cross_entropy_with_logits_smooths_labels_correctly(self) -> None:
        tensor: torch.Tensor = torch.rand([1, 3, 4])
        targets: torch.Tensor = torch.LongTensor(numpy.random.randint(0, 3, [1, 3]))
        weights: torch.Tensor = torch.ones([2, 3])
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights, label_smoothing=0.1)
        correct_loss: float = 0.0
        for prediction, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            prediction: torch.Tensor = torch.nn.functional.log_softmax(prediction, dim=-1)
            correct_loss += prediction[label] * 0.9
            correct_loss += prediction.sum() * 0.1 / 4
        correct_loss: torch.Tensor = -correct_loss / 3
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())

    def test_sequence_cross_entropy_with_logits_averages_batch_correctly(self) -> None:
        tensor: torch.Tensor = torch.rand([5, 7, 4])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 2:, :] = 0
        tensor[3, :, :] = 0
        weights: torch.Tensor = (tensor != 0.0)[:, :, 0].long().squeeze(-1)
        targets: torch.Tensor = torch.LongTensor(numpy.random.randint(0, 3, [5, 7]))
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
        targets: torch.Tensor = torch.LongTensor(numpy.random.randint(0, 3, [5, 7]))
        targets *= weights
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights, average='token')
        vector_loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights, average=None)
        total_token_loss: float = (vector_loss * weights.float().sum(dim=-1)).sum()
        average_token_loss: float = (total_token_loss / weights.float().sum()).detach()
        assert_almost_equal(loss.detach().item(), average_token_loss.item(), decimal=5)

    def test_sequence_cross_entropy_with_logits_gamma_correctly(self) -> None:
        batch: int = 1
        length: int = 3
        classes: int = 4
        gamma: float = abs(numpy.random.randn())
        tensor: torch.Tensor = torch.rand([batch, length, classes])
        targets: torch.Tensor = torch.LongTensor(numpy.random.randint(0, classes, [batch, length]))
        weights: torch.Tensor = torch.ones([batch, length])
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor, targets, weights, gamma=gamma)
        correct_loss: float = 0.0
        for logit, label in zip(tensor.squeeze(0), targets.squeeze(0)):
            p: torch.Tensor = torch.nn.functional.softmax(logit, dim=-1)
            pt: float = p[label]
            ft: float = (1 - pt) ** gamma
            correct_loss += -pt.log() * ft
        correct_loss: torch.Tensor = correct_loss / length
        numpy.testing.assert_array_almost_equal(loss.data.numpy(), correct_loss.data.numpy())

    def test_sequence_cross_entropy_with_logits_alpha_float_correctly(self) -> None:
        batch: int = 1
        length: int = 3
        classes: int = 2
        alpha: float = numpy.random.rand() if numpy.random.rand() > 0.5 else 1.0 - numpy.random.rand()
        tensor: torch.Tensor = torch.rand([batch, length, classes])
        targets: torch.Tensor = torch.LongTensor(numpy.random.randint(0, classes, [batch, length]))
        weights: torch.Tensor = torch.ones([batch, length])
        loss: torch.Tensor = util.sequence_cross_entropy_with_logits(tensor