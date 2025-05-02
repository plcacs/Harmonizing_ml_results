import json
import random
from typing import NamedTuple, Any, Union, Callable, Dict, List, Tuple, Optional, Set
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
        binary_mask = torch.tensor([[True, True, True, False, False, False], [True, True, False, False, False, False], [True, True, True, True, True, True], [True, False, False, False, False, False]])
        lengths = util.get_lengths_from_binary_sequence_mask(binary_mask)
        numpy.testing.assert_array_equal(lengths.numpy(), numpy.array([3, 2, 6, 1]))

    def test_get_mask_from_sequence_lengths(self) -> None:
        sequence_lengths = torch.LongTensor([4, 3, 1, 4, 2])
        mask = util.get_mask_from_sequence_lengths(sequence_lengths, 5).data.numpy()
        assert_almost_equal(mask, [[1, 1, 1, 1, 0], [1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 0]])

    def test_get_sequence_lengths_converts_to_long_tensor_and_avoids_variable_overflow(self) -> None:
        binary_mask = torch.ones(2, 260).bool()
        lengths = util.get_lengths_from_binary_sequence_mask(binary_mask)
        numpy.testing.assert_array_equal(lengths.data.numpy(), numpy.array([260, 260]))

    def test_clamp_tensor(self) -> None:
        i = torch.LongTensor([[0, 1, 1, 0], [2, 0, 2, 2]])
        v = torch.FloatTensor([3, 4, -5, 3])
        tensor = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
        clamped_tensor = util.clamp_tensor(tensor, minimum=-3, maximum=3).to_dense()
        assert_almost_equal(clamped_tensor, [[0, 0, 3], [3, 0, -3]])
        i = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
        v = torch.FloatTensor([3, 4, -5])
        tensor = torch.sparse.FloatTensor(i, v, torch.Size([2, 3]))
        clamped_tensor = util.clamp_tensor(tensor, minimum=-3, maximum=3).to_dense()
        assert_almost_equal(clamped_tensor, [[0, 0, 3], [3, 0, -3]])
        tensor = torch.tensor([[5, -4, 3], [-3, 0, -30]])
        clamped_tensor = util.clamp_tensor(tensor, minimum=-3, maximum=3)
        assert_almost_equal(clamped_tensor, [[3, -3, 3], [-3, 0, -3]])

    def test_sort_tensor_by_length(self) -> None:
        tensor = torch.rand([5, 7, 9])
        tensor[0, 3:, :] = 0
        tensor[1, 4:, :] = 0
        tensor[2, 1:, :] = 0
        tensor[3, 5:, :] = 0
        sequence_lengths = torch.LongTensor([3, 4, 1, 5, 7])
        sorted_tensor, sorted_lengths, reverse_indices, _ = util.sort_batch_by_length(tensor, sequence_lengths)
        numpy.testing.assert_array_equal(sorted_tensor[1, 5:, :].data.numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[2, 4:, :].data.numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[3, 3:, :].data.numpy(), 0.0)
        numpy.testing.assert_array_equal(sorted_tensor[4, 1:, :].data.numpy(), 0.0)
        assert sorted_lengths.data.equal(torch.LongTensor([7, 5, 4, 3, 1]))
        assert sorted_tensor.index_select(0, reverse_indices).data.equal(tensor.data)

    def test_get_final_encoder_states(self) -> None:
        encoder_outputs = torch.Tensor([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]])
        mask = torch.tensor([[True, True, True], [True, True, False]])
        final_states = util.get_final_encoder_states(encoder_outputs, mask, bidirectional=False)
        assert_almost_equal(final_states.data.numpy(), [[9, 10, 11, 12], [17, 18, 19, 20]])
        final_states = util.get_final_encoder_states(encoder_outputs, mask, bidirectional=True)
        assert_almost_equal(final_states.data.numpy(), [[9, 10, 3, 4], [17, 18, 15, 16]])

    def test_masked_softmax_no_mask(self) -> None:
        vector_1d = torch.FloatTensor([[1.0, 2.0, 3.0]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, None).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.090031, 0.244728, 0.665241]]))
        assert_almost_equal(1.0, numpy.sum(vector_1d_softmaxed), decimal=6)
        vector_1d = torch.FloatTensor([[1.0, 2.0, 5.0]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, None).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.017148, 0.046613, 0.93624]]))
        vector_zero = torch.FloatTensor([[0.0, 0.0, 0.0]])
        vector_zero_softmaxed = util.masked_softmax(vector_zero, None).data.numpy()
        assert_array_almost_equal(vector_zero_softmaxed, numpy.array([[0.33333334, 0.33333334, 0.33333334]]))
        matrix = torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, None).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.01714783, 0.04661262, 0.93623955], [0.09003057, 0.24472847, 0.66524096]]))
        matrix = torch.FloatTensor([[1.0, 2.0, 5.0], [0.0, 0.0, 0.0]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, None).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.01714783, 0.04661262, 0.93623955], [0.33333334, 0.33333334, 0.33333334]]))

    def test_masked_softmax_masked(self) -> None:
        vector_1d = torch.FloatTensor([[1.0, 2.0, 5.0]])
        mask_1d = torch.tensor([[True, False, True]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.01798621, 0.0, 0.98201382]]))
        vector_1d = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d = torch.tensor([[True, False, True, True]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.01321289, 0.0, 0.26538793, 0.72139918]]))
        vector_1d = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d = torch.tensor([[False, False, False, True]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0, 0, 0, 1]]))
        vector_1d = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d = torch.FloatTensor([[1.0, 1.0, 100000.0]])
        mask_1d = torch.tensor([[True, True, False]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.5, 0.5, 0]]))
        matrix = torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, False, True], [True, True, True]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.01798621, 0.0, 0.98201382], [0.090031, 0.244728, 0.665241]]))
        matrix = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, False, True], [True, True, True]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.5, 0.0, 0.5], [0.090031, 0.244728, 0.665241]]))
        matrix = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, False, True], [False, False, False]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.5, 0.0, 0.5], [0.0, 0.0, 0.0]]))
        matrix = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask = torch.tensor([[False, False, False], [True, False, True]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, mask).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.0, 0.0, 0.0], [0.11920292, 0.0, 0.88079708]]))

    def test_masked_softmax_memory_efficient_masked(self) -> None:
        vector_1d = torch.FloatTensor([[1.0, 2.0, 5.0]])
        mask_1d = torch.tensor([[True, False, True]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.01798621, 0.0, 0.98201382]]))
        vector_1d = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d = torch.tensor([[True, False, True, True]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.01321289, 0.0, 0.26538793, 0.72139918]]))
        vector_1d = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d = torch.tensor([[False, False, False, True]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0, 0, 0, 1]]))
        vector_1d = torch.FloatTensor([[0.0, 2.0, 3.0, 4.0]])
        mask_1d = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.25, 0.25, 0.25, 0.25]]))
        vector_1d = torch.FloatTensor([[0.0, 0.0, 0.0, 0.0]])
        mask_1d = torch.tensor([[False, False, False, False]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.25, 0.25, 0.25, 0.25]]))
        vector_1d = torch.FloatTensor([[1.0, 1.0, 100000.0]])
        mask_1d = torch.tensor([[True, True, False]])
        vector_1d_softmaxed = util.masked_softmax(vector_1d, mask_1d, memory_efficient=True).data.numpy()
        assert_array_almost_equal(vector_1d_softmaxed, numpy.array([[0.5, 0.5, 0]]))
        matrix = torch.FloatTensor([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, False, True], [True, True, True]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, mask, memory_efficient=True).data.numpy()
        assert_array_almost_equal(masked_matrix_softmaxed, numpy.array([[0.01798621, 0.0, 0.98201382], [0.090031, 0.244728, 0.665241]]))
        matrix = torch.FloatTensor([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        mask = torch.tensor([[True, False, True], [True, True, True]])
        masked_matrix_softmaxed = util.masked_softmax(matrix, mask, memory_efficient=True).data.numpy()
        assert_array_almost_equal(masked_matrix