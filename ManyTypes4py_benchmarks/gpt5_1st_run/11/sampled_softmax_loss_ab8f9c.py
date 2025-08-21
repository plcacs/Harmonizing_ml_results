from typing import Set, Tuple, Optional, Callable, Union
import numpy as np
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.nn import util

def _choice(num_words: int, num_samples: int) -> Tuple[np.ndarray, int]:
    """
    Chooses `num_samples` samples without replacement from [0, ..., num_words).
    Returns a tuple (samples, num_tries).
    """
    num_tries: int = 0
    num_chosen: int = 0

    def get_buffer() -> np.ndarray:
        log_samples = np.random.rand(num_samples) * np.log(num_words + 1)
        samples = np.exp(log_samples).astype('int64') - 1
        return np.clip(samples, a_min=0, a_max=num_words - 1)

    sample_buffer: np.ndarray = get_buffer()
    buffer_index: int = 0
    samples: Set[int] = set()
    while num_chosen < num_samples:
        num_tries += 1
        sample_id = sample_buffer[buffer_index]
        if sample_id not in samples:
            samples.add(int(sample_id))
            num_chosen += 1
        buffer_index += 1
        if buffer_index == num_samples:
            sample_buffer = get_buffer()
            buffer_index = 0
    return (np.array(list(samples)), num_tries)

class SampledSoftmaxLoss(torch.nn.Module):
    """
    Based on the default log_uniform_candidate_sampler in tensorflow.

    !!! NOTE
        num_words DOES NOT include padding id.

    !!! NOTE
        In all cases except (tie_embeddings=True and use_character_inputs=False)
        the weights are dimensioned as num_words and do not include an entry for the padding (0) id.
        For the (tie_embeddings=True and use_character_inputs=False) case,
        then the embeddings DO include the extra 0 padding, to be consistent with the word embedding layer.

    # Parameters

    num_words, `int`, required
        The number of words in the vocabulary
    embedding_dim, `int`, required
        The dimension to softmax over
    num_samples, `int`, required
        During training take this many samples. Must be less than num_words.
    sparse, `bool`, optional (default = `False`)
        If this is true, we use a sparse embedding matrix.
    unk_id, `int`, optional (default = `None`)
        If provided, the id that represents unknown characters.
    use_character_inputs, `bool`, optional (default = `True`)
        Whether to use character inputs
    use_fast_sampler, `bool`, optional (default = `False`)
        Whether to use the fast cython sampler.
    """

    tie_embeddings: bool
    choice_func: Callable[[int, int], Tuple[np.ndarray, int]]
    softmax_w: Union[torch.nn.Embedding, torch.nn.Parameter]
    softmax_b: Union[torch.nn.Embedding, torch.nn.Parameter]
    sparse: bool
    use_character_inputs: bool
    _unk_id: Optional[int]
    _num_samples: int
    _embedding_dim: int
    _num_words: int
    _log_num_words_p1: float
    _probs: np.ndarray

    def __init__(
        self,
        num_words: int,
        embedding_dim: int,
        num_samples: int,
        sparse: bool = False,
        unk_id: Optional[int] = None,
        use_character_inputs: bool = True,
        use_fast_sampler: bool = False
    ) -> None:
        super().__init__()
        self.tie_embeddings = False
        assert num_samples < num_words
        if use_fast_sampler:
            raise ConfigurationError('fast sampler is not implemented')
        else:
            self.choice_func = _choice
        if sparse:
            self.softmax_w = torch.nn.Embedding(num_embeddings=num_words, embedding_dim=embedding_dim, sparse=True)
            self.softmax_w.weight.data.normal_(mean=0.0, std=1.0 / np.sqrt(embedding_dim))
            self.softmax_b = torch.nn.Embedding(num_embeddings=num_words, embedding_dim=1, sparse=True)
            self.softmax_b.weight.data.fill_(0.0)
        else:
            self.softmax_w = torch.nn.Parameter(torch.randn(num_words, embedding_dim) / np.sqrt(embedding_dim))
            self.softmax_b = torch.nn.Parameter(torch.zeros(num_words))
        self.sparse = sparse
        self.use_character_inputs = use_character_inputs
        if use_character_inputs:
            self._unk_id = unk_id
        else:
            self._unk_id = None
        self._num_samples = num_samples
        self._embedding_dim = embedding_dim
        self._num_words = num_words
        self.initialize_num_words()

    def initialize_num_words(self) -> None:
        if self.sparse:
            num_words = self.softmax_w.weight.size(0)  # type: ignore[union-attr]
        else:
            num_words = self.softmax_w.size(0)  # type: ignore[union-attr]
        self._num_words = int(num_words)
        self._log_num_words_p1 = float(np.log(num_words + 1))
        self._probs = (np.log(np.arange(num_words) + 2) - np.log(np.arange(num_words) + 1)) / self._log_num_words_p1

    def forward(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        target_token_embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if embeddings.shape[0] == 0:
            return torch.tensor(0.0, device=embeddings.device)
        if not self.training:
            return self._forward_eval(embeddings, targets)
        else:
            return self._forward_train(embeddings, targets, target_token_embedding)

    def _forward_train(
        self,
        embeddings: torch.Tensor,
        targets: torch.Tensor,
        target_token_embedding: Optional[torch.Tensor]
    ) -> torch.Tensor:
        sampled_ids, target_expected_count, sampled_expected_count = self.log_uniform_candidate_sampler(
            targets, choice_func=self.choice_func
        )
        long_targets = targets.long()
        long_targets.requires_grad_(False)
        all_ids = torch.cat([long_targets, sampled_ids], dim=0)
        if self.sparse:
            all_ids_1 = all_ids.unsqueeze(1)
            all_w = self.softmax_w(all_ids_1).squeeze(1)  # type: ignore[operator]
            all_b = self.softmax_b(all_ids_1).squeeze(2).squeeze(1)  # type: ignore[operator]
        else:
            all_w = torch.nn.functional.embedding(all_ids, self.softmax_w)  # type: ignore[arg-type]
            all_b = torch.nn.functional.embedding(all_ids, self.softmax_b.unsqueeze(1)).squeeze(1)  # type: ignore[arg-type, union-attr]
        batch_size = long_targets.size(0)
        true_w = all_w[:batch_size, :]
        sampled_w = all_w[batch_size:, :]
        true_b = all_b[:batch_size]
        sampled_b = all_b[batch_size:]
        true_logits = (true_w * embeddings).sum(dim=1) + true_b - torch.log(
            target_expected_count + util.tiny_value_of_dtype(target_expected_count.dtype)
        )
        sampled_logits = torch.matmul(embeddings, sampled_w.t()) + sampled_b - torch.log(
            sampled_expected_count + util.tiny_value_of_dtype(sampled_expected_count.dtype)
        )
        true_in_sample_mask = sampled_ids == long_targets.unsqueeze(1)
        masked_sampled_logits = sampled_logits.masked_fill(true_in_sample_mask, -10000.0)
        logits = torch.cat([true_logits.unsqueeze(1), masked_sampled_logits], dim=1)
        log_softmax = torch.nn.functional.log_softmax(logits, dim=1)
        nll_loss = -1.0 * log_softmax[:, 0].sum()
        return nll_loss

    def _forward_eval(self, embeddings: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.sparse:
            w = self.softmax_w.weight  # type: ignore[union-attr]
            b = self.softmax_b.weight.squeeze(1)  # type: ignore[union-attr]
        else:
            w = self.softmax_w  # type: ignore[assignment]
            b = self.softmax_b  # type: ignore[assignment]
        log_softmax = torch.nn.functional.log_softmax(torch.matmul(embeddings, w.t()) + b, dim=-1)
        if self.tie_embeddings and (not self.use_character_inputs):
            targets_ = targets + 1
        else:
            targets_ = targets
        return torch.nn.functional.nll_loss(log_softmax, targets_.long(), reduction='sum')

    def log_uniform_candidate_sampler(
        self,
        targets: torch.Tensor,
        choice_func: Callable[[int, int], Tuple[np.ndarray, int]] = _choice
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        np_sampled_ids, num_tries = choice_func(self._num_words, self._num_samples)
        sampled_ids = torch.from_numpy(np_sampled_ids).to(targets.device)
        target_probs = torch.log((targets.float() + 2.0) / (targets.float() + 1.0)) / self._log_num_words_p1
        target_expected_count = -1.0 * (torch.exp(num_tries * torch.log1p(-target_probs)) - 1.0)
        sampled_probs = torch.log((sampled_ids.float() + 2.0) / (sampled_ids.float() + 1.0)) / self._log_num_words_p1
        sampled_expected_count = -1.0 * (torch.exp(num_tries * torch.log1p(-sampled_probs)) - 1.0)
        sampled_ids.requires_grad_(False)
        target_expected_count.requires_grad_((False))
        sampled_expected_count.requires_grad_(False)
        return (sampled_ids, target_expected_count, sampled_expected_count)