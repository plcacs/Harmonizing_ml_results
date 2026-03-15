"""
A suite of metrics to quantify how much bias is encoded by word embeddings
and determine the effectiveness of bias mitigation.

Bias metrics are based on:

1. Caliskan, A., Bryson, J., & Narayanan, A. (2017). [Semantics derived automatically
from language corpora contain human-like biases](https://api.semanticscholar.org/CorpusID:23163324).
Science, 356, 183 - 186.

2. Dev, S., & Phillips, J.M. (2019). [Attenuating Bias in Word Vectors]
(https://api.semanticscholar.org/CorpusID:59158788). AISTATS.

3. Dev, S., Li, T., Phillips, J.M., & Srikumar, V. (2020). [On Measuring and Mitigating
Biased Inferences of Word Embeddings](https://api.semanticscholar.org/CorpusID:201670701).
ArXiv, abs/1908.09369.

4. Rathore, A., Dev, S., Phillips, J.M., Srikumar, V., Zheng, Y., Yeh, C.M., Wang, J., Zhang,
W., & Wang, B. (2021). [VERB: Visualizing and Interpreting Bias Mitigation Techniques for
Word Representations](https://api.semanticscholar.org/CorpusID:233168618).
ArXiv, abs/2104.02797.

5. Aka, O.; Burke, K.; Bäuerle, A.; Greer, C.; and Mitchell, M. 2021.
[Measuring model biases in the absence of ground truth](https://api.semanticscholar.org/CorpusID:232135043).
arXiv preprint arXiv:2103.03417.

"""
from typing import Optional, Dict, Union, List
import torch
import torch.distributed as dist
from allennlp.common.util import is_distributed
from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import dist_reduce_sum
from allennlp.training.metrics.metric import Metric

class WordEmbeddingAssociationTest:
    """
    Word Embedding Association Test (WEAT) score measures the unlikelihood there is no
    difference between two sets of target words in terms of their relative similarity
    to two sets of attribute words by computing the probability that a random
    permutation of attribute words would produce the observed (or greater) difference
    in sample means. Analog of Implicit Association Test from psychology for word embeddings.

    Based on: Caliskan, A., Bryson, J., & Narayanan, A. (2017). [Semantics derived automatically
    from language corpora contain human-like biases](https://api.semanticscholar.org/CorpusID:23163324).
    Science, 356, 183 - 186.
    """

    def __call__(self, target_embeddings1: torch.Tensor, target_embeddings2: torch.Tensor, attribute_embeddings1: torch.Tensor, attribute_embeddings2: torch.Tensor) -> torch.FloatTensor:
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        target_embeddings1 : `torch.Tensor`, required.
            A tensor of size (target_embeddings_batch_size, ..., dim) containing target word
            embeddings related to a concept group. For example, if the concept is gender,
            target_embeddings1 could contain embeddings for linguistically masculine words, e.g.
            "man", "king", "brother", etc. Represented as X.

        target_embeddings2 : `torch.Tensor`, required.
            A tensor of the same size as target_embeddings1 containing target word
            embeddings related to a different group for the same concept. For example,
            target_embeddings2 could contain embeddings for linguistically feminine words, e.g.
            "woman", "queen", "sister", etc. Represented as Y.

        attribute_embeddings1 : `torch.Tensor`, required.
            A tensor of size (attribute_embeddings1_batch_size, ..., dim) containing attribute word
            embeddings related to a concept group associated with the concept group for target_embeddings1.
            For example, if the concept is professions, attribute_embeddings1 could contain embeddings for
            stereotypically male professions, e.g. "doctor", "banker", "engineer", etc. Represented as A.

        attribute_embeddings2 : `torch.Tensor`, required.
            A tensor of size (attribute_embeddings2_batch_size, ..., dim) containing attribute word
            embeddings related to a concept group associated with the concept group for target_embeddings2.
            For example, if the concept is professions, attribute_embeddings2 could contain embeddings for
            stereotypically female professions, e.g. "nurse", "receptionist", "homemaker", etc. Represented as B.

        !!! Note
            While target_embeddings1 and target_embeddings2 must be the same size, attribute_embeddings1 and
            attribute_embeddings2 need not be the same size.

        # Returns

        weat_score : `torch.FloatTensor`
            The unlikelihood there is no difference between target_embeddings1 and target_embeddings2 in
            terms of their relative similarity to attribute_embeddings1 and attribute_embeddings2.
            Typical values are around [-1, 1], with values closer to 0 indicating less biased associations.

        """
        if target_embeddings1.ndim < 2 or target_embeddings2.ndim < 2:
            raise ConfigurationError('target_embeddings1 and target_embeddings2 must have at least two dimensions.')
        if attribute_embeddings1.ndim < 2 or attribute_embeddings2.ndim < 2:
            raise ConfigurationError('attribute_embeddings1 and attribute_embeddings2 must have at least two dimensions.')
        if target_embeddings1.size() != target_embeddings2.size():
            raise ConfigurationError('target_embeddings1 and target_embeddings2 must be of the same size.')
        if attribute_embeddings1.size(dim=-1) != attribute_embeddings2.size(dim=-1) or attribute_embeddings1.size(dim=-1) != target_embeddings1.size(dim=-1):
            raise ConfigurationError('All embeddings must have the same dimensionality.')
        target_embeddings1 = target_embeddings1.flatten(end_dim=-2)
        target_embeddings2 = target_embeddings2.flatten(end_dim=-2)
        attribute_embeddings1 = attribute_embeddings1.flatten(end_dim=-2)
        attribute_embeddings2 = attribute_embeddings2.flatten(end_dim=-2)
        target_embeddings1 = torch.nn.functional.normalize(target_embeddings1, p=2, dim=-1)
        target_embeddings2 = torch.nn.functional.normalize(target_embeddings2, p=2, dim=-1)
        attribute_embeddings1 = torch.nn.functional.normalize(attribute_embeddings1, p=2, dim=-1)
        attribute_embeddings2 = torch.nn.functional.normalize(attribute_embeddings2, p=2, dim=-1)
        X_sim_A = torch.mm(target_embeddings1, attribute_embeddings1.t())
        X_sim_B = torch.mm(target_embeddings1, attribute_embeddings2.t())
        Y_sim_A = torch.mm(target_embeddings2, attribute_embeddings1.t())
        Y_sim_B = torch.mm(target_embeddings2, attribute_embeddings2.t())
        X_union_Y_sim_A = torch.cat([X_sim_A, Y_sim_A])
        X_union_Y_sim_B = torch.cat([X_sim_B, Y_sim_B])
        s_X_A_B = torch.mean(X_sim_A, dim=-1) - torch.mean(X_sim_B, dim=-1)
        s_Y_A_B = torch.mean(Y_sim_A, dim=-1) - torch.mean(Y_sim_B, dim=-1)
        s_X_Y_A_B = torch.mean(s_X_A_B) - torch.mean(s_Y_A_B)
        S_X_union_Y_A_B = torch.mean(X_union_Y_sim_A, dim=-1) - torch.mean(X_union_Y_sim_B, dim=-1)
        return s_X_Y_A_B / torch.std(S_X_union_Y_A_B, unbiased=False)

class EmbeddingCoherenceTest:
    """
    Embedding Coherence Test (ECT) score measures if groups of words
    have stereotypical associations by computing the Spearman Coefficient
    of lists of attribute embeddings sorted based on their similarity to
    target embeddings.

    Based on: Dev, S., & Phillips, J.M. (2019). [Attenuating Bias in Word Vectors]
    (https://api.semanticscholar.org/CorpusID:59158788). AISTATS.
    """

    def __call__(self, target_embeddings1: torch.Tensor, target_embeddings2: torch.Tensor, attribute_embeddings: torch.Tensor) -> torch.FloatTensor:
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        target_embeddings1 : `torch.Tensor`, required.
            A tensor of size (target_embeddings_batch_size, ..., dim) containing target word
            embeddings related to a concept group. For example, if the concept is gender,
            target_embeddings1 could contain embeddings for linguistically masculine words, e.g.
            "man", "king", "brother", etc. Represented as X.

        target_embeddings2 : `torch.Tensor`, required.
            A tensor of the same size as target_embeddings1 containing target word
            embeddings related to a different group for the same concept. For example,
            target_embeddings2 could contain embeddings for linguistically feminine words, e.g.
            "woman", "queen", "sister", etc. Represented as Y.

        attribute_embeddings : `torch.Tensor`, required.
            A tensor of size (attribute_embeddings_batch_size, ..., dim) containing attribute word
            embeddings related to a concept associated with target_embeddings1 and target_embeddings2.
            For example, if the concept is professions, attribute_embeddings could contain embeddings for
            "doctor", "banker", "engineer", etc. Represented as AB.

        # Returns

        ect_score : `torch.FloatTensor`
            The Spearman Coefficient measuring the similarity of lists of attribute embeddings sorted
            based on their similarity to the target embeddings. Ranges from [-1, 1], with values closer
            to 1 indicating less biased associations.

        """
        if target_embeddings1.ndim < 2 or target_embeddings2.ndim < 2:
            raise ConfigurationError('target_embeddings1 and target_embeddings2 must have at least two dimensions.')
        if attribute_embeddings.ndim < 2:
            raise ConfigurationError('attribute_embeddings must have at least two dimensions.')
        if target_embeddings1.size() != target_embeddings2.size():
            raise ConfigurationError('target_embeddings1 and target_embeddings2 must be of the same size.')
        if attribute_embeddings.size(dim=-1) != target_embeddings1.size(dim=-1):
            raise ConfigurationError('All embeddings must have the same dimensionality.')
        mean_target_embedding1 = target_embeddings1.flatten(end_dim=-2).mean(dim=0)
        mean_target_embedding2 = target_embeddings2.flatten(end_dim=-2).mean(dim=0)
        attribute_embeddings = attribute_embeddings.flatten(end_dim=-2)
        mean_target_embedding1 = torch.nn.functional.normalize(mean_target_embedding1, p=2, dim=-1)
        mean_target_embedding2 = torch.nn.functional.normalize(mean_target_embedding2, p=2, dim=-1)
        attribute_embeddings = torch.nn.functional.normalize(attribute_embeddings, p=2, dim=-1)
        AB_sim_m = torch.matmul(attribute_embeddings, mean_target_embedding1)
        AB_sim_f = torch.matmul(attribute_embeddings, mean_target_embedding2)
        return self.spearman_correlation(AB_sim_m, AB_sim_f)

    def _get_ranks(self, x: torch.Tensor) -> torch.Tensor:
        tmp = x.argsort()
        ranks = torch.zeros_like(tmp)
        ranks[tmp] = torch.arange(x.size(0), device=ranks.device)
        return ranks

    def spearman_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:
        x_rank = self._get_ranks(x)
        y_rank = self._get_ranks(y)
        n = x.size(0)
        upper = 6 * torch.sum((x_rank - y_rank).pow(2))
        down = n * (n ** 2 - 1.0)
        return 1.0 - upper / down

@Metric.register('nli')
class NaturalLanguageInference(Metric):
    """
    Natural language inference scores measure the effect biased associations have on decisions
    made downstream, given neutrally-constructed pairs of sentences differing only in the subject.

    1. Net Neutral (NN): The average probability of the neutral label
    across all sentence pairs.

    2. Fraction Neutral (FN): The fraction of sentence pairs predicted neutral.

    3. Threshold:tau (T:tau): A parameterized measure that reports the fraction
    of examples whose probability of neutral is above tau.

    # Parameters

    neutral_label : `int`, optional (default=`2`)
        The discrete integer label corresponding to a neutral entailment prediction.
    taus : `List[float]`, optional (default=`[0.5, 0.7]`)
        All the taus for which to compute Threshold:tau.

    Based on: Dev, S., Li, T., Phillips, J.M., & Srikumar, V. (2020). [On Measuring and Mitigating
    Biased Inferences of Word Embeddings](https://api.semanticscholar.org/CorpusID:201670701).
    ArXiv, abs/1908.09369.
    """

    def __init__(self, neutral_label: int = 2, taus: List[float] = [0.5, 0.7]):
        self.neutral_label = neutral_label
        self.taus = taus
        self._nli_probs_sum = 0.0
        self._num_neutral_predictions = 0.0
        self._num_neutral_above_taus = {tau: 0.0 for tau in taus}
        self._total_predictions = 0

    def __call__(self, nli_probabilities: torch.Tensor) -> None:
        """

        # Parameters

        !!! Note
            In the examples below, we treat gender identity as binary, which does not accurately
            characterize gender in real life.

        nli_probabilities : `torch.Tensor`, required.
            A tensor of size (batch_size, ..., 3) containing natural language inference
            (i.e. entailment, contradiction, and neutral) probabilities for neutrally-constructed
            pairs of sentences differing only in the subject. For example, if the concept is gender,
            nli_probabilities could contain the natural language inference probabilities of:

            - "The driver owns a cabinet." -> "The man owns a cabinet."

            - "The driver owns a cabinet." -> "The woman owns a cabinet."

            - "The doctor eats an apple." -> "The man eats an apple."

            - "The doctor eats an apple." -> "The woman eats an apple."
        """
        nli_probabilities = nli_probabilities.detach()
        if nli_probabilities.dim() < 2:
            raise ConfigurationError('nli_probabilities must have at least two dimensions but found tensor of shape: {}'.format(nli_probabilities.size()))
        if nli_probabilities.size(-1) != 3:
            raise ConfigurationError('Last dimension of nli_probabilities must have dimensionality of 3 but found tensor of shape: {}'.format(nli_probabilities.size()))
        _nli_neutral_probs = nli_probabilities[..., self.neutral_label]
        self._nli_probs_sum += dist_reduce_sum(_nli_neutral_probs.sum().item())
        self._num_neutral_predictions += dist_reduce_sum((nli_probabilities.argmax(dim=-1) == self.neutral_label).float().sum().item())
        for tau in self.taus:
            self._num_neutral_above_taus[tau] += dist_reduce_sum((_nli_neutral_probs > tau).float().sum().item())
        self._total_predictions += dist_reduce_sum(_nli_neutral_probs.numel())

    def get_metric(self, reset: bool = False) -> Dict[str, float]:
        """
        # Returns

        nli_scores : `Dict[str, float]`
            Contains the following keys:

            1. "`net_neutral`" : The average probability of the neutral label across
            all sentence pairs. A value closer to 1 suggests lower bias, as bias will result in a higher
            probability of entailment or contradiction.

            2. "`fraction_neutral`" : The fraction of sentence pairs predicted neutral.
            A value closer to 1 suggests lower bias, as bias will result in a higher
            probability of entailment or contradiction.

            3. "`threshold_{taus}`" : For each tau, the fraction of examples whose probability of
            neutral is above tau. For each tau, a value closer to 1 suggests lower bias, as bias
            will result in a higher probability of entailment or contradiction.

        """
        if self._total_predictions == 0:
            nli_s