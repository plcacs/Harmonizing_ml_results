import torch
from typing import Union, Optional, List, Tuple, Any
from os import PathLike
from allennlp.fairness.bias_direction import BiasDirection, PCABiasDirection, PairedPCABiasDirection, TwoMeansBiasDirection, ClassificationNormalBiasDirection
from allennlp.fairness.bias_utils import load_word_pairs, load_words
from allennlp.common import Registrable
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data import Vocabulary
from torch import nn

class BiasDirectionWrapper(Registrable):
    """
    Parent class for bias direction wrappers.
    """
    direction: Optional[torch.Tensor]
    noise: Optional[float]

    def __init__(self) -> None:
        self.direction = None
        self.noise = None

    def __call__(self, module: nn.Module) -> torch.Tensor:
        raise NotImplementedError

    def train(self, mode: bool = True) -> None:
        """

        # Parameters

        mode : `bool`, optional (default=`True`)
            Sets `requires_grad` to value of `mode` for bias direction.
        """
        if self.direction is not None:
            self.direction.requires_grad = mode

    def add_noise(self, t: torch.Tensor) -> torch.Tensor:
        """

        # Parameters

        t : `torch.Tensor`
            Tensor to which to add small amount of Gaussian noise.
        """
        if self.noise is None:
            return t
        return t + self.noise * torch.randn(t.size(), device=t.device)

@BiasDirectionWrapper.register('pca')
class PCABiasDirectionWrapper(BiasDirectionWrapper):
    """

    # Parameters

    seed_words_file : `Union[PathLike, str]`
        Path of file containing seed words.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize seed words.
    direction_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of direction_vocab to use when tokenizing.
        Disregarded when direction_vocab is `None`.
    requires_grad : `bool`, optional (default=`False`)
        Option to enable gradient calculation for bias direction.
    noise : `float`, optional (default=`1e-10`)
        To avoid numerical instability if embeddings are initialized uniformly.
    """
    ids: List[torch.Tensor]

    def __init__(
        self,
        seed_words_file: Union[PathLike, str],
        tokenizer: Tokenizer,
        direction_vocab: Optional[Vocabulary] = None,
        namespace: str = 'tokens',
        requires_grad: bool = False,
        noise: float = 1e-10
    ) -> None:
        super().__init__()
        self.ids = load_words(seed_words_file, tokenizer, direction_vocab, namespace)
        self.direction = PCABiasDirection(requires_grad=requires_grad)
        self.noise = noise

    def __call__(self, module: nn.Module) -> torch.Tensor:
        ids_embeddings: List[torch.Tensor] = []
        for i in self.ids:
            i = i.to(module.weight.device)
            ids_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids_embeddings_tensor = torch.cat(ids_embeddings)
        ids_embeddings_noised = self.add_noise(ids_embeddings_tensor)
        return self.direction(ids_embeddings_noised)

@BiasDirectionWrapper.register('paired_pca')
class PairedPCABiasDirectionWrapper(BiasDirectionWrapper):
    """

    # Parameters

    seed_word_pairs_file : `Union[PathLike, str]`
        Path of file containing seed word pairs.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize seed words.
    direction_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of direction_vocab to use when tokenizing.
        Disregarded when direction_vocab is `None`.
    requires_grad : `bool`, optional (default=`False`)
        Option to enable gradient calculation for bias direction.
    noise : `float`, optional (default=`1e-10`)
        To avoid numerical instability if embeddings are initialized uniformly.
    """
    ids1: List[torch.Tensor]
    ids2: List[torch.Tensor]

    def __init__(
        self,
        seed_word_pairs_file: Union[PathLike, str],
        tokenizer: Tokenizer,
        direction_vocab: Optional[Vocabulary] = None,
        namespace: str = 'tokens',
        requires_grad: bool = False,
        noise: float = 1e-10
    ) -> None:
        super().__init__()
        self.ids1, self.ids2 = load_word_pairs(seed_word_pairs_file, tokenizer, direction_vocab, namespace)
        self.direction = PairedPCABiasDirection(requires_grad=requires_grad)
        self.noise = noise

    def __call__(self, module: nn.Module) -> torch.Tensor:
        ids1_embeddings: List[torch.Tensor] = []
        for i in self.ids1:
            i = i.to(module.weight.device)
            ids1_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids2_embeddings: List[torch.Tensor] = []
        for i in self.ids2:
            i = i.to(module.weight.device)
            ids2_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids1_embeddings_tensor = torch.cat(ids1_embeddings)
        ids2_embeddings_tensor = torch.cat(ids2_embeddings)
        ids1_embeddings_noised = self.add_noise(ids1_embeddings_tensor)
        ids2_embeddings_noised = self.add_noise(ids2_embeddings_tensor)
        return self.direction(ids1_embeddings_noised, ids2_embeddings_noised)

@BiasDirectionWrapper.register('two_means')
class TwoMeansBiasDirectionWrapper(BiasDirectionWrapper):
    """

    # Parameters

    seed_word_pairs_file : `Union[PathLike, str]`
        Path of file containing seed word pairs.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize seed words.
    direction_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of direction_vocab to use when tokenizing.
        Disregarded when direction_vocab is `None`.
    requires_grad : `bool`, optional (default=`False`)
        Option to enable gradient calculation for bias direction.
    noise : `float`, optional (default=`1e-10`)
        To avoid numerical instability if embeddings are initialized uniformly.
    """
    ids1: List[torch.Tensor]
    ids2: List[torch.Tensor]

    def __init__(
        self,
        seed_word_pairs_file: Union[PathLike, str],
        tokenizer: Tokenizer,
        direction_vocab: Optional[Vocabulary] = None,
        namespace: str = 'tokens',
        requires_grad: bool = False,
        noise: float = 1e-10
    ) -> None:
        super().__init__()
        self.ids1, self.ids2 = load_word_pairs(seed_word_pairs_file, tokenizer, direction_vocab, namespace)
        self.direction = TwoMeansBiasDirection(requires_grad=requires_grad)
        self.noise = noise

    def __call__(self, module: nn.Module) -> torch.Tensor:
        ids1_embeddings: List[torch.Tensor] = []
        for i in self.ids1:
            i = i.to(module.weight.device)
            ids1_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids2_embeddings: List[torch.Tensor] = []
        for i in self.ids2:
            i = i.to(module.weight.device)
            ids2_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids1_embeddings_tensor = torch.cat(ids1_embeddings)
        ids2_embeddings_tensor = torch.cat(ids2_embeddings)
        ids1_embeddings_noised = self.add_noise(ids1_embeddings_tensor)
        ids2_embeddings_noised = self.add_noise(ids2_embeddings_tensor)
        return self.direction(ids1_embeddings_noised, ids2_embeddings_noised)

@BiasDirectionWrapper.register('classification_normal')
class ClassificationNormalBiasDirectionWrapper(BiasDirectionWrapper):
    """

    # Parameters

    seed_word_pairs_file : `Union[PathLike, str]`
        Path of file containing seed word pairs.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize seed words.
    direction_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of direction_vocab to use when tokenizing.
        Disregarded when direction_vocab is `None`.
    noise : `float`, optional (default=`1e-10`)
        To avoid numerical instability if embeddings are initialized uniformly.
    """
    ids1: List[torch.Tensor]
    ids2: List[torch.Tensor]

    def __init__(
        self,
        seed_word_pairs_file: Union[PathLike, str],
        tokenizer: Tokenizer,
        direction_vocab: Optional[Vocabulary] = None,
        namespace: str = 'tokens',
        noise: float = 1e-10
    ) -> None:
        super().__init__()
        self.ids1, self.ids2 = load_word_pairs(seed_word_pairs_file, tokenizer, direction_vocab, namespace)
        self.direction = ClassificationNormalBiasDirection()
        self.noise = noise

    def __call__(self, module: nn.Module) -> torch.Tensor:
        ids1_embeddings: List[torch.Tensor] = []
        for i in self.ids1:
            i = i.to(module.weight.device)
            ids1_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids2_embeddings: List[torch.Tensor] = []
        for i in self.ids2:
            i = i.to(module.weight.device)
            ids2_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids1_embeddings_tensor = torch.cat(ids1_embeddings)
        ids2_embeddings_tensor = torch.cat(ids2_embeddings)
        ids1_embeddings_noised = self.add_noise(ids1_embeddings_tensor)
        ids2_embeddings_noised = self.add_noise(ids2_embeddings_tensor)
        return self.direction(ids1_embeddings_noised, ids2_embeddings_noised)
