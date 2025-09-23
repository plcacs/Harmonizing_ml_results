import torch
from typing import Union, Optional, List, Tuple
from os import PathLike
from allennlp.fairness.bias_mitigators import HardBiasMitigator, LinearBiasMitigator, INLPBiasMitigator, OSCaRBiasMitigator
from allennlp.fairness.bias_direction_wrappers import BiasDirectionWrapper
from allennlp.fairness.bias_utils import load_word_pairs
from allennlp.common import Registrable
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data import Vocabulary

class BiasMitigatorWrapper(Registrable):
    """
    Parent class for bias mitigator wrappers.
    """

    def train(self, mode: bool = True) -> None:
        """

        # Parameters

        mode : `bool`, optional (default=`True`)
            Sets `requires_grad` to value of `mode` for bias mitigator
            and associated bias direction.
        """
        raise NotImplementedError

@BiasMitigatorWrapper.register('hard')
class HardBiasMitigatorWrapper(BiasMitigatorWrapper):
    """

    # Parameters

    bias_direction : `BiasDirectionWrapper`
        Bias direction used by mitigator.
    embedding_layer : `torch.nn.Embedding`
        Embedding layer of base model.
    equalize_word_pairs_file : `Union[PathLike, str]`
        Path of file containing equalize word pairs.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize equalize words.
    mitigator_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of mitigator_vocab to use when tokenizing.
        Disregarded when mitigator_vocab is `None`.
    requires_grad : `bool`, optional (default=`True`)
        Option to enable gradient calculation for bias mitigator.
    """

    def __init__(
        self, 
        bias_direction: BiasDirectionWrapper, 
        embedding_layer: torch.nn.Embedding, 
        equalize_word_pairs_file: Union[PathLike, str], 
        tokenizer: Tokenizer, 
        mitigator_vocab: Optional[Vocabulary] = None, 
        namespace: str = 'tokens', 
        requires_grad: bool = True
    ) -> None:
        self.bias_direction: BiasDirectionWrapper = bias_direction
        self.predetermined_bias_direction: torch.Tensor = self.bias_direction(embedding_layer)
        self.ids1: List[torch.Tensor]
        self.ids2: List[torch.Tensor]
        (self.ids1, self.ids2) = load_word_pairs(equalize_word_pairs_file, tokenizer, mitigator_vocab, namespace)
        self.mitigator: HardBiasMitigator = HardBiasMitigator(requires_grad=requires_grad)

    def __call__(
        self, 
        module: torch.nn.Module, 
        module_in: Tuple[torch.Tensor, ...], 
        module_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Called as forward hook.
        """
        ids1_embeddings: List[torch.Tensor] = []
        for i in self.ids1:
            i = i.to(module.weight.device)
            ids1_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids2_embeddings: List[torch.Tensor] = []
        for i in self.ids2:
            i = i.to(module.weight.device)
            ids2_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids1_embeddings_tensor: torch.Tensor = torch.cat(ids1_embeddings)
        ids2_embeddings_tensor: torch.Tensor = torch.cat(ids2_embeddings)
        module_out_size: torch.Size = module_out.size()
        module_out_flat: torch.Tensor = module_out.flatten(end_dim=-2)
        module_out_mitigated: torch.Tensor = self.mitigator(
            module_out_flat, 
            self.predetermined_bias_direction.to(module_out_flat.device), 
            ids1_embeddings_tensor.to(module_out_flat.device), 
            ids2_embeddings_tensor.to(module_out_flat.device)
        )[:module_out_flat.size(0)]
        return module_out_mitigated.reshape(module_out_size)

    def train(self, mode: bool = True) -> None:
        self.mitigator.requires_grad = mode
        self.bias_direction.train(mode)

@BiasMitigatorWrapper.register('linear')
class LinearBiasMitigatorWrapper(BiasMitigatorWrapper):
    """

    # Parameters

    bias_direction : `BiasDirectionWrapper`
        Bias direction used by mitigator.
    embedding_layer : `torch.nn.Embedding`
        Embedding layer of base model.
    requires_grad : `bool`, optional (default=`True`)
        Option to enable gradient calculation for bias mitigator.
    """

    def __init__(
        self, 
        bias_direction: BiasDirectionWrapper, 
        embedding_layer: torch.nn.Embedding, 
        requires_grad: bool = True
    ) -> None:
        self.bias_direction: BiasDirectionWrapper = bias_direction
        self.predetermined_bias_direction: torch.Tensor = self.bias_direction(embedding_layer)
        self.mitigator: LinearBiasMitigator = LinearBiasMitigator(requires_grad=requires_grad)

    def __call__(
        self, 
        module: torch.nn.Module, 
        module_in: Tuple[torch.Tensor, ...], 
        module_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Called as forward hook.
        """
        module_out_size: torch.Size = module_out.size()
        module_out_flat: torch.Tensor = module_out.flatten(end_dim=-2)
        module_out_mitigated: torch.Tensor = self.mitigator(
            module_out_flat, 
            self.predetermined_bias_direction.to(module_out_flat.device)
        )
        return module_out_mitigated.reshape(module_out_size)

    def train(self, mode: bool = True) -> None:
        self.mitigator.requires_grad = mode
        self.bias_direction.train(mode)

@BiasMitigatorWrapper.register('inlp')
class INLPBiasMitigatorWrapper(BiasMitigatorWrapper):
    """

    # Parameters

    embedding_layer : `torch.nn.Embedding`
        Embedding layer of base model.
    seed_word_pairs_file : `Union[PathLike, str]`
        Path of file containing seed word pairs.
    tokenizer : `Tokenizer`
        Tokenizer used to tokenize seed words.
    mitigator_vocab : `Vocabulary`, optional (default=`None`)
        Vocabulary of tokenizer. If `None`, assumes tokenizer is of
        type `PreTrainedTokenizer` and uses tokenizer's `vocab` attribute.
    namespace : `str`, optional (default=`"tokens"`)
        Namespace of mitigator_vocab to use when tokenizing.
        Disregarded when mitigator_vocab is `None`.
    """

    def __init__(
        self, 
        embedding_layer: torch.nn.Embedding, 
        seed_word_pairs_file: Union[PathLike, str], 
        tokenizer: Tokenizer, 
        mitigator_vocab: Optional[Vocabulary] = None, 
        namespace: str = 'tokens'
    ) -> None:
        self.ids1: List[torch.Tensor]
        self.ids2: List[torch.Tensor]
        (self.ids1, self.ids2) = load_word_pairs(seed_word_pairs_file, tokenizer, mitigator_vocab, namespace)
        self.mitigator: INLPBiasMitigator = INLPBiasMitigator()

    def __call__(
        self, 
        module: torch.nn.Module, 
        module_in: Tuple[torch.Tensor, ...], 
        module_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Called as forward hook.
        """
        ids1_embeddings: List[torch.Tensor] = []
        for i in self.ids1:
            i = i.to(module.weight.device)
            ids1_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids2_embeddings: List[torch.Tensor] = []
        for i in self.ids2:
            i = i.to(module.weight.device)
            ids2_embeddings.append(torch.mean(module.forward(i), dim=0, keepdim=True))
        ids1_embeddings_tensor: torch.Tensor = torch.cat(ids1_embeddings)
        ids2_embeddings_tensor: torch.Tensor = torch.cat(ids2_embeddings)
        module_out_size: torch.Size = module_out.size()
        module_out_flat: torch.Tensor = module_out.flatten(end_dim=-2)
        module_out_mitigated: torch.Tensor = self.mitigator(
            module_out_flat, 
            ids1_embeddings_tensor.to(module_out_flat.device), 
            ids2_embeddings_tensor.to(module_out_flat.device)
        )
        return module_out_mitigated.reshape(module_out_size)

    def train(self, mode: bool = True) -> None:
        pass

@BiasMitigatorWrapper.register('oscar')
class OSCaRBiasMitigatorWrapper(BiasMitigatorWrapper):
    """

    # Parameters

    bias_direction1 : `BiasDirectionWrapper`
        Bias direction of first concept subspace used by mitigator.
    bias_direction2 : `BiasDirectionWrapper`
        Bias direction of second concept subspace used by mitigator.
    embedding_layer : `torch.nn.Embedding`
        Embedding layer of base model.
    requires_grad : `bool`, optional (default=`True`)
        Option to enable gradient calculation for bias mitigator.
    """

    def __init__(
        self, 
        bias_direction1: BiasDirectionWrapper, 
        bias_direction2: BiasDirectionWrapper, 
        embedding_layer: torch.nn.Embedding, 
        requires_grad: bool = True
    ) -> None:
        self.bias_direction1: BiasDirectionWrapper = bias_direction1
        self.predetermined_bias_direction1: torch.Tensor = self.bias_direction1(embedding_layer)
        self.bias_direction2: BiasDirectionWrapper = bias_direction2
        self.predetermined_bias_direction2: torch.Tensor = self.bias_direction2(embedding_layer)
        self.mitigator: OSCaRBiasMitigator = OSCaRBiasMitigator(requires_grad=requires_grad)

    def __call__(
        self, 
        module: torch.nn.Module, 
        module_in: Tuple[torch.Tensor, ...], 
        module_out: torch.Tensor
    ) -> torch.Tensor:
        """
        Called as forward hook.
        """
        module_out_size: torch.Size = module_out.size()
        module_out_flat: torch.Tensor = module_out.flatten(end_dim=-2)
        module_out_mitigated: torch.Tensor = self.mitigator(
            module_out_flat, 
            self.predetermined_bias_direction1.to(module_out_flat.device), 
            self.predetermined_bias_direction2.to(module_out_flat.device)
        )
        return module_out_mitigated.reshape(module_out_size)

    def train(self, mode: bool = True) -> None:
        self.mitigator.requires_grad = mode
        self.bias_direction1.train(mode)
        self.bias_direction2.train(mode)
