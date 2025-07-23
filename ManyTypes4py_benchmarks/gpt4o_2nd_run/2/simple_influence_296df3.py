import logging
from typing import List, Optional, Tuple, Union, Sequence
import numpy as np
import torch
import torch.autograd as autograd
from allennlp.common import Lazy
from allennlp.common.tqdm import Tqdm
from allennlp.data import DatasetReader, DatasetReaderInput, Instance
from allennlp.data.data_loaders import DataLoader, SimpleDataLoader
from allennlp.interpret.influence_interpreters.influence_interpreter import InfluenceInterpreter
from allennlp.models.model import Model

logger = logging.getLogger(__name__)

@InfluenceInterpreter.register('simple-influence')
class SimpleInfluence(InfluenceInterpreter):
    def __init__(
        self,
        model: Model,
        train_data_path: str,
        train_dataset_reader: DatasetReader,
        *,
        test_dataset_reader: Optional[DatasetReader] = None,
        train_data_loader: Lazy[SimpleDataLoader] = Lazy(SimpleDataLoader.from_dataset_reader),
        test_data_loader: Lazy[SimpleDataLoader] = Lazy(SimpleDataLoader.from_dataset_reader),
        params_to_freeze: Optional[List[str]] = None,
        cuda_device: int = -1,
        lissa_batch_size: int = 8,
        damping: float = 0.003,
        num_samples: int = 1,
        recursion_depth: Union[float, int] = 0.25,
        scale: float = 10000.0
    ) -> None:
        super().__init__(
            model=model,
            train_data_path=train_data_path,
            train_dataset_reader=train_dataset_reader,
            test_dataset_reader=test_dataset_reader,
            train_data_loader=train_data_loader,
            test_data_loader=test_data_loader,
            params_to_freeze=params_to_freeze,
            cuda_device=cuda_device
        )
        self._lissa_dataloader = SimpleDataLoader(
            list(self._train_loader.iter_instances()),
            lissa_batch_size,
            shuffle=True,
            vocab=self.vocab
        )
        self._lissa_dataloader.set_target_device(self.device)
        if isinstance(recursion_depth, float) and recursion_depth > 0.0:
            self._lissa_dataloader.batches_per_epoch = int(len(self._lissa_dataloader) * recursion_depth)
        elif isinstance(recursion_depth, int) and recursion_depth > 0:
            self._lissa_dataloader.batches_per_epoch = recursion_depth
        else:
            raise ValueError("'recursion_depth' should be a positive int or float")
        self._damping = damping
        self._num_samples = num_samples
        self._recursion_depth = recursion_depth
        self._scale = scale

    def _calculate_influence_scores(
        self,
        test_instance: Instance,
        test_loss: torch.Tensor,
        test_grads: Sequence[torch.Tensor]
    ) -> List[float]:
        inv_hvp = get_inverse_hvp_lissa(
            test_grads,
            self.model,
            self.used_params,
            self._lissa_dataloader,
            self._damping,
            self._num_samples,
            self._scale
        )
        return [
            torch.dot(inv_hvp, _flatten_tensors(x.grads)).item()
            for x in Tqdm.tqdm(self.train_instances, desc='scoring train instances')
        ]

def get_inverse_hvp_lissa(
    vs: Sequence[torch.Tensor],
    model: Model,
    used_params: Sequence[torch.Tensor],
    lissa_data_loader: DataLoader,
    damping: float,
    num_samples: int,
    scale: float
) -> torch.Tensor:
    inverse_hvps = [torch.tensor(0) for _ in vs]
    for _ in Tqdm.tqdm(range(num_samples), desc='LiSSA samples', total=num_samples):
        cur_estimates = vs
        recursion_iter = Tqdm.tqdm(lissa_data_loader, desc='LiSSA depth', total=len(lissa_data_loader))
        for j, training_batch in enumerate(recursion_iter):
            model.zero_grad()
            train_output_dict = model(**training_batch)
            hvps = get_hvp(train_output_dict['loss'], used_params, cur_estimates)
            cur_estimates = [
                v + (1 - damping) * cur_estimate - hvp / scale
                for v, cur_estimate, hvp in zip(vs, cur_estimates, hvps)
            ]
            if j % 50 == 0 or j == len(lissa_data_loader) - 1:
                norm = np.linalg.norm(_flatten_tensors(cur_estimates).cpu().numpy())
                recursion_iter.set_description(desc=f'calculating inverse HVP, norm = {norm:.5f}')
        inverse_hvps = [
            inverse_hvp + cur_estimate / scale
            for inverse_hvp, cur_estimate in zip(inverse_hvps, cur_estimates)
        ]
    return_ihvp = _flatten_tensors(inverse_hvps)
    return_ihvp /= num_samples
    return return_ihvp

def get_hvp(
    loss: torch.Tensor,
    params: Sequence[torch.Tensor],
    vectors: Sequence[torch.Tensor]
) -> Tuple[torch.Tensor, ...]:
    assert len(params) == len(vectors)
    assert all((p.size() == v.size() for p, v in zip(params, vectors)))
    grads = autograd.grad(loss, params, create_graph=True, retain_graph=True)
    hvp = autograd.grad(grads, params, grad_outputs=vectors)
    return hvp

def _flatten_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    views = []
    for p in tensors:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)
