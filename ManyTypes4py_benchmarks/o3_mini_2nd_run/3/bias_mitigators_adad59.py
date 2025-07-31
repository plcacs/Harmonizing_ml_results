import torch
import numpy as np
import scipy
import sklearn
from allennlp.common.checks import ConfigurationError
from torch import Tensor
from typing import List

class BiasMitigator:
    """
    Parent class for bias mitigator classes.

    # Parameters

    requires_grad : `bool`, optional (default=`False`)
        Option to enable gradient calculation.
    """
    def __init__(self, requires_grad: bool = False) -> None:
        self.requires_grad: bool = requires_grad

    def _proj(self, u: Tensor, v: Tensor, normalize: bool = False) -> Tensor:
        proj: Tensor = torch.matmul(u, v.reshape(-1, 1)) * v
        if normalize:
            return proj / torch.dot(v, v)
        return proj

    def _remove_component(self, embeddings: Tensor, bias_direction: Tensor, normalize: bool = False) -> Tensor:
        return embeddings - self._proj(embeddings, bias_direction, normalize)

class HardBiasMitigator(BiasMitigator):
    """
    Hard bias mitigator. Mitigates bias in embeddings by:

    1. Neutralizing: ensuring protected variable-neutral words remain equidistant
    from the bias direction by removing component of embeddings
    in the bias direction.

    2. Equalizing: ensuring that protected variable-related words are averaged
    out to have the same norm.
    """
    def __call__(self, 
                 evaluation_embeddings: Tensor, 
                 bias_direction: Tensor, 
                 equalize_embeddings1: Tensor, 
                 equalize_embeddings2: Tensor) -> Tensor:
        if equalize_embeddings1.size() != equalize_embeddings2.size():
            raise ConfigurationError('equalize_embeddings1 and equalize_embeddings2 must be the same size.')
        if equalize_embeddings1.ndim < 2:
            raise ConfigurationError('equalize_embeddings1 and equalize_embeddings2 must have at least two dimensions.')
        if evaluation_embeddings.ndim < 2:
            raise ConfigurationError('evaluation_embeddings must have at least two dimensions.')
        if evaluation_embeddings.size()[1:] != equalize_embeddings1.size()[1:]:
            raise ConfigurationError('evaluation_embeddings, equalize_embeddings1, and equalize_embeddings2 must have same size                 except for 0th dim (i.e. batch dimension).')
        if bias_direction.ndim != 1:
            raise ConfigurationError('bias_direction must be one-dimensional.')
        if evaluation_embeddings.size(-1) != bias_direction.size(-1):
            raise ConfigurationError('All embeddings and bias_direction must have the same dimensionality.')
        with torch.set_grad_enabled(self.requires_grad):
            bias_direction = bias_direction / torch.linalg.norm(bias_direction)
            bias_mitigated_embeddings: Tensor = self._remove_component(evaluation_embeddings, bias_direction, normalize=True)
            mean_equalize_embeddings: Tensor = (equalize_embeddings1 + equalize_embeddings2) / 2
            y: Tensor = self._remove_component(mean_equalize_embeddings, bias_direction, normalize=True)
            z: Tensor = torch.sqrt(1 - torch.square(torch.linalg.norm(y, dim=-1, keepdim=True)))
            z = torch.where(torch.matmul(equalize_embeddings1 - equalize_embeddings2, bias_direction.reshape(-1, 1)) < 0, -z, z)
            return torch.cat([bias_mitigated_embeddings, z * bias_direction + y, -z * bias_direction + y])

class LinearBiasMitigator(BiasMitigator):
    """
    Linear bias mitigator. Mitigates bias in embeddings by removing component
    in the bias direction.
    """
    def __call__(self, 
                 evaluation_embeddings: Tensor, 
                 bias_direction: Tensor) -> Tensor:
        if evaluation_embeddings.ndim < 2:
            raise ConfigurationError('evaluation_embeddings must have at least two dimensions.')
        if bias_direction.ndim != 1:
            raise ConfigurationError('bias_direction must be one-dimensional.')
        if evaluation_embeddings.size(-1) != bias_direction.size(-1):
            raise ConfigurationError('All embeddings and bias_direction must have the same dimensionality.')
        with torch.set_grad_enabled(self.requires_grad):
            bias_direction = bias_direction / torch.linalg.norm(bias_direction)
            return self._remove_component(evaluation_embeddings, bias_direction)

class INLPBiasMitigator(BiasMitigator):
    """
    Iterative Nullspace Projection. It mitigates bias by repeatedly building
    a linear classifier that separates concept groups and linearly
    projecting all words along the classifier normal.
    """
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, 
                 evaluation_embeddings: Tensor, 
                 seed_embeddings1: Tensor, 
                 seed_embeddings2: Tensor, 
                 num_iters: int = 35) -> Tensor:
        if seed_embeddings1.ndim < 2 or seed_embeddings2.ndim < 2:
            raise ConfigurationError('seed_embeddings1 and seed_embeddings2 must have at least two dimensions.')
        if seed_embeddings1.size(-1) != seed_embeddings2.size(-1):
            raise ConfigurationError('All seed embeddings must have same dimensionality.')
        if evaluation_embeddings.ndim < 2:
            raise ConfigurationError('evaluation_embeddings must have at least two dimensions.')
        if evaluation_embeddings.size(-1) != seed_embeddings1.size(-1) or evaluation_embeddings.size(-1) != seed_embeddings2.size(-1):
            raise ConfigurationError('evaluation_embeddings, seed_embeddings1, and seed_embeddings2 must have the same dimensionality.')
        device = seed_embeddings1.device
        seed_embeddings1_np: np.ndarray = seed_embeddings1.flatten(end_dim=-2).detach().cpu().numpy()
        seed_embeddings2_np: np.ndarray = seed_embeddings2.flatten(end_dim=-2).detach().cpu().numpy()
        X: np.ndarray = np.vstack([seed_embeddings1_np, seed_embeddings2_np])
        Y: np.ndarray = np.concatenate([[0] * seed_embeddings1_np.shape[0], [1] * seed_embeddings2_np.shape[0]])
        rowspace_projs: List[np.ndarray] = []
        for iter_idx in range(num_iters):
            classifier = sklearn.svm.SVC(kernel='linear').fit(X, Y)
            weights: np.ndarray = np.expand_dims(classifier.coef_[0], 0)
            if (np.linalg.norm(weights) < 1e-10 or classifier.score(X, Y) < 0.55) and iter_idx > 1:
                break
            rowspace_projs.append(self._get_rowspace_proj(weights))
            sum_proj: np.ndarray = np.sum(rowspace_projs, axis=0)
            nullspace_proj: np.ndarray = np.eye(seed_embeddings1.shape[1]) - self._get_rowspace_proj(sum_proj)
            evaluation_embeddings = torch.matmul(evaluation_embeddings, torch.from_numpy(nullspace_proj).float().t().to(device))
            X = nullspace_proj.dot(X.T).T
        return evaluation_embeddings

    def _get_rowspace_proj(self, weights: np.ndarray) -> np.ndarray:
        if np.allclose(weights, 0):
            weights_basis: np.ndarray = np.zeros_like(weights.T)
        else:
            weights_basis = scipy.linalg.orth(weights.T)
        return weights_basis.dot(weights_basis.T)

class OSCaRBiasMitigator(BiasMitigator):
    """
    OSCaR bias mitigator. Mitigates bias in embeddings by dissociating concept subspaces
    through subspace orthogonalization.
    """
    def __call__(self, 
                 evaluation_embeddings: Tensor, 
                 bias_direction1: Tensor, 
                 bias_direction2: Tensor) -> Tensor:
        if evaluation_embeddings.ndim < 2:
            raise ConfigurationError('evaluation_embeddings must have at least two dimensions.')
        if bias_direction1.ndim != 1 or bias_direction2.ndim != 1:
            raise ConfigurationError('bias_direction1 and bias_direction2 must be one-dimensional.')
        if evaluation_embeddings.size(-1) != bias_direction1.size(-1) or evaluation_embeddings.size(-1) != bias_direction2.size(-1):
            raise ConfigurationError('All embeddings, bias_direction1, and bias_direction2 must have the same dimensionality.')
        if bias_direction1.size(-1) < 2:
            raise ConfigurationError('Dimensionality of all embeddings, bias_direction1, and bias_direction2 must                 be >= 2.')
        with torch.set_grad_enabled(self.requires_grad):
            bias_direction1 = bias_direction1 / torch.linalg.norm(bias_direction1)
            bias_direction2 = bias_direction2 / torch.linalg.norm(bias_direction2)
            bias_direction2_orth: Tensor = self._remove_component(bias_direction2.reshape(1, -1), bias_direction1)[0]
            bias_direction2_orth = bias_direction2_orth / torch.linalg.norm(bias_direction2_orth)
            init_orth_matrix: Tensor = torch.eye(bias_direction1.size(0), device=evaluation_embeddings.device, requires_grad=self.requires_grad)
            rotation_matrix: Tensor = torch.zeros((bias_direction1.size(0), bias_direction1.size(0)), device=evaluation_embeddings.device, requires_grad=self.requires_grad)
            rotation_matrix = torch.cat([bias_direction1.reshape(1, -1), bias_direction2_orth.reshape(1, -1), rotation_matrix[2:]])
            for i in range(len(rotation_matrix) - 2):
                subspace_proj: Tensor = torch.sum(self._proj(rotation_matrix[:i + 2].clone(), init_orth_matrix[i], normalize=True), dim=0)
                rotation_matrix[i + 2] = (init_orth_matrix[i] - subspace_proj) / torch.linalg.norm(init_orth_matrix[i] - subspace_proj)
            mask: Tensor = ~(evaluation_embeddings == 0).all(dim=-1)
            rotated_evaluation_embeddings: Tensor = torch.matmul(evaluation_embeddings[mask], rotation_matrix.t())
            fixed_rotated_evaluation_embeddings: Tensor = rotated_evaluation_embeddings[..., 2:]
            restricted_rotated_evaluation_embeddings: Tensor = torch.cat([
                torch.matmul(rotated_evaluation_embeddings, bias_direction1.reshape(-1, 1)),
                torch.matmul(rotated_evaluation_embeddings, bias_direction2_orth.reshape(-1, 1))
            ], dim=-1)
            restricted_bias_direction1: Tensor = torch.tensor([1.0, 0.0], device=evaluation_embeddings.device, requires_grad=self.requires_grad)
            bias_direction_inner_prod: Tensor = torch.dot(bias_direction1, bias_direction2)
            restricted_bias_direction2: Tensor = torch.tensor([bias_direction_inner_prod, torch.sqrt(1 - torch.square(bias_direction_inner_prod))], device=evaluation_embeddings.device, requires_grad=self.requires_grad)
            restricted_bias_direction2_orth: Tensor = torch.tensor([0.0, 1.0], device=evaluation_embeddings.device, requires_grad=self.requires_grad)
            restricted_bias_direction_inner_prod: Tensor = torch.dot(restricted_bias_direction1, restricted_bias_direction2)
            theta: Tensor = torch.abs(torch.arccos(restricted_bias_direction_inner_prod))
            theta_proj: float = np.pi / 2 - theta.item()
            phi: Tensor = torch.arccos(torch.matmul((restricted_rotated_evaluation_embeddings / torch.linalg.norm(restricted_rotated_evaluation_embeddings, dim=-1, keepdim=True)), restricted_bias_direction1))
            d: Tensor = torch.matmul((restricted_rotated_evaluation_embeddings / torch.linalg.norm(restricted_rotated_evaluation_embeddings, dim=-1, keepdim=True)), restricted_bias_direction2_orth)
            theta_x: Tensor = torch.zeros_like(phi, requires_grad=self.requires_grad)
            theta_x = torch.where((d > 0) & (phi < theta_proj), theta * (phi / (theta_proj + 1e-10)), theta_x)
            theta_x = torch.where((d > 0) & (phi > theta_proj), theta * ((np.pi - phi) / (np.pi - theta_proj + 1e-10)), theta_x)
            theta_x = torch.where((d < 0) & (phi >= np.pi - theta_proj), theta * ((np.pi - phi) / (theta_proj + 1e-10)), theta_x)
            theta_x = torch.where((d < 0) & (phi < np.pi - theta_proj), theta * (phi / (np.pi - theta_proj + 1e-10)), theta_x)
            f_matrix: Tensor = torch.cat([
                torch.cos(theta_x).unsqueeze(-1), 
                -torch.sin(theta_x).unsqueeze(-1), 
                torch.sin(theta_x).unsqueeze(-1), 
                torch.cos(theta_x).unsqueeze(-1)
            ], dim=-1)
            f_matrix = f_matrix.reshape(f_matrix.size()[:-1] + (2, 2))
            evaluation_embeddings_clone: Tensor = evaluation_embeddings.clone()
            evaluation_embeddings_clone[mask] = torch.cat([
                torch.bmm(f_matrix, restricted_rotated_evaluation_embeddings.unsqueeze(-1)).squeeze(-1), 
                fixed_rotated_evaluation_embeddings
            ], dim=-1)
            return torch.matmul(evaluation_embeddings_clone, rotation_matrix)