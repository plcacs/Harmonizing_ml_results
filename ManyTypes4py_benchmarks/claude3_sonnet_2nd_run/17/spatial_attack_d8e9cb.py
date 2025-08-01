from typing import Union, Any, Tuple, Generator, Optional, cast
import eagerpy as ep
import numpy as np
from ..devutils import atleast_kd
from ..criteria import Criterion
from .base import Model
from .base import T
from .base import get_is_adversarial
from .base import get_criterion
from .base import Attack
from .spatial_attack_transformations import rotate_and_shift
from .base import raise_if_kwargs
from .base import verify_input_bounds

class SpatialAttack(Attack):
    """Adversarially chosen rotations and translations. [#Engs]
    This implementation is based on the reference implementation by
    Madry et al.: https://github.com/MadryLab/adversarial_spatial

    References:
    .. [#Engs] Logan Engstrom*, Brandon Tran*, Dimitris Tsipras*,
           Ludwig Schmidt, Aleksander Mądry: "A Rotation and a
           Translation Suffice: Fooling CNNs with Simple Transformations",
           http://arxiv.org/abs/1712.02779
    """

    def __init__(self, max_translation: float = 3, max_rotation: float = 30, num_translations: int = 5, num_rotations: int = 5, grid_search: bool = True, random_steps: int = 100) -> None:
        self.max_trans = max_translation
        self.max_rot = max_rotation
        self.grid_search = grid_search
        self.num_trans = num_translations
        self.num_rots = num_rotations
        self.random_steps = random_steps

    def __call__(self, model: Model, inputs: T, criterion: Union[Criterion, Any], **kwargs: Any) -> Tuple[T, T, T]:
        x, restore_type = ep.astensor_(inputs)
        del inputs
        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)
        if x.ndim != 4:
            raise NotImplementedError('only implemented for inputs with two spatial dimensions (and one channel and one batch dimension)')
        xp = self.run(model, x, criterion)
        success = is_adversarial(xp)
        xp_ = restore_type(xp)
        return (xp_, xp_, restore_type(success))

    def run(self, model: Model, inputs: Any, criterion: Union[Criterion, Any], **kwargs: Any) -> Any:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        del inputs, kwargs
        verify_input_bounds(x, model)
        criterion = get_criterion(criterion)
        is_adversarial = get_is_adversarial(criterion, model)
        found = is_adversarial(x)
        results = x

        def grid_search_generator() -> Generator[Tuple[float, float, float], None, None]:
            dphis = np.linspace(-self.max_rot, self.max_rot, self.num_rots)
            dxs = np.linspace(-self.max_trans, self.max_trans, self.num_trans)
            dys = np.linspace(-self.max_trans, self.max_trans, self.num_trans)
            for dphi in dphis:
                for dx in dxs:
                    for dy in dys:
                        yield (dphi, dx, dy)

        def random_search_generator() -> Generator[Tuple[float, float, float], None, None]:
            dphis = np.random.uniform(-self.max_rot, self.max_rot, self.random_steps)
            dxs = np.random.uniform(-self.max_trans, self.max_trans, self.random_steps)
            dys = np.random.uniform(-self.max_trans, self.max_trans, self.random_steps)
            for dphi, dx, dy in zip(dphis, dxs, dys):
                yield (dphi, dx, dy)
        gen = grid_search_generator() if self.grid_search else random_search_generator()
        for dphi, dx, dy in gen:
            x_p = rotate_and_shift(x, translation=(dx, dy), rotation=dphi)
            is_adv = is_adversarial(x_p)
            new_adv = ep.logical_and(is_adv, found.logical_not())
            results = ep.where(atleast_kd(new_adv, x_p.ndim), x_p, results)
            found = ep.logical_or(new_adv, found)
            if found.all():
                break
        return restore_type(results)

    def repeat(self, times: int) -> "SpatialAttack":
        if self.grid_search:
            raise ValueError('repeat is not supported if attack is deterministic')
        else:
            random_steps = self.random_steps * times
            return SpatialAttack(max_translation=self.max_trans, max_rotation=self.max_rot, num_translations=self.num_trans, num_rotations=self.num_rots, grid_search=self.grid_search, random_steps=random_steps)
