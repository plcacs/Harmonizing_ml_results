import numpy as np
from functools import partial
from nevergrad.parametrization import parameter as p
from ..base import ExperimentFunction
import nevergrad as ng

class OlympusSurface(ExperimentFunction):
    SURFACE_KINDS = ('Michalewicz', 'AckleyPath', 'Dejong', 'HyperEllipsoid', 'Levy', 'Michalewicz', 'Rastrigin', 'Rosenbrock', 'Schwefel', 'StyblinskiTang', 'Zakharov', 'DiscreteAckley', 'DiscreteDoubleWell', 'DiscreteMichalewicz', 'LinearFunnel', 'NarrowFunnel', 'GaussianMixture')

    def __init__(self, kind, dimension=10, noise_kind='GaussianNoise', noise_scale=1):
        self.kind = kind
        self.param_dim = dimension
        self.noise_kind = noise_kind
        assert self.kind in OlympusSurface.SURFACE_KINDS
        assert self.noise_kind in ['GaussianNoise', 'UniformNoise', 'GammaNoise']
        self.noise_scale = noise_scale
        self.surface = partial(self._simulate_surface, noise=True)
        self.surface_without_noise = partial(self._simulate_surface, noise=False)
        parametrization = p.Array(shape=(dimension,))
        parametrization.function.deterministic = False
        super().__init__(self.surface, parametrization)
        self.shift = self.parametrization.random_state.normal(size=self.dimension)

    def _simulate_surface(self, x, noise=True):
        try:
            from olympus.surfaces import import_surface
            from olympus import noises
        except ImportError as e:
            raise ng.errors.UnsupportedExperiment('Please install olympus for Olympus experiments') from e
        if noise:
            noise = noises.Noise(kind=self.noise_kind, scale=self.noise_scale)
            surface = import_surface(self.kind)(param_dim=self.param_dim, noise=noise)
        else:
            surface = import_surface(self.kind)(param_dim=self.param_dim)
        return surface.run(x - self.shift)[0][0]

    def evaluation_function(self, *recommendations):
        """Averages multiple evaluations if necessary"""
        x = recommendations[0].value
        return self.surface_without_noise(x - self.shift)

class OlympusEmulator(ExperimentFunction):
    DATASETS = ('suzuki', 'fullerenes', 'colors_bob', 'photo_wf3', 'snar', 'alkox', 'benzylation', 'photo_pce10', 'hplc', 'colors_n9')

    def __init__(self, dataset_kind='alkox', model_kind='NeuralNet'):
        self.dataset_kind = dataset_kind
        self.model_kind = model_kind
        assert self.dataset_kind in OlympusEmulator.DATASETS
        assert self.model_kind in ['BayesNeuralNet', 'NeuralNet']
        parametrization = self._get_parametrization()
        parametrization.function.deterministic = False
        parametrization.set_name('')
        super().__init__(self._simulate_emulator, parametrization)

    def _get_parametrization(self):
        try:
            from olympus.datasets import Dataset
        except ImportError as e:
            raise ng.errors.UnsupportedExperiment('Please install olympus for Olympus experiments') from e
        dataset = Dataset(self.dataset_kind)
        dimension = dataset.shape[1] - 1
        bounds = list(zip(*dataset.param_space.param_bounds))
        return p.Array(shape=(dimension,), lower=bounds[0], upper=bounds[1])

    def _simulate_emulator(self, x):
        try:
            from olympus import Emulator
        except ImportError as e:
            raise ng.errors.UnsupportedExperiment('Please install olympus for Olympus experiments') from e
        emulator = Emulator(dataset=self.dataset_kind, model=self.model_kind)
        return emulator.run(x)[0][0] * (-1 if emulator.get_goal() == 'maximize' else 1)