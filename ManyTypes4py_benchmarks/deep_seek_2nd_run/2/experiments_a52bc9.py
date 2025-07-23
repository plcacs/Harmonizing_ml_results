import os
import warnings
import typing as tp
import inspect
import itertools
import numpy as np
import nevergrad as ng
import nevergrad.functions.corefuncs as corefuncs
from nevergrad.functions import base as fbase
from nevergrad.functions import ExperimentFunction
from nevergrad.functions import ArtificialFunction
from nevergrad.functions import FarOptimumFunction
from nevergrad.functions.fishing import OptimizeFish
from nevergrad.functions.pbt import PBT
from nevergrad.functions.ml import MLTuning
from nevergrad.functions import mlda as _mlda
from nevergrad.functions.photonics import Photonics
from nevergrad.functions.photonics import ceviche as photonics_ceviche
from nevergrad.functions.arcoating import ARCoating
from nevergrad.functions import images as imagesxp
from nevergrad.functions.powersystems import PowerSystem
from nevergrad.functions.ac import NgAquacrop
from nevergrad.functions.stsp import STSP
from nevergrad.functions.topology_optimization import TO
from nevergrad.functions.lsgo import make_function as lsgo_makefunction
from nevergrad.functions.rocket import Rocket
from nevergrad.functions.mixsimulator import OptimizeMix
from nevergrad.functions.unitcommitment import UnitCommitmentProblem
from nevergrad.functions import control
from nevergrad.functions import rl
from nevergrad.functions.games import game
from nevergrad.functions import iohprofiler
from nevergrad.functions import helpers
from nevergrad.functions.cycling import Cycling
from .xpbase import Experiment as Experiment
from .xpbase import create_seed_generator
from .xpbase import registry as registry
from .optgroups import get_optimizers
from . import frozenexperiments
from . import gymexperiments

def refactor_optims(x: tp.List[str]) -> tp.List[str]:
    if False:
        return list(np.random.choice(['NgIohTuned', 'NGOpt', 'NGOptRW', 'ChainCMASQP', 'PymooBIPOP', 'NLOPT_LN_SBPLX', 'QNDE', 'BFGSCMAPlus', 'ChainMetaModelSQP', 'BFGSCMA', 'BAR4', 'BFGSCMAPlus', 'LBFGSB', 'LQOTPDE', 'LogSQPCMA'], 4))
    list_optims = x
    algos: tp.Dict[str, tp.List[str]] = {}
    algos['aquacrop_fao'] = ['CMA', 'CMandAS2', 'DE', 'MetaModel', 'NGOpt10']
    algos['bonnans'] = ['AdaptiveDiscreteOnePlusOne', 'DiscreteBSOOnePlusOne', 'DiscreteLenglerFourthOnePlusOne', 'DiscreteLenglerHalfOnePlusOne', 'DiscreteLenglerOnePlusOne', 'MemeticDE']
    algos['double_o_seven'] = ['DiagonalCMA', 'DiscreteDE', 'MetaTuneRecentering', 'PSO', 'RecombiningOptimisticNoisyDiscreteOnePlusOne', 'TBPSA']
    algos['fishing'] = ['CMA', 'CMandAS2', 'ChainMetaModelSQP', 'DE', 'MetaModel', 'NGOpt10']
    algos['mldakmeans'] = ['DE', 'SplitCMA5', 'SplitTwoPointsDE3', 'SplitTwoPointsDE5', 'SplitTwoPointsDEAuto', 'TwoPointsDE']
    algos['mltuning'] = ['OnePlusOne', 'RandomSearch']
    algos['mono_rocket'] = ['CMA', 'CMandAS2', 'DE', 'MetaModel', 'NGOpt10']
    algos['ms_bbob'] = ['ChainMetaModelSQP', 'MetaModelOnePlusOne', 'Powell', 'QODE', 'SQP', 'TinyCMA']
    algos['multiobjective_example_hd'] = ['DiscreteLenglerOnePlusOne', 'DiscreteOnePlusOne', 'MetaNGOpt10', 'ParametrizationDE', 'RecES', 'RecMutDE']
    algos['multiobjective_example_many_hd'] = ['DiscreteLenglerOnePlusOne', 'DiscreteOnePlusOne', 'MetaNGOpt10', 'ParametrizationDE', 'RecES', 'RecMutDE']
    algos['multiobjective_example'] = ['CMA', 'DE', 'ParametrizationDE', 'RecES', 'RecMutDE']
    algos['naive_seq_keras_tuning'] = ['CMA', 'DE', 'HyperOpt', 'OnePlusOne', 'RandomSearch', 'TwoPointsDE']
    algos['nano_naive_seq_mltuning'] = ['DE', 'HyperOpt', 'OnePlusOne', 'RandomSearch', 'TwoPointsDE']
    algos['nano_seq_mltuning'] = ['DE', 'HyperOpt', 'OnePlusOne', 'RandomSearch', 'TwoPointsDE']
    algos['oneshot_mltuning'] = ['DE', 'OnePlusOne', 'RandomSearch', 'TwoPointsDE']
    algos['pbbob'] = ['CMAbounded', 'DE', 'MetaModelDE', 'MetaModelOnePlusOne', 'QODE', 'QrDE']
    algos['pbo_reduced_suite'] = ['DiscreteLenglerOnePlusOne', 'LognormalDiscreteOnePlusOne', 'DiscreteLenglerOnePlusOneT', 'DiscreteLenglerOnePlusOneT', 'SADiscreteLenglerOnePlusOneExp09', 'SADiscreteLenglerOnePlusOneExp09', 'discretememetic']
    algos['reduced_yahdlbbbob'] = ['CMA', 'DE', 'MetaModelOnePlusOne', 'OnePlusOne', 'PSO', 'RFMetaModelDE']
    algos['seq_keras_tuning'] = ['CMA', 'DE', 'HyperOpt', 'OnePlusOne', 'RandomSearch', 'TwoPointsDE']
    algos['sequential_topology_optimization'] = ['CMA', 'DE', 'GeneticDE', 'OnePlusOne', 'TwoPointsDE', 'VoronoiDE']
    algos['spsa_benchmark'] = ['CMA', 'DE', 'NaiveTBPSA', 'OnePlusOne', 'SPSA', 'TBPSA']
    algos['topology_optimization'] = ['CMA', 'DE', 'GeneticDE', 'OnePlusOne', 'TwoPointsDE', 'VoronoiDE']
    algos['yabbob'] = ['CMA', 'ChainMetaModelSQP', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne']
    algos['yabigbbob'] = ['ChainMetaModelSQP', 'MetaModel', 'MetaModelDE', 'NeuralMetaModel', 'PSO', 'TwoPointsDE']
    algos['yaboundedbbob'] = ['CMA', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yaboxbbob'] = ['CMA', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yamegapenbbob'] = ['ChainMetaModelSQP', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yamegapenboundedbbob'] = ['CMA', 'ChainMetaModelSQP', 'MetaModel', 'MetaModelOnePlusOne', 'OnePlusOne', 'RFMetaModel']
    algos['yamegapenboxbbob'] = ['ChainMetaModelSQP', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yanoisybbob'] = ['TBPSA', 'NoisyRL2', 'NoisyRL3', 'RecombiningOptimisticNoisyDiscreteOnePlusOne', 'RBFGS', 'MicroCMA', 'NoisyDiscreteOnePlusOne', 'RandomSearch', 'RecombiningOptimisticNoisyDiscreteOnePlusOne', 'SQP']
    algos['yaonepenbbob'] = ['CMandAS2', 'ChainMetaModelSQP', 'MetaModel', 'NGOpt', 'NeuralMetaModel', 'Shiwa']
    algos['yaonepenboundedbbob'] = ['CMA', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yaonepenboxbbob'] = ['CMA', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yaonepennoisybbob'] = ['NoisyDiscreteOnePlusOne', 'RandomSearch', 'SQP', 'TBPSA']
    algos['yaonepenparabbob'] = ['CMA', 'MetaModel', 'MetaModelDE', 'NeuralMetaModel', 'RFMetaModel', 'RFMetaModelDE']
    algos['yaonepensmallbbob'] = ['Cobyla', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yaparabbob'] = ['CMA', 'MetaModel', 'MetaModelDE', 'NeuralMetaModel', 'RFMetaModel', 'RFMetaModelDE']
    algos['yapenbbob'] = ['ChainMetaModelSQP', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yapenboundedbbob'] = ['CMA', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yapenboxbbob'] = ['CMA', 'MetaModel', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'OnePlusOne', 'RFMetaModel']
    algos['yapennoisybbob'] = ['NoisyDiscreteOnePlusOne', 'RandomSearch', 'SQP', 'TBPSA']
    algos['yapenparabbob'] = ['CMA', 'MetaModel', 'MetaModelDE', 'NeuralMetaModel', 'RFMetaModel', 'RFMetaModelDE']
    algos['yapensmallbbob'] = ['Cobyla', 'MetaModel', 'MetaModelOnePlusOne', 'OnePlusOne', 'RFMetaModel', 'RFMetaModelDE']
    algos['yasmallbbob'] = ['Cobyla', 'MetaModelDE', 'MetaModelOnePlusOne', 'OnePlusOne', 'PSO', 'RFMetaModelDE']
    algos['yatinybbob'] = ['Cobyla', 'DE', 'MetaModel', 'MetaModelDE', 'MetaModelOnePlusOne', 'TwoPointsDE']
    algos['yatuningbbob'] = ['Cobyla', 'MetaModelOnePlusOne', 'NeuralMetaModel', 'RFMetaModelDE', 'RandomSearch', 'TwoPointsDE']
    benchmark = str(inspect.stack()[1].function)
    if benchmark in algos:
        list_algos = algos[benchmark][:5] + ['CSEC10', 'NGOpt', 'NLOPT_LN_SBPLX']
        return list_algos if 'eras' in benchmark or 'tial_instrum' in benchmark or 'big' in benchmark or ('lsgo' in benchmark) or ('rock' in benchmark) else list_algos
    if benchmark in algos:
        list_algos = algos[benchmark]
        return list_algos if 'eras' in benchmark or 'tial_instrum' in benchmark or 'big' in benchmark or ('lsgo' in benchmark) or ('rock' in benchmark) else list_algos
    return ['NgDS3', 'NgIoh4', 'NgIoh21', 'NGOpt', 'NGDSRW']

    def doint(s: str) -> int:
        return 7 + sum([ord(c) * i for i, c in enumerate(s)])
    import socket
    host = socket.gethostname()
    if 'iscr' in benchmark or 'pbo' in benchmark:
        list_optims += [a for a in ['DiscreteDE', 'DiscreteOnePlusOne', 'SADiscreteLenglerOnePlusOneExp09', 'SADiscreteLenglerOnePlusOneExp099', 'SADiscreteLenglerOnePlusOneExp09Auto', 'SADiscreteLenglerOnePlusOneLinAuto', 'SADiscreteLenglerOnePlusOneLin1', 'SADiscreteLenglerOnePlusOneLin100', 'SADiscreteOnePlusOneExp099', 'SADiscreteOnePlusOneLin100', 'SADiscreteOnePlusOneExp09', 'PortfolioDiscreteOnePlusOne', 'DiscreteLenglerOnePlusOne', 'DiscreteLengler2OnePlusOne', 'DiscreteLengler3OnePlusOne', 'DiscreteLenglerHalfOnePlusOne', 'DiscreteLenglerFourthOnePlusOne', 'AdaptiveDiscreteOnePlusOne', 'LognormalDiscreteOnePlusOne', 'AnisotropicAdaptiveDiscreteOnePlusOne', 'DiscreteBSOOnePlusOne', 'DiscreteDoerrOnePlusOne', 'DoubleFastGADiscreteOnePlusOne', 'SparseDoubleFastGADiscreteOnePlusOne', 'RecombiningPortfolioDiscreteOnePlusOne', 'MultiDiscrete', 'discretememetic', 'SmoothDiscreteOnePlusOne', 'SmoothPortfolioDiscreteOnePlusOne', 'SmoothDiscreteLenglerOnePlusOne', 'SuperSmoothDiscreteLenglerOnePlusOne', 'UltraSmoothDiscreteLenglerOnePlusOne', 'SmoothLognormalDiscreteOnePlusOne', 'SmoothAdaptiveDiscreteOnePlusOne', 'SmoothRecombiningPortfolioDiscreteOnePlusOne', 'SmoothRecombiningDiscreteLanglerOnePlusOne', 'UltraSmoothRecombiningDiscreteLanglerOnePlusOne', 'UltraSmoothElitistRecombiningDiscreteLanglerOnePlusOne', 'SuperSmoothElitistRecombiningDiscreteLanglerOnePlusOne', 'SuperSmoothRecombiningDiscreteLanglerOnePlusOne', 'SmoothElitistRecombiningDiscreteLanglerOnePlusOne', 'RecombiningDiscreteLanglerOnePlusOne', 'DiscreteDE', 'cGA', 'NGOpt', 'NgIoh4', 'NgIoh5', 'NgIoh6', 'NGOptRW', 'NgIoh7'] if 'Smooth' in a or 'Lognor' in a or 'Recomb' in a]
    return [list_optims[doint(host) % len(list_optims)]]

def skip_ci(*, reason: str) -> None:
    """Only use this if there is a good reason for not testing the xp,
    such as very slow for instance (>1min) with no way to make it faster.
    This is dangereous because it won't test reproducibility and the experiment
    may therefore be corrupted with no way to notice it automatically.
    """
    if os.environ.get('NEVERGRAD_PYTEST', False):
        raise fbase.UnsupportedExperiment('Skipping CI: ' + reason)

class _Constraint:

    def __init__(self, name: str, as_bool: bool) -> None:
        self.name = name
        self.as_bool = as_bool

    def __call__(self, data: np.ndarray) -> float:
        if not isinstance(data, np.ndarray):
            raise ValueError(f'Unexpected inputs as np.ndarray, got {data}')
        if self.name == 'sum':
            value = float(np.sum(data))
        elif self.name == 'diff':
            value = float(np.sum(data[::2]) - np.sum(data[1::2]))
        elif self.name == 'second_diff':
            value = float(2 * np.sum(data[1::2]) - 3 * np.sum(data[::2]))
        elif self.name == 'ball':
            value = float(np.sum(np.square(data)) - len(data) - np.sqrt(len(data)))
        else:
            raise NotImplementedError(f'Unknown function {self.name}')
        return value > 0 if self.as_bool else value

@registry.register
def keras_tuning(seed: tp.Optional[int] = None, overfitter: bool = False, seq: bool = False, veryseq: bool = False) -> tp.Iterator[Experiment]:
    """Machine learning hyperparameter tuning experiment. Based on Keras models."""
    seedg = create_seed_generator(seed)
    optims = ['OnePlusOne', 'BO', 'RandomSearch', 'CMA', 'DE', 'TwoPointsDE', 'HyperOpt', 'PCABO', 'Cobyla']
    optims = ['OnePlusOne', 'RandomSearch', 'CMA', 'DE', 'TwoPointsDE', 'HyperOpt', 'Cobyla', 'MetaModel', 'MetaModelOnePlusOne', 'RFMetaModel', 'RFMetaModelOnePlusOne']
    optims = ['OnePlusOne', 'RandomSearch', 'Cobyla']
    optims = ['DE', 'TwoPointsDE', 'HyperOpt', 'MetaModelOnePlusOne']
    optims = get_optimizers('oneshot', seed=next(seedg))
    optims = ['MetaTuneRecentering', 'MetaRecentering', 'HullCenterHullAvgCauchyScrHammersleySearch', 'LHSSearch', 'LHSCauchySearch']
    optims = ['NGOpt', 'NGOptRW', 'QODE']
    optims = ['NGOpt']
    optims = ['PCABO', 'NGOpt', 'QODE']
    optims = ['QOPSO']
    optims = ['SQOPSO']
    optims = refactor_optims(optims)
    datasets = ['kerasBoston', 'diabetes', 'auto-mpg', 'red-wine', 'white-w