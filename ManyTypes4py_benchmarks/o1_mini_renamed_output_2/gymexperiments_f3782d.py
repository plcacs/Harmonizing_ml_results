import os
import typing as tp
from nevergrad.functions import gym as nevergrad_gym
from .xpbase import registry
from .xpbase import create_seed_generator
from .xpbase import Experiment
from .optgroups import get_optimizers


def func_jn4v4yju(specific_problem: str) -> str:
    specific_problem = os.environ.get('TARGET_GYM_ENV', specific_problem)
    print('problem=', specific_problem)
    return specific_problem


def func_r2zgrjs6(optims: tp.List[str]) -> tp.List[str]:
    optimizer_env = os.environ.get('GYM_OPTIMIZER')
    if optimizer_env is not None:
        optimizer_string = optimizer_env
        print(f'Considering optimizers with {optimizer_string} in their name.')
        optims = [o for o in optims if optimizer_string in str(o)]
        if len(optims) == 0:
            optims = [optimizer_string]
    print('optims=', optims)
    return optims


def func_nyicn0ba(budgets: tp.List[int]) -> tp.List[int]:
    budget_env = os.environ.get('MAX_GYM_BUDGET')
    if budget_env is not None:
        budget_string = budget_env
        budgets = [b for b in budgets if b < int(budget_string)]
    print('budgets=', budgets)
    return budgets


@registry.register
def func_ojfytlf7(
    seed: tp.Optional[int] = None,
    randomized: bool = True,
    multi: bool = False,
    big: bool = False,
    memory: bool = False,
    ng_gym: bool = False,
    conformant: bool = False,
    gp: bool = False,
    sparse: bool = False,
    multi_scale: bool = False,
    small: bool = False,
    tiny: bool = False,
    structured: bool = False
) -> tp.Generator[Experiment, None, None]:
    """Gym simulator. Maximize reward.  Many distinct problems.

    Parameters:
        seed: int
           random seed.
        randomized: bool
           whether we keep the problem's stochasticity
        multi: bool
           do we have one neural net per time step
        big: bool
           do we consider big budgets
        memory: bool
           do we use recurrent nets
        ng_gym: bool
           do we restrict to ng-gym
        conformant: bool
           do we restrict to conformant planning, i.e. deterministic controls.
    """
    env_names: tp.List[str] = nevergrad_gym.GymMulti.get_env_names()
    assert int(ng_gym) + int(gp) <= 1, 'At most one specific list of environments.'
    if ng_gym:
        env_names = nevergrad_gym.GymMulti.ng_gym
    if gp:
        import pybullet_envs
        env_names = [
            'CartPole-v1', 'Acrobot-v1',
            'MountainCarContinuous-v0', 'Pendulum-v1', 'BipedalWalker-v3',
            'BipedalWalkerHardcore-v3', 'HopperBulletEnv-v0',
            'LunarLanderContinuous-v2'
        ]
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = [
        'DiagonalCMA', 'GeneticDE', 'NoisyRL1', 'NoisyRL2',
        'NoisyRL3', 'MixDeterministicRL', 'SpecialRL', 'PSO', 'NGOpt',
        'NgIohTuned'
    ]
    if multi:
        controls: tp.List[str] = ['multi_neural']
    else:
        controls = (
            ['noisy_semideep_neural',
             'noisy_scrambled_semideep_neural', 'noisy_deep_neural',
             'noisy_scrambled_deep_neural', 'neural',
             'stackingmemory_neural', 'deep_neural', 'semideep_neural',
             'noisy_neural', 'noisy_scrambled_neural', 'resid_neural',
             'resid_semideep_neural', 'resid_deep_neural']
            if not big else ['resid_neural']
        )
    if structured:
        controls = ['neural', 'structured_neural']
    if memory:
        controls = [
            'stackingmemory_neural', 'deep_stackingmemory_neural',
            'semideep_stackingmemory_neural'
        ]
        controls += [
            'memory_neural', 'deep_memory_neural',
            'semideep_memory_neural'
        ]
        controls += [
            'extrapolatestackingmemory_neural',
            'deep_extrapolatestackingmemory_neural',
            'semideep_extrapolatestackingmemory_neural'
        ]
        assert not multi
    if conformant:
        controls = ['stochastic_conformant']
    optimization_scales: tp.List[int] = [0]
    if multi_scale:
        optimization_scales = [-6, -4, -2, 0]
    budgets: tp.List[int] = [50, 200, 100, 25, 400]
    budgets = func_nyicn0ba(budgets)
    for control in controls:
        if conformant or control == 'linear':
            neural_factors: tp.List[tp.Optional[int]] = [None]
        elif 'memory' in control:
            neural_factors = [1]
        elif big:
            neural_factors = [3]
        elif tiny or small:
            neural_factors = [1]
        else:
            neural_factors = [1, 2, 3]
        for neural_factor in neural_factors:
            for name in env_names:
                sparse_limits: tp.List[tp.Optional[int]] = [None]
                if sparse:
                    sparse_limits += [10, 100, 1000]
                for sparse_limit in sparse_limits:
                    for optimization_scale in optimization_scales:
                        try:
                            func = nevergrad_gym.GymMulti(
                                name,
                                control=control,
                                neural_factor=neural_factor,
                                randomized=randomized,
                                optimization_scale=optimization_scale,
                                sparse_limit=sparse_limit
                            )
                            if not randomized:
                                func.parametrization.function.deterministic = True  # type: ignore
                                func.parametrization.enforce_determinism = True  # type: ignore
                        except MemoryError:
                            continue
                        for budget in budgets:
                            for algo in optims:
                                xp = Experiment(
                                    func,
                                    algo,
                                    budget,
                                    num_workers=1,
                                    seed=next(seedg)
                                )
                                xp.function.parametrization.real_world = True  # type: ignore
                                xp.function.parametrization.neural = True  # type: ignore
                                if (xp.function.parametrization.dimension > 40 and small):  # type: ignore
                                    continue
                                if (xp.function.parametrization.dimension > 20 and tiny):  # type: ignore
                                    continue
                                if not xp.is_incoherent:
                                    yield xp


@registry.register
def func_kfusrxo7(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """Counterpart of ng_full_gym with one neural net per time step.

    Each neural net is used for many problems, but only for one of the time steps."""
    return func_ojfytlf7(seed, multi=True)


@registry.register
def func_4ygvu6xt(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """Counterpart of ng_full_gym with one neural net per time step.

    Each neural net is used for many problems, but only for one of the time steps."""
    return func_ojfytlf7(seed, multi=True, structured=True)


@registry.register
def func_g1ht87ma(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """Counterpart of ng_full_gym with fixed, predetermined actions for each time step.

    This is conformant: we optimize directly the actions for a given context.
    This does not prevent stochasticity, but actions do not depend on observations."""
    return func_ojfytlf7(seed, conformant=True)


@registry.register
def func_zx4kwcy7(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """Counterpart of ng_full_gym with a specific, reduced list of problems."""
    return func_ojfytlf7(seed, ng_gym=True)


@registry.register
def func_0yccyuym(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """GP benchmark.

    Counterpart of ng_full_gym with a specific, reduced list of problems for matching
    a genetic programming benchmark."""
    return func_ojfytlf7(seed, gp=True, multi_scale=True)


@registry.register
def func_ks5jy7p9(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """GP benchmark.

    Counterpart of ng_full_gym with a specific, reduced list of problems for matching
    a genetic programming benchmark."""
    return func_ojfytlf7(seed, conformant=True, gp=True)


@registry.register
def func_4vjk3f2g(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """GP benchmark.

    Counterpart of ng_full_gym with a specific, reduced list of problems for matching
    a genetic programming benchmark."""
    return func_ojfytlf7(seed, gp=True, sparse=True, multi_scale=True)


@registry.register
def func_ifwsxrpw(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """Counterpart of ng_gym with a recurrent network."""
    return func_ojfytlf7(seed, ng_gym=True, memory=True)


@registry.register
def func_3pop2sl1(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """Counterpart of ng_full_gym with bigger nets."""
    return func_ojfytlf7(seed, big=True)


@registry.register
def func_sxjgw7h9(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """Counterpart of ng_full_gym with fixed seeds (so that the problem becomes deterministic)."""
    return func_ojfytlf7(seed, randomized=False)


@registry.register
def func_k83y9i2m(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """Counterpart of ng_full_gym with fixed seeds (so that the problem becomes deterministic)."""
    return func_ojfytlf7(seed, randomized=False, tiny=True)


@registry.register
def func_rq7vbwtp(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """Counterpart of ng_full_gym with fixed seeds (so that the problem becomes deterministic)."""
    return func_ojfytlf7(seed, randomized=False, small=True)


def func_96p1q2v6(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    """Gym simulator for Active Network Management."""
    func = nevergrad_gym.GymMulti('multifidLANM')
    seedg = create_seed_generator(seed)
    optims = get_optimizers(
        'basics', 'progressive', 'splitters',
        'baselines', seed=next(seedg)
    )
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]:
        for num_workers in [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    xp = Experiment(
                        func,
                        algo,
                        budget,
                        num_workers=num_workers,
                        seed=next(seedg)
                    )
                    if not xp.is_incoherent:
                        yield xp


def func_4glc8kln(
    seed: tp.Optional[int] = None,
    specific_problem: str = 'LANM',
    conformant: bool = False,
    big_noise: bool = False,
    multi_scale: bool = False,
    greedy_bias: bool = False
) -> tp.Generator[Experiment, None, None]:
    """Gym simulator for Active Network Management (default) or other pb.

    seed: int
        random seed for determinizing the problem
    specific_problem: string
        name of the problem we are working on
    conformant: bool
        do we focus on conformant planning
    big_noise: bool
        do we switch to specific optimizers, dedicated to noise
    multi_scale: boolean
        do we check multiple scales
    greedy_bias: boolean
        do we use greedy reward estimates for biasing the decisions.
    """
    if conformant:
        funcs: tp.List[nevergrad_gym.GymMulti] = [
            nevergrad_gym.GymMulti(
                specific_problem,
                control='conformant',
                neural_factor=None
            )
        ]
    else:
        funcs = [
            nevergrad_gym.GymMulti(
                specific_problem,
                control=control,
                neural_factor=1 if control != 'linear' else None,
                optimization_scale=scale,
                greedy_bias=greedy_bias
            )
            for scale in ([-6, -4, -2, 0] if multi_scale else [0])
            for control in (
                ['deep_neural', 'semideep_neural', 'neural', 'linear']
                if not greedy_bias else ['neural']
            )
        ]
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = [
        'TwoPointsDE', 'GeneticDE', 'PSO', 'DiagonalCMA',
        'DoubleFastGADiscreteOnePlusOne', 'DiscreteLenglerOnePlusOne',
        'PortfolioDiscreteOnePlusOne', 'MixDeterministicRL', 'NoisyRL2',
        'NoisyRL3', 'SpecialRL', 'NGOpt39', 'CMA', 'DE'
    ]
    if 'stochastic' in specific_problem:
        optims = ['DiagonalCMA', 'TBPSA'] if big_noise else ['DiagonalCMA']
    if specific_problem == 'EnergySavingsGym-v0' and conformant:
        optims = [
            'DiscreteOnePlusOne', 'PortfolioDiscreteOnePlusOne',
            'DiscreteLenglerOnePlusOne', 'AdaptiveDiscreteOnePlusOne',
            'AnisotropicAdaptiveDiscreteOnePlusOne',
            'DiscreteBSOOnePlusOne', 'DiscreteDoerrOnePlusOne',
            'OptimisticDiscreteOnePlusOne', 'NoisyDiscreteOnePlusOne',
            'DoubleFastGADiscreteOnePlusOne',
            'SparseDoubleFastGADiscreteOnePlusOne',
            'RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne',
            'RecombiningPortfolioDiscreteOnePlusOne', 'MultiDiscrete', 'NGOpt'
        ]
    optims = func_r2zgrjs6(optims)
    budgets: tp.List[int] = [25, 50, 100, 200]
    budgets = func_nyicn0ba(budgets)
    for func in funcs:
        for budget in budgets:
            for num_workers in [1]:
                if num_workers < budget:
                    for algo in optims:
                        xp = Experiment(
                            func,
                            algo,
                            budget,
                            num_workers=num_workers,
                            seed=next(seedg)
                        )
                        xp.function.parametrization.real_world = True  # type: ignore
                        xp.function.parametrization.neural = not conformant  # type: ignore
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def func_6web0mp1(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    specific_problem: str = 'EnergySavingsGym-v0'
    return func_4glc8kln(
        seed,
        specific_problem=func_jn4v4yju(specific_problem),
        conformant=True,
        big_noise=False
    )


@registry.register
def func_m0iz9kz8(seed: tp.Optional[int] = None) -> tp.Generator[Experiment, None, None]:
    specific_problem: str = 'EnergySavingsGym-v0'
    return func_4glc8kln(
        seed,
        specific_problem=func_jn4v4yju(specific_problem),
        conformant=False,
        big_noise=False
    )
