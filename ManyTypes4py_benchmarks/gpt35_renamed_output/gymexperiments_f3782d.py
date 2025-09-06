import os
from typing import List, Generator
from nevergrad.functions import gym as nevergrad_gym
from .xpbase import registry
from .xpbase import create_seed_generator
from .xpbase import Experiment
from .optgroups import get_optimizers

def func_jn4v4yju(specific_problem: str) -> str:
    specific_problem = os.environ.get('TARGET_GYM_ENV', specific_problem)
    print('problem=', specific_problem)
    return specific_problem

def func_r2zgrjs6(optims: List[str]) -> List[str]:
    if os.environ.get('GYM_OPTIMIZER') is not None:
        optimizer_string = os.environ.get('GYM_OPTIMIZER')
        print(f'Considering optimizers with {optimizer_string} in their name.')
        optims = [o for o in optims if optimizer_string in str(o)]
        if len(optims) == 0:
            optims = [optimizer_string]
    print('optims=', optims)
    return optims

def func_nyicn0ba(budgets: List[int]) -> List[int]:
    if os.environ.get('MAX_GYM_BUDGET') is not None:
        budget_string = os.environ.get('MAX_GYM_BUDGET')
        budgets = [b for b in budgets if b < int(budget_string)]
    print('budgets=', budgets)
    return budgets

@registry.register
def func_ojfytlf7(seed=None, randomized=True, multi=False, big=False,
    memory=False, ng_gym=False, conformant=False, gp=False, sparse=False,
    multi_scale=False, small=False, tiny=False, structured=False) -> Generator:
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
    env_names = nevergrad_gym.GymMulti.get_env_names()
    assert int(ng_gym) + int(gp) <= 1, 'At most one specific list of environments.'
    if ng_gym:
        env_names = nevergrad_gym.GymMulti.ng_gym
    if gp:
        import pybullet_envs
        env_names = ['CartPole-v1', 'Acrobot-v1', 'MountainCarContinuous-v0', 'Pendulum-v1', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3', 'HopperBulletEnv-v0', 'LunarLanderContinuous-v2']
    seedg = create_seed_generator(seed)
    optims = ['DiagonalCMA', 'GeneticDE', 'NoisyRL1', 'NoisyRL2', 'NoisyRL3', 'MixDeterministicRL', 'SpecialRL', 'PSO', 'NGOpt', 'NgIohTuned']
    if multi:
        controls = ['multi_neural']
    else:
        controls = ['noisy_semideep_neural', 'noisy_scrambled_semideep_neural', 'noisy_deep_neural', 'noisy_scrambled_deep_neural', 'neural', 'stackingmemory_neural', 'deep_neural', 'semideep_neural', 'noisy_neural', 'noisy_scrambled_neural', 'resid_neural', 'resid_semideep_neural', 'resid_deep_neural'] if not big else ['resid_neural']
    if structured:
        controls = ['neural', 'structured_neural']
    if memory:
        controls = ['stackingmemory_neural', 'deep_stackingmemory_neural', 'semideep_stackingmemory_neural']
        controls += ['memory_neural', 'deep_memory_neural', 'semideep_memory_neural']
        controls += ['extrapolatestackingmemory_neural', 'deep_extrapolatestackingmemory_neural', 'semideep_extrapolatestackingmemory_neural']
        assert not multi
    if conformant:
        controls = ['stochastic_conformant']
    optimization_scales = [0]
    if multi_scale:
        optimization_scales = [-6, -4, -2, 0]
    budgets = [50, 200, 100, 25, 400]
    budgets = func_nyicn0ba(budgets)
    for control in controls:
        neural_factors = [None] if conformant or control == 'linear' else [1] if 'memory' in control else [3] if big else [1] if tiny or small else [1, 2, 3]
        for neural_factor in neural_factors:
            for name in env_names:
                sparse_limits = [None]
                if sparse:
                    sparse_limits += [10, 100, 1000]
                for sparse_limit in sparse_limits:
                    for optimization_scale in optimization_scales:
                        try:
                            func = nevergrad_gym.GymMulti(name, control=control, neural_factor=neural_factor, randomized=randomized, optimization_scale=optimization_scale, sparse_limit=sparse_limit)
                            if not randomized:
                                func.parametrization.function.deterministic = True
                                func.parametrization.enforce_determinism = True
                        except MemoryError:
                            continue
                        for budget in budgets:
                            for algo in optims:
                                xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                                xp.function.parametrization.real_world = True
                                xp.function.parametrization.neural = True
                                if xp.function.parametrization.dimension > 40 and small:
                                    continue
                                if xp.function.parametrization.dimension > 20 and tiny:
                                    continue
                                if not xp.is_incoherent:
                                    yield xp
