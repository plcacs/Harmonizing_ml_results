# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import copy
import scipy.stats
import typing as tp
import numpy as np
import gym

import nevergrad as ng

from nevergrad.parametrization import parameter
from ..base import ExperimentFunction

# pylint: disable=unused-import,import-outside-toplevel


GUARANTEED_GYM_ENV_NAMES: tp.List[str] = [
    # "ReversedAddition-v0",
    # "ReversedAddition3-v0",
    # "DuplicatedInput-v0",
    # "Reverse-v0",
    "CartPole-v0",
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
    # "Blackjack-v0",
    # "FrozenLake-v0",   # deprecated
    # "FrozenLake8x8-v0",
    "CliffWalking-v0",
    # "NChain-v0",
    # "Roulette-v0",
    "Taxi-v3",
    # "CubeCrash-v0",
    # "CubeCrashSparse-v0",
    # "CubeCrashScreenBecomesBlack-v0",
    # "MemorizeDigits-v0",
]


# We do not use "conformant" which is not consistent with the rest.
CONTROLLERS: tp.List[str] = [
    "resid_neural",
    "resid_semideep_neural",
    "resid_deep_neural",
    "resid_scrambled_neural",
    "resid_scrambled_semideep_neural",
    "resid_scrambled_deep_neural",
    "resid_noisy_scrambled_neural",
    "resid_noisy_scrambled_semideep_neural",
    "resid_noisy_scrambled_deep_neural",
    "linear",  # Simple linear controller.
    "neural",  # Simple neural controller.
    "deep_neural",  # Deeper neural controller.
    "semideep_neural",  # Deep, but not very deep.
    "structured_neural",  # Structured optimization of a neural net.
    "memory_neural",  # Uses a memory (i.e. recurrent net).
    "deep_memory_neural",
    "stackingmemory_neural",  # Uses a memory and stacks a heuristic and the memory as inputs.
    "deep_stackingmemory_neural",
    "semideep_stackingmemory_neural",
    "extrapolatestackingmemory_neural",  # Same as stackingmemory_neural + suffix-based extrapolation.
    "deep_extrapolatestackingmemory_neural",
    "semideep_extrapolatestackingmemory_neural",
    "semideep_memory_neural",
    "noisy_semideep_neural",
    "noisy_scrambled_semideep_neural",  # Scrambling: why not perturbating the order of variables ?
    "noisy_deep_neural",
    "noisy_scrambled_deep_neural",
    "multi_neural",  # One neural net per time step.
    "noisy_neural",  # Do not start at 0 but at a random point.
    "noisy_scrambled_neural",
    "stochastic_conformant",  # Conformant planning, but still not deterministic.
]


NO_LENGTH: tp.List[str] = ["ANM", "Blackjack", "CliffWalking", "Cube", "Memorize", "llvm"]


class GymMulti(ExperimentFunction):
    """Class for converting a gym environment, a controller style, and others into a black-box optimization benchmark."""

    @staticmethod
    def get_env_names() -> tp.List[str]:
        # import gym_anm  # noqa

        gym_env_names: tp.List[str] = []
        max_displays: int = 10
        for e in gym.envs.registry.values():  # .all():
            try:
                assert not any(
                    x in str(e.id)
                    for x in "Kelly Copy llvm BulletEnv Minitaur Kuka InvertedPendulumSwingupBulletEnv".split()
                )  # We should have another check than that.
                assert (
                    "RacecarZedBulletEnv-v0" != e.id
                ), "This specific environment causes X11 error when using pybullet_envs."
                assert "CarRacing-v" not in str(e.id), "Pixel based task not supported yet"
                env = gym.make(e.id)
                env.reset()
                env.step(env.action_space.sample())
                a1 = np.asarray(env.action_space.sample())
                a2 = np.asarray(env.action_space.sample())
                a3 = np.asarray(env.action_space.sample())
                a1 = a1 + a2 + a3
                if hasattr(a1, "size"):
                    try:
                        assert a1.size < 15000
                    except Exception:  # pylint: disable=broad-except
                        assert a1.size() < 15000  # type: ignore
                gym_env_names.append(e.id)
            except Exception as exception:  # pylint: disable=broad-except
                max_displays -= 1
                if max_displays > 0:
                    print(f"{e.id} not included in full list because of {exception}.")
                if max_displays == 0:
                    print("(similar issue for other environments)")
        return gym_env_names

    controllers: tp.List[str] = CONTROLLERS

    ng_gym: tp.List[str] = [
        # "Reverse-v0",
        "CartPole-v0",
        "CartPole-v1",
        "Acrobot-v1",
        # "FrozenLake-v0",  # deprecated
        # "FrozenLake8x8-v0",
        # "NChain-v0",
        # "Roulette-v0",
    ]

    def wrap_env(self, input_env: gym.Env) -> gym.Env:
        env = gym.wrappers.TimeLimit(
            env=input_env,
            max_episode_steps=self.num_episode_steps,
        )
        return env

    def create_env(self) -> gym.Env:
        env = gym.make(self.short_name)
        try:
            env.reset()
        except:
            assert False, f"Maybe check if {self.short_name} has a problem in reset / observation."
        return env

    def __init__(
        self,
        name: str = "CartPole-v0",
        control: str = "conformant",
        neural_factor: tp.Optional[int] = 1,
        randomized: bool = True,
        optimization_scale: int = 0,
        greedy_bias: bool = False,
        sparse_limit: tp.Optional[int] = None,  # if not None, we penalize solutions with more than sparse_limit weights !=0
    ) -> None:
        self.num_calls: int = 0
        self.optimization_scale: int = optimization_scale
        self.stochastic_problem: bool = "stoc" in name
        self.greedy_bias: bool = greedy_bias
        self.sparse_limit: tp.Optional[int] = sparse_limit
        if "conformant" in control or control == "linear":
            assert neural_factor is None
        if os.name == "nt":
            raise ng.errors.UnsupportedExperiment("Windows is not supported")
        # self.env = None  # self.create_env() let us have no self.env

        # Build various attributes.
        self.short_name: str = name  # Just the environment name.
        env = self.create_env()
        self.name: str = name + "__" + control + "__" + str(neural_factor)
        if sparse_limit is not None:
            self.name += f"__{sparse_limit}"
        if randomized:
            self.name += "_unseeded"
        self.randomized: bool = randomized
        try:
            try:
                self.num_time_steps: int = env._max_episode_steps  # I know! This is a private variable.
            except AttributeError:  # Second chance! Some environments use self.horizon.
                self.num_time_steps: int = env.horizon
        except AttributeError:  # Not all environements have a max number of episodes!
            assert any(x in name for x in NO_LENGTH), name
            if "LANM" not in name:  # Most cases: let's say 5000 time steps.
                self.num_time_steps: int = 200 if control == "conformant" else 5000
            else:  # LANM is a special case with 3000 time steps.
                self.num_time_steps: int = 3000
        self.gamma: float = 0.995 if "LANM" in name else 1.0
        self.neural_factor: tp.Optional[int] = neural_factor

        # Infer the action space.
        self.arities: tp.List[tp.Any] = []
        if isinstance(env.action_space, gym.spaces.Tuple):
            assert all(
                isinstance(p, gym.spaces.MultiDiscrete) for p in env.action_space
            ), f"{name} has a too complicated structure."
            self.arities = [p.nvec for p in env.action_space]
            if control == "conformant":
                output_dim: int = sum(len(a) for a in self.arities)
            else:
                output_dim: int = sum(sum(a) for a in self.arities)
            output_shape: tp.Tuple[int, ...] = (output_dim,)
            discrete: bool = True
        elif isinstance(env.action_space, gym.spaces.Discrete):
            output_dim: int = env.action_space.n
            output_shape: tp.Tuple[int, ...] = (output_dim,)
            discrete: bool = True
            assert output_dim is not None, env.action_space.n
        else:  # Continuous action space
            output_shape: tp.Tuple[int, ...] = env.action_space.shape
            if output_shape is None:
                output_shape = tuple(np.asarray(env.action_space.sample()).shape)
            # When the shape is not available we might do:
            # output_shape = tuple(np.asarray(env.action_space.sample()).shape
            discrete: bool = False
            output_dim: int = np.prod(output_shape)

        self.discrete: bool = discrete

        # Infer the observation space.
        assert env.observation_space is not None or "llvm" in name, "An observation space should be defined."
        if env.observation_space is not None and env.observation_space.dtype == int:
            # Direct inference for corner cases:
            # if "int" in str(type(o)):
            input_dim: int = env.observation_space.n
            assert input_dim is not None, env.observation_space.n
            self.discrete_input: bool = True
        else:
            input_dim: int = np.prod(env.observation_space.shape) if env.observation_space is not None else 0
            if input_dim is None:
                o = env.reset()
                input_dim = np.prod(np.asarray(o).shape)
            self.discrete_input: bool = False

        # Infer the action type.
        a = env.action_space.sample()
        self.action_type: type = type(a)
        self.subaction_type: tp.Optional[type] = None
        if hasattr(a, "__iter__"):
            self.subaction_type = type(a[0])

        # Prepare the policy shape.
        if neural_factor is None:
            assert (
                control == "linear" or "conformant" in control
            ), f"{control} has neural_factor {neural_factor}"
            neural_factor = 1
        self.output_shape: tp.Tuple[int, ...] = output_shape
        self.num_stacking: int = 1
        self.memory_len: int = neural_factor * input_dim if "memory" in control else 0
        self.extended_input_len: int = (input_dim + output_dim) * self.num_stacking if "stacking" in control else 0
        input_dim = input_dim + self.memory_len + self.extended_input_len
        self.extended_input: np.ndarray = np.zeros(self.extended_input_len)
        output_dim = output_dim + self.memory_len
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.num_neurons: int = neural_factor * (input_dim - self.extended_input_len)
        self.num_internal_layers: int = 1 if "semi" in control else 3
        internal: int = self.num_internal_layers * (self.num_neurons**2) if "deep" in control else 0
        unstructured_neural_size: tp.Tuple[int, ...] = (
            output_dim * self.num_neurons + self.num_neurons * (input_dim + 1) + internal,
        )
        neural_size: tp.Tuple[int, ...] = unstructured_neural_size
        if self.greedy_bias:
            neural_size = (unstructured_neural_size[0] + 1,)
            assert "multi" not in control
            assert "structured" not in control
        assert control in CONTROLLERS or control == "conformant", f"{control} not known as a form of control"
        self.control: str = control
        if "neural" in control:
            self.first_size: int = self.num_neurons * (self.input_dim + 1)
            self.second_size: int = self.num_neurons * self.output_dim
            self.first_layer_shape: tp.Tuple[int, ...] = (self.input_dim + 1, self.num_neurons)
            self.second_layer_shape: tp.Tuple[int, ...] = (self.num_neurons, self.output_dim)
        shape_dict: tp.Dict[str, tp.Tuple[int, ...]] = {
            "conformant": (self.num_time_steps,) + output_shape,
            "stochastic_conformant": (self.num_time_steps,) + output_shape,
            "linear": (input_dim + 1, output_dim),
            "multi_neural": (min(self.num_time_steps, 50),) + unstructured_neural_size,
        }
        shape: tp.Tuple[int, ...] = tuple(map(int, shape_dict.get(control, neural_size)))
        self.policy_shape: tp.Optional[tp.Tuple[int, ...]] = shape if "structured" not in control else None

        # Create the parametrization.
        parametrization = parameter.Array(shape=shape).set_name("ng_default")
        if sparse_limit is not None:
            parametrization1 = parameter.Array(shape=shape)
            repetitions = int(np.prod(shape))
            assert isinstance(repetitions, int), f"{repetitions}"
            parametrization2 = ng.p.Choice([0, 1], repetitions=repetitions)  # type: ignore
            parametrization = ng.p.Instrumentation(  # type: ignore
                weights=parametrization1,
                enablers=parametrization2,
            )
            parametrization.set_name("ng_sparse" + str(sparse_limit))
            assert "conformant" not in control and "structured" not in control

        if "structured" in control and "neural" in control and "multi" not in control:
            parametrization = parameter.Instrumentation(  # type: ignore
                parameter.Array(shape=tuple(map(int, self.first_layer_shape))),
                parameter.Array(shape=tuple(map(int, self.second_layer_shape))),
            ).set_name("ng_struct")
        elif "conformant" in control:
            try:
                if env.action_space.low is not None and env.action_space.high is not None:
                    low = np.repeat(np.expand_dims(env.action_space.low, 0), self.num_time_steps, axis=0)
                    high = np.repeat(np.expand_dims(env.action_space.high, 0), self.num_time_steps, axis=0)
                    init = 0.5 * (low + high)
                    parametrization = parameter.Array(init=init)
                    parametrization.set_bounds(low, high)
            except AttributeError:  # Not all env.action_space have a low and a high.
                pass
            if self.subaction_type == int:
                parametrization.set_integer_casting()
            parametrization.set_name("conformant")

        # Now initializing.
        super().__init__(
            self.sparse_gym_multi_function if sparse_limit is not None else self.gym_multi_function,  # type: ignore
            parametrization=parametrization,
        )
        self.greedy_coefficient: float = 0.0
        self.parametrization.function.deterministic = False
        self.archive: tp.List[tp.Tuple[tp.List[np.ndarray], tp.List[np.ndarray], float]] = []
        self.mean_loss: float = 0.0
        self.num_losses: int = 0

    def evaluation_function(self, *recommendations: tp.Any) -> float:
        """Averages multiple evaluations if necessary."""
        if self.sparse_limit is None:  # Life is simple here, we directly have the weights.
            x = recommendations[0].value
        else:  # Here 0 in the enablers means that the weight is forced to 0.
            # assert np.prod(recommendations[0].value["weights"].shape) == np.prod(recommendations[0].value["enablers"].shape)
            weights = recommendations[0].kwargs["weights"]
            enablers = np.asarray(recommendations[0].kwargs["enablers"])
            assert all(x_ in [0, 1] for x_ in enablers), f"non-binary enablers: {enablers}."
            enablers = enablers.reshape(weights.shape)
            x = weights * enablers
        if not self.randomized:
            return self.gym_multi_function(x, limited_fidelity=False)
        # We want to reduce noise by averaging without
        # spending more than 20% of the whole experiment,
        # hence the line below:
        num = max(self.num_calls // 5, 23)
        # Pb_index >= 0 refers to the test set.
        return np.sum([self.gym_multi_function(x