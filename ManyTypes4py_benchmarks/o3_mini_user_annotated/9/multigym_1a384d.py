import os
import copy
import scipy.stats
import typing as tp
import numpy as np
import gym

import nevergrad as ng

from nevergrad.parametrization import parameter
from ..base import ExperimentFunction

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
        gym_env_names: tp.List[str] = []
        max_displays: int = 10
        for e in gym.envs.registry.values():
            try:
                assert not any(
                    x in str(e.id)
                    for x in "Kelly Copy llvm BulletEnv Minitaur Kuka InvertedPendulumSwingupBulletEnv".split()
                )
                assert (
                    "RacecarZedBulletEnv-v0" != e.id
                ), "This specific environment causes X11 error when using pybullet_envs."
                assert "CarRacing-v" not in str(e.id), "Pixel based task not supported yet"
                env: gym.Env = gym.make(e.id)
                env.reset()
                env.step(env.action_space.sample())
                a1: np.ndarray = np.asarray(env.action_space.sample())
                a2: np.ndarray = np.asarray(env.action_space.sample())
                a3: np.ndarray = np.asarray(env.action_space.sample())
                a1 = a1 + a2 + a3
                if hasattr(a1, "size"):
                    try:
                        assert a1.size < 15000
                    except Exception:
                        assert a1.size() < 15000  # type: ignore
                gym_env_names.append(e.id)
            except Exception as exception:
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
        env: gym.Env = gym.wrappers.TimeLimit(
            env=input_env,
            max_episode_steps=self.num_episode_steps,  # type: ignore
        )
        return env

    def create_env(self) -> gym.Env:
        env: gym.Env = gym.make(self.short_name)
        try:
            env.reset()
        except Exception:
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
        sparse_limit: tp.Optional[int] = None,
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
        self.short_name: str = name
        env: gym.Env = self.create_env()
        self.name: str = name + "__" + control + "__" + str(neural_factor)
        if sparse_limit is not None:
            self.name += f"__{sparse_limit}"
        if randomized:
            self.name += "_unseeded"
        self.randomized: bool = randomized
        try:
            try:
                self.num_time_steps: int = env._max_episode_steps  # type: ignore
            except AttributeError:
                self.num_time_steps = env.horizon  # type: ignore
        except AttributeError:
            assert any(x in name for x in NO_LENGTH), name
            if "LANM" not in name:
                self.num_time_steps = 200 if control == "conformant" else 5000
            else:
                self.num_time_steps = 3000
        self.gamma: float = 0.995 if "LANM" in name else 1.0
        self.neural_factor: tp.Optional[int] = neural_factor

        self.arities: tp.List = []
        if isinstance(env.action_space, gym.spaces.Tuple):
            assert all(
                isinstance(p, gym.spaces.MultiDiscrete) for p in env.action_space
            ), f"{name} has a too complicated structure."
            self.arities = [p.nvec for p in env.action_space]
            if control == "conformant":
                output_dim = sum(len(a) for a in self.arities)
            else:
                output_dim = sum(sum(a) for a in self.arities)
            output_shape = (output_dim,)
            discrete = True
        elif isinstance(env.action_space, gym.spaces.Discrete):
            output_dim = env.action_space.n
            output_shape = (output_dim,)
            discrete = True
            assert output_dim is not None, env.action_space.n
        else:  # Continuous action space
            output_shape = env.action_space.shape
            if output_shape is None:
                output_shape = tuple(np.asarray(env.action_space.sample()).shape)
            discrete = False
            output_dim = np.prod(output_shape)
        self.discrete: bool = discrete

        assert env.observation_space is not None or "llvm" in name, "An observation space should be defined."
        if env.observation_space is not None and env.observation_space.dtype == int:
            input_dim = env.observation_space.n
            assert input_dim is not None, env.observation_space.n
            self.discrete_input: bool = True
        else:
            input_dim = np.prod(env.observation_space.shape) if env.observation_space is not None else 0
            if input_dim is None:
                o_tmp = env.reset()
                input_dim = np.prod(np.asarray(o_tmp).shape)
            self.discrete_input = False

        a_sample = env.action_space.sample()
        self.action_type = type(a_sample)
        self.subaction_type: tp.Optional[type] = None
        if hasattr(a_sample, "__iter__"):
            self.subaction_type = type(a_sample[0])

        if neural_factor is None:
            assert (
                control == "linear" or "conformant" in control
            ), f"{control} has neural_factor {neural_factor}"
            neural_factor = 1
        self.output_shape = output_shape
        self.num_stacking: int = 1
        self.memory_len: int = neural_factor * input_dim if "memory" in control else 0  # type: ignore
        self.extended_input_len: int = (input_dim + output_dim) * self.num_stacking if "stacking" in control else 0
        input_dim = input_dim + self.memory_len + self.extended_input_len
        self.extended_input: np.ndarray = np.zeros(self.extended_input_len)
        output_dim = output_dim + self.memory_len
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.num_neurons: int = neural_factor * (input_dim - self.extended_input_len)  # type: ignore
        self.num_internal_layers: int = 1 if "semi" in control else 3
        internal = self.num_internal_layers * (self.num_neurons**2) if "deep" in control else 0
        unstructured_neural_size = (
            output_dim * self.num_neurons + self.num_neurons * (input_dim + 1) + internal,
        )
        neural_size = unstructured_neural_size
        if self.greedy_bias:
            neural_size = (unstructured_neural_size[0] + 1,)
            assert "multi" not in control
            assert "structured" not in control
        assert control in CONTROLLERS or control == "conformant", f"{control} not known as a form of control"
        self.control: str = control
        if "neural" in control:
            self.first_size: int = self.num_neurons * (self.input_dim + 1)
            self.second_size: int = self.num_neurons * self.output_dim
            self.first_layer_shape: tp.Tuple[int, int] = (self.input_dim + 1, self.num_neurons)
            self.second_layer_shape: tp.Tuple[int, int] = (self.num_neurons, self.output_dim)
        shape_dict: dict = {
            "conformant": (self.num_time_steps,) + output_shape,
            "stochastic_conformant": (self.num_time_steps,) + output_shape,
            "linear": (input_dim + 1, output_dim),
            "multi_neural": (min(self.num_time_steps, 50),) + unstructured_neural_size,
        }
        shape: tp.Tuple[int, ...] = tuple(map(int, shape_dict.get(control, neural_size)))
        self.policy_shape: tp.Optional[tp.Tuple[int, ...]] = shape if "structured" not in control else None

        parametrization: tp.Any = parameter.Array(shape=shape).set_name("ng_default")
        if sparse_limit is not None:
            parametrization1: tp.Any = parameter.Array(shape=shape)
            repetitions: int = int(np.prod(shape))
            assert isinstance(repetitions, int), f"{repetitions}"
            parametrization2: tp.Any = ng.p.Choice([0, 1], repetitions=repetitions)  # type: ignore
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
                    low: np.ndarray = np.repeat(np.expand_dims(env.action_space.low, 0), self.num_time_steps, axis=0)
                    high: np.ndarray = np.repeat(np.expand_dims(env.action_space.high, 0), self.num_time_steps, axis=0)
                    init: np.ndarray = 0.5 * (low + high)
                    parametrization = parameter.Array(init=init)
                    parametrization.set_bounds(low, high)
            except AttributeError:
                pass
            if self.subaction_type == int:
                parametrization.set_integer_casting()
            parametrization.set_name("conformant")

        super().__init__(
            self.sparse_gym_multi_function if sparse_limit is not None else self.gym_multi_function,
            parametrization=parametrization,
        )
        self.greedy_coefficient: float = 0.0
        self.parametrization.function.deterministic = False
        self.archive: tp.List[tp.Any] = []
        self.mean_loss: float = 0.0
        self.num_losses: int = 0

    def evaluation_function(self, *recommendations: tp.Any) -> float:
        if self.sparse_limit is None:
            x: np.ndarray = recommendations[0].value
        else:
            weights: np.ndarray = recommendations[0].kwargs["weights"]
            enablers: np.ndarray = np.asarray(recommendations[0].kwargs["enablers"])
            assert all(x_ in [0, 1] for x_ in enablers), f"non-binary enablers: {enablers}."
            enablers = enablers.reshape(weights.shape)
            x = weights * enablers
        if not self.randomized:
            return self.gym_multi_function(x, limited_fidelity=False)
        num: int = max(self.num_calls // 5, 23)
        return np.sum([self.gym_multi_function(x, limited_fidelity=False) for _ in range(num)]) / num

    def softmax(self, a: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(a, copy=False, nan=-1e20, posinf=1e20, neginf=-1e20)
        probabilities: np.ndarray = np.exp(a - max(a))
        probabilities = probabilities / sum(probabilities)
        return probabilities

    def discretize(self, a: np.ndarray, env: gym.Env) -> tp.Any:
        if len(self.arities) > 0:
            if self.control == "conformant":
                index: int = 0
                output: tp.List[tp.Any] = []
                for arities in self.arities:
                    local_output: tp.List[int] = []
                    for arity in arities:
                        local_output += [min(int(scipy.stats.norm.cdf(a[index]) * arity), arity - 1)]
                        index += 1
                    output += [local_output]
                assert index == len(a)
                return np.array(output)
            else:
                index = 0
                output = []
                for arities in self.arities:
                    local_output = []
                    for arity in arities:
                        local_output += [
                            int(
                                list(np.random.multinomial(1, self.softmax(a[index: index + arity]))).index(1)
                            )
                        ]
                        index += arity
                    output += [local_output]
                assert index == len(a)
                return np.array(output)

        if self.greedy_bias:
            a = np.asarray(a, dtype=np.float32)
            for i, action in enumerate(range(len(a))):
                tmp_env = copy.deepcopy(env)
                _, r, _, _ = tmp_env.step(action)
                a[i] += self.greedy_coefficient * r
        probabilities: np.ndarray = self.softmax(a)
        assert sum(probabilities) <= 1.0 + 1e-7, f"{probabilities} with greediness {self.greedy_coefficient}."
        return int(list(np.random.multinomial(1, probabilities)).index(1))

    def neural(self, x: np.ndarray, o: np.ndarray) -> tp.Tuple[np.ndarray, np.ndarray]:
        if self.greedy_bias:
            assert "multi" not in self.control
            assert "structured" not in self.control
            self.greedy_coefficient = x[-1:]
            x = x[:-1]
        o = o.ravel()
        my_scale: float = 2**self.optimization_scale
        if "structured" not in self.name and self.optimization_scale != 0:
            x = np.asarray(my_scale * x, dtype=np.float32)
        if self.control == "linear":
            output = np.matmul(o, x[1:, :])
            output += x[0]
            return output.reshape(self.output_shape), np.zeros(0)
        if "structured" not in self.control:
            first_matrix: np.ndarray = x[: self.first_size].reshape(self.first_layer_shape) / np.sqrt(len(o))
            second_matrix: np.ndarray = x[self.first_size: (self.first_size + self.second_size)].reshape(self.second_layer_shape) / np.sqrt(self.num_neurons)
        else:
            assert len(x) == 2
            first_matrix = np.asarray(x[0][0])
            second_matrix = np.asarray(x[0][1])
            assert first_matrix.shape == self.first_layer_shape, f"{first_matrix} does not match {self.first_layer_shape}"
            assert second_matrix.shape == self.second_layer_shape, f"{second_matrix} does not match {self.second_layer_shape}"
        if "resid" in self.control:
            first_matrix += my_scale * np.eye(*first_matrix.shape)
            second_matrix += my_scale * np.eye(*second_matrix.shape)
        assert len(o) == len(first_matrix[1:]), f"{o.shape} coming in matrix of shape {first_matrix.shape}"
        output = np.matmul(o, first_matrix[1:])
        if "deep" in self.control:
            current_index: int = self.first_size + self.second_size
            internal_layer_size: int = self.num_neurons**2
            s: tp.Tuple[int, int] = (self.num_neurons, self.num_neurons)
            for _ in range(self.num_internal_layers):
                output = np.tanh(output)
                layer = x[current_index: current_index + internal_layer_size].reshape(s)
                if "resid" in self.control:
                    layer += my_scale * np.eye(*layer.shape)
                output = np.matmul(output, layer) / np.sqrt(self.num_neurons)
                current_index += internal_layer_size
            assert current_index == len(x)
        output = np.matmul(np.tanh(output + first_matrix[0]), second_matrix)
        return output[self.memory_len :].reshape(self.output_shape), output[: self.memory_len]

    def sparse_gym_multi_function(
        self,
        weights: np.ndarray,
        enablers: np.ndarray,
        limited_fidelity: bool = False,
    ) -> float:
        assert all(x_ in [0, 1] for x_ in enablers)
        x: np.ndarray = weights * enablers
        loss: float = self.gym_multi_function(x, limited_fidelity=limited_fidelity)
        sparse_penalty: float = 0.0
        if self.sparse_limit is not None:
            sparse_penalty = (1 + np.abs(loss)) * max(0, np.sum(enablers) - self.sparse_limit)
        return loss + sparse_penalty

    def gym_multi_function(self, x: np.ndarray, limited_fidelity: bool = False) -> float:
        self.num_calls += 1
        num_simulations: int = 7 if self.control != "conformant" and not self.randomized else 1
        loss: float = 0.0
        for simulation_index in range(num_simulations):
            loss += self.gym_simulate(
                x,
                seed=(
                    simulation_index
                    if not self.randomized
                    else self.parametrization.random_state.randint(500000)
                ),
                limited_fidelity=limited_fidelity,
            )
        return loss / num_simulations

    def action_cast(self, a: tp.Any, env: gym.Env) -> tp.Any:
        if self.action_type == tuple:
            a_cast = self.discretize(a, env)
            return tuple(a_cast)
        if type(a) == np.float64:
            a = np.asarray((a,))
        if self.discrete:
            a = self.discretize(a, env)
        else:
            if type(a) != self.action_type:
                a = self.action_type(a)
            try:
                if env.action_space.low is not None and env.action_space.high is not None:
                    a = 0.5 * (1.0 + np.tanh(a))
                    a = env.action_space.low + (env.action_space.high - env.action_space.low) * a
            except AttributeError:
                pass
            if self.subaction_type is not None:
                if type(a) == tuple:
                    a = tuple(int(_a + 0.5) for _a in a)
                else:
                    for i in range(len(a)):
                        a[i] = self.subaction_type(a[i])
        if not np.isscalar(a):
            a = np.asarray(a, dtype=env.action_space.sample().dtype)
        assert type(a) == self.action_type, f"{a} should have type {self.action_type} "
        try:
            assert env.action_space.contains(a), (
                f"In {self.name}, high={env.action_space.high} low={env.action_space.low} {a} "
                f"is not sufficiently close to {[env.action_space.sample() for _ in range(10)]}"
                f"Action space = {env.action_space} (sample has type {type(env.action_space.sample())})"
                f"and a={a} with type {type(a)}"
            )
        except AttributeError:
            pass
        return a

    def step(self, a: tp.Any, env: gym.Env) -> tp.Tuple[np.ndarray, float, bool, tp.Any]:
        o, r, done, info = env.step(a)
        return o, r, done, info

    def heuristic(self, o: np.ndarray, current_observations: tp.List[np.ndarray]) -> tp.Optional[np.ndarray]:
        current_observations_arr = np.asarray(current_observations + [o], dtype=np.float32)
        self.archive = [
            self.archive[i] for i in range(len(self.archive)) if self.archive[i][2] <= self.mean_loss
        ]
        self.archive = sorted(self.archive, key=lambda trace: -len(trace[0]))
        for trace in self.archive:
            to, ta, _ = trace
            assert len(to) == len(ta)
            if len(current_observations_arr) > len(to) and "extrapolate" not in self.control:
                continue
            to_arr = np.asarray(to[(-len(current_observations_arr)) :], dtype=np.float32)
            if np.array_equal(to_arr, current_observations_arr):
                return np.asarray(ta[len(current_observations_arr) - 1], dtype=np.float32)
        return None

    def gym_simulate(
        self,
        x: np.ndarray,
        seed: int,
        limited_fidelity: bool = True,
    ) -> float:
        current_time_index: int = 0
        current_reward: float = 0.0
        current_observations: tp.List[np.ndarray] = []
        current_actions: tp.List[np.ndarray] = []
        try:
            if self.policy_shape is not None:
                x = x.reshape(self.policy_shape)
        except Exception:
            assert False, f"x has shape {x.shape} and needs {self.policy_shape} for control {self.control}"
        assert seed == 0 or self.control != "conformant" or self.randomized
        env: gym.Env = self.create_env()
        env.seed(seed=seed)
        o: np.ndarray = env.reset()
        control: str = self.control
        if "conformant" in control:
            return self.gym_conformant(x, env)
        if "scrambled" in control:
            x = x.copy()
            np.random.RandomState(1234).shuffle(x)
        if "noisy" in control:
            x = x + 0.01 * np.random.RandomState(1234).normal(size=x.shape)
        reward: float = 0.0
        memory: np.ndarray = np.zeros(self.memory_len)
        for i in range(self.num_time_steps):
            if self.discrete_input:
                obs = np.zeros(shape=self.input_dim - self.extended_input_len - len(memory))
                obs[o] = 1
                o = obs
            previous_o: np.ndarray = np.asarray(o)
            o = np.concatenate([previous_o.ravel(), memory.ravel(), self.extended_input])
            assert len(o) == self.input_dim, (
                f"o has shape {o.shape} whereas input_dim={self.input_dim} "
                f"({control} / {env} {self.name})"
            )
            a_temp, memory = self.neural(x[i % len(x)] if "multi" in control else x, o)
            a_cast = self.action_cast(a_temp, env)
            try:
                o, r, done, _ = self.step(a_cast, env)
                current_time_index += 1
                if ("multifidLANM" in self.name) and current_time_index > 500 and limited_fidelity:
                    done = True
                current_reward *= self.gamma
                current_reward += r
                current_observations += [np.asarray(o).copy()]
                current_actions += [np.asarray(a_cast).copy()]
                if done and "stacking" in self.control:
                    self.archive_observations(current_actions, current_observations, current_reward)
            except AssertionError:
                return 1e20 / (1.0 + i)
            if "stacking" in control:
                attention_a = self.heuristic(o, current_observations)
                a_final = attention_a if attention_a is not None else 0.0 * np.asarray(a_temp)
                previous_o = previous_o.ravel()
                additional_input = np.concatenate([np.asarray(a_final).ravel(), previous_o])
                shift = len(additional_input)
                self.extended_input[: (len(self.extended_input) - shift)] = self.extended_input[shift:]
                self.extended_input[(len(self.extended_input) - shift) :] = additional_input
            reward += r
            if done:
                break
        return -reward

    def gym_conformant(self, x: np.ndarray, env: gym.Env) -> float:
        reward: float = 0.0
        for _, a in enumerate(x):
            a_cast = self.action_cast(a, env)
            _, r, done, _ = self.step(a_cast, env)
            reward *= self.gamma
            reward += r
            if done:
                break
        return -reward

    def archive_observations(self, current_actions: tp.List[np.ndarray], current_observations: tp.List[np.ndarray], current_reward: float) -> None:
        self.num_losses += 1
        tau: float = 1.0 / self.num_losses
        self.mean_loss = (
            ((1.0 - tau) * self.mean_loss + tau * current_reward)
            if self.mean_loss is not None
            else current_reward
        )
        found: bool = False
        for trace in self.archive:
            to, _, _ = trace
            if np.array_equal(
                np.asarray(current_observations, dtype=np.float32),
                np.asarray(to, dtype=np.float32),
            ):
                found = True
                break
        if not found:
            self.archive += [(current_observations, current_actions, current_reward)]