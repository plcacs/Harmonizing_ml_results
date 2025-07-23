import os
import copy
import scipy.stats
import typing as tp
import numpy as np
import gym
import nevergrad as ng
from nevergrad.parametrization import parameter
from ..base import ExperimentFunction
from gym import Env
from typing import Any, Dict, List, Optional, Tuple, Union

GUARANTEED_GYM_ENV_NAMES: List[str] = [
    'CartPole-v0', 'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1',
    'CliffWalking-v0', 'Taxi-v3'
]
CONTROLLERS: List[str] = [
    'resid_neural', 'resid_semideep_neural', 'resid_deep_neural',
    'resid_scrambled_neural', 'resid_scrambled_semideep_neural',
    'resid_scrambled_deep_neural', 'resid_noisy_scrambled_neural',
    'resid_noisy_scrambled_semideep_neural', 'resid_noisy_scrambled_deep_neural',
    'linear', 'neural', 'deep_neural', 'semideep_neural', 'structured_neural',
    'memory_neural', 'deep_memory_neural', 'stackingmemory_neural',
    'deep_stackingmemory_neural','semideep_stackingmemory_neural',
    'extrapolatestackingmemory_neural','deep_extrapolatestackingmemory_neural',
    'semideep_extrapolatestackingmemory_neural','semideep_memory_neural',
    'noisy_semideep_neural','noisy_scrambled_semideep_neural','noisy_deep_neural',
    'noisy_scrambled_deep_neural','multi_neural','noisy_neural',
    'noisy_scrambled_neural','stochastic_conformant'
]
NO_LENGTH: List[str] = ['ANM', 'Blackjack', 'CliffWalking', 'Cube', 'Memorize', 'llvm']


class GymMulti(ExperimentFunction):
    """Class for converting a gym environment, a controller style, and others into a black-box optimization benchmark."""

    @staticmethod
    def get_env_names() -> List[str]:
        gym_env_names: List[str] = []
        max_displays: int = 10
        for e in gym.envs.registry.values():
            try:
                assert not any(
                    (x in str(e.id) for x in 'Kelly Copy llvm BulletEnv Minitaur Kuka InvertedPendulumSwingupBulletEnv'.split())
                )
                assert 'RacecarZedBulletEnv-v0' != e.id, 'This specific environment causes X11 error when using pybullet_envs.'
                assert 'CarRacing-v' not in str(e.id), 'Pixel based task not supported yet'
                env: Env = gym.make(e.id)
                env.reset()
                env.step(env.action_space.sample())
                a1: np.ndarray = np.asarray(env.action_space.sample())
                a2: np.ndarray = np.asarray(env.action_space.sample())
                a3: np.ndarray = np.asarray(env.action_space.sample())
                a1 = a1 + a2 + a3
                if hasattr(a1, 'size'):
                    try:
                        assert a1.size < 15000
                    except Exception:
                        assert a1.size() < 15000
                gym_env_names.append(e.id)
            except Exception as exception:
                max_displays -= 1
                if max_displays > 0:
                    print(f'{e.id} not included in full list because of {exception}.')
                if max_displays == 0:
                    print('(similar issue for other environments)')
        return gym_env_names

    controllers: List[str] = CONTROLLERS
    ng_gym: List[str] = ['CartPole-v0', 'CartPole-v1', 'Acrobot-v1']

    def wrap_env(self, input_env: Env) -> Env:
        env: Env = gym.wrappers.TimeLimit(env=input_env, max_episode_steps=self.num_episode_steps)
        return env

    def create_env(self) -> Env:
        env: Env = gym.make(self.short_name)
        try:
            env.reset()
        except:
            assert False, f'Maybe check if {self.short_name} has a problem in reset / observation.'
        return env

    def __init__(
        self,
        name: str = 'CartPole-v0',
        control: str = 'conformant',
        neural_factor: int = 1,
        randomized: bool = True,
        optimization_scale: int = 0,
        greedy_bias: bool = False,
        sparse_limit: Optional[int] = None
    ) -> None:
        self.num_calls: int = 0
        self.optimization_scale: int = optimization_scale
        self.stochastic_problem: bool = 'stoc' in name
        self.greedy_bias: bool = greedy_bias
        self.sparse_limit: Optional[int] = sparse_limit
        if 'conformant' in control or control == 'linear':
            assert neural_factor is None
        if os.name == 'nt':
            raise ng.errors.UnsupportedExperiment('Windows is not supported')
        self.short_name: str = name
        env: Env = self.create_env()
        self.name: str = name + '__' + control + '__' + str(neural_factor)
        if sparse_limit is not None:
            self.name += f'__{sparse_limit}'
        if randomized:
            self.name += '_unseeded'
        self.randomized: bool = randomized
        try:
            try:
                self.num_time_steps: int = env._max_episode_steps
            except AttributeError:
                self.num_time_steps: int = env.horizon
        except AttributeError:
            assert any((x in name for x in NO_LENGTH)), name
            if 'LANM' not in name:
                self.num_time_steps = 200 if control == 'conformant' else 5000
            else:
                self.num_time_steps = 3000
        self.gamma: float = 0.995 if 'LANM' in name else 1.0
        self.neural_factor: int = neural_factor
        self.arities: List[np.ndarray] = []
        if isinstance(env.action_space, gym.spaces.Tuple):
            assert all(
                isinstance(p, gym.spaces.MultiDiscrete) for p in env.action_space
            ), f'{name} has a too complicated structure.'
            self.arities = [p.nvec for p in env.action_space]
            if control == 'conformant':
                output_dim: int = sum(len(a) for a in self.arities)
            else:
                output_dim = sum(sum(a) for a in self.arities)
            output_shape: Tuple[int, ...] = (output_dim,)
            discrete: bool = True
        elif isinstance(env.action_space, gym.spaces.Discrete):
            output_dim: int = env.action_space.n
            output_shape: Tuple[int, ...] = (output_dim,)
            discrete: bool = True
            assert output_dim is not None, env.action_space.n
        else:
            output_shape = env.action_space.shape
            if output_shape is None:
                output_shape = tuple(np.asarray(env.action_space.sample()).shape)
            discrete = False
            output_dim = int(np.prod(output_shape))
        self.discrete: bool = discrete
        assert env.observation_space is not None or 'llvm' in name, 'An observation space should be defined.'
        if env.observation_space is not None and env.observation_space.dtype == int:
            input_dim: int = env.observation_space.n
            assert input_dim is not None, env.observation_space.n
            self.discrete_input: bool = True
        else:
            input_dim = int(np.prod(env.observation_space.shape)) if env.observation_space is not None else 0
            if input_dim is None:
                o = env.reset()
                input_dim = int(np.prod(np.asarray(o).shape))
            self.discrete_input = False
        a = env.action_space.sample()
        self.action_type: type = type(a)
        self.subaction_type: Optional[type] = None
        if hasattr(a, '__iter__'):
            self.subaction_type = type(a[0])
        if neural_factor is None:
            assert control == 'linear' or 'conformant' in control, f'{control} has neural_factor {neural_factor}'
            neural_factor = 1
        self.output_shape: Tuple[int, ...] = output_shape
        self.num_stacking: int = 1
        self.memory_len: int = neural_factor * input_dim if 'memory' in control else 0
        self.extended_input_len: int = (input_dim + output_dim) * self.num_stacking if 'stacking' in control else 0
        input_dim = input_dim + self.memory_len + self.extended_input_len
        self.extended_input: np.ndarray = np.zeros(self.extended_input_len)
        output_dim = output_dim + self.memory_len
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.num_neurons: int = neural_factor * (input_dim - self.extended_input_len)
        self.num_internal_layers: int = 1 if 'semi' in control else 3
        internal: int = self.num_internal_layers * self.num_neurons ** 2 if 'deep' in control else 0
        unstructured_neural_size: Tuple[int, ...] = (
            output_dim * self.num_neurons + self.num_neurons * (input_dim + 1) + internal,
        )
        neural_size: Tuple[int, ...] = unstructured_neural_size
        if self.greedy_bias:
            neural_size = (unstructured_neural_size[0] + 1,)
            assert 'multi' not in control
            assert 'structured' not in control
        assert control in CONTROLLERS or control == 'conformant', f'{control} not known as a form of control'
        self.control: str = control
        if 'neural' in control:
            self.first_size: int = self.num_neurons * (self.input_dim + 1)
            self.second_size: int = self.num_neurons * self.output_dim
            self.first_layer_shape: Tuple[int, ...] = (self.input_dim + 1, self.num_neurons)
            self.second_layer_shape: Tuple[int, ...] = (self.num_neurons, self.output_dim)
        shape_dict: Dict[str, Tuple[int, ...]] = {
            'conformant': (self.num_time_steps,) + output_shape,
            'stochastic_conformant': (self.num_time_steps,) + output_shape,
            'linear': (input_dim + 1, output_dim),
            'multi_neural': (min(self.num_time_steps, 50),) + unstructured_neural_size
        }
        shape: Tuple[int, ...] = tuple(map(int, shape_dict.get(control, neural_size)))
        self.policy_shape: Optional[Tuple[int, ...]] = shape if 'structured' not in control else None
        parametrization: parameter.Parameter = parameter.Array(shape=shape).set_name('ng_default')
        if sparse_limit is not None:
            parametrization1: parameter.Parameter = parameter.Array(shape=shape)
            repetitions: int = int(np.prod(shape))
            assert isinstance(repetitions, int), f'{repetitions}'
            parametrization2: parameter.Parameter = ng.p.Choice([0, 1], repetitions=repetitions)
            parametrization = ng.p.Instrumentation(weights=parametrization1, enablers=parametrization2)
            parametrization.set_name('ng_sparse' + str(sparse_limit))
            assert 'conformant' not in control and 'structured' not in control
        if 'structured' in control and 'neural' in control and ('multi' not in control):
            parametrization = parameter.Instrumentation(
                parameter.Array(shape=tuple(map(int, self.first_layer_shape))),
                parameter.Array(shape=tuple(map(int, self.second_layer_shape)))
            ).set_name('ng_struct')
        elif 'conformant' in control:
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
            parametrization.set_name('conformant')
        super().__init__(
            self.sparse_gym_multi_function if sparse_limit is not None else self.gym_multi_function,
            parametrization=parametrization
        )
        self.greedy_coefficient: float = 0.0
        self.parametrization.function.deterministic = False
        self.archive: List[Tuple[List[np.ndarray], List[np.ndarray], float]] = []
        self.mean_loss: float = 0.0
        self.num_losses: int = 0

    def evaluation_function(self, *recommendations: ng.optimizers.recommendations.Recommendation) -> float:
        """Averages multiple evaluations if necessary."""
        if self.sparse_limit is None:
            x: np.ndarray = recommendations[0].value
        else:
            weights: np.ndarray = recommendations[0].kwargs['weights']
            enablers: np.ndarray = np.asarray(recommendations[0].kwargs['enablers'])
            assert all((x_ in [0, 1] for x_ in enablers)), f'non-binary enablers: {enablers}.'
            enablers = enablers.reshape(weights.shape)
            x = weights * enablers
        if not self.randomized:
            return self.gym_multi_function(x, limited_fidelity=False)
        num: int = max(self.num_calls // 5, 23)
        return float(np.sum([
            self.gym_multi_function(x, limited_fidelity=False) for _ in range(num)
        ]) / num)

    def softmax(self, a: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(a, copy=False, nan=-1e+20, posinf=1e+20, neginf=-1e+20)
        probabilities = np.exp(a - np.max(a))
        probabilities = probabilities / np.sum(probabilities)
        return probabilities

    def discretize(self, a: np.ndarray, env: Env) -> Union[int, np.ndarray]:
        """Transforms a logit into an int obtained through softmax."""
        if len(self.arities) > 0:
            if self.control == 'conformant':
                index: int = 0
                output: List[List[int]] = []
                for arities in self.arities:
                    local_output: List[int] = []
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
                        logits = a[index:index + arity]
                        probabilities = self.softmax(logits)
                        sampled: int = int(list(np.random.multinomial(1, probabilities)).index(1))
                        local_output += [sampled]
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
        probabilities = self.softmax(a)
        assert float(np.sum(probabilities)) <= 1.0 + 1e-07, f'{probabilities} with greediness {self.greedy_coefficient}.'
        sampled_action: int = int(list(np.random.multinomial(1, probabilities)).index(1))
        return sampled_action

    def neural(self, x: np.ndarray, o: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies a neural net parametrized by x to an observation o. Returns an action or logits of actions."""
        if self.greedy_bias:
            assert 'multi' not in self.control
            assert 'structured' not in self.control
            self.greedy_coefficient = float(x[-1:])
            x = x[:-1]
        o = o.ravel()
        my_scale: float = 2 ** self.optimization_scale
        if 'structured' not in self.name and self.optimization_scale != 0:
            x = np.asarray(my_scale * x, dtype=np.float32)
        if self.control == 'linear':
            output = np.matmul(o, x[1:, :])
            output += x[0]
            return (output.reshape(self.output_shape), np.zeros(0))
        if 'structured' not in self.control:
            first_matrix: np.ndarray = x[:self.first_size].reshape(self.first_layer_shape) / np.sqrt(len(o))
            second_matrix: np.ndarray = x[self.first_size:self.first_size + self.second_size].reshape(self.second_layer_shape) / np.sqrt(self.num_neurons)
        else:
            assert len(x) == 2
            first_matrix = np.asarray(x[0][0])
            second_matrix = np.asarray(x[0][1])
            assert first_matrix.shape == self.first_layer_shape, f'{first_matrix} does not match {self.first_layer_shape}'
            assert second_matrix.shape == self.second_layer_shape, f'{second_matrix} does not match {self.second_layer_shape}'
        if 'resid' in self.control:
            first_matrix += my_scale * np.eye(*first_matrix.shape)
            second_matrix += my_scale * np.eye(*second_matrix.shape)
        assert len(o) == len(first_matrix[1:]), f'{o.shape} coming in matrix of shape {first_matrix.shape}'
        output = np.matmul(o, first_matrix[1:])
        if 'deep' in self.control:
            current_index: int = self.first_size + self.second_size
            internal_layer_size: int = self.num_neurons ** 2
            s: Tuple[int, ...] = (self.num_neurons, self.num_neurons)
            for _ in range(self.num_internal_layers):
                output = np.tanh(output)
                layer: np.ndarray = x[current_index:current_index + internal_layer_size].reshape(s)
                if 'resid' in self.control:
                    layer += my_scale * np.eye(*layer.shape)
                output = np.matmul(output, layer) / np.sqrt(self.num_neurons)
                current_index += internal_layer_size
            assert current_index == len(x)
        output = np.matmul(np.tanh(output + first_matrix[0]), second_matrix)
        return (
            output[self.memory_len:].reshape(self.output_shape),
            output[:self.memory_len]
        )

    def sparse_gym_multi_function(
        self,
        weights: np.ndarray,
        enablers: np.ndarray,
        limited_fidelity: bool = False
    ) -> float:
        assert all((x_ in [0, 1] for x_ in enablers))
        x = weights * enablers
        loss: float = self.gym_multi_function(x, limited_fidelity=limited_fidelity)
        sparse_penalty: float = 0.0
        if self.sparse_limit is not None:
            sparse_penalty = (1 + np.abs(loss)) * max(0, int(np.sum(enablers)) - self.sparse_limit)
        return loss + sparse_penalty

    def gym_multi_function(self, x: np.ndarray, limited_fidelity: bool = False) -> float:
        """Do a simulation with parametrization x and return the result.

        Parameters:
            limited_fidelity: bool
                whether we use a limited version for the beginning of the training.
        """
        self.num_calls += 1
        num_simulations: int = 7 if self.control != 'conformant' and (not self.randomized) else 1
        loss: float = 0.0
        for simulation_index in range(num_simulations):
            seed: int = simulation_index if not self.randomized else self.parametrization.random_state.randint(500000)
            loss += self.gym_simulate(x, seed=seed, limited_fidelity=limited_fidelity)
        return loss / num_simulations

    def action_cast(self, a: Any, env: Env) -> Any:
        """Transforms an action into an action of type as expected by the gym step function."""
        if self.action_type == tuple:
            a = self.discretize(a, env)
            return tuple(a)
        if isinstance(a, np.float64):
            a = np.asarray((a,))
        if self.discrete:
            a = self.discretize(a, env)
        else:
            if not isinstance(a, self.action_type):
                a = self.action_type(a)
            try:
                if env.action_space.low is not None and env.action_space.high is not None:
                    a = 0.5 * (1.0 + np.tanh(a))
                    a = env.action_space.low + (env.action_space.high - env.action_space.low) * a
            except AttributeError:
                pass
            if self.subaction_type is not None:
                if isinstance(a, tuple):
                    a = tuple(int(_a + 0.5) for _a in a)
                else:
                    a = np.array([self.subaction_type(_a) for _a in a])
        if not np.isscalar(a):
            a = np.asarray(a, dtype=env.action_space.sample().dtype)
        assert isinstance(a, self.action_type), f'{a} should have type {self.action_type} '
        try:
            assert env.action_space.contains(a), (
                f'In {self.name}, high={env.action_space.high} low={env.action_space.low} '
                f'{a} is not sufficiently close to {[env.action_space.sample() for _ in range(10)]}'
                f'Action space = {env.action_space} '
                f'(sample has type {type(env.action_space.sample())})and a={a} '
                f'with type {type(a)}'
            )
        except AttributeError:
            pass
        return a

    def step(self, a: Any, env: Env) -> Tuple[Any, float, bool, Dict[str, Any]]:
        """Apply an action.

        We have a step on top of Gym's step for possibly storing some statistics."""
        o, r, done, info = env.step(a)
        return (o, r, done, info)

    def heuristic(
        self,
        o: Any,
        current_observations: List[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Returns a heuristic action in the given context.

        Parameters:
           o: gym observation
               new observation
           current_observations: list of gym observations
               current observations up to the present time step.
        """
        current_observations = np.asarray(current_observations + [o], dtype=np.float32)
        self.archive = [
            self.archive[i] for i in range(len(self.archive))
            if self.archive[i][2] <= self.mean_loss
        ]
        self.archive = sorted(self.archive, key=lambda trace: -len(trace[0]))
        for trace in self.archive:
            to, ta, _ = trace
            assert len(to) == len(ta)
            if len(current_observations) > len(to) and 'extrapolate' not in self.control:
                continue
            to_slice = np.asarray(to[-len(current_observations):], dtype=np.float32)
            if np.array_equal(to_slice, current_observations):
                return np.asarray(ta[len(current_observations) - 1], dtype=np.float32)
        return None

    def gym_simulate(
        self,
        x: np.ndarray,
        seed: int,
        limited_fidelity: bool = True
    ) -> float:
        """Single simulation with parametrization x."""
        current_time_index: int = 0
        current_reward: float = 0.0
        current_observations: List[np.ndarray] = []
        current_actions: List[np.ndarray] = []
        try:
            if self.policy_shape is not None:
                x = x.reshape(self.policy_shape)
        except:
            assert False, f'x has shape {x.shape} and needs {self.policy_shape} for control {self.control}'
        assert seed == 0 or self.control != 'conformant' or self.randomized
        env: Env = self.create_env()
        env.seed(seed=seed)
        o: Any = env.reset()
        control: str = self.control
        if 'conformant' in control:
            return self.gym_conformant(x, env)
        if 'scrambled' in control:
            x = x.copy()
            np.random.RandomState(1234).shuffle(x)
        if 'noisy' in control:
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
                f'o has shape {o.shape} whereas input_dim={self.input_dim} ({control} / {env} {self.name})'
            )
            if 'multi' in control:
                policy_input = x[i % len(x)]
            else:
                policy_input = x
            a, memory = self.neural(policy_input, o)
            a = self.action_cast(a, env)
            try:
                o, r, done, _ = self.step(a, env)
                current_time_index += 1
                if 'multifidLANM' in self.name and current_time_index > 500 and limited_fidelity:
                    done = True
                current_reward *= self.gamma
                current_reward += r
                current_observations.append(np.asarray(o).copy())
                current_actions.append(np.asarray(a).copy())
                if done and 'stacking' in self.control:
                    self.archive_observations(current_actions, current_observations, current_reward)
            except AssertionError:
                return 1e+20 / (1.0 + i)
            if 'stacking' in control:
                attention_a = self.heuristic(o, current_observations)
                a = attention_a if attention_a is not None else 0.0 * np.asarray(a)
                previous_o = previous_o.ravel()
                additional_input = np.concatenate([np.asarray(a).ravel(), previous_o])
                shift: int = len(additional_input)
                self.extended_input[:len(self.extended_input) - shift] = self.extended_input[shift:]
                self.extended_input[len(self.extended_input) - shift:] = additional_input
            reward += r
            if done:
                break
        return -reward

    def gym_conformant(self, x: np.ndarray, env: Env) -> float:
        """Conformant: we directly optimize inputs, not parameters of a policy."""
        reward: float = 0.0
        for _, a in enumerate(x):
            a_casted = self.action_cast(a, env)
            _, r, done, _ = self.step(a_casted, env)
            reward *= self.gamma
            reward += r
            if done:
                break
        return -reward

    def archive_observations(
        self,
        current_actions: List[np.ndarray],
        current_observations: List[np.ndarray],
        current_reward: float
    ) -> None:
        self.num_losses += 1
        tau: float = 1.0 / self.num_losses
        self.mean_loss = (1.0 - tau) * self.mean_loss + tau * current_reward if self.mean_loss is not None else current_reward
        found: bool = False
        for trace in self.archive:
            to, _, _ = trace
            if np.array_equal(np.asarray(current_observations, dtype=np.float32), np.asarray(to, dtype=np.float32)):
                found = True
                break
        if not found:
            self.archive.append((current_observations, current_actions, current_reward))
