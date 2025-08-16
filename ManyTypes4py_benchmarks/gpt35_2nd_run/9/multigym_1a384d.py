import os
import copy
import scipy.stats
from typing import List, Tuple
import numpy as np
import gym
import nevergrad as ng
from nevergrad.parametrization import parameter

GUARANTEED_GYM_ENV_NAMES: List[str] = ['CartPole-v0', 'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'CliffWalking-v0', 'Taxi-v3']
CONTROLLERS: List[str] = ['resid_neural', 'resid_semideep_neural', 'resid_deep_neural', 'resid_scrambled_neural', 'resid_scrambled_semideep_neural', 'resid_scrambled_deep_neural', 'resid_noisy_scrambled_neural', 'resid_noisy_scrambled_semideep_neural', 'resid_noisy_scrambled_deep_neural', 'linear', 'neural', 'deep_neural', 'semideep_neural', 'structured_neural', 'memory_neural', 'deep_memory_neural', 'stackingmemory_neural', 'deep_stackingmemory_neural', 'semideep_stackingmemory_neural', 'extrapolatestackingmemory_neural', 'deep_extrapolatestackingmemory_neural', 'semideep_extrapolatestackingmemory_neural', 'semideep_memory_neural', 'noisy_semideep_neural', 'noisy_scrambled_semideep_neural', 'noisy_deep_neural', 'noisy_scrambled_deep_neural', 'multi_neural', 'noisy_neural', 'noisy_scrambled_neural', 'stochastic_conformant']
NO_LENGTH: List[str] = ['ANM', 'Blackjack', 'CliffWalking', 'Cube', 'Memorize', 'llvm']

class GymMulti(ExperimentFunction):
    """Class for converting a gym environment, a controller style, and others into a black-box optimization benchmark."""

    @staticmethod
    def get_env_names() -> List[str]:
        gym_env_names = []
        max_displays: int = 10
        for e in gym.envs.registry.values():
            try:
                assert not any((x in str(e.id) for x in 'Kelly Copy llvm BulletEnv Minitaur Kuka InvertedPendulumSwingupBulletEnv'.split()))
                assert 'RacecarZedBulletEnv-v0' != e.id, 'This specific environment causes X11 error when using pybullet_envs.'
                assert 'CarRacing-v' not in str(e.id), 'Pixel based task not supported yet'
                env = gym.make(e.id)
                env.reset()
                env.step(env.action_space.sample())
                a1 = np.asarray(env.action_space.sample())
                a2 = np.asarray(env.action_space.sample())
                a3 = np.asarray(env.action_space.sample())
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

    def wrap_env(self, input_env) -> gym.wrappers.TimeLimit:
        env = gym.wrappers.TimeLimit(env=input_env, max_episode_steps=self.num_episode_steps)
        return env

    def create_env(self) -> gym.Env:
        env = gym.make(self.short_name)
        try:
            env.reset()
        except:
            assert False, f'Maybe check if {self.short_name} has a problem in reset / observation.'
        return env

    def __init__(self, name: str = 'CartPole-v0', control: str = 'conformant', neural_factor: int = 1, randomized: bool = True, optimization_scale: int = 0, greedy_bias: bool = False, sparse_limit: int = None):
        self.num_calls: int = 0
        self.optimization_scale: int = optimization_scale
        self.stochastic_problem: bool = 'stoc' in name
        self.greedy_bias: bool = greedy_bias
        self.sparse_limit: int = sparse_limit
        if 'conformant' in control or control == 'linear':
            assert neural_factor is None
        if os.name == 'nt':
            raise ng.errors.UnsupportedExperiment('Windows is not supported')
        self.short_name: str = name
        env = self.create_env()
        self.name: str = name + '__' + control + '__' + str(neural_factor)
        if sparse_limit is not None:
            self.name += f'__{sparse_limit}'
        if randomized:
            self.name += '_unseeded'
        self.randomized: bool = randomized
        try:
            try:
                self.num_time_steps = env._max_episode_steps
            except AttributeError:
                self.num_time_steps = env.horizon
        except AttributeError:
            assert any((x in name for x in NO_LENGTH)), name
            if 'LANM' not in name:
                self.num_time_steps = 200 if control == 'conformant' else 5000
            else:
                self.num_time_steps = 3000
        self.gamma: float = 0.995 if 'LANM' in name else 1.0
        self.neural_factor: int = neural_factor
        self.arities: List[List[int]] = []
        if isinstance(env.action_space, gym.spaces.Tuple):
            assert all((isinstance(p, gym.spaces.MultiDiscrete) for p in env.action_space)), f'{name} has a too complicated structure.'
            self.arities = [p.nvec for p in env.action_space]
            if control == 'conformant':
                output_dim = sum((len(a) for a in self.arities))
            else:
                output_dim = sum((sum(a) for a in self.arities))
            output_shape = (output_dim,)
            discrete = True
        elif isinstance(env.action_space, gym.spaces.Discrete):
            output_dim = env.action_space.n
            output_shape = (output_dim,)
            discrete = True
            assert output_dim is not None, env.action_space.n
        else:
            output_shape = env.action_space.shape
            if output_shape is None:
                output_shape = tuple(np.asarray(env.action_space.sample()).shape)
            discrete = False
            output_dim = np.prod(output_shape)
        self.discrete: bool = discrete
        assert env.observation_space is not None or 'llvm' in name, 'An observation space should be defined.'
        if env.observation_space is not None and env.observation_space.dtype == int:
            input_dim = env.observation_space.n
            assert input_dim is not None, env.observation_space.n
            self.discrete_input: bool = True
        else:
            input_dim = np.prod(env.observation_space.shape) if env.observation_space is not None else 0
            if input_dim is None:
                o = env.reset()
                input_dim = np.prod(np.asarray(o).shape)
            self.discrete_input: bool = False
        a = env.action_space.sample()
        self.action_type = type(a)
        self.subaction_type = None
        if hasattr(a, '__iter__'):
            self.subaction_type = type(a[0])
        if neural_factor is None:
            assert control == 'linear' or 'conformant' in control, f'{control} has neural_factor {neural_factor}'
        self.output_shape: Tuple = output_shape
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
        unstructured_neural_size: Tuple[int] = (output_dim * self.num_neurons + self.num_neurons * (input_dim + 1) + internal,)
        neural_size: Tuple[int] = unstructured_neural_size
        if self.greedy_bias:
            neural_size = (unstructured_neural_size[0] + 1,)
            assert 'multi' not in control
            assert 'structured' not in control
        assert control in CONTROLLERS or control == 'conformant', f'{control} not known as a form of control'
        self.control: str = control
        if 'neural' in control:
            self.first_size: int = self.num_neurons * (self.input_dim + 1)
            self.second_size: int = self.num_neurons * self.output_dim
            self.first_layer_shape: Tuple[int, int] = (self.input_dim + 1, self.num_neurons)
            self.second_layer_shape: Tuple[int, int] = (self.num_neurons, self.output_dim)
        shape_dict: dict = {'conformant': (self.num_time_steps,) + output_shape, 'stochastic_conformant': (self.num_time_steps,) + output_shape, 'linear': (input_dim + 1, output_dim), 'multi_neural': (min(self.num_time_steps, 50),) + unstructured_neural_size}
        shape: Tuple[int] = tuple(map(int, shape_dict.get(control, neural_size)))
        self.policy_shape: Tuple[int] = shape if 'structured' not in control else None
        parametrization: parameter.Array = parameter.Array(shape=shape).set_name('ng_default')
        if sparse_limit is not None:
            parametrization1: parameter.Array = parameter.Array(shape=shape)
            repetitions: int = int(np.prod(shape))
            assert isinstance(repetitions, int), f'{repetitions}'
            parametrization2: ng.p.Choice = ng.p.Choice([0, 1], repetitions=repetitions)
            parametrization: ng.p.Instrumentation = ng.p.Instrumentation(weights=parametrization1, enablers=parametrization2)
            parametrization.set_name('ng_sparse' + str(sparse_limit))
            assert 'conformant' not in control and 'structured' not in control
        if 'structured' in control and 'neural' in control and ('multi' not in control):
            parametrization: parameter.Instrumentation = parameter.Instrumentation(parameter.Array(shape=tuple(map(int, self.first_layer_shape))), parameter.Array(shape=tuple(map(int, self.second_layer_shape))).set_name('ng_struct')
        elif 'conformant' in control:
            try:
                if env.action_space.low is not None and env.action_space.high is not None:
                    low: np.ndarray = np.repeat(np.expand_dims(env.action_space.low, 0), self.num_time_steps, axis=0)
                    high: np.ndarray = np.repeat(np.expand_dims(env.action_space.high, 0), self.num_time_steps, axis=0)
                    init: np.ndarray = 0.5 * (low + high)
                    parametrization: parameter.Array = parameter.Array(init=init)
                    parametrization.set_bounds(low, high)
            except AttributeError:
                pass
            if self.subaction_type == int:
                parametrization.set_integer_casting()
            parametrization.set_name('conformant')
        super().__init__(self.sparse_gym_multi_function if sparse_limit is not None else self.gym_multi_function, parametrization=parametrization)
        self.greedy_coefficient: float = 0.0
        self.parametrization.function.deterministic: bool = False
        self.archive: List[Tuple[List[np.ndarray], List[np.ndarray], float]] = []
        self.mean_loss: float = 0.0
        self.num_losses: int = 0

    def evaluation_function(self, *recommendations) -> float:
        """Averages multiple evaluations if necessary."""
        if self.sparse_limit is None:
            x: np.ndarray = recommendations[0].value
        else:
            weights: np.ndarray = recommendations[0].kwargs['weights']
            enablers: np.ndarray = np.asarray(recommendations[0].kwargs['enablers'])
            assert all((x_ in [0, 1] for x_ in enablers)), f'non-binary enablers: {enablers}.'
            enablers = enablers.reshape(weights.shape)
            x: np.ndarray = weights * enablers
        if not self.randomized:
            return self.gym_multi_function(x, limited_fidelity=False)
        num: int = max(self.num_calls // 5, 23)
        return np.sum([self.gym_multi_function(x, limited_fidelity=False) for _ in range(num)]) / num

    def softmax(self, a: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(a, copy=False, nan=-1e+20, posinf=1e+20, neginf=-1e+20)
        probabilities: np.ndarray = np.exp(a - max(a))
        probabilities = probabilities / sum(probabilities)
        return probabilities

    def discretize(self, a: np.ndarray, env: gym.Env) -> np.ndarray:
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
                index: int = 0
                output: List[List[int]] = []
                for arities in self.arities:
                    local_output: List[int] = []
                    for arity in arities:
                        local_output += [int(list(np.random.multinomial(1, self.softmax(a[index:index + arity]))).index(1))]
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
        assert sum(probabilities) <= 1.0 + 1e-07, f'{probabilities} with greediness {self.greedy_coefficient}.'
        return int(list(np.random.multinomial(1, probabilities)).index(1))

    def neural(self, x: np.ndarray, o: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Applies a neural net parametrized by x to an observation o. Returns an action or logits of actions."""
        if self.greedy_bias:
            assert 'multi' not in self.control
            assert 'structured' not in self.control
            self.greedy_coefficient = x[-1:]
            x = x[:-1]
        o = o.ravel()
        my_scale: float = 2 ** self.optimization_scale
        if 'structured' not in self.name and self.optimization_scale != 0:
            x = np.asarray(my_scale * x, dtype=np.float32)
        if self.control == 'linear':
            output: np.ndarray = np.matmul(o, x[1:, :])
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
        output: np.ndarray = np.matmul(o, first_matrix[1:])
        if 'deep' in self.control:
            current_index: int = self.first_size + self.second_size
            internal_layer_size: int = self.num_neurons ** 2
            s: Tuple[int, int] = (self.num_neurons, self.num_neurons)
            for _ in range(self.num_internal_layers):
                output = np.tanh(output)
                layer: np.ndarray = x[current_index:current_index + internal_layer_size].reshape(s)
                if 'resid' in self.control:
                    layer += my_scale * np.eye(*layer.shape)
                output = np.matmul(output, layer) / np.sqrt(self.num_neurons)
                current_index += internal_layer_size
            assert current_index == len(x)
        output = np.matmul(np.tanh(output + first_matrix[0]), second_matrix)
        return (output[self.memory_len:].reshape(self.output_shape), output[:self.memory_len])

    def sparse_gym_multi_function(self, weights: np.ndarray, enablers: np.ndarray, limited_fidelity: bool = False) -> float:
        assert all((x_ in [0, 1] for x_ in enablers))
        x: np.ndarray = weights * enablers
        loss: float = self.gym_multi_function(x, limited_fidelity=limited_fidelity)
        sparse_penalty: float = 0
        if self.sparse_limit is not None:
            sparse_penalty = (1 + np.abs(loss)) * max(0, np