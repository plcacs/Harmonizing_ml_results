import warnings
import operator
import copy as _copy
import typing as tp

import gym
import numpy as np
from nevergrad.common.tools import pytorch_import_fix
from nevergrad.parametrization import parameter as p
from ..base import ExperimentFunction
from . import base
from . import envs

pytorch_import_fix()
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import WeightedRandomSampler

AnyEnv = tp.Union[gym.Env, base.MultiAgentEnv]


class RandomAgent(base.Agent):
    """Agent that plays randomly."""

    def __init__(self, env: AnyEnv) -> None:
        self.env = env
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.num_outputs: int = env.action_space.n

    def act(
        self, observation: np.ndarray, reward: float, done: bool, info: tp.Optional[tp.Any] = None
    ) -> int:
        return int(np.random.randint(self.num_outputs))

    def copy(self) -> "RandomAgent":
        return self.__class__(self.env)


class Agent007(base.Agent):
    """Agent that plays slightly better than random on the 007 game."""

    def __init__(self, env: tp.Union[envs.DoubleOSeven, base.SingleAgentEnv]) -> None:
        self.env = env
        assert isinstance(env, envs.DoubleOSeven) or (
            isinstance(env, base.SingleAgentEnv) and isinstance(env.env, envs.DoubleOSeven)
        )

    def act(
        self, observation: tp.Tuple[int, int, int, int], reward: float, done: bool, info: tp.Optional[tp.Any] = None
    ) -> int:
        my_amm, my_prot, their_amm, their_prot = observation
        if their_prot == 4 and my_amm:
            action: str = "fire"
        elif their_amm == 0:
            action = np.random.choice(["fire", "reload"])
        else:
            action = np.random.choice(["fire", "protect", "reload"])
        return envs.JamesBond.actions.index(action)

    def copy(self) -> "Agent007":
        return self.__class__(self.env)


class TorchAgent(base.Agent):
    """Agent that plays through a torch neural network."""

    def __init__(self, module: nn.Module, deterministic: bool = True, instrumentation_std: float = 0.1) -> None:
        super().__init__()
        self.deterministic: bool = deterministic
        self.module: nn.Module = module
        kwargs: tp.Dict[str, p.Array] = {
            name: p.Array(shape=value.shape).set_mutation(sigma=instrumentation_std).set_bounds(-10, 10, method="arctan")
            for name, value in module.state_dict().items()
        }
        self.instrumentation: p.Instrumentation = p.Instrumentation(**kwargs)

    @classmethod
    def from_module_maker(
        cls,
        env: gym.Env,
        module_maker: tp.Callable[[tp.Tuple[int, ...], int], nn.Module],
        deterministic: bool = True,
    ) -> "TorchAgent":
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Box)
        module: nn.Module = module_maker(env.observation_space.shape, env.action_space.n)
        return cls(module, deterministic=deterministic)

    def act(
        self, observation: np.ndarray, reward: float, done: bool, info: tp.Optional[tp.Any] = None
    ) -> int:
        obs: torch.Tensor = torch.from_numpy(observation.astype(np.float32))
        forward: torch.Tensor = self.module.forward(obs)
        probas: torch.Tensor = F.softmax(forward, dim=0)
        if self.deterministic:
            return int(probas.max(0)[1].view(1, 1).item())
        else:
            # WeightedRandomSampler expects weights to be a Tensor and returns an iterator over indices.
            sampler: WeightedRandomSampler = WeightedRandomSampler(probas, 1)
            return int(next(iter(sampler)))

    def copy(self) -> "TorchAgent":
        return TorchAgent(_copy.deepcopy(self.module), self.deterministic)

    def load_state_dict(self, state_dict: tp.Dict[str, np.ndarray]) -> None:
        converted: tp.Dict[str, torch.Tensor] = {
            x: torch.tensor(y.astype(np.float32)) for x, y in state_dict.items()
        }
        self.module.load_state_dict(converted)


class TorchAgentFunction(ExperimentFunction):
    """Instrumented function which plays the agent using an environment runner"""
    _num_test_evaluations: int = 1000

    def __init__(
        self,
        agent: TorchAgent,
        env_runner: tp.Any,
        reward_postprocessing: tp.Callable[[float], float] = operator.neg,
    ) -> None:
        assert isinstance(env_runner.env, gym.Env)
        self.agent: TorchAgent = agent.copy()
        self.runner: tp.Any = env_runner.copy()
        self.reward_postprocessing: tp.Callable[[float], float] = reward_postprocessing
        super().__init__(self.compute, self.agent.instrumentation.copy().set_name(""))
        self.parametrization.function.deterministic = False
        self.add_descriptors(num_repetitions=self.runner.num_repetitions, archi=self.agent.module.__class__.__name__)

    def compute(self, **kwargs: tp.Any) -> float:
        self.agent.load_state_dict(kwargs)
        try:
            with torch.no_grad():
                reward: float = self.runner.run(self.agent)
        except RuntimeError as e:
            warnings.warn(f"Returning 0 after error: {e}")
            reward = 0.0
        assert isinstance(reward, (int, float))
        return self.reward_postprocessing(float(reward))

    def evaluation_function(self, *recommendations: tp.Any) -> float:
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """
        assert len(recommendations) == 1, "Should not be a pareto set for a singleobjective function"
        num_tests: int = max(1, int(self._num_test_evaluations / self.runner.num_repetitions))
        total: float = sum((self.compute(**recommendations[0].kwargs) for _ in range(num_tests)))
        return total / num_tests


class Perceptron(nn.Module):
    def __init__(self, input_shape: tp.Tuple[int, ...], output_size: int) -> None:
        super().__init__()
        assert len(input_shape) == 1
        self.head: nn.Linear = nn.Linear(input_shape[0], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class DenseNet(nn.Module):
    def __init__(self, input_shape: tp.Tuple[int, ...], output_size: int) -> None:
        super().__init__()
        assert len(input_shape) == 1
        self.lin1: nn.Linear = nn.Linear(input_shape[0], 16)
        self.lin2: nn.Linear = nn.Linear(16, 16)
        self.lin3: nn.Linear = nn.Linear(16, 16)
        self.head: nn.Linear = nn.Linear(16, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return self.head(x)