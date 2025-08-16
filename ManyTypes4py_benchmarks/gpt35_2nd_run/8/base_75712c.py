import typing as tp
from collections import defaultdict
import gym

StepReturn = tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, int], tp.Dict[str, bool], tp.Dict[str, tp.Any]]

class StepOutcome:
    def __init__(self, observation, reward=None, done=False, info=None) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = {} if info is None else info

    def __iter__(self) -> tp.Iterator:
        return iter((self.observation, self.reward, self.done, self.info))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(observation={self.observation}, reward={self.reward}, done={self.done}, info={self.info})'

    @classmethod
    def from_multiagent_step(cls, obs, reward, done, info) -> tp.Tuple[tp.Dict[str, 'StepOutcome'], bool]:
        outcomes = {agent: cls(obs[agent], reward.get(agent, None), done.get(agent, done.get('__all__', False)), info.get(agent, {})) for agent in obs}
        return (outcomes, done.get('__all__', False))

    @staticmethod
    def to_multiagent_step(outcomes, done=False) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, int], tp.Dict[str, bool], tp.Dict[str, tp.Any]]:
        names = ['observation', 'reward', 'done', 'info']
        obs, reward, done_dict, info = ({agent: getattr(outcome, name) for agent, outcome in outcomes.items()} for name in names)
        done_dict['__all__'] = done
        return (obs, reward, done_dict, info)

class Agent:
    def act(self, observation, reward, done, info=None) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def copy(self) -> 'Agent':
        return self.__class__()

class MultiAgentEnv:
    def reset(self) -> None:
        raise NotImplementedError

    def step(self, action_dict) -> tp.Dict[str, 'StepOutcome']:
        raise NotImplementedError

    def copy(self) -> 'MultiAgentEnv':
        raise NotImplementedError

    def with_agent(self, **agents) -> 'PartialMultiAgentEnv':
        return PartialMultiAgentEnv(self, **agents)

class PartialMultiAgentEnv(MultiAgentEnv):
    def __init__(self, env, **agents) -> None:
        ...

    def reset(self) -> tp.Dict[str, tp.Any]:
        ...

    def step(self, action_dict) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, int], tp.Dict[str, bool], tp.Dict[str, tp.Any]]:
        ...

    def copy(self) -> 'PartialMultiAgentEnv':
        ...

    def as_single_agent(self) -> 'SingleAgentEnv':
        ...

class SingleAgentEnv(gym.Env):
    def __init__(self, env) -> None:
        ...

    def reset(self) -> tp.Any:
        ...

    def step(self, action) -> tp.Tuple[tp.Any, int, bool, tp.Dict[str, tp.Any]]:
        ...

    def copy(self) -> 'SingleAgentEnv':
        ...

class EnvironmentRunner:
    def __init__(self, env, num_repetitions=1, max_step=float('inf')) -> None:
        ...

    def run(self, *agent, **agents) -> tp.Union[float, tp.Dict[str, float]]:
        ...

    def _run_once(self, *single_agent, **agents) -> tp.Dict[str, float]:
        ...

    def copy(self) -> 'EnvironmentRunner':
        ...
