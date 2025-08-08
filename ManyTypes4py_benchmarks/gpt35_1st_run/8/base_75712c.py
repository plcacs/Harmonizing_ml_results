import typing as tp
from collections import defaultdict
import gym

StepReturn = tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, int], tp.Dict[str, bool], tp.Dict[str, tp.Any]]

class StepOutcome:
    def __init__(self, observation, reward=None, done=False, info=None) -> None:
    @classmethod
    def from_multiagent_step(cls, obs, reward, done, info) -> tp.Tuple[tp.Dict[str, 'StepOutcome'], bool]:
    @staticmethod
    def to_multiagent_step(outcomes, done=False) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, int], tp.Dict[str, bool], tp.Dict[str, tp.Any]]:
    
class Agent:
    def act(self, observation, reward, done, info=None) -> tp.Any:
    def reset(self) -> None:
    def copy(self) -> 'Agent':

class MultiAgentEnv:
    def reset(self) -> None:
    def step(self, action_dict) -> tp.Dict[str, 'StepOutcome']:
    def copy(self) -> 'MultiAgentEnv':
    def with_agent(self, **agents) -> 'PartialMultiAgentEnv':

class PartialMultiAgentEnv(MultiAgentEnv):
    def __init__(self, env, **agents) -> None:
    def reset(self) -> tp.Dict[str, tp.Any]:
    def step(self, action_dict) -> tp.Tuple[tp.Dict[str, tp.Any], tp.Dict[str, int], tp.Dict[str, bool], tp.Dict[str, tp.Any]]:
    def copy(self) -> 'PartialMultiAgentEnv':
    def as_single_agent(self) -> 'SingleAgentEnv':

class SingleAgentEnv(gym.Env):
    def __init__(self, env) -> None:
    def reset(self) -> tp.Any:
    def step(self, action) -> tp.Tuple[tp.Any, int, bool, tp.Dict[str, tp.Any]]:
    def copy(self) -> 'SingleAgentEnv':

class EnvironmentRunner:
    def __init__(self, env, num_repetitions=1, max_step=float('inf')) -> None:
    def run(self, *agent, **agents) -> tp.Union[float, tp.Dict[str, float]]:
    def _run_once(self, *single_agent, **agents) -> tp.Dict[str, float]:
    def copy(self) -> 'EnvironmentRunner':
