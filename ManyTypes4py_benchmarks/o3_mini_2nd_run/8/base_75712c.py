import typing as tp
from typing import Any, Dict, Iterator, Optional, Tuple, Union
from collections import defaultdict
import gym

StepReturn = Tuple[Dict[str, Any], Dict[str, Any], Dict[str, bool], Dict[str, Any]]


class StepOutcome:
    """Handle for dealing with environment (and especially multi-agent) outputs more easily"""

    def __init__(self, observation: Any, reward: Optional[Any] = None, done: bool = False, info: Optional[Dict[str, Any]] = None) -> None:
        self.observation: Any = observation
        self.reward: Optional[Any] = reward
        self.done: bool = done
        self.info: Dict[str, Any] = {} if info is None else info

    def __iter__(self) -> Iterator[Any]:
        return iter((self.observation, self.reward, self.done, self.info))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(observation={self.observation}, reward={self.reward}, done={self.done}, info={self.info})'

    @classmethod
    def from_multiagent_step(cls, obs: Dict[str, Any], reward: Dict[str, Any], done: Dict[str, bool], info: Dict[str, Any]) -> Tuple[Dict[str, "StepOutcome"], bool]:
        outcomes: Dict[str, StepOutcome] = {
            agent: cls(obs[agent], reward.get(agent, None), done.get(agent, done.get('__all__', False)), info.get(agent, {}))
            for agent in obs
        }
        return outcomes, done.get('__all__', False)

    @staticmethod
    def to_multiagent_step(outcomes: Dict[str, "StepOutcome"], done: bool = False) -> StepReturn:
        names = ['observation', 'reward', 'done', 'info']
        obs, reward, done_dict, info = (
            {agent: getattr(outcome, name) for agent, outcome in outcomes.items()} for name in names
        )
        done_dict['__all__'] = done
        return obs, reward, done_dict, info


class Agent:
    """Base class for an Agent operating in an environment."""

    def act(self, observation: Any, reward: Optional[Any], done: bool, info: Optional[Dict[str, Any]] = None) -> Any:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def copy(self) -> "Agent":
        return self.__class__()


class MultiAgentEnv:
    """Base class for a multi-agent environment (in a ray-like fashion)."""

    def reset(self) -> Any:
        raise NotImplementedError

    def step(self, action_dict: Dict[str, Any]) -> StepReturn:
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to StepOutcome. The
        number of agents in the env can vary over time.
        """
        raise NotImplementedError

    def copy(self) -> "MultiAgentEnv":
        """Used to create new instances of Ray MultiAgentEnv"""
        raise NotImplementedError

    def with_agent(self, **agents: Agent) -> "PartialMultiAgentEnv":
        return PartialMultiAgentEnv(self, **agents)


class PartialMultiAgentEnv(MultiAgentEnv):
    """Multi agent environment for which some of the agents have been fixed"""

    def __init__(self, env: MultiAgentEnv, **agents: Agent) -> None:
        self.env: MultiAgentEnv = env.copy()
        self.agents: Dict[str, Agent] = {name: agent.copy() for name, agent in agents.items()}
        self.env.reset()
        # Assuming env has an attribute 'agent_names'
        unknown = set(agents) - set(getattr(env, 'agent_names'))
        if unknown:
            raise ValueError(f'Unkwnon agents: {unknown}')
        self.agent_names: tp.List[str] = [an for an in getattr(env, 'agent_names') if an not in self.agents]
        self.observation_space: Any = getattr(env, 'observation_space')
        self.action_space: Any = getattr(env, 'action_space')
        self._agents_outcome: Dict[str, StepOutcome] = {}

    def reset(self) -> Dict[str, Any]:
        outcomes, _ = StepOutcome.from_multiagent_step(self.env.reset(), {}, {}, {})
        self._agents_outcome = {name: outcomes[name] for name in self.agents}
        return {name: outcomes[name].observation for name in outcomes if name not in self.agents}

    def step(self, action_dict: Dict[str, Any]) -> StepReturn:
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to StepOutcome. The
        number of agents in the env can vary over time.
        """
        full_action_dict: Dict[str, Any] = {
            name: self.agents[name].act(*outcome) for name, outcome in self._agents_outcome.items()
        }
        full_action_dict.update(action_dict)
        outcomes, done = StepOutcome.from_multiagent_step(*self.env.step(full_action_dict))
        self._agents_outcome = {name: outcomes[name] for name in self.agents}
        outcomes_remaining: Dict[str, StepOutcome] = {name: outcomes[name] for name in outcomes if name not in self.agents}
        return StepOutcome.to_multiagent_step(outcomes_remaining, done)

    def copy(self) -> "PartialMultiAgentEnv":
        return self.__class__(self.env, **self.agents)

    def as_single_agent(self) -> "SingleAgentEnv":
        return SingleAgentEnv(self)


class SingleAgentEnv(gym.Env):
    """Single-agent gym-like environment based on a multi-agent environment for which
    all but one agent has been fixed.
    """

    def __init__(self, env: PartialMultiAgentEnv) -> None:
        assert len(env.agent_names) == 1, f'Too many remaining agents: {env.agent_names}'
        self.env: PartialMultiAgentEnv = env.copy()
        self.env.reset()
        self.observation_space: Any = getattr(env, 'observation_space')
        self.action_space: Any = getattr(env, 'action_space')
        self._agent_name: str = env.agent_names[0]
        self.env = env

    def reset(self) -> Any:
        return self.env.reset()[self._agent_name]

    def step(self, action: Any) -> Tuple[Any, Any, bool, Dict[str, Any]]:
        an: str = self._agent_name
        obs, reward, done, info = self.env.step({an: action})
        return (obs[an], reward[an], done[an] or done['__all__'], info.get(an, {}))

    def copy(self) -> "SingleAgentEnv":
        return self.__class__(self.env.copy())


class EnvironmentRunner:
    """Helper for running environments

    Parameters
    ----------
    env: gym.Env or MultiAgentEnv
        a possibly multi agent environment
    num_repetations: int
        number of repetitions to play the environment (smoothes the output)
    max_step: int
        maximum number of steps to play the environment before breaking
    """

    def __init__(self, env: Union[gym.Env, MultiAgentEnv], num_repetitions: int = 1, max_step: float = float('inf')) -> None:
        self.env: Union[gym.Env, MultiAgentEnv] = env
        self.num_repetitions: int = num_repetitions
        self.max_step: float = max_step

    def run(self, *agent: Agent, **agents: Agent) -> Union[float, Dict[str, float]]:
        """Run one agent or multiple named agents

        Parameters
        ----------
        *agent: Agent (optional)
            the agent to play a single-agent environment
        **agents: Agent
            the named agents to play a multi-agent environment

        Returns
        -------
        float:
            the mean reward (possibly for each agent)
        """
        san: str = 'single_agent_name'
        if agents:
            sum_rewards: Dict[str, float] = {name: 0.0 for name in agents}
        else:
            sum_rewards = {san: 0.0}  # type: Dict[str, float]
        for _ in range(self.num_repetitions):
            rewards: Dict[str, float] = self._run_once(*agent, **agents)
            for name, value in rewards.items():
                sum_rewards[name] += value
        mean_rewards: Dict[str, float] = {name: float(value) / self.num_repetitions for name, value in sum_rewards.items()}
        if isinstance(self.env, gym.Env):
            return mean_rewards[san]
        return mean_rewards

    def _run_once(self, *single_agent: Agent, **agents: Agent) -> Dict[str, float]:
        san: str = 'single_agent_name'
        if len(single_agent) == 1 and (not agents):
            agents = {san: single_agent[0]}
        elif single_agent or not agents:
            raise ValueError('Either provide 1 unnamed agent or several named agents')
        for agent in agents.values():
            agent.reset()
        if isinstance(self.env, gym.Env):
            outcomes: Dict[str, StepOutcome] = {san: StepOutcome(self.env.reset())}
            done: bool = False
        else:
            outcomes, done = StepOutcome.from_multiagent_step(self.env.reset(), {}, {}, {})
        reward_sum: tp.DefaultDict[str, float] = defaultdict(float)
        step: int = 0
        while step < self.max_step and (not done):
            actions: Dict[str, Any] = {}
            for name, outcome in outcomes.items():
                actions[name] = agents[name].act(*outcome)
            if isinstance(self.env, gym.Env):
                outcomes = {san: StepOutcome(*self.env.step(actions[san]))}
                done = outcomes[san].done
            else:
                outcomes, done = StepOutcome.from_multiagent_step(*self.env.step(actions))
            for name, outcome in outcomes.items():
                assert outcome.reward is not None
                reward_sum[name] += outcome.reward
            step += 1
        for name, outcome in outcomes.items():
            agents[name].act(*outcome)
        return dict(reward_sum)

    def copy(self) -> "EnvironmentRunner":
        return EnvironmentRunner(self.env.copy(), num_repetitions=self.num_repetitions, max_step=self.max_step)