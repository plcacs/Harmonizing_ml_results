from typing import Any, Dict
import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
from pandas import DataFrame
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.freqai_interface import IFreqaiModel
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv
from freqtrade.freqai.RL.BaseEnvironment import BaseActions, BaseEnvironment, Positions
from freqtrade.freqai.tensorboard.TensorboardCallback import TensorboardCallback
from freqtrade.persistence import Trade

class BaseReinforcementLearningModel(IFreqaiModel):
    def __init__(self, **kwargs: Dict[str, Any]):
        self.train_env: gym.Env = gym.Env()
        self.eval_env: gym.Env = gym.Env()
        self.eval_callback: MaskableEvalCallback = None
        self.MODELCLASS: Any = None
        self.tensorboard_callback: TensorboardCallback = TensorboardCallback(verbose=1, actions=BaseActions)

    def set_train_and_eval_environments(self, data_dictionary: Dict[str, Any], prices_train: DataFrame, prices_test: DataFrame, dk: FreqaiDataKitchen):
        pass

    def pack_env_dict(self, pair: str) -> Dict[str, Any]:
        pass

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs: Any):
        pass

    def get_state_info(self, pair: str) -> Tuple[float, float, int]:
        pass

    def predict(self, unfiltered_df: DataFrame, dk: FreqaiDataKitchen, **kwargs: Any) -> Tuple[DataFrame, np.ndarray]:
        pass

    def rl_model_predict(self, dataframe: DataFrame, dk: FreqaiDataKitchen, model: Any):
        pass

    def build_ohlc_price_dataframes(self, data_dictionary: Dict[str, Any], pair: str, dk: FreqaiDataKitchen) -> Tuple[DataFrame, DataFrame]:
        pass

    def drop_ohlc_from_df(self, df: DataFrame, dk: FreqaiDataKitchen) -> DataFrame:
        pass

    def load_model_from_disk(self, dk: FreqaiDataKitchen) -> Any:
        pass

    def _on_stop(self):
        pass

class MyRLEnv(Base5ActionRLEnv):
    def calculate_reward(self, action: int) -> float:
        pass

def make_env(MyRLEnv, env_id: str, rank: int, seed: int, train_df: DataFrame, price: DataFrame, env_info: Dict[str, Any]) -> Callable:
    pass
