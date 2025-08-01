from datetime import datetime
from typing import Any, Dict, List, Optional
import talib.abstract as ta
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import BooleanParameter, DecimalParameter, IntParameter, IStrategy, RealParameter


class StrategyTestV3(IStrategy):
    """
    Strategy used by tests freqtrade bot.
    Please do not modify this strategy, it's  intended for internal use only.
    Please look at the SampleStrategy in the user_data/strategy directory
    or strategy repository https://github.com/freqtrade/freqtrade-strategies
    for samples and inspiration.
    """
    INTERFACE_VERSION = 3
    minimal_roi: Dict[str, float] = {'40': 0.0, '30': 0.01, '20': 0.02, '0': 0.04}
    max_open_trades: int = -1
    stoploss: float = -0.1
    timeframe: str = '5m'
    order_types: Dict[str, Any] = {
        'entry': 'limit', 
        'exit': 'limit', 
        'stoploss': 'limit', 
        'stoploss_on_exchange': False
    }
    startup_candle_count: int = 20
    order_time_in_force: Dict[str, str] = {'entry': 'gtc', 'exit': 'gtc'}
    buy_params: Dict[str, Any] = {'buy_rsi': 35}
    sell_params: Dict[str, Any] = {'sell_rsi': 74, 'sell_minusdi': 0.4}
    buy_rsi: IntParameter = IntParameter([0, 50], default=30, space='buy')
    buy_plusdi: RealParameter = RealParameter(low=0, high=1, default=0.5, space='buy')
    sell_rsi: IntParameter = IntParameter(low=50, high=100, default=70, space='sell')
    sell_minusdi: DecimalParameter = DecimalParameter(low=0, high=1, default=0.5001, decimals=3, space='sell', load=False)
    protection_enabled: BooleanParameter = BooleanParameter(default=True)
    protection_cooldown_lookback: IntParameter = IntParameter([0, 50], default=30)

    bot_started: bool = False

    @property
    def protections(self) -> List[Any]:
        prot: List[Any] = []
        if self.protection_enabled.value:
            prot = self.config.get('_strategy_protections', {})
        return prot

    def bot_start(self) -> None:
        self.bot_started = True

    def informative_pairs(self) -> List[Any]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        dataframe['adx'] = ta.ADX(dataframe)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['rsi'] < self.buy_rsi.value) &
                (dataframe['fastd'] < 35) &
                (dataframe['adx'] > 30) &
                (dataframe['plus_di'] > self.buy_plusdi.value)
            ) | (
                (dataframe['adx'] > 65) &
                (dataframe['plus_di'] > self.buy_plusdi.value)
            ),
            'enter_long'
        ] = 1
        dataframe.loc[
            qtpylib.crossed_below(dataframe['rsi'], self.sell_rsi.value),
            ('enter_short', 'enter_tag')
        ] = (1, 'short_Tag')
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        dataframe.loc[
            (
                (qtpylib.crossed_above(dataframe['rsi'], self.sell_rsi.value) | qtpylib.crossed_above(dataframe['fastd'], 70)) &
                (dataframe['adx'] > 10) &
                (dataframe['minus_di'] > 0)
            ) | (
                (dataframe['adx'] > 70) &
                (dataframe['minus_di'] > self.sell_minusdi.value)
            ),
            'exit_long'
        ] = 1
        dataframe.loc[
            qtpylib.crossed_above(dataframe['rsi'], self.buy_rsi.value),
            ('exit_short', 'exit_tag')
        ] = (1, 'short_Tag')
        return dataframe

    def leverage(
        self, 
        pair: str, 
        current_time: datetime, 
        current_rate: float, 
        proposed_leverage: float, 
        max_leverage: float, 
        entry_tag: str, 
        side: str, 
        **kwargs: Any
    ) -> float:
        return 3.0

    def adjust_trade_position(
        self, 
        trade: Trade, 
        current_time: datetime, 
        current_rate: float, 
        current_profit: float, 
        min_stake: float, 
        max_stake: float, 
        current_entry_rate: float, 
        current_exit_rate: float, 
        current_entry_profit: float, 
        current_exit_profit: float, 
        **kwargs: Any
    ) -> Optional[float]:
        if current_profit < -0.0075:
            orders = trade.select_filled_orders(trade.entry_side)
            return round(orders[0].stake_amount, 0)
        return None


class StrategyTestV3Futures(StrategyTestV3):
    can_short: bool = True