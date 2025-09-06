from typing import Any, Literal, TypedDict, List, Dict
from freqtrade.constants import Config
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import Exchange, market_is_active
from freqtrade.exchange.exchange_types import Ticker, Tickers
from freqtrade.mixins import LoggingMixin
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum

logger: logging.Logger = logging.getLogger(__name__)

class __PairlistParameterBase(TypedDict):
    pass

class __NumberPairlistParameter(__PairlistParameterBase):
    pass

class __StringPairlistParameter(__PairlistParameterBase):
    pass

class __OptionPairlistParameter(__PairlistParameterBase):
    pass

class __ListPairListParamenter(__PairlistParameterBase):
    pass

class __BoolPairlistParameter(__PairlistParameterBase):
    pass

PairlistParameter: type = __NumberPairlistParameter | __StringPairlistParameter | __OptionPairlistParameter | __BoolPairlistParameter | __ListPairListParamenter

class SupportsBacktesting(str, Enum):
    YES: str = 'yes'
    NO: str = 'no'
    NO_ACTION: str = 'no_action'
    BIASED: str = 'biased'

class IPairList(LoggingMixin, ABC):
    is_pairlist_generator: bool = False
    supports_backtesting: SupportsBacktesting = SupportsBacktesting.NO

    def __init__(self, exchange: Exchange, pairlistmanager: Any, config: Config, pairlistconfig: Dict[str, Any], pairlist_pos: int) -> None:
        self._enabled: bool = True
        self._exchange: Exchange = exchange
        self._pairlistmanager: Any = pairlistmanager
        self._config: Config = config
        self._pairlistconfig: Dict[str, Any] = pairlistconfig
        self._pairlist_pos: int = pairlist_pos
        self.refresh_period: int = self._pairlistconfig.get('refresh_period', 1800)
        LoggingMixin.__init__(self, logger, self.refresh_period)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    @abstractmethod
    def needstickers(self) -> bool:
        return False

    @staticmethod
    @abstractmethod
    def description() -> str:
        return ''

    @staticmethod
    def available_parameters() -> Dict[str, Dict[str, Any]]:
        return {}

    @staticmethod
    def refresh_period_parameter() -> Dict[str, Dict[str, Any]]:
        return {'refresh_period': {'type': 'number', 'default': 1800, 'description': 'Refresh period', 'help': 'Refresh period in seconds'}}

    @abstractmethod
    def short_desc(self) -> str:
        return ''

    def _validate_pair(self, pair: str, ticker: Ticker) -> bool:
        raise NotImplementedError()

    def gen_pairlist(self, tickers: Tickers) -> List[str]:
        raise OperationalException('This Pairlist Handler should not be used at the first position in the list of Pairlist Handlers.')

    def filter_pairlist(self, pairlist: List[str], tickers: Tickers) -> List[str]:
        if self._enabled:
            for p in deepcopy(pairlist):
                if not self._validate_pair(p, tickers[p] if p in tickers else None):
                    pairlist.remove(p)
        return pairlist

    def verify_blacklist(self, pairlist: List[str], logmethod: Any) -> List[str]:
        return self._pairlistmanager.verify_blacklist(pairlist, logmethod)

    def verify_whitelist(self, pairlist: List[str], logmethod: Any, keep_invalid: bool = False) -> List[str]:
        return self._pairlistmanager.verify_whitelist(pairlist, logmethod, keep_invalid)

    def _whitelist_for_active_markets(self, pairlist: List[str]) -> List[str]:
        markets = self._exchange.markets
        if not markets:
            raise OperationalException('Markets not loaded. Make sure that exchange is initialized correctly.')
        sanitized_whitelist: List[str] = []
        for pair in pairlist:
            if pair not in markets:
                self.log_once(f'Pair {pair} is not compatible with exchange {self._exchange.name}. Removing it from whitelist..', logger.warning)
                continue
            if not self._exchange.market_is_tradable(markets[pair]):
                self.log_once(f'Pair {pair} is not tradable with Freqtrade. Removing it from whitelist..', logger.warning)
                continue
            if self._exchange.get_pair_quote_currency(pair) != self._config['stake_currency']:
                self.log_once(f'Pair {pair} is not compatible with your stake currency {self._config["stake_currency"]}. Removing it from whitelist..', logger.warning)
                continue
            market = markets[pair]
            if not market_is_active(market):
                self.log_once(f'Ignoring {pair} from whitelist. Market is not active.', logger.info)
                continue
            if pair not in sanitized_whitelist:
                sanitized_whitelist.append(pair)
        return sanitized_whitelist
