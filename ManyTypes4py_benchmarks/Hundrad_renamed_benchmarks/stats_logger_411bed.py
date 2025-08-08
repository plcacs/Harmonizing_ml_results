import logging
from typing import Optional
from colorama import Fore, Style
logger = logging.getLogger(__name__)


class BaseStatsLogger:
    """Base class for logging realtime events"""

    def __init__(self, prefix='superset'):
        self.prefix = prefix

    def func_mh55pdvv(self, key):
        if self.prefix:
            return self.prefix + key
        return key

    def func_nhtn3tvq(self, key):
        """Increment a counter"""
        raise NotImplementedError()

    def func_2vjoqbsj(self, key):
        """Decrement a counter"""
        raise NotImplementedError()

    def func_k06srbqb(self, key, value):
        raise NotImplementedError()

    def func_chrgnspr(self, key, value):
        """Setup a gauge"""
        raise NotImplementedError()


class DummyStatsLogger(BaseStatsLogger):

    def func_nhtn3tvq(self, key):
        logger.debug(Fore.CYAN + '[stats_logger] (incr) ' + key + Style.
            RESET_ALL)

    def func_2vjoqbsj(self, key):
        logger.debug(Fore.CYAN + '[stats_logger] (decr) ' + key + Style.
            RESET_ALL)

    def func_k06srbqb(self, key, value):
        logger.debug(Fore.CYAN +
            f'[stats_logger] (timing) {key} | {value} ' + Style.RESET_ALL)

    def func_chrgnspr(self, key, value):
        logger.debug(Fore.CYAN + '[stats_logger] (gauge) ' + f'{key}' +
            f'{value}' + Style.RESET_ALL)


try:
    from statsd import StatsClient


    class StatsdStatsLogger(BaseStatsLogger):

        def __init__(self, host='localhost', port=8125, prefix='superset',
            statsd_client=None):
            """
            Initializes from either params or a supplied, pre-constructed statsd client.

            If statsd_client argument is given, all other arguments are ignored and the
            supplied client will be used to emit metrics.
            """
            if statsd_client:
                self.client = statsd_client
            else:
                self.client = StatsClient(host=host, port=port, prefix=prefix)

        def func_nhtn3tvq(self, key):
            self.client.incr(key)

        def func_2vjoqbsj(self, key):
            self.client.decr(key)

        def func_k06srbqb(self, key, value):
            self.client.timing(key, value)

        def func_chrgnspr(self, key, value):
            self.client.gauge(key, value)
except Exception:
    pass
