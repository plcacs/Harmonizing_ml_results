# pragma pylint: disable=missing-docstring,C0103,protected-access

import logging
import time
from copy import deepcopy
from datetime import timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import MagicMock, PropertyMock

import pandas as pd
import pytest
import time_machine

from freqtrade.constants import AVAILABLE_PAIRLISTS
from freqtrade.data.dataprovider import DataProvider
from freqtrade.enums import CandleType, RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.persistence import LocalTrade, Trade
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist, expand_pairlist
from freqtrade.plugins.pairlistmanager import PairListManager
from freqtrade.resolvers import PairListResolver
from freqtrade.util.datetime_helpers import dt_now
from tests.conftest import (
    EXMS,
    create_mock_trades_usdt,
    generate_test_data,
    get_patched_exchange,
    get_patched_freqtradebot,
    log_has,
    log_has_re,
    num_log_has,
)
