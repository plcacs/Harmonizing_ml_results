from freqtrade.constants import DECIMAL_PER_COIN_FALLBACK, DECIMALS_PER_COIN
from typing import Union

def decimals_per_coin(coin: str) -> Union[int, float]:
    return DECIMALS_PER_COIN.get(coin, DECIMAL_PER_COIN_FALLBACK)

def strip_trailing_zeros(value: str) -> str:
    return value.rstrip('0').rstrip('.')

def round_value(value: Union[int, float], decimals: int, keep_trailing_zeros: bool = False) -> str:
    val = f'{value:.{decimals}f}'
    if not keep_trailing_zeros:
        val = strip_trailing_zeros(val)
    return val

def fmt_coin(value: Union[int, float], coin: str, show_coin_name: bool = True, keep_trailing_zeros: bool = False) -> str:
    val = round_value(value, decimals_per_coin(coin), keep_trailing_zeros)
    if show_coin_name:
        val = f'{val} {coin}'
    return val

def fmt_coin2(value: Union[int, float], coin: str, decimals: int = 8, *, show_coin_name: bool = True, keep_trailing_zeros: bool = False) -> str:
    val = round_value(value, decimals, keep_trailing_zeros)
    if show_coin_name:
        val = f'{val} {coin}'
    return val
