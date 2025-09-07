from typing import Union
from . import boc, daykline, hkquote, jsl, sina, tencent, timekline

def use(source: str) -> Union[sina.Sina, jsl.Jsl, tencent.Tencent, boc.Boc, timekline.TimeKline, daykline.Daykline, hkquote.HKQuote]:
    if source in ['sina']:
        return sina.Sina()
    if source in ['jsl']:
        return jsl.Jsl()
    if source in ['qq', 'tencent']:
        return tencent.Tencent()
    if source in ['boc']:
        return boc.Boc()
    if source in ['timekline']:
        return timekline.TimeKline()
    if source in ['daykline']:
        return daykline.Daykline()
    if source in ['hkquote']:
        return hkquote.HKQuote()
    raise NotImplementedError