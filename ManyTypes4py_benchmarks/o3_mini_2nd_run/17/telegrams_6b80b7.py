"""KNX Telegram handler."""
from __future__ import annotations
from collections import deque
from typing import Any, Deque, Dict, List, Optional, TypedDict
from xknx import XKNX
from xknx.dpt import DPTArray, DPTBase, DPTBinary
from xknx.dpt.dpt import DPTComplexData, DPTEnumData
from xknx.exceptions import XKNXException
from xknx.telegram import Telegram, TelegramDirection
from xknx.telegram.apci import GroupValueResponse, GroupValueWrite
from homeassistant.core import HomeAssistant
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.storage import Store
from homeassistant.util import dt as dt_util
from homeassistant.util.signal_type import SignalType
from .const import DOMAIN
from .project import KNXProject

STORAGE_VERSION: int = 1
STORAGE_KEY: str = f"{DOMAIN}/telegrams_history.json"
SIGNAL_KNX_TELEGRAM: SignalType = SignalType("knx_telegram")


class DecodedTelegramPayload(TypedDict, total=False):
    dpt_main: Optional[int]
    dpt_sub: Optional[int]
    dpt_name: Optional[str]
    unit: Optional[str]
    value: Optional[Any]


class TelegramDict(DecodedTelegramPayload):
    destination: str
    destination_name: str
    direction: str
    payload: Optional[Any]
    source: str
    source_name: str
    telegramtype: str
    timestamp: str


class Telegrams:
    """Class to handle KNX telegrams."""

    def __init__(
        self,
        hass: HomeAssistant,
        xknx: XKNX,
        project: KNXProject,
        log_size: int,
    ) -> None:
        """Initialize Telegrams class."""
        self.hass: HomeAssistant = hass
        self.project: KNXProject = project
        self._history_store: Store[List[TelegramDict]] = Store(
            hass, STORAGE_VERSION, STORAGE_KEY
        )
        self._xknx_telegram_cb_handle = xknx.telegram_queue.register_telegram_received_cb(
            telegram_received_cb=self._xknx_telegram_cb, match_for_outgoing=True
        )
        self.recent_telegrams: Deque[TelegramDict] = deque(maxlen=log_size)
        self.last_ga_telegrams: Dict[str, TelegramDict] = {}

    async def load_history(self) -> None:
        """Load history from store."""
        telegrams: Optional[List[TelegramDict]] = await self._history_store.async_load()
        if telegrams is None:
            return
        if self.recent_telegrams.maxlen == 0:
            await self._history_store.async_remove()
            return
        for telegram in telegrams:
            if isinstance(telegram["payload"], list):
                telegram["payload"] = tuple(telegram["payload"])
        self.recent_telegrams.extend(telegrams)
        self.last_ga_telegrams = {
            t["destination"]: t for t in telegrams if t["payload"] is not None
        }

    async def save_history(self) -> None:
        """Save history to store."""
        if self.recent_telegrams:
            await self._history_store.async_save(list(self.recent_telegrams))

    def _xknx_telegram_cb(self, telegram: Telegram) -> None:
        """Handle incoming and outgoing telegrams from xknx."""
        telegram_dict: TelegramDict = self.telegram_to_dict(telegram)
        self.recent_telegrams.append(telegram_dict)
        if telegram_dict["payload"] is not None:
            self.last_ga_telegrams[telegram_dict["destination"]] = telegram_dict
        async_dispatcher_send(self.hass, SIGNAL_KNX_TELEGRAM, telegram, telegram_dict)

    def telegram_to_dict(self, telegram: Telegram) -> TelegramDict:
        """Convert a Telegram to a dict."""
        dst_name: str = ""
        payload_data: Optional[Any] = None
        src_name: str = ""
        transcoder: Optional[DPTBase] = None
        value: Optional[Any] = None

        ga_info: Any = self.project.group_addresses.get(f"{telegram.destination_address}")
        if ga_info is not None:
            dst_name = ga_info.name

        device: Any = self.project.devices.get(f"{telegram.source_address}")
        if device is not None:
            src_name = f"{device['manufacturer_name']} {device['name']}"
        elif telegram.direction is TelegramDirection.OUTGOING:
            src_name = "Home Assistant"

        if isinstance(telegram.payload, (GroupValueWrite, GroupValueResponse)):
            payload_data = telegram.payload.value.value

        if telegram.decoded_data is not None:
            transcoder = telegram.decoded_data.transcoder
            value = _serializable_decoded_data(telegram.decoded_data.value)

        return TelegramDict(
            destination=f"{telegram.destination_address}",
            destination_name=dst_name,
            direction=telegram.direction.value,
            dpt_main=transcoder.dpt_main_number if transcoder is not None else None,
            dpt_sub=transcoder.dpt_sub_number if transcoder is not None else None,
            dpt_name=transcoder.value_type if transcoder is not None else None,
            payload=payload_data,
            source=f"{telegram.source_address}",
            source_name=src_name,
            telegramtype=telegram.payload.__class__.__name__,
            timestamp=dt_util.now().isoformat(),
            unit=transcoder.unit if transcoder is not None else None,
            value=value,
        )


def _serializable_decoded_data(value: Any) -> Any:
    """Return a serializable representation of decoded data."""
    if isinstance(value, DPTComplexData):
        return value.as_dict()
    if isinstance(value, DPTEnumData):
        return value.name.lower()
    return value


def decode_telegram_payload(payload: Any, transcoder: DPTBase) -> DecodedTelegramPayload:
    """Decode the payload of a KNX telegram with custom transcoder."""
    try:
        value = transcoder.from_knx(payload)
    except XKNXException:
        value = "Error decoding value"
    value = _serializable_decoded_data(value)
    return DecodedTelegramPayload(
        dpt_main=transcoder.dpt_main_number,
        dpt_sub=transcoder.dpt_sub_number,
        dpt_name=transcoder.value_type,
        unit=transcoder.unit,
        value=value,
    )