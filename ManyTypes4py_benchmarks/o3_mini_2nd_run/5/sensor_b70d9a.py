from __future__ import annotations
import logging
from typing import Any, Dict, List
from homeassistant.components.sensor import SensorEntity, SensorStateClass
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from . import CoinbaseConfigEntry, CoinbaseData
from .const import (
    ACCOUNT_IS_VAULT,
    API_ACCOUNT_AMOUNT,
    API_ACCOUNT_CURRENCY,
    API_ACCOUNT_ID,
    API_ACCOUNT_NAME,
    API_RATES,
    CONF_CURRENCIES,
    CONF_EXCHANGE_PRECISION,
    CONF_EXCHANGE_PRECISION_DEFAULT,
    CONF_EXCHANGE_RATES,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)

ATTR_NATIVE_BALANCE: str = "Balance in native currency"
ATTR_API_VERSION: str = "API Version"
CURRENCY_ICONS: Dict[str, str] = {
    "BTC": "mdi:currency-btc",
    "ETH": "mdi:currency-eth",
    "EUR": "mdi:currency-eur",
    "LTC": "mdi:litecoin",
    "USD": "mdi:currency-usd",
}
DEFAULT_COIN_ICON: str = "mdi:cash"
ATTRIBUTION: str = "Data provided by coinbase.com"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: CoinbaseConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up Coinbase sensor platform."""
    instance: CoinbaseData = config_entry.runtime_data
    entities: List[SensorEntity] = []
    provided_currencies: List[str] = [
        account[API_ACCOUNT_CURRENCY]
        for account in instance.accounts
        if not account[ACCOUNT_IS_VAULT]
    ]
    desired_currencies: List[str] = []
    if CONF_CURRENCIES in config_entry.options:
        desired_currencies = config_entry.options[CONF_CURRENCIES]
    exchange_base_currency: str = instance.exchange_rates[API_ACCOUNT_CURRENCY]
    exchange_precision: int = config_entry.options.get(
        CONF_EXCHANGE_PRECISION, CONF_EXCHANGE_PRECISION_DEFAULT
    )
    for currency in desired_currencies:
        _LOGGER.debug("Attempting to set up %s account sensor with %s API", currency, instance.api_version)
        if currency not in provided_currencies:
            _LOGGER.warning(
                "The currency %s is no longer provided by your account, please check your settings in Coinbase's developer tools",
                currency,
            )
            continue
        entities.append(AccountSensor(instance, currency))
    if CONF_EXCHANGE_RATES in config_entry.options:
        for rate in config_entry.options[CONF_EXCHANGE_RATES]:
            _LOGGER.debug("Attempting to set up %s account sensor with %s API", rate, instance.api_version)
            entities.append(ExchangeRateSensor(instance, rate, exchange_base_currency, exchange_precision))
    async_add_entities(entities)


class AccountSensor(SensorEntity):
    """Representation of a Coinbase.com sensor."""

    _attr_attribution: str = ATTRIBUTION

    def __init__(self, coinbase_data: CoinbaseData, currency: str) -> None:
        """Initialize the sensor."""
        self._coinbase_data: CoinbaseData = coinbase_data
        self._currency: str = currency
        for account in coinbase_data.accounts:
            if account[API_ACCOUNT_CURRENCY] != currency or account[ACCOUNT_IS_VAULT]:
                continue
            self._attr_name = f"Coinbase {account[API_ACCOUNT_NAME]}"
            self._attr_unique_id = f"coinbase-{account[API_ACCOUNT_ID]}-wallet-{account[API_ACCOUNT_CURRENCY]}"
            self._attr_native_value = account[API_ACCOUNT_AMOUNT]
            self._attr_native_unit_of_measurement = account[API_ACCOUNT_CURRENCY]
            self._attr_icon = CURRENCY_ICONS.get(account[API_ACCOUNT_CURRENCY], DEFAULT_COIN_ICON)
            self._native_balance: float = round(
                float(account[API_ACCOUNT_AMOUNT])
                / float(coinbase_data.exchange_rates[API_RATES][currency]),
                2,
            )
            break
        self._attr_state_class = SensorStateClass.TOTAL
        self._attr_device_info = DeviceInfo(
            configuration_url="https://www.coinbase.com/settings/api",
            entry_type=DeviceEntryType.SERVICE,
            identifiers={(DOMAIN, self._coinbase_data.user_id)},
            manufacturer="Coinbase.com",
            name=f"Coinbase {self._coinbase_data.user_id[-4:]}",
        )

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes of the sensor."""
        return {
            ATTR_NATIVE_BALANCE: f"{self._native_balance} {self._coinbase_data.exchange_base}",
            ATTR_API_VERSION: self._coinbase_data.api_version,
        }

    def update(self) -> None:
        """Get the latest state of the sensor."""
        _LOGGER.debug("Updating %s account sensor with %s API", self._currency, self._coinbase_data.api_version)
        self._coinbase_data.update()
        for account in self._coinbase_data.accounts:
            if account[API_ACCOUNT_CURRENCY] != self._currency or account[ACCOUNT_IS_VAULT]:
                continue
            self._attr_native_value = account[API_ACCOUNT_AMOUNT]
            self._native_balance = round(
                float(account[API_ACCOUNT_AMOUNT])
                / float(self._coinbase_data.exchange_rates[API_RATES][self._currency]),
                2,
            )
            break


class ExchangeRateSensor(SensorEntity):
    """Representation of a Coinbase.com sensor."""

    _attr_attribution: str = ATTRIBUTION

    def __init__(
        self,
        coinbase_data: CoinbaseData,
        exchange_currency: str,
        exchange_base: str,
        precision: int,
    ) -> None:
        """Initialize the sensor."""
        self._coinbase_data: CoinbaseData = coinbase_data
        self._currency: str = exchange_currency
        self._attr_name = f"{exchange_currency} Exchange Rate"
        self._attr_unique_id = f"coinbase-{coinbase_data.user_id}-xe-{exchange_currency}"
        self._precision: int = precision
        self._attr_icon = CURRENCY_ICONS.get(exchange_currency, DEFAULT_COIN_ICON)
        self._attr_native_value = round(
            1 / float(coinbase_data.exchange_rates[API_RATES][exchange_currency]), precision
        )
        self._attr_native_unit_of_measurement = exchange_base
        self._attr_state_class = SensorStateClass.MEASUREMENT
        self._attr_device_info = DeviceInfo(
            configuration_url="https://www.coinbase.com/settings/api",
            entry_type=DeviceEntryType.SERVICE,
            identifiers={(DOMAIN, self._coinbase_data.user_id)},
            manufacturer="Coinbase.com",
            name=f"Coinbase {self._coinbase_data.user_id[-4:]}",
        )

    def update(self) -> None:
        """Get the latest state of the sensor."""
        _LOGGER.debug("Updating %s rate sensor with %s API", self._currency, self._coinbase_data.api_version)
        self._coinbase_data.update()
        self._attr_native_value = round(
            1 / float(self._coinbase_data.exchange_rates[API_RATES][self._currency]), self._precision
        )