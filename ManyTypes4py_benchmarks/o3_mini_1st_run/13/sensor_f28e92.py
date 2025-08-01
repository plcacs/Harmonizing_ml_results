from __future__ import annotations
from collections import namedtuple
from datetime import timedelta
import logging
from typing import Any, Dict, List, Optional, Tuple
from fints.client import FinTS3PinTanClient
from fints.models import SEPAAccount
from propcache.api import cached_property
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import CONF_NAME, CONF_PIN, CONF_URL, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER = logging.getLogger(__name__)
SCAN_INTERVAL = timedelta(hours=4)
ICON = 'mdi:currency-eur'

BankCredentials = namedtuple('BankCredentials', 'blz login pin url')

CONF_BIN = 'bank_identification_number'
CONF_ACCOUNTS = 'accounts'
CONF_HOLDINGS = 'holdings'
CONF_ACCOUNT = 'account'
ATTR_ACCOUNT = CONF_ACCOUNT
ATTR_BANK = 'bank'
ATTR_ACCOUNT_TYPE = 'account_type'

SCHEMA_ACCOUNTS = vol.Schema({
    vol.Required(CONF_ACCOUNT): cv.string, 
    vol.Optional(CONF_NAME, default=None): vol.Any(None, cv.string)
})

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_BIN): cv.string,
    vol.Required(CONF_USERNAME): cv.string,
    vol.Required(CONF_PIN): cv.string,
    vol.Required(CONF_URL): cv.string,
    vol.Optional(CONF_NAME): cv.string,
    vol.Optional(CONF_ACCOUNTS, default=[]): cv.ensure_list(SCHEMA_ACCOUNTS),
    vol.Optional(CONF_HOLDINGS, default=[]): cv.ensure_list(SCHEMA_ACCOUNTS)
})


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the sensors.

    Login to the bank and get a list of existing accounts. Create a
    sensor for each account.
    """
    credentials: BankCredentials = BankCredentials(
        config[CONF_BIN], config[CONF_USERNAME], config[CONF_PIN], config[CONF_URL]
    )
    fints_name: str = config.get(CONF_NAME, config[CONF_BIN])
    account_config: Dict[str, Optional[str]] = {acc[CONF_ACCOUNT]: acc[CONF_NAME] for acc in config[CONF_ACCOUNTS]}
    holdings_config: Dict[str, Optional[str]] = {acc[CONF_ACCOUNT]: acc[CONF_NAME] for acc in config[CONF_HOLDINGS]}
    client = FinTsClient(credentials, fints_name, account_config, holdings_config)
    balance_accounts, holdings_accounts = client.detect_accounts()
    accounts: List[SensorEntity] = []
    for account in balance_accounts:
        if config[CONF_ACCOUNTS] and account.iban not in account_config:
            _LOGGER.debug('Skipping account %s for bank %s', account.iban, fints_name)
            continue
        account_name: str = account_config.get(account.iban) or f'{fints_name} - {account.iban}'
        accounts.append(FinTsAccount(client, account, account_name))
        _LOGGER.debug('Creating account %s for bank %s', account.iban, fints_name)
    for account in holdings_accounts:
        if config[CONF_HOLDINGS] and account.accountnumber not in holdings_config:
            _LOGGER.debug('Skipping holdings %s for bank %s', account.accountnumber, fints_name)
            continue
        account_name: str = holdings_config.get(account.accountnumber) or f'{fints_name} - {account.accountnumber}'
        accounts.append(FinTsHoldingsAccount(client, account, account_name))
        _LOGGER.debug('Creating holdings %s for bank %s', account.accountnumber, fints_name)
    add_entities(accounts, True)


class FinTsClient:
    """Wrapper around the FinTS3PinTanClient.

    Use this class as Context Manager to get the FinTS3Client object.
    """

    def __init__(
        self, 
        credentials: BankCredentials, 
        name: str, 
        account_config: Dict[str, Optional[str]], 
        holdings_config: Dict[str, Optional[str]]
    ) -> None:
        """Initialize a FinTsClient."""
        self._credentials: BankCredentials = credentials
        self._account_information: Dict[str, Any] = {}
        self._account_information_fetched: bool = False
        self.name: str = name
        self.account_config: Dict[str, Optional[str]] = account_config
        self.holdings_config: Dict[str, Optional[str]] = holdings_config

    @cached_property
    def client(self) -> FinTS3PinTanClient:
        """Get the FinTS client object.

        The FinTS library persists the current dialog with the bank
        and stores bank capabilities. So caching the client is beneficial.
        """
        return FinTS3PinTanClient(
            self._credentials.blz, 
            self._credentials.login, 
            self._credentials.pin, 
            self._credentials.url
        )

    def get_account_information(self, iban: str) -> Optional[Dict[str, Any]]:
        """Get a dictionary of account IBANs as key and account information as value."""
        if not self._account_information_fetched:
            info = self.client.get_information()['accounts']
            self._account_information = {account['iban']: account for account in info}
            self._account_information_fetched = True
        return self._account_information.get(iban, None)

    def is_balance_account(self, account: SEPAAccount) -> bool:
        """Determine if the given account is of type balance account."""
        if not account.iban:
            return False
        account_information = self.get_account_information(account.iban)
        if not account_information:
            return False
        if (account_type := account_information.get('type')):
            return 1 <= account_type <= 9
        if account_information['iban'] in self.account_config or account_information['account_number'] in self.account_config:
            return True
        return False

    def is_holdings_account(self, account: SEPAAccount) -> bool:
        """Determine if the given account is of type holdings account."""
        if not account.iban:
            return False
        account_information = self.get_account_information(account.iban)
        if not account_information:
            return False
        if (account_type := account_information.get('type')):
            return 30 <= account_type <= 39
        if account_information['iban'] in self.holdings_config or account_information['account_number'] in self.holdings_config:
            return True
        return False

    def detect_accounts(self) -> Tuple[List[SEPAAccount], List[SEPAAccount]]:
        """Identify the accounts of the bank."""
        balance_accounts: List[SEPAAccount] = []
        holdings_accounts: List[SEPAAccount] = []
        for account in self.client.get_sepa_accounts():
            if self.is_balance_account(account):
                balance_accounts.append(account)
            elif self.is_holdings_account(account):
                holdings_accounts.append(account)
            else:
                _LOGGER.warning('Could not determine type of account %s from %s', account.iban, self.client.user_id)
        return (balance_accounts, holdings_accounts)


class FinTsAccount(SensorEntity):
    """Sensor for a FinTS balance account.

    A balance account contains an amount of money (=balance). The amount may
    also be negative.
    """

    def __init__(self, client: FinTsClient, account: SEPAAccount, name: str) -> None:
        """Initialize a FinTs balance account."""
        self._client: FinTsClient = client
        self._account: SEPAAccount = account
        self._attr_name: str = name
        self._attr_icon: str = ICON
        self._attr_extra_state_attributes: Dict[str, Any] = {ATTR_ACCOUNT: self._account.iban, ATTR_ACCOUNT_TYPE: 'balance'}
        if self._client.name:
            self._attr_extra_state_attributes[ATTR_BANK] = self._client.name

    def update(self) -> None:
        """Get the current balance and currency for the account."""
        bank: FinTS3PinTanClient = self._client.client
        balance = bank.get_balance(self._account)
        self._attr_native_value = balance.amount.amount
        self._attr_native_unit_of_measurement = balance.amount.currency
        _LOGGER.debug('updated balance of account %s', self._attr_name)


class FinTsHoldingsAccount(SensorEntity):
    """Sensor for a FinTS holdings account.

    A holdings account does not contain money but rather some financial
    instruments, e.g. stocks.
    """

    def __init__(self, client: FinTsClient, account: SEPAAccount, name: str) -> None:
        """Initialize a FinTs holdings account."""
        self._client: FinTsClient = client
        self._attr_name: str = name
        self._account: SEPAAccount = account
        self._holdings: List[Any] = []
        self._attr_icon: str = ICON
        self._attr_native_unit_of_measurement: str = 'EUR'

    def update(self) -> None:
        """Get the current holdings for the account."""
        bank: FinTS3PinTanClient = self._client.client
        self._holdings = bank.get_holdings(self._account)
        self._attr_native_value = sum((h.total_value for h in self._holdings))

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Additional attributes of the sensor.

        Lists each holding of the account with the current value.
        """
        attributes: Dict[str, Any] = {ATTR_ACCOUNT: self._account.accountnumber, ATTR_ACCOUNT_TYPE: 'holdings'}
        if self._client.name:
            attributes[ATTR_BANK] = self._client.name
        for holding in self._holdings:
            total_name: str = f'{holding.name} total'
            attributes[total_name] = holding.total_value
            pieces_name: str = f'{holding.name} pieces'
            attributes[pieces_name] = holding.pieces
            price_name: str = f'{holding.name} price'
            attributes[price_name] = holding.market_value
        return attributes