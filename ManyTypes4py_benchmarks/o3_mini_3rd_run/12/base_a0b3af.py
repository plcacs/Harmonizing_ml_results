from __future__ import annotations
from contextlib import suppress
from dataclasses import dataclass
from datetime import timedelta
import logging
import re
from typing import Any, Dict, List, Optional, Union

from fritzconnection.lib.fritzphonebook import FritzPhonebook
from homeassistant.util import Throttle
from .const import REGEX_NUMBER, UNKNOWN_NAME

_LOGGER: logging.Logger = logging.getLogger(__name__)
MIN_TIME_PHONEBOOK_UPDATE: timedelta = timedelta(hours=6)

@dataclass
class Contact:
    """Store details for one phonebook contact."""
    name: str
    numbers: List[str]
    vip: bool

    def __init__(self, name: str, numbers: Optional[List[str]] = None, category: Optional[str] = None) -> None:
        """Initialize the class."""
        self.name = name
        self.numbers = [re.sub(REGEX_NUMBER, '', nr) for nr in numbers or []]
        self.vip = category == '1'

unknown_contact: Contact = Contact(UNKNOWN_NAME)

class FritzBoxPhonebook:
    """Connects to a FritzBox router and downloads its phone book."""

    def __init__(
        self,
        host: str,
        username: str,
        password: str,
        phonebook_id: Optional[str] = None,
        prefixes: Optional[List[str]] = None
    ) -> None:
        """Initialize the class."""
        self.host: str = host
        self.username: str = username
        self.password: str = password
        self.phonebook_id: Optional[str] = phonebook_id
        self.prefixes: Optional[List[str]] = prefixes
        self.fph: Optional[FritzPhonebook] = None
        self.contacts: List[Contact] = []
        self.number_dict: Dict[str, Contact] = {}

    def init_phonebook(self) -> None:
        """Establish a connection to the FRITZ!Box and check if phonebook_id is valid."""
        self.fph = FritzPhonebook(address=self.host, user=self.username, password=self.password)
        self.update_phonebook()

    @Throttle(MIN_TIME_PHONEBOOK_UPDATE)
    def update_phonebook(self) -> None:
        """Update the phone book dictionary."""
        if self.phonebook_id is None:
            return
        assert self.fph is not None  # For type checker, self.fph must be initialized.
        self.fph.get_all_name_numbers(self.phonebook_id)
        self.contacts = [
            Contact(c.name, c.numbers, getattr(c, 'category', None))
            for c in self.fph.phonebook.contacts  # type: ignore
        ]
        self.number_dict = {nr: c for c in self.contacts for nr in c.numbers}
        _LOGGER.debug('Fritz!Box phone book successfully updated')

    def get_phonebook_ids(self) -> List[str]:
        """Return list of phonebook ids."""
        assert self.fph is not None  # Ensure that fph is initialized.
        return self.fph.phonebook_ids

    def get_contact(self, number: Union[str, int]) -> Contact:
        """Return a contact for a given phone number."""
        number_str: str = re.sub(REGEX_NUMBER, '', str(number))
        with suppress(KeyError):
            return self.number_dict[number_str]
        if not self.prefixes:
            return unknown_contact
        for prefix in self.prefixes:
            with suppress(KeyError):
                return self.number_dict[prefix + number_str]
            with suppress(KeyError):
                return self.number_dict[prefix + number_str.lstrip('0')]
        return unknown_contact