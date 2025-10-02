"""Support for Reddit."""
from __future__ import annotations
from datetime import timedelta
import logging
from typing import Any, Dict, List, Optional
import praw
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import ATTR_ID, CONF_CLIENT_ID, CONF_CLIENT_SECRET, CONF_MAXIMUM, CONF_PASSWORD, CONF_USERNAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType

_LOGGER: logging.Logger = logging.getLogger(__name__)

CONF_SORT_BY: str = 'sort_by'
CONF_SUBREDDITS: str = 'subreddits'
ATTR_BODY: str = 'body'
ATTR_COMMENTS_NUMBER: str = 'comms_num'
ATTR_CREATED: str = 'created'
ATTR_POSTS: str = 'posts'
ATTR_SUBREDDIT: str = 'subreddit'
ATTR_SCORE: str = 'score'
ATTR_TITLE: str = 'title'
ATTR_URL: str = 'url'
DEFAULT_NAME: str = 'Reddit'
DOMAIN: str = 'reddit'
LIST_TYPES: List[str] = ['top', 'controversial', 'hot', 'new']
SCAN_INTERVAL: timedelta = timedelta(seconds=300)
PLATFORM_SCHEMA: vol.Schema = SENSOR_PLATFORM_SCHEMA.extend({
    vol.Required(CONF_CLIENT_ID): cv.string,
    vol.Required(CONF_CLIENT_SECRET): cv.string,
    vol.Required(CONF_USERNAME): cv.string,
    vol.Required(CONF_PASSWORD): cv.string,
    vol.Required(CONF_SUBREDDITS): vol.All(cv.ensure_list, [cv.string]),
    vol.Optional(CONF_SORT_BY, default='hot'): vol.All(cv.string, vol.In(LIST_TYPES)),
    vol.Optional(CONF_MAXIMUM, default=10): cv.positive_int
})

def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None
) -> None:
    """Set up the Reddit sensor platform."""
    subreddits: List[str] = config[CONF_SUBREDDITS]
    user_agent: str = f'{config[CONF_USERNAME]}_home_assistant_sensor'
    limit: int = config[CONF_MAXIMUM]
    sort_by: str = config[CONF_SORT_BY]
    try:
        reddit: praw.Reddit = praw.Reddit(
            client_id=config[CONF_CLIENT_ID],
            client_secret=config[CONF_CLIENT_SECRET],
            username=config[CONF_USERNAME],
            password=config[CONF_PASSWORD],
            user_agent=user_agent
        )
        _LOGGER.debug('Connected to praw')
    except praw.exceptions.PRAWException as err:
        _LOGGER.error('Reddit error %s', err)
        return
    sensors: List[RedditSensor] = [RedditSensor(reddit, subreddit, limit, sort_by) for subreddit in subreddits]
    add_entities(sensors, True)

class RedditSensor(SensorEntity):
    """Representation of a Reddit sensor."""

    def __init__(
        self,
        reddit: praw.Reddit,
        subreddit: str,
        limit: int,
        sort_by: str
    ) -> None:
        """Initialize the Reddit sensor."""
        self._reddit: praw.Reddit = reddit
        self._subreddit: str = subreddit
        self._limit: int = limit
        self._sort_by: str = sort_by
        self._subreddit_data: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        """Return the name of the sensor."""
        return f'reddit_{self._subreddit}'

    @property
    def native_value(self) -> int:
        """Return the state of the sensor."""
        return len(self._subreddit_data)

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        """Return the state attributes."""
        return {
            ATTR_SUBREDDIT: self._subreddit,
            ATTR_POSTS: self._subreddit_data,
            CONF_SORT_BY: self._sort_by
        }

    @property
    def icon(self) -> str:
        """Return the icon to use in the frontend."""
        return 'mdi:reddit'

    def update(self) -> None:
        """Update data from Reddit API."""
        self._subreddit_data = []
        try:
            subreddit: Any = self._reddit.subreddit(self._subreddit)
            if hasattr(subreddit, self._sort_by):
                method_to_call: Any = getattr(subreddit, self._sort_by)
                for submission in method_to_call(limit=self._limit):
                    self._subreddit_data.append({
                        ATTR_ID: submission.id,
                        ATTR_URL: submission.url,
                        ATTR_TITLE: submission.title,
                        ATTR_SCORE: submission.score,
                        ATTR_COMMENTS_NUMBER: submission.num_comments,
                        ATTR_CREATED: submission.created,
                        ATTR_BODY: submission.selftext
                    })
        except praw.exceptions.PRAWException as err:
            _LOGGER.error('Reddit error %s', err)
