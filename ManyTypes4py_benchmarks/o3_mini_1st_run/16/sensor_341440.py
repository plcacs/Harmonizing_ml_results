from __future__ import annotations
import datetime
import logging
import os
import threading
from typing import Any, Dict, Optional
import pygtfs
from sqlalchemy.engine import Result
from sqlalchemy.sql import text
import voluptuous as vol
from homeassistant.components.sensor import PLATFORM_SCHEMA as SENSOR_PLATFORM_SCHEMA, SensorDeviceClass, SensorEntity
from homeassistant.const import CONF_NAME, CONF_OFFSET, STATE_UNKNOWN
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.util import dt as dt_util, slugify

_LOGGER = logging.getLogger(__name__)

ATTR_ARRIVAL = "arrival"
ATTR_BICYCLE = "trip_bikes_allowed_state"
ATTR_DAY = "day"
ATTR_FIRST = "first"
ATTR_DROP_OFF_DESTINATION = "destination_stop_drop_off_type_state"
ATTR_DROP_OFF_ORIGIN = "origin_stop_drop_off_type_state"
ATTR_INFO = "info"
ATTR_OFFSET = CONF_OFFSET
ATTR_LAST = "last"
ATTR_LOCATION_DESTINATION = "destination_station_location_type_name"
ATTR_LOCATION_ORIGIN = "origin_station_location_type_name"
ATTR_PICKUP_DESTINATION = "destination_stop_pickup_type_state"
ATTR_PICKUP_ORIGIN = "origin_stop_pickup_type_state"
ATTR_ROUTE_TYPE = "route_type_name"
ATTR_TIMEPOINT_DESTINATION = "destination_stop_timepoint_exact"
ATTR_TIMEPOINT_ORIGIN = "origin_stop_timepoint_exact"
ATTR_WHEELCHAIR = "trip_wheelchair_access_available"
ATTR_WHEELCHAIR_DESTINATION = "destination_station_wheelchair_boarding_available"
ATTR_WHEELCHAIR_ORIGIN = "origin_station_wheelchair_boarding_available"
CONF_DATA = "data"
CONF_DESTINATION = "destination"
CONF_ORIGIN = "origin"
CONF_TOMORROW = "include_tomorrow"
DEFAULT_NAME = "GTFS Sensor"
DEFAULT_PATH = "gtfs"
BICYCLE_ALLOWED_DEFAULT = STATE_UNKNOWN
BICYCLE_ALLOWED_OPTIONS: Dict[int, bool] = {1: True, 2: False}
DROP_OFF_TYPE_DEFAULT = STATE_UNKNOWN
DROP_OFF_TYPE_OPTIONS: Dict[int, str] = {0: "Regular", 1: "Not Available", 2: "Call Agency", 3: "Contact Driver"}
ICON = "mdi:train"
ICONS: Dict[int, str] = {
    0: "mdi:tram",
    1: "mdi:subway",
    2: "mdi:train",
    3: "mdi:bus",
    4: "mdi:ferry",
    5: "mdi:train-variant",
    6: "mdi:gondola",
    7: "mdi:stairs",
    100: "mdi:train",
    101: "mdi:train",
    102: "mdi:train",
    103: "mdi:train",
    104: "mdi:train-car",
    105: "mdi:train",
    106: "mdi:train",
    107: "mdi:train",
    108: "mdi:train",
    109: "mdi:train",
    110: "mdi:train-variant",
    111: "mdi:train-variant",
    112: "mdi:train-variant",
    113: "mdi:train-variant",
    114: "mdi:train-variant",
    115: "mdi:train-variant",
    116: "mdi:train-variant",
    117: "mdi:train-variant",
    200: "mdi:bus",
    201: "mdi:bus",
    202: "mdi:bus",
    203: "mdi:bus",
    204: "mdi:bus",
    205: "mdi:bus",
    206: "mdi:bus",
    207: "mdi:bus",
    208: "mdi:bus",
    209: "mdi:bus",
    400: "mdi:subway-variant",
    401: "mdi:subway-variant",
    402: "mdi:subway",
    403: "mdi:subway-variant",
    404: "mdi:subway-variant",
    405: "mdi:subway-variant",
    700: "mdi:bus",
    701: "mdi:bus",
    702: "mdi:bus",
    703: "mdi:bus",
    704: "mdi:bus",
    705: "mdi:bus",
    706: "mdi:bus",
    707: "mdi:bus",
    708: "mdi:bus",
    709: "mdi:bus",
    710: "mdi:bus",
    711: "mdi:bus",
    712: "mdi:bus-school",
    713: "mdi:bus-school",
    714: "mdi:bus",
    715: "mdi:bus",
    716: "mdi:bus",
    800: "mdi:bus",
    900: "mdi:tram",
    901: "mdi:tram",
    902: "mdi:tram",
    903: "mdi:tram",
    904: "mdi:tram",
    905: "mdi:tram",
    906: "mdi:tram",
    1000: "mdi:ferry",
    1100: "mdi:airplane",
    1200: "mdi:ferry",
    1300: "mdi:airplane",
    1400: "mdi:gondola",
    1500: "mdi:taxi",
    1501: "mdi:taxi",
    1502: "mdi:ferry",
    1503: "mdi:train-variant",
    1504: "mdi:bicycle-basket",
    1505: "mdi:taxi",
    1506: "mdi:car-multiple",
    1507: "mdi:taxi",
    1700: "mdi:train-car",
    1702: "mdi:horse-variant",
}
LOCATION_TYPE_DEFAULT = "Stop"
LOCATION_TYPE_OPTIONS: Dict[int, str] = {0: "Station", 1: "Stop", 2: "Station Entrance/Exit", 3: "Other"}
PICKUP_TYPE_DEFAULT = STATE_UNKNOWN
PICKUP_TYPE_OPTIONS: Dict[int, str] = {0: "Regular", 1: "None Available", 2: "Call Agency", 3: "Contact Driver"}
ROUTE_TYPE_OPTIONS: Dict[int, str] = {
    0: "Tram",
    1: "Subway",
    2: "Rail",
    3: "Bus",
    4: "Ferry",
    5: "Cable Tram",
    6: "Aerial Lift",
    7: "Funicular",
    100: "Railway Service",
    101: "High Speed Rail Service",
    102: "Long Distance Trains",
    103: "Inter Regional Rail Service",
    104: "Car Transport Rail Service",
    105: "Sleeper Rail Service",
    106: "Regional Rail Service",
    107: "Tourist Railway Service",
    108: "Rail Shuttle (Within Complex)",
    109: "Suburban Railway",
    110: "Replacement Rail Service",
    111: "Special Rail Service",
    112: "Lorry Transport Rail Service",
    113: "All Rail Services",
    114: "Cross-Country Rail Service",
    115: "Vehicle Transport Rail Service",
    116: "Rack and Pinion Railway",
    117: "Additional Rail Service",
    200: "Coach Service",
    201: "International Coach Service",
    202: "National Coach Service",
    203: "Shuttle Coach Service",
    204: "Regional Coach Service",
    205: "Special Coach Service",
    206: "Sightseeing Coach Service",
    207: "Tourist Coach Service",
    208: "Commuter Coach Service",
    209: "All Coach Services",
    400: "Urban Railway Service",
    401: "Metro Service",
    402: "Underground Service",
    403: "Urban Railway Service",
    404: "All Urban Railway Services",
    405: "Monorail",
    700: "Bus Service",
    701: "Regional Bus Service",
    702: "Express Bus Service",
    703: "Stopping Bus Service",
    704: "Local Bus Service",
    705: "Night Bus Service",
    706: "Post Bus Service",
    707: "Special Needs Bus",
    708: "Mobility Bus Service",
    709: "Mobility Bus for Registered Disabled",
    710: "Sightseeing Bus",
    711: "Shuttle Bus",
    712: "School Bus",
    713: "School and Public Service Bus",
    714: "Rail Replacement Bus Service",
    715: "Demand and Response Bus Service",
    716: "All Bus Services",
    800: "Trolleybus Service",
    900: "Tram Service",
    901: "City Tram Service",
    902: "Local Tram Service",
    903: "Regional Tram Service",
    904: "Sightseeing Tram Service",
    905: "Shuttle Tram Service",
    906: "All Tram Services",
    1000: "Water Transport Service",
    1100: "Air Service",
    1200: "Ferry Service",
    1300: "Aerial Lift Service",
    1400: "Funicular Service",
    1500: "Taxi Service",
    1501: "Communal Taxi Service",
    1502: "Water Taxi Service",
    1503: "Rail Taxi Service",
    1504: "Bike Taxi Service",
    1505: "Licensed Taxi Service",
    1506: "Private Hire Service Vehicle",
    1507: "All Taxi Services",
    1700: "Miscellaneous Service",
    1702: "Horse-drawn Carriage",
}
TIMEPOINT_DEFAULT = True
TIMEPOINT_OPTIONS: Dict[int, bool] = {0: False, 1: True}
WHEELCHAIR_ACCESS_DEFAULT = STATE_UNKNOWN
WHEELCHAIR_ACCESS_OPTIONS: Dict[int, bool] = {1: True, 2: False}
WHEELCHAIR_BOARDING_DEFAULT = STATE_UNKNOWN
WHEELCHAIR_BOARDING_OPTIONS: Dict[int, bool] = {1: True, 2: False}

PLATFORM_SCHEMA = SENSOR_PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_ORIGIN): cv.string,
        vol.Required(CONF_DESTINATION): cv.string,
        vol.Required(CONF_DATA): cv.string,
        vol.Optional(CONF_NAME): cv.string,
        vol.Optional(CONF_OFFSET, default=0): cv.time_period,
        vol.Optional(CONF_TOMORROW, default=False): cv.boolean,
    }
)


def get_next_departure(
    schedule: pygtfs.Schedule,
    start_station_id: str,
    end_station_id: str,
    offset: datetime.timedelta,
    include_tomorrow: bool = False,
) -> Dict[str, Any]:
    now = dt_util.now().replace(tzinfo=None) + offset
    now_date: str = now.strftime(dt_util.DATE_STR_FORMAT)
    yesterday: datetime.datetime = now - datetime.timedelta(days=1)
    yesterday_date: str = yesterday.strftime(dt_util.DATE_STR_FORMAT)
    tomorrow: datetime.datetime = now + datetime.timedelta(days=1)
    tomorrow_date: str = tomorrow.strftime(dt_util.DATE_STR_FORMAT)
    limit: int = 24 * 60 * 60 * 2
    tomorrow_select: str = ""
    tomorrow_where: str = ""
    tomorrow_order: str = ""
    if include_tomorrow:
        limit = int(limit / 2 * 3)
        tomorrow_name: str = tomorrow.strftime("%A").lower()
        tomorrow_select = f"calendar.{tomorrow_name} AS tomorrow,"
        tomorrow_where = f"OR calendar.{tomorrow_name} = 1"
        tomorrow_order = f"calendar.{tomorrow_name} DESC,"
    sql_query: str = f"""
        SELECT trip.trip_id, trip.route_id,
               time(origin_stop_time.arrival_time) AS origin_arrival_time,
               time(origin_stop_time.departure_time) AS origin_depart_time,
               date(origin_stop_time.departure_time) AS origin_depart_date,
               origin_stop_time.drop_off_type AS origin_drop_off_type,
               origin_stop_time.pickup_type AS origin_pickup_type,
               origin_stop_time.shape_dist_traveled AS origin_dist_traveled,
               origin_stop_time.stop_headsign AS origin_stop_headsign,
               origin_stop_time.stop_sequence AS origin_stop_sequence,
               origin_stop_time.timepoint AS origin_stop_timepoint,
               time(destination_stop_time.arrival_time) AS dest_arrival_time,
               time(destination_stop_time.departure_time) AS dest_depart_time,
               destination_stop_time.drop_off_type AS dest_drop_off_type,
               destination_stop_time.pickup_type AS dest_pickup_type,
               destination_stop_time.shape_dist_traveled AS dest_dist_traveled,
               destination_stop_time.stop_headsign AS dest_stop_headsign,
               destination_stop_time.stop_sequence AS dest_stop_sequence,
               destination_stop_time.timepoint AS dest_stop_timepoint,
               calendar.{yesterday.strftime("%A").lower()} AS yesterday,
               calendar.{now.strftime("%A").lower()} AS today,
               {tomorrow_select}
               calendar.start_date AS start_date,
               calendar.end_date AS end_date
        FROM trips trip
        INNER JOIN calendar calendar
                   ON trip.service_id = calendar.service_id
        INNER JOIN stop_times origin_stop_time
                   ON trip.trip_id = origin_stop_time.trip_id
        INNER JOIN stops start_station
                   ON origin_stop_time.stop_id = start_station.stop_id
        INNER JOIN stop_times destination_stop_time
                   ON trip.trip_id = destination_stop_time.trip_id
        INNER JOIN stops end_station
                   ON destination_stop_time.stop_id = end_station.stop_id
        WHERE (calendar.{yesterday.strftime("%A").lower()} = 1
               OR calendar.{now.strftime("%A").lower()} = 1
               {tomorrow_where}
               )
        AND start_station.stop_id = :origin_station_id
                   AND end_station.stop_id = :end_station_id
        AND origin_stop_sequence < dest_stop_sequence
        AND calendar.start_date <= :today
        AND calendar.end_date >= :today
        ORDER BY calendar.{yesterday.strftime("%A").lower()} DESC,
                 calendar.{now.strftime("%A").lower()} DESC,
                 {tomorrow_order}
                 origin_stop_time.departure_time
        LIMIT :limit
        """
    result: Result = schedule.engine.connect().execute(
        text(sql_query),
        {
            "origin_station_id": start_station_id,
            "end_station_id": end_station_id,
            "today": now_date,
            "limit": limit,
        },
    )
    timetable: Dict[str, Any] = {}
    yesterday_start: Optional[str] = None
    today_start: Optional[str] = None
    tomorrow_start: Optional[str] = None
    yesterday_last: str = ""
    today_last: str = ""
    for row_cursor in result:
        row: Dict[str, Any] = row_cursor._asdict()
        if row["yesterday"] == 1 and yesterday_date >= row["start_date"]:
            extras: Dict[str, Any] = {"day": "yesterday", "first": None, "last": False}
            if yesterday_start is None:
                yesterday_start = row["origin_depart_date"]
            if yesterday_start != row["origin_depart_date"]:
                idx: str = f"{now_date} {row['origin_depart_time']}"
                timetable[idx] = {**row, **extras}
                yesterday_last = idx
        if row["today"] == 1:
            extras = {"day": "today", "first": False, "last": False}
            if today_start is None:
                today_start = row["origin_depart_date"]
                extras["first"] = True
            if today_start == row["origin_depart_date"]:
                idx_prefix: str = now_date
            else:
                idx_prefix = tomorrow_date
            idx = f"{idx_prefix} {row['origin_depart_time']}"
            timetable[idx] = {**row, **extras}
            today_last = idx
        if "tomorrow" in row and row["tomorrow"] == 1 and (tomorrow_date <= row["end_date"]):
            extras = {"day": "tomorrow", "first": False, "last": None}
            if tomorrow_start is None:
                tomorrow_start = row["origin_depart_date"]
                extras["first"] = True
            if tomorrow_start == row["origin_depart_date"]:
                idx = f"{tomorrow_date} {row['origin_depart_time']}"
                timetable[idx] = {**row, **extras}
    for idx in filter(None, [yesterday_last, today_last]):
        timetable[idx]["last"] = True
    _LOGGER.debug("Timetable: %s", sorted(timetable.keys()))
    item: Dict[str, Any] = {}
    for key in sorted(timetable.keys()):
        if (value := dt_util.parse_datetime(key)) is not None and value > now:
            item = timetable[key]
            _LOGGER.debug("Departure found for station %s @ %s -> %s", start_station_id, key, item)
            break
    if item == {}:
        return {}
    origin_arrival: datetime.datetime = now
    if item["origin_arrival_time"] > item["origin_depart_time"]:
        origin_arrival -= datetime.timedelta(days=1)
    origin_arrival_time: str = f"{origin_arrival.strftime(dt_util.DATE_STR_FORMAT)} {item['origin_arrival_time']}"
    origin_depart_time: str = f"{now_date} {item['origin_depart_time']}"
    dest_arrival: datetime.datetime = now
    if item["dest_arrival_time"] < item["origin_depart_time"]:
        dest_arrival += datetime.timedelta(days=1)
    dest_arrival_time: str = f"{dest_arrival.strftime(dt_util.DATE_STR_FORMAT)} {item['dest_arrival_time']}"
    dest_depart: datetime.datetime = dest_arrival
    if item["dest_depart_time"] < item["dest_arrival_time"]:
        dest_depart += datetime.timedelta(days=1)
    dest_depart_time: str = f"{dest_depart.strftime(dt_util.DATE_STR_FORMAT)} {item['dest_depart_time']}"
    depart_time: Optional[datetime.datetime] = dt_util.parse_datetime(origin_depart_time)
    arrival_time: Optional[datetime.datetime] = dt_util.parse_datetime(dest_arrival_time)
    origin_stop_time: Dict[str, Any] = {
        "Arrival Time": origin_arrival_time,
        "Departure Time": origin_depart_time,
        "Drop Off Type": item["origin_drop_off_type"],
        "Pickup Type": item["origin_pickup_type"],
        "Shape Dist Traveled": item["origin_dist_traveled"],
        "Headsign": item["origin_stop_headsign"],
        "Sequence": item["origin_stop_sequence"],
        "Timepoint": item["origin_stop_timepoint"],
    }
    destination_stop_time: Dict[str, Any] = {
        "Arrival Time": dest_arrival_time,
        "Departure Time": dest_depart_time,
        "Drop Off Type": item["dest_drop_off_type"],
        "Pickup Type": item["dest_pickup_type"],
        "Shape Dist Traveled": item["dest_dist_traveled"],
        "Headsign": item["dest_stop_headsign"],
        "Sequence": item["dest_stop_sequence"],
        "Timepoint": item["dest_stop_timepoint"],
    }
    return {
        "trip_id": item["trip_id"],
        "route_id": item["route_id"],
        "day": item["day"],
        "first": item["first"],
        "last": item["last"],
        "departure_time": depart_time,
        "arrival_time": arrival_time,
        "origin_stop_time": origin_stop_time,
        "destination_stop_time": destination_stop_time,
    }


def setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    add_entities: AddEntitiesCallback,
    discovery_info: Optional[DiscoveryInfoType] = None,
) -> None:
    gtfs_dir: str = hass.config.path(DEFAULT_PATH)
    data: str = config[CONF_DATA]
    origin: str = config.get(CONF_ORIGIN)
    destination: str = config.get(CONF_DESTINATION)
    name: Optional[str] = config.get(CONF_NAME)
    offset: datetime.timedelta = config[CONF_OFFSET]
    include_tomorrow: bool = config[CONF_TOMORROW]
    os.makedirs(gtfs_dir, exist_ok=True)
    if not os.path.exists(os.path.join(gtfs_dir, data)):
        _LOGGER.error("The given GTFS data file/folder was not found")
        return
    gtfs_root, _ = os.path.splitext(data)
    sqlite_file: str = f"{gtfs_root}.sqlite?check_same_thread=False"
    joined_path: str = os.path.join(gtfs_dir, sqlite_file)
    gtfs: pygtfs.Schedule = pygtfs.Schedule(joined_path)
    if not gtfs.feeds:
        pygtfs.append_feed(gtfs, os.path.join(gtfs_dir, data))
    add_entities([GTFSDepartureSensor(gtfs, name, origin, destination, offset, include_tomorrow)])


class GTFSDepartureSensor(SensorEntity):
    _attr_device_class = SensorDeviceClass.TIMESTAMP

    def __init__(
        self,
        gtfs: pygtfs.Schedule,
        name: Optional[str],
        origin: str,
        destination: str,
        offset: datetime.timedelta,
        include_tomorrow: bool,
    ) -> None:
        self._pygtfs: pygtfs.Schedule = gtfs
        self.origin: str = origin
        self.destination: str = destination
        self._include_tomorrow: bool = include_tomorrow
        self._offset: datetime.timedelta = offset
        self._custom_name: Optional[str] = name
        self._available: bool = False
        self._icon: str = ICON
        self._name: str = ""
        self._state: Optional[datetime.datetime] = None
        self._attributes: Dict[str, Any] = {}
        self._agency: Optional[Any] = None
        self._departure: Dict[str, Any] = {}
        self._destination: Optional[Any] = None
        self._origin: Optional[Any] = None
        self._route: Optional[Any] = None
        self._trip: Optional[Any] = None
        self.lock: threading.Lock = threading.Lock()
        self.update()

    @property
    def name(self) -> str:
        return self._name

    @property
    def native_value(self) -> Optional[datetime.datetime]:
        return self._state

    @property
    def available(self) -> bool:
        return self._available

    @property
    def extra_state_attributes(self) -> Dict[str, Any]:
        return self._attributes

    @property
    def icon(self) -> str:
        return self._icon

    def update(self) -> None:
        with self.lock:
            if not self._origin:
                stops = self._pygtfs.stops_by_id(self.origin)
                if not stops:
                    self._available = False
                    _LOGGER.warning("Origin stop ID %s not found", self.origin)
                    return
                self._origin = stops[0]
            if not self._destination:
                stops = self._pygtfs.stops_by_id(self.destination)
                if not stops:
                    self._available = False
                    _LOGGER.warning("Destination stop ID %s not found", self.destination)
                    return
                self._destination = stops[0]
            self._available = True
            self._departure = get_next_departure(
                self._pygtfs, self.origin, self.destination, self._offset, self._include_tomorrow
            )
            if not self._departure:
                self._trip = None
            else:
                trip_id: Any = self._departure["trip_id"]
                if not self._trip or self._trip.trip_id != trip_id:
                    _LOGGER.debug("Fetching trip details for %s", trip_id)
                    self._trip = self._pygtfs.trips_by_id(trip_id)[0]
                route_id: Any = self._departure["route_id"]
                if not self._route or self._route.route_id != route_id:
                    _LOGGER.debug("Fetching route details for %s", route_id)
                    self._route = self._pygtfs.routes_by_id(route_id)[0]
            if self._agency is None and self._route:
                _LOGGER.debug("Fetching agency details for %s", self._route.agency_id)
                try:
                    self._agency = self._pygtfs.agencies_by_id(self._route.agency_id)[0]
                except IndexError:
                    _LOGGER.warning(
                        "Agency ID '%s' was not found in agency table, you may want to update the routes database table to fix this missing reference",
                        self._route.agency_id,
                    )
                    self._agency = False
            if not self._departure:
                self._state = None
            elif self._agency:
                self._state = self._departure["departure_time"].replace(
                    tzinfo=dt_util.get_time_zone(self._agency.agency_timezone)
                )
            else:
                self._state = self._departure["departure_time"].replace(tzinfo=dt_util.UTC)
            self.update_attributes()
            if self._agency:
                self._attr_attribution = self._agency.agency_name
            else:
                self._attr_attribution = None
            if self._route:
                self._icon = ICONS.get(self._route.route_type, ICON)
            else:
                self._icon = ICON
            name_str: str = f"{getattr(self._agency, 'agency_name', DEFAULT_NAME)} {self.origin} to {self.destination} next departure"
            if not self._departure:
                name_str = f"{DEFAULT_NAME}"
            self._name = self._custom_name or name_str

    def update_attributes(self) -> None:
        if self._departure:
            self._attributes[ATTR_ARRIVAL] = dt_util.as_utc(self._departure["arrival_time"]).isoformat()
            self._attributes[ATTR_DAY] = self._departure["day"]
            if self._departure[ATTR_FIRST] is not None:
                self._attributes[ATTR_FIRST] = self._departure["first"]
            elif ATTR_FIRST in self._attributes:
                del self._attributes[ATTR_FIRST]
            if self._departure[ATTR_LAST] is not None:
                self._attributes[ATTR_LAST] = self._departure["last"]
            elif ATTR_LAST in self._attributes:
                del self._attributes[ATTR_LAST]
        else:
            if ATTR_ARRIVAL in self._attributes:
                del self._attributes[ATTR_ARRIVAL]
            if ATTR_DAY in self._attributes:
                del self._attributes[ATTR_DAY]
            if ATTR_FIRST in self._attributes:
                del self._attributes[ATTR_FIRST]
            if ATTR_LAST in self._attributes:
                del self._attributes[ATTR_LAST]
        self._attributes[ATTR_OFFSET] = self._offset.total_seconds() / 60
        if self._state is None:
            self._attributes[ATTR_INFO] = "No more departures" if self._include_tomorrow else "No more departures today"
        elif ATTR_INFO in self._attributes:
            del self._attributes[ATTR_INFO]
        key: str = "agency_id"
        if self._agency and key not in self._attributes:
            self.append_keys(self.dict_for_table(self._agency), "Agency")
        key = "origin_station_stop_id"
        if self._origin and key not in self._attributes:
            self.append_keys(self.dict_for_table(self._origin), "Origin Station")
            self._attributes[ATTR_LOCATION_ORIGIN] = LOCATION_TYPE_OPTIONS.get(self._origin.location_type, LOCATION_TYPE_DEFAULT)
            self._attributes[ATTR_WHEELCHAIR_ORIGIN] = WHEELCHAIR_BOARDING_OPTIONS.get(self._origin.wheelchair_boarding, WHEELCHAIR_BOARDING_DEFAULT)
        key = "destination_station_stop_id"
        if self._destination and key not in self._attributes:
            self.append_keys(self.dict_for_table(self._destination), "Destination Station")
            self._attributes[ATTR_LOCATION_DESTINATION] = LOCATION_TYPE_OPTIONS.get(self._destination.location_type, LOCATION_TYPE_DEFAULT)
            self._attributes[ATTR_WHEELCHAIR_DESTINATION] = WHEELCHAIR_BOARDING_OPTIONS.get(self._destination.wheelchair_boarding, WHEELCHAIR_BOARDING_DEFAULT)
        key = "route_id"
        if not self._route and key in self._attributes:
            self.remove_keys("Route")
        elif self._route and (key not in self._attributes or self._attributes[key] != self._route.route_id):
            self.append_keys(self.dict_for_table(self._route), "Route")
            self._attributes[ATTR_ROUTE_TYPE] = ROUTE_TYPE_OPTIONS[self._route.route_type]
        key = "trip_id"
        if not self._trip and key in self._attributes:
            self.remove_keys("Trip")
        elif self._trip and (key not in self._attributes or self._attributes[key] != self._trip.trip_id):
            self.append_keys(self.dict_for_table(self._trip), "Trip")
            self._attributes[ATTR_BICYCLE] = BICYCLE_ALLOWED_OPTIONS.get(self._trip.bikes_allowed, BICYCLE_ALLOWED_DEFAULT)
            self._attributes[ATTR_WHEELCHAIR] = WHEELCHAIR_ACCESS_OPTIONS.get(self._trip.wheelchair_accessible, WHEELCHAIR_ACCESS_DEFAULT)
        prefix: str = "origin_stop"
        if self._departure:
            self.append_keys(self._departure["origin_stop_time"], prefix)
            self._attributes[ATTR_DROP_OFF_ORIGIN] = DROP_OFF_TYPE_OPTIONS.get(self._departure["origin_stop_time"]["Drop Off Type"], DROP_OFF_TYPE_DEFAULT)
            self._attributes[ATTR_PICKUP_ORIGIN] = PICKUP_TYPE_OPTIONS.get(self._departure["origin_stop_time"]["Pickup Type"], PICKUP_TYPE_DEFAULT)
            self._attributes[ATTR_TIMEPOINT_ORIGIN] = TIMEPOINT_OPTIONS.get(self._departure["origin_stop_time"]["Timepoint"], TIMEPOINT_DEFAULT)
        else:
            self.remove_keys(prefix)
        prefix = "destination_stop"
        if self._departure:
            self.append_keys(self._departure["destination_stop_time"], prefix)
            self._attributes[ATTR_DROP_OFF_DESTINATION] = DROP_OFF_TYPE_OPTIONS.get(self._departure["destination_stop_time"]["Drop Off Type"], DROP_OFF_TYPE_DEFAULT)
            self._attributes[ATTR_PICKUP_DESTINATION] = PICKUP_TYPE_OPTIONS.get(self._departure["destination_stop_time"]["Pickup Type"], PICKUP_TYPE_DEFAULT)
            self._attributes[ATTR_TIMEPOINT_DESTINATION] = TIMEPOINT_OPTIONS.get(self._departure["destination_stop_time"]["Timepoint"], TIMEPOINT_DEFAULT)
        else:
            self.remove_keys(prefix)

    @staticmethod
    def dict_for_table(resource: Any) -> Dict[str, str]:
        _dict: Dict[str, str] = {}
        for column in resource.__table__.columns:
            _dict[column.name] = str(getattr(resource, column.name))
        return _dict

    def append_keys(self, resource: Dict[str, Any], prefix: Optional[str] = None) -> None:
        for attr, val in resource.items():
            if val == "" or val is None or attr == "feed_id":
                continue
            key: str = attr
            if prefix and (not key.startswith(prefix)):
                key = f"{prefix} {key}"
            key = slugify(key)
            self._attributes[key] = val

    def remove_keys(self, prefix: str) -> None:
        self._attributes = {k: v for k, v in self._attributes.items() if not k.startswith(prefix)}