"""ONVIF device abstraction."""
from __future__ import annotations
import asyncio
from contextlib import suppress
import datetime as dt
import os
import time
from typing import Any, List, Optional
from httpx import RequestError
import onvif
from onvif import ONVIFCamera
from onvif.exceptions import ONVIFError
from zeep.exceptions import Fault, TransportError, XMLParseError, XMLSyntaxError
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PASSWORD, CONF_PORT, CONF_USERNAME, Platform
from homeassistant.core import HomeAssistant, callback
from homeassistant.util import dt as dt_util
from .const import ABSOLUTE_MOVE, CONF_ENABLE_WEBHOOKS, CONTINUOUS_MOVE, DEFAULT_ENABLE_WEBHOOKS, GET_CAPABILITIES_EXCEPTIONS, GOTOPRESET_MOVE, LOGGER, PAN_FACTOR, RELATIVE_MOVE, STOP_MOVE, TILT_FACTOR, ZOOM_FACTOR
from .event import EventManager
from .models import PTZ, Capabilities, DeviceInfo, Profile, Resolution, Video

class ONVIFDevice:
    """Manages an ONVIF device."""

    def __init__(self, hass: HomeAssistant, config_entry: ConfigEntry) -> None:
        """Initialize the device."""
        self.hass = hass
        self.config_entry = config_entry
        self._original_options = dict(config_entry.options)
        self.available = True
        self.info = DeviceInfo()
        self.capabilities = Capabilities()
        self.onvif_capabilities: Optional[dict] = None
        self.profiles: List[Profile] = []
        self.max_resolution = 0
        self.platforms: List[Platform] = []
        self._dt_diff_seconds = 0

    async def _async_update_listener(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Handle options update."""
        if self._original_options != entry.options:
            hass.async_create_task(hass.config_entries.async_reload(entry.entry_id))

    @property
    def name(self) -> str:
        """Return the name of this device."""
        return self.config_entry.data[CONF_NAME]

    @property
    def host(self) -> str:
        """Return the host of this device."""
        return self.config_entry.data[CONF_HOST]

    @property
    def port(self) -> int:
        """Return the port of this device."""
        return self.config_entry.data[CONF_PORT]

    @property
    def username(self) -> str:
        """Return the username of this device."""
        return self.config_entry.data[CONF_USERNAME]

    @property
    def password(self) -> str:
        """Return the password of this device."""
        return self.config_entry.data[CONF_PASSWORD]

    async def async_setup(self) -> None:
        """Set up the device."""
        self.device = get_device(self.hass, host=self.config_entry.data[CONF_HOST], port=self.config_entry.data[CONF_PORT], username=self.config_entry.data[CONF_USERNAME], password=self.config_entry.data[CONF_PASSWORD])
        await self.device.update_xaddrs()
        LOGGER.debug('%s: xaddrs = %s', self.name, self.device.xaddrs)
        self.onvif_capabilities = await self.device.get_capabilities()
        await self.async_check_date_and_time()
        assert self.config_entry.unique_id
        self.events = EventManager(self.hass, self.device, self.config_entry, self.name)
        self.info = await self.async_get_device_info()
        LOGGER.debug('%s: camera info = %s', self.name, self.info)
        LOGGER.debug('%s: fetching initial capabilities', self.name)
        self.capabilities = await self.async_get_capabilities()
        LOGGER.debug('%s: fetching profiles', self.name)
        self.profiles = await self.async_get_profiles()
        LOGGER.debug('Camera %s profiles = %s', self.name, self.profiles)
        if not self.profiles:
            raise ONVIFError('No camera profiles found')
        if self.capabilities.ptz:
            LOGGER.debug('%s: creating PTZ service', self.name)
            await self.device.create_ptz_service()
        self.max_resolution = max((profile.video.resolution.width for profile in self.profiles if profile.video.encoding == 'H264'))
        LOGGER.debug('%s: starting events', self.name)
        self.capabilities.events = await self.async_start_events()
        LOGGER.debug('Camera %s capabilities = %s', self.name, self.capabilities)
        self.config_entry.async_on_unload(self.config_entry.add_update_listener(self._async_update_listener))

    async def async_stop(self, event: Optional[Any] = None) -> None:
        """Shut it all down."""
        if self.events:
            await self.events.async_stop()
        await self.device.close()

    async def async_manually_set_date_and_time(self) -> None:
        """Set Date and Time Manually using SetSystemDateAndTime command."""
        device_mgmt = await self.device.create_devicemgmt_service()
        device_time = await device_mgmt.GetSystemDateAndTime()
        system_date = dt_util.utcnow()
        LOGGER.debug('System date (UTC): %s', system_date)
        dt_param = device_mgmt.create_type('SetSystemDateAndTime')
        dt_param.DateTimeType = 'Manual'
        dt_param.DaylightSavings = bool(time.localtime().tm_isdst)
        dt_param.UTCDateTime = {'Date': {'Year': system_date.year, 'Month': system_date.month, 'Day': system_date.day}, 'Time': {'Hour': system_date.hour, 'Minute': system_date.minute, 'Second': system_date.second}}
        system_timezone = str(system_date.astimezone().tzinfo)
        timezone_names = [system_timezone]
        if (time_zone := device_time.TimeZone) and system_timezone != time_zone.TZ:
            timezone_names.append(time_zone.TZ)
        timezone_names.append(None)
        timezone_max_idx = len(timezone_names) - 1
        LOGGER.debug('%s: SetSystemDateAndTime: timezone_names:%s', self.name, timezone_names)
        for idx, timezone_name in enumerate(timezone_names):
            dt_param.TimeZone = timezone_name
            LOGGER.debug('%s: SetSystemDateAndTime: %s', self.name, dt_param)
            try:
                await device_mgmt.SetSystemDateAndTime(dt_param)
                LOGGER.debug('%s: SetSystemDateAndTime: success', self.name)
            except (IndexError, Fault):
                if idx == timezone_max_idx:
                    raise
            else:
                return

    async def async_check_date_and_time(self) -> None:
        """Warns if device and system date not synced."""
        LOGGER.debug('%s: Setting up the ONVIF device management service', self.name)
        device_mgmt = await self.device.create_devicemgmt_service()
        system_date = dt_util.utcnow()
        LOGGER.debug('%s: Retrieving current device date/time', self.name)
        try:
            device_time = await device_mgmt.GetSystemDateAndTime()
        except RequestError as err:
            LOGGER.warning("Couldn't get device '%s' date/time. Error: %s", self.name, err)
            return
        if not device_time:
            LOGGER.debug("Couldn't get device '%s' date/time.\n                GetSystemDateAndTime() return null/empty", self.name)
            return
        LOGGER.debug('%s: Device time: %s', self.name, device_time)
        tzone = dt_util.get_default_time_zone()
        cdate = device_time.LocalDateTime
        if device_time.UTCDateTime:
            tzone = dt_util.UTC
            cdate = device_time.UTCDateTime
        elif device_time.TimeZone:
            tzone = await dt_util.async_get_time_zone(device_time.TimeZone.TZ) or tzone
        if cdate is None:
            LOGGER.warning('%s: Could not retrieve date/time on this camera', self.name)
            return
        try:
            cam_date = dt.datetime(cdate.Date.Year, cdate.Date.Month, cdate.Date.Day, cdate.Time.Hour, cdate.Time.Minute, cdate.Time.Second, 0, tzone)
        except ValueError as err:
            LOGGER.warning('%s: Could not parse date/time from camera: %s', self.name, err)
            return
        cam_date_utc = cam_date.astimezone(dt_util.UTC)
        LOGGER.debug('%s: Device date/time: %s | System date/time: %s', self.name, cam_date_utc, system_date)
        dt_diff = cam_date - system_date
        self._dt_diff_seconds = dt_diff.total_seconds()
        if abs(self._dt_diff_seconds) < 5:
            return
        if device_time.DateTimeType != 'Manual':
            self._async_log_time_out_of_sync(cam_date_utc, system_date)
            return
        try:
            await self.async_manually_set_date_and_time()
        except (RequestError, TransportError, IndexError, Fault):
            LOGGER.warning('%s: Could not sync date/time on this camera', self.name)
            self._async_log_time_out_of_sync(cam_date_utc, system_date)

    @callback
    def _async_log_time_out_of_sync(self, cam_date_utc: dt.datetime, system_date: dt.datetime) -> None:
        """Log a warning if the camera and system date/time are not synced."""
        LOGGER.warning("The date/time on %s (UTC) is '%s', which is different from the system '%s', this could lead to authentication issues", self.name, cam_date_utc, system_date)

    async def async_get_device_info(self) -> DeviceInfo:
        """Obtain information about this device."""
        device_mgmt = await self.device.create_devicemgmt_service()
        manufacturer = None
        model = None
        firmware_version = None
        serial_number = None
        try:
            device_info = await device_mgmt.GetDeviceInformation()
        except (XMLParseError, XMLSyntaxError, TransportError) as ex:
            LOGGER.warning('%s: Failed to fetch device information: %s', self.name, ex)
        else:
            manufacturer = device_info.Manufacturer
            model = device_info.Model
            firmware_version = device_info.FirmwareVersion
            serial_number = device_info.SerialNumber
        mac = None
        try:
            network_interfaces = await device_mgmt.GetNetworkInterfaces()
            for interface in network_interfaces:
                if interface.Enabled:
                    mac = interface.Info.HwAddress
        except Fault as fault:
            if 'not implemented' not in fault.message:
                raise
            LOGGER.debug("Couldn't get network interfaces from ONVIF device '%s'. Error: %s", self.name, fault)
        return DeviceInfo(manufacturer, model, firmware_version, serial_number, mac)

    async def async_get_capabilities(self) -> Capabilities:
        """Obtain information about the available services on the device."""
        snapshot = False
        with suppress(*GET_CAPABILITIES_EXCEPTIONS):
            media_service = await self.device.create_media_service()
            media_capabilities = await media_service.GetServiceCapabilities()
            snapshot = media_capabilities and media_capabilities.SnapshotUri
        ptz = False
        with suppress(*GET_CAPABILITIES_EXCEPTIONS):
            self.device.get_definition('ptz')
            ptz = True
        imaging = False
        with suppress(*GET_CAPABILITIES_EXCEPTIONS):
            await self.device.create_imaging_service()
            imaging = True
        return Capabilities(snapshot=snapshot, ptz=ptz, imaging=imaging)

    async def async_start_events(self) -> bool:
        """Start the event handler."""
        with suppress(*GET_CAPABILITIES_EXCEPTIONS, XMLParseError):
            onvif_capabilities = self.onvif_capabilities or {}
            pull_point_support = (onvif_capabilities.get('Events') or {}).get('WSPullPointSupport')
            LOGGER.debug('%s: WSPullPointSupport: %s', self.name, pull_point_support)
            return await self.events.async_start(True, self.config_entry.options.get(CONF_ENABLE_WEBHOOKS, DEFAULT_ENABLE_WEBHOOKS))
        return False

    async def async_get_profiles(self) -> List[Profile]:
        """Obtain media profiles for this device."""
        media_service = await self.device.create_media_service()
        LOGGER.debug('%s: xaddr for media_service: %s', self.name, media_service.xaddr)
        try:
            result = await media_service.GetProfiles()
        except GET_CAPABILITIES_EXCEPTIONS:
            LOGGER.debug('%s: Could not get profiles from ONVIF device', self.name, exc_info=True)
            raise
        profiles = []
        if not isinstance(result, list):
            return profiles
        for key, onvif_profile in enumerate(result):
            if not onvif_profile.VideoEncoderConfiguration or onvif_profile.VideoEncoderConfiguration.Encoding != 'H264':
                continue
            profile = Profile(key, onvif_profile.token, onvif_profile.Name, Video(onvif_profile.VideoEncoderConfiguration.Encoding, Resolution(onvif_profile.VideoEncoderConfiguration.Resolution.Width, onvif_profile.VideoEncoderConfiguration.Resolution.Height)))
            if self.capabilities.ptz and onvif_profile.PTZConfiguration:
                profile.ptz = PTZ(onvif_profile.PTZConfiguration.DefaultContinuousPanTiltVelocitySpace is not None, onvif_profile.PTZConfiguration.DefaultRelativePanTiltTranslationSpace is not None, onvif_profile.PTZConfiguration.DefaultAbsolutePantTiltPositionSpace is not None)
                try:
                    ptz_service = await self.device.create_ptz_service()
                    presets = await ptz_service.GetPresets(profile.token)
                    profile.ptz.presets = [preset.token for preset in presets if preset]
                except GET_CAPABILITIES_EXCEPTIONS:
                    profile.ptz.presets = []
            if self.capabilities.imaging and onvif_profile.VideoSourceConfiguration:
                profile.video_source_token = onvif_profile.VideoSourceConfiguration.SourceToken
            profiles.append(profile)
        return profiles

    async def async_get_stream_uri(self, profile: Profile) -> str:
        """Get the stream URI for a specified profile."""
        media_service = await self.device.create_media_service()
        req = media_service.create_type('GetStreamUri')
        req.ProfileToken = profile.token
        req.StreamSetup = {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}}
        result = await media_service.GetStreamUri(req)
        return result.Uri

    async def async_perform_ptz(self, profile: Profile, distance: float, speed: float, move_mode: str, continuous_duration: float, preset: Optional[str], pan: Optional[str] = None, tilt: Optional[str] = None, zoom: Optional[str] = None) -> None:
        """Perform a PTZ action on the camera."""
        if not self.capabilities.ptz:
            LOGGER.warning("PTZ actions are not supported on device '%s'", self.name)
            return
        ptz_service = await self.device.create_ptz_service()
        pan_val = distance * PAN_FACTOR.get(pan, 0)
        tilt_val = distance * TILT_FACTOR.get(tilt, 0)
        zoom_val = distance * ZOOM_FACTOR.get(zoom, 0)
        speed_val = speed
        preset_val = preset
        LOGGER.debug('Calling %s PTZ | Pan = %4.2f | Tilt = %4.2f | Zoom = %4.2f | Speed = %4.2f | Preset = %s', move_mode, pan_val, tilt_val, zoom_val, speed_val, preset_val)
        try:
            req = ptz_service.create_type(move_mode)
            req.ProfileToken = profile.token
            if move_mode == CONTINUOUS_MOVE:
                if not profile.ptz or not profile.ptz.continuous:
                    LOGGER.warning("ContinuousMove not supported on device '%s'", self.name)
                    return
                velocity = {}
                if pan is not None or tilt is not None:
                    velocity['PanTilt'] = {'x': pan_val, 'y': tilt_val}
                if zoom is not None:
                    velocity['Zoom'] = {'x': zoom_val}
                req.Velocity = velocity
                await ptz_service.ContinuousMove(req)
                await asyncio.sleep(continuous_duration)
                req = ptz_service.create_type('Stop')
                req.ProfileToken = profile.token
                await ptz_service.Stop({'ProfileToken': req.ProfileToken, 'PanTilt': True, 'Zoom': False})
            elif move_mode == RELATIVE_MOVE:
                if not profile.ptz or not profile.ptz.relative:
                    LOGGER.warning("RelativeMove not supported on device '%s'", self.name)
                    return
                req.Translation = {'PanTilt': {'x': pan_val, 'y': tilt_val}, 'Zoom': {'x': zoom_val}}
                req.Speed = {'PanTilt': {'x': speed_val, 'y': speed_val}, 'Zoom': {'x': speed_val}}
                await ptz_service.RelativeMove(req)
            elif move_mode == ABSOLUTE_MOVE:
                if not profile.ptz or not profile.ptz.absolute:
                    LOGGER.warning("AbsoluteMove not supported on device '%s'", self.name)
                    return
                req.Position = {'PanTilt': {'x': pan_val, 'y': tilt_val}, 'Zoom': {'x': zoom_val}}
                req.Speed = {'PanTilt': {'x': speed_val, 'y': speed_val}, 'Zoom': {'x': speed_val}}
                await ptz_service.AbsoluteMove(req)
            elif move_mode == GOTOPRESET_MOVE:
                if not profile.ptz or not profile.ptz.presets:
                    LOGGER.warning("Absolute Presets not supported on device '%s'", self.name)
                    return
                if preset_val not in profile.ptz.presets:
                    LOGGER.warning("PTZ preset '%s' does not exist on device '%s'. Available Presets: %s", preset_val, self.name, ', '.join(profile.ptz.presets))
                    return
                req.PresetToken = preset_val
                req.Speed = {'PanTilt': {'x': speed_val, 'y': speed_val}, 'Zoom': {'x': speed_val}}
                await ptz_service.GotoPreset(req)
            elif move_mode == STOP_MOVE:
                await ptz_service.Stop(req)
        except ONVIFError as err:
            if 'Bad Request' in err.reason:
                LOGGER.warning("Device '%s' doesn't support PTZ", self.name)
            else:
                LOGGER.error('Error trying to perform PTZ action: %s', err)

    async def async_run_aux_command(self, profile: Profile, cmd: str) -> None:
        """Execute a PTZ auxiliary command on the camera."""
        if not self.capabilities.ptz:
            LOGGER.warning("PTZ actions are not supported on device '%s'", self.name)
            return
        ptz_service = await self.device.create_ptz_service()
        LOGGER.debug('Running Aux Command | Cmd = %s', cmd)
        try:
            req = ptz_service.create_type('SendAuxiliaryCommand')
            req.ProfileToken = profile.token
            req.AuxiliaryData = cmd
            await ptz_service.SendAuxiliaryCommand(req)
        except ONVIFError as err:
            if 'Bad Request' in err.reason:
                LOGGER.warning("Device '%s' doesn't support PTZ", self.name)
            else:
                LOGGER.error('Error trying to send PTZ auxiliary command: %s', err)

    async def async_set_imaging_settings(self, profile: Profile, settings: dict) -> None:
        """Set an imaging setting on the ONVIF imaging service."""
        if not self.capabilities.imaging:
            LOGGER.warning("The imaging service is not supported on device '%s'", self.name)
            return
        imaging_service = await self.device.create_imaging_service()
        LOGGER.debug('Setting Imaging Setting | Settings = %s', settings)
        try:
            req = imaging_service.create_type('SetImagingSettings')
            req.VideoSourceToken = profile.video_source_token
            req.ImagingSettings = settings
            await imaging_service.SetImagingSettings(req)
        except ONVIFError as err:
            if 'Bad Request' in err.reason:
                LOGGER.warning("Device '%s' doesn't support the Imaging Service", self.name)
            else:
                LOGGER.error('Error trying to set Imaging settings: %s', err)

def get_device(hass: HomeAssistant, host: str, port: int, username: str, password: str) -> ONVIFCamera:
    """Get ONVIFCamera instance."""
    return ONVIFCamera(host, port, username, password, f'{os.path.dirname(onvif.__file__)}/wsdl/', no_cache=True)
