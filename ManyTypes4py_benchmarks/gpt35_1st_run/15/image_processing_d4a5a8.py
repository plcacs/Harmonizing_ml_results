def setup_platform(hass: HomeAssistant, config: ConfigType, add_entities: AddEntitiesCallback, discovery_info: DiscoveryInfoType = None) -> None:
    api_key: str = config[CONF_API_KEY]
    account_type: str = config[CONF_ACCOUNT_TYPE]
    api: hound = hound.cloud(api_key, account_type)
    try:
        api.detect(b'Test')
    except hound.SimplehoundException as exc:
        _LOGGER.error('Sighthound error %s setup aborted', exc)
        return
    if (save_file_folder := config.get(CONF_SAVE_FILE_FOLDER)):
        save_file_folder: Path = Path(save_file_folder)
    entities: List[SighthoundEntity] = []
    for camera in config[CONF_SOURCE]:
        sighthound: SighthoundEntity = SighthoundEntity(api, camera[CONF_ENTITY_ID], camera.get(CONF_NAME), save_file_folder, config[CONF_SAVE_TIMESTAMPTED_FILE])
        entities.append(sighthound)
    add_entities(entities)

class SighthoundEntity(ImageProcessingEntity):
    _attr_should_poll: bool = False
    _attr_unit_of_measurement: str = ATTR_PEOPLE

    def __init__(self, api: hound, camera_entity: str, name: str, save_file_folder: Path, save_timestamped_file: bool) -> None:
        self._api: hound = api
        self._camera: str = camera_entity
        if name:
            self._name: str = name
        else:
            camera_name: str = split_entity_id(camera_entity)[1]
            self._name: str = f'sighthound_{camera_name}'
        self._state: Optional[int] = None
        self._last_detection: Optional[str] = None
        self._image_width: Optional[int] = None
        self._image_height: Optional[int] = None
        self._save_file_folder: Path = save_file_folder
        self._save_timestamped_file: bool = save_timestamped_file

    def process_image(self, image: bytes) -> None:
        detections: List[hound] = self._api.detect(image)
        people: List[hound] = hound.get_people(detections)
        self._state: int = len(people)
        if self._state > 0:
            self._last_detection: str = dt_util.now().strftime(DATETIME_FORMAT)
        metadata: Dict[str, int] = hound.get_metadata(detections)
        self._image_width: int = metadata['image_width']
        self._image_height: int = metadata['image_height']
        for person in people:
            self.fire_person_detected_event(person)
        if self._save_file_folder and self._state > 0:
            self.save_image(image, people, self._save_file_folder)

    def fire_person_detected_event(self, person: hound) -> None:
        self.hass.bus.fire(EVENT_PERSON_DETECTED, {ATTR_ENTITY_ID: self.entity_id, ATTR_BOUNDING_BOX: hound.bbox_to_tf_style(person['boundingBox'], self._image_width, self._image_height)})

    def save_image(self, image: bytes, people: List[hound], directory: Path) -> None:
        try:
            img: Image = Image.open(io.BytesIO(bytearray(image))).convert('RGB')
        except UnidentifiedImageError:
            _LOGGER.warning('Sighthound unable to process image, bad data')
            return
        draw: ImageDraw = ImageDraw.Draw(img)
        for person in people:
            box: Tuple[int, int, int, int] = hound.bbox_to_tf_style(person['boundingBox'], self._image_width, self._image_height)
            draw_box(draw, box, self._image_width, self._image_height)
        latest_save_path: Path = directory / f'{self._name}_latest.jpg'
        img.save(latest_save_path)
        if self._save_timestamped_file:
            timestamp_save_path: Path = directory / f'{self._name}_{self._last_detection}.jpg'
            img.save(timestamp_save_path)
            _LOGGER.debug('Sighthound saved file %s', timestamp_save_path)

    @property
    def camera_entity(self) -> str:
        return self._camera

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> Optional[int]:
        return self._state

    @property
    def extra_state_attributes(self) -> Dict[str, str]:
        if not self._last_detection:
            return {}
        return {'last_person': self._last_detection}
