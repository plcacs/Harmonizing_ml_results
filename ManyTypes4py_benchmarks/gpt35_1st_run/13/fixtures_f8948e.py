def get_start_dttm(annotation_id: int) -> datetime:
    return datetime(1990 + annotation_id, 1, 1)

def get_end_dttm(annotation_id: int) -> datetime:
    return datetime(1990 + annotation_id, 7, 1)

def _insert_annotation_layer(name: str = '', descr: str = '') -> AnnotationLayer:

def _insert_annotation(layer: AnnotationLayer, short_descr: str, long_descr: str, json_metadata: str = '', start_dttm: Optional[datetime] = None, end_dttm: Optional[datetime] = None) -> Annotation:
