from datetime import datetime
from typing import Optional, Generator, List
import pytest
from flask.ctx import AppContext
from superset import db
from superset.models.annotations import Annotation, AnnotationLayer

ANNOTATION_LAYERS_COUNT: int = 10
ANNOTATIONS_COUNT: int = 5

def get_start_dttm(annotation_id: int) -> datetime:
    return datetime(1990 + annotation_id, 1, 1)

def get_end_dttm(annotation_id: int) -> datetime:
    return datetime(1990 + annotation_id, 7, 1)

def _insert_annotation_layer(name: str = '', descr: str = '') -> AnnotationLayer:
    annotation_layer = AnnotationLayer(name=name, descr=descr)
    db.session.add(annotation_layer)
    db.session.commit()
    return annotation_layer

def _insert_annotation(
    layer: AnnotationLayer,
    short_descr: str,
    long_descr: str,
    json_metadata: str = '',
    start_dttm: Optional[datetime] = None,
    end_dttm: Optional[datetime] = None,
) -> Annotation:
    annotation = Annotation(
        layer=layer,
        short_descr=short_descr,
        long_descr=long_descr,
        json_metadata=json_metadata,
        start_dttm=start_dttm,
        end_dttm=end_dttm,
    )
    db.session.add(annotation)
    db.session.commit()
    return annotation

@pytest.fixture
def create_annotation_layers(app_context: AppContext) -> Generator[List[AnnotationLayer], None, None]:
    """
    Creates ANNOTATION_LAYERS_COUNT-1 layers with no annotations
    and a final one with ANNOTATION_COUNT children
    :return:
    """
    annotation_layers: List[AnnotationLayer] = []
    annotations: List[Annotation] = []
    for cx in range(ANNOTATION_LAYERS_COUNT - 1):
        annotation_layers.append(_insert_annotation_layer(name=f'name{cx}', descr=f'descr{cx}'))
    layer_with_annotations = _insert_annotation_layer('layer_with_annotations')
    annotation_layers.append(layer_with_annotations)
    for cx in range(ANNOTATIONS_COUNT):
        annotations.append(
            _insert_annotation(
                layer_with_annotations,
                short_descr=f'short_descr{cx}',
                long_descr=f'long_descr{cx}',
                start_dttm=get_start_dttm(cx),
                end_dttm=get_end_dttm(cx),
            )
        )
    yield annotation_layers
    for annotation_layer in annotation_layers:
        db.session.delete(annotation_layer)
    for annotation in annotations:
        db.session.delete(annotation)
    db.session.commit()