from __future__ import annotations
import enum
from typing import Any, ClassVar, List, Optional, Set, TYPE_CHECKING, Type, Union
from flask_appbuilder import Model
from markupsafe import escape
from sqlalchemy import Column, Enum, exists, ForeignKey, Integer, orm, String, Table, Text
from sqlalchemy.engine.base import Connection
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.orm.mapper import Mapper
from sqlalchemy.schema import UniqueConstraint
from superset import security_manager
from superset.models.helpers import AuditMixinNullable
if TYPE_CHECKING:
    from superset.connectors.sqla.models import SqlaTable
    from superset.models.core import FavStar
    from superset.models.dashboard import Dashboard
    from superset.models.slice import Slice
    from superset.models.sql_lab import Query
Session = sessionmaker()
user_favorite_tag_table = Table('user_favorite_tag', Model.metadata, Column('user_id', Integer, ForeignKey('ab_user.id')), Column('tag_id', Integer, ForeignKey('tag.id')))

class TagType(enum.Enum):
    """
    Types for tags.

    Objects (queries, charts, dashboards, and datasets) will have with implicit tags based
    on metadata: types, owners and who favorited them. This way, user "alice"
    can find all their objects by querying for the tag `owner:alice`.
    """
    custom = 1
    type = 2
    owner = 3
    favorited_by = 4

class ObjectType(enum.Enum):
    """Object types."""
    query = 1
    chart = 2
    dashboard = 3
    dataset = 4

class Tag(Model, AuditMixinNullable):
    """A tag attached to an object (query, chart, dashboard, or dataset)."""
    __tablename__ = 'tag'
    id: int = Column(Integer, primary_key=True)
    name: str = Column(String(250), unique=True)
    type: TagType = Column(Enum(TagType))
    description: Optional[str] = Column(Text)
    objects: List[TaggedObject] = relationship('TaggedObject', back_populates='tag', overlaps='objects,tags')
    users_favorited: List[Any] = relationship(security_manager.user_model, secondary=user_favorite_tag_table)

class TaggedObject(Model, AuditMixinNullable):
    """An association between an object and a tag."""
    __tablename__ = 'tagged_object'
    id: int = Column(Integer, primary_key=True)
    tag_id: int = Column(Integer, ForeignKey('tag.id'))
    object_id: int = Column(Integer, ForeignKey('dashboards.id'), ForeignKey('slices.id'), ForeignKey('saved_query.id'))
    object_type: ObjectType = Column(Enum(ObjectType))
    tag: Tag = relationship('Tag', back_populates='objects', overlaps='tags')
    __table_args__ = (UniqueConstraint('tag_id', 'object_id', 'object_type', name='uix_tagged_object'),)

    def __str__(self) -> str:
        return f'<TaggedObject: {self.object_type}:{self.object_id} TAG:{self.tag_id}>'

def get_tag(name: str, session: Any, type_: TagType) -> Tag:
    tag_name = name.strip()
    tag = session.query(Tag).filter_by(name=tag_name, type=type_).one_or_none()
    if tag is None:
        tag = Tag(name=escape(tag_name), type=type_)
        session.add(tag)
        session.commit()
    return tag

def get_object_type(class_name: str) -> ObjectType:
    mapping: dict[str, ObjectType] = {'slice': ObjectType.chart, 'dashboard': ObjectType.dashboard, 'query': ObjectType.query, 'dataset': ObjectType.dataset}
    try:
        return mapping[class_name.lower()]
    except KeyError as ex:
        raise Exception(f'No mapping found for {class_name}') from ex

class ObjectUpdater:
    object_type: ClassVar[str] = 'default'

    @classmethod
    def get_owners_ids(cls, target: Any) -> List[int]:
        raise NotImplementedError('Subclass should implement `get_owners_ids`')

    @classmethod
    def get_owner_tag_ids(cls, session: Any, target: Any) -> Set[int]:
        tag_ids: Set[int] = set()
        for owner_id in cls.get_owners_ids(target):
            name = f'owner:{owner_id}'
            tag = get_tag(name, session, TagType.owner)
            tag_ids.add(tag.id)
        return tag_ids

    @classmethod
    def _add_owners(cls, session: Any, target: Any) -> None:
        for owner_id in cls.get_owners_ids(target):
            name = f'owner:{owner_id}'
            tag = get_tag(name, session, TagType.owner)
            cls.add_tag_object_if_not_tagged(session, tag_id=tag.id, object_id=target.id, object_type=cls.object_type)

    @classmethod
    def add_tag_object_if_not_tagged(cls, session: Any, tag_id: int, object_id: int, object_type: str) -> None:
        exists_query = exists().where(TaggedObject.tag_id == tag_id, TaggedObject.object_id == object_id, TaggedObject.object_type == object_type)
        already_tagged = session.query(exists_query).scalar()
        if not already_tagged:
            tagged_object = TaggedObject(tag_id=tag_id, object_id=object_id, object_type=object_type)
            session.add(tagged_object)

    @classmethod
    def after_insert(cls, _mapper: Mapper, connection: Connection, target: Any) -> None:
        with Session(bind=connection) as session:
            cls._add_owners(session, target)
            tag = get_tag(f'type:{cls.object_type}', session, TagType.type)
            cls.add_tag_object_if_not_tagged(session, tag_id=tag.id, object_id=target.id, object_type=cls.object_type)
            session.commit()

    @classmethod
    def after_update(cls, _mapper: Mapper, connection: Connection, target: Any) -> None:
        with Session(bind=connection) as session:
            existing_tags = session.query(TaggedObject).join(Tag).filter(TaggedObject.object_type == cls.object_type, TaggedObject.object_id == target.id, Tag.type == TagType.owner).all()
            existing_owner_tag_ids = {tag.tag_id for tag in existing_tags}
            new_owner_tag_ids = cls.get_owner_tag_ids(session, target)
            for owner_tag_id in new_owner_tag_ids - existing_owner_tag_ids:
                tagged_object = TaggedObject(tag_id=owner_tag_id, object_id=target.id, object_type=cls.object_type)
                session.add(tagged_object)
            for tag in existing_tags:
                if tag.tag_id not in new_owner_tag_ids:
                    session.delete(tag)
            session.commit()

    @classmethod
    def after_delete(cls, _mapper: Mapper, connection: Connection, target: Any) -> None:
        with Session(bind=connection) as session:
            session.query(TaggedObject).filter(TaggedObject.object_type == cls.object_type, TaggedObject.object_id == target.id).delete()
            session.commit()

class ChartUpdater(ObjectUpdater):
    object_type: ClassVar[str] = 'chart'

    @classmethod
    def get_owners_ids(cls, target: 'Slice') -> List[int]:
        return [owner.id for owner in target.owners]

class DashboardUpdater(ObjectUpdater):
    object_type: ClassVar[str] = 'dashboard'

    @classmethod
    def get_owners_ids(cls, target: 'Dashboard') -> List[int]:
        return [owner.id for owner in target.owners]

class QueryUpdater(ObjectUpdater):
    object_type: ClassVar[str] = 'query'

    @classmethod
    def get_owners_ids(cls, target: 'Query') -> List[int]:
        return [target.user_id]

class DatasetUpdater(ObjectUpdater):
    object_type: ClassVar[str] = 'dataset'

    @classmethod
    def get_owners_ids(cls, target: 'SqlaTable') -> List[int]:
        return [owner.id for owner in target.owners]

class FavStarUpdater:

    @classmethod
    def after_insert(cls, _mapper: Mapper, connection: Connection, target: 'FavStar') -> None:
        with Session(bind=connection) as session:
            name = f'favorited_by:{target.user_id}'
            tag = get_tag(name, session, TagType.favorited_by)
            tagged_object = TaggedObject(tag_id=tag.id, object_id=target.obj_id, object_type=get_object_type(target.class_name))
            session.add(tagged_object)
            session.commit()

    @classmethod
    def after_delete(cls, _mapper: Mapper, connection: Connection, target: 'FavStar') -> None:
        with Session(bind=connection) as session:
            name = f'favorited_by:{target.user_id}'
            query = session.query(TaggedObject.id).join(Tag).filter(TaggedObject.object_id == target.obj_id, Tag.type == TagType.favorited_by, Tag.name == name)
            ids = [row[0] for row in query]
            session.query(TaggedObject).filter(TaggedObject.id.in_(ids)).delete(synchronize_session=False)
            session.commit()
