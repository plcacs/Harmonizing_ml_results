from __future__ import annotations
import enum
from typing import TYPE_CHECKING, Any, ClassVar
from flask_appbuilder import Model
from markupsafe import escape
from sqlalchemy import Column, Enum, exists, ForeignKey, Integer, orm, String, Table, Text
from sqlalchemy.engine.base import Connection
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.orm import Session as SASession
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
Session: sessionmaker[SASession] = sessionmaker()
user_favorite_tag_table: Table = Table('user_favorite_tag', Model.metadata, Column('user_id', Integer, ForeignKey('ab_user.id')), Column('tag_id', Integer, ForeignKey('tag.id')))

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
    id = Column(Integer, primary_key=True)
    name = Column(String(250), unique=True)
    type = Column(Enum(TagType))
    description = Column(Text)
    objects = relationship('TaggedObject', back_populates='tag', overlaps='objects,tags')
    users_favorited = relationship(security_manager.user_model, secondary=user_favorite_tag_table)

class TaggedObject(Model, AuditMixinNullable):
    """An association between an object and a tag."""
    __tablename__ = 'tagged_object'
    id = Column(Integer, primary_key=True)
    tag_id = Column(Integer, ForeignKey('tag.id'))
    object_id = Column(Integer, ForeignKey('dashboards.id'), ForeignKey('slices.id'), ForeignKey('saved_query.id'))
    object_type = Column(Enum(ObjectType))
    tag = relationship('Tag', back_populates='objects', overlaps='tags')
    __table_args__ = (UniqueConstraint('tag_id', 'object_id', 'object_type', name='uix_tagged_object'),)

    def __str__(self) -> str:
        return f'<TaggedObject: {self.object_type}:{self.object_id} TAG:{self.tag_id}>'

def get_tag(name: str, session: SASession, type_: TagType) -> Tag:
    tag_name = name.strip()
    tag: Tag | None = session.query(Tag).filter_by(name=tag_name, type=type_).one_or_none()
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
    def get_owners_ids(cls, target: Any) -> list[int]:
        raise NotImplementedError('Subclass should implement `get_owners_ids`')

    @classmethod
    def get_owner_tag_ids(cls, session: SASession, target: Any) -> set[int]:
        tag_ids: set[int] = set()
        for owner_id in cls.get_owners_ids(target):
            name = f'owner:{owner_id}'
            tag = get_tag(name, session, TagType.owner)
            tag_ids.add(tag.id)  # type: ignore[arg-type]
        return tag_ids

    @classmethod
    def _add_owners(cls, session: SASession, target: Any) -> None:
        for owner_id in cls.get_owners_ids(target):
            name = f'owner:{owner_id}'
            tag = get_tag(name, session, TagType.owner)
            cls.add_tag_object_if_not_tagged(session, tag_id=tag.id, object_id=target.id, object_type=cls.object_type)  # type: ignore[attr-defined]

    @classmethod
    def add_tag_object_if_not_tagged(cls, session: SASession, tag_id: int, object_id: int, object_type: str) -> None:
        exists_query = exists().where(TaggedObject.tag_id == tag_id, TaggedObject.object_id == object_id, TaggedObject.object_type == object_type)
        already_tagged: bool = bool(session.query(exists_query).scalar())
        if not already_tagged:
            tagged_object = TaggedObject(tag_id=tag_id, object_id=object_id, object_type=object_type)
            session.add(tagged_object)

    @classmethod
    def after_insert(cls, _mapper: Mapper, connection: Connection, target: Any) -> None:
        with Session(bind=connection) as session:  # type: ignore[call-arg]
            cls._add_owners(session, target)
            tag = get_tag(f'type:{cls.object_type}', session, TagType.type)
            cls.add_tag_object_if_not_tagged(session, tag_id=tag.id, object_id=target.id, object_type=cls.object_type)  # type: ignore[attr-defined]
            session.commit()

    @classmethod
    def after_update(cls, _mapper: Mapper, connection: Connection, target: Any) -> None:
        with Session(bind=connection) as session:  # type: ignore[call-arg]
            existing_tags: list[TaggedObject] = session.query(TaggedObject).join(Tag).filter(TaggedObject.object_type == cls.object_type, TaggedObject.object_id == target.id, Tag.type == TagType.owner).all()  # type: ignore[attr-defined]
            existing_owner_tag_ids: set[int] = {tag.tag_id for tag in existing_tags}  # type: ignore[assignment]
            new_owner_tag_ids: set[int] = cls.get_owner_tag_ids(session, target)
            for owner_tag_id in new_owner_tag_ids - existing_owner_tag_ids:
                tagged_object = TaggedObject(tag_id=owner_tag_id, object_id=target.id, object_type=cls.object_type)  # type: ignore[attr-defined]
                session.add(tagged_object)
            for tag in existing_tags:
                if tag.tag_id not in new_owner_tag_ids:
                    session.delete(tag)
            session.commit()

    @classmethod
    def after_delete(cls, _mapper: Mapper, connection: Connection, target: Any) -> None:
        with Session(bind=connection) as session:  # type: ignore[call-arg]
            session.query(TaggedObject).filter(TaggedObject.object_type == cls.object_type, TaggedObject.object_id == target.id).delete()  # type: ignore[attr-defined]
            session.commit()

class ChartUpdater(ObjectUpdater):
    object_type: ClassVar[str] = 'chart'

    @classmethod
    def get_owners_ids(cls, target: Any) -> list[int]:
        return [owner.id for owner in target.owners]  # type: ignore[attr-defined]

class DashboardUpdater(ObjectUpdater):
    object_type: ClassVar[str] = 'dashboard'

    @classmethod
    def get_owners_ids(cls, target: Any) -> list[int]:
        return [owner.id for owner in target.owners]  # type: ignore[attr-defined]

class QueryUpdater(ObjectUpdater):
    object_type: ClassVar[str] = 'query'

    @classmethod
    def get_owners_ids(cls, target: Any) -> list[int]:
        return [target.user_id]  # type: ignore[attr-defined]

class DatasetUpdater(ObjectUpdater):
    object_type: ClassVar[str] = 'dataset'

    @classmethod
    def get_owners_ids(cls, target: Any) -> list[int]:
        return [owner.id for owner in target.owners]  # type: ignore[attr-defined]

class FavStarUpdater:

    @classmethod
    def after_insert(cls, _mapper: Mapper, connection: Connection, target: FavStar) -> None:
        with Session(bind=connection) as session:  # type: ignore[call-arg]
            name = f'favorited_by:{target.user_id}'
            tag = get_tag(name, session, TagType.favorited_by)
            tagged_object = TaggedObject(tag_id=tag.id, object_id=target.obj_id, object_type=get_object_type(target.class_name))  # type: ignore[arg-type]
            session.add(tagged_object)
            session.commit()

    @classmethod
    def after_delete(cls, _mapper: Mapper, connection: Connection, target: FavStar) -> None:
        with Session(bind=connection) as session:  # type: ignore[call-arg]
            name = f'favorited_by:{target.user_id}'
            query = session.query(TaggedObject.id).join(Tag).filter(TaggedObject.object_id == target.obj_id, Tag.type == TagType.favorited_by, Tag.name == name)
            ids: list[int] = [row[0] for row in query]
            session.query(TaggedObject).filter(TaggedObject.id.in_(ids)).delete(synchronize_session=False)
            session.commit()