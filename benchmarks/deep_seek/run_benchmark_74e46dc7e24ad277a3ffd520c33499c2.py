import pyperf
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy import create_engine
from typing import Any, List, Tuple, Optional
import argparse

Base = declarative_base()  # type: Any

class Person(Base):
    __tablename__ = 'person'
    id = Column(Integer, primary_key=True)  # type: Column[int]
    name = Column(String(250), nullable=False)  # type: Column[str]

class Address(Base):
    __tablename__ = 'address'
    id = Column(Integer, primary_key=True)  # type: Column[int]
    street_name = Column(String(250))  # type: Column[Optional[str]]
    street_number = Column(String(250))  # type: Column[Optional[str]]
    post_code = Column(String(250), nullable=False)  # type: Column[str]
    person_id = Column(Integer, ForeignKey('person.id'))  # type: Column[int]
    person = relationship(Person)  # type: relationship[Person]

engine = create_engine('sqlite://')
Base.metadata.create_all(engine)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

def bench_sqlalchemy(loops: int, npeople: int) -> float:
    total_dt = 0.0
    for _ in range(loops):
        session.query(Person).delete(synchronize_session=False)
        session.query(Address).delete(synchronize_session=False)
        t0 = pyperf.perf_counter()
        for i in range(npeople):
            new_person = Person(name=('name %i' % i))
            session.add(new_person)
            session.commit()
            new_address = Address(post_code=('%05i' % i), person=new_person)
            session.add(new_address)
            session.commit()
        for i in range(npeople):
            session.query(Person).all()
        total_dt += (pyperf.perf_counter() - t0)
    return total_dt

def add_cmdline_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.extend(('--rows', str(args.rows)))

if __name__ == '__main__':
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'SQLAlchemy Declarative benchmark using SQLite'
    runner.argparser.add_argument('--rows', type=int, default=100, help='Number of rows (default: 100)')
    args = runner.parse_args()
    runner.bench_time_func('sqlalchemy_declarative', bench_sqlalchemy, args.rows)
