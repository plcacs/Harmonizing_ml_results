import pyperf
from sqlalchemy import Column, ForeignKey, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from typing import List
import argparse

Base = declarative_base()

class Person(Base):
    __tablename__ = 'person'
    id: int = Column(Integer, primary_key=True)
    name: str = Column(String(250), nullable=False)

class Address(Base):
    __tablename__ = 'address'
    id: int = Column(Integer, primary_key=True)
    street_name: str = Column(String(250))
    street_number: str = Column(String(250))
    post_code: str = Column(String(250), nullable=False)
    person_id: int = Column(Integer, ForeignKey('person.id'))
    person: Person = relationship(Person)

engine = create_engine('sqlite://')
Base.metadata.create_all(engine)
Base.metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

def bench_sqlalchemy(loops: int, npeople: int) -> float:
    total_dt: float = 0.0
    for _ in range(loops):
        session.query(Person).delete(synchronize_session=False)
        session.query(Address).delete(synchronize_session=False)
        t0: float = pyperf.perf_counter()
        for i in range(npeople):
            new_person: Person = Person(name=('name %i' % i))
            session.add(new_person)
            session.commit()
            new_address: Address = Address(post_code=('%05i' % i), person=new_person)
            session.add(new_address)
            session.commit()
        for _ in range(npeople):
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