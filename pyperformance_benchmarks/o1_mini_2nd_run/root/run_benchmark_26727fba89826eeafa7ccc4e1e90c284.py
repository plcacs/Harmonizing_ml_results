import pyperf
from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String,
    Table,
    MetaData,
    create_engine,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.base import Engine, Connection
from sqlalchemy.sql import Delete, Insert, Select
from typing import List
import argparse

metadata: MetaData = MetaData()

Person: Table = Table(
    'person',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('name', String(250), nullable=False)
)

Address: Table = Table(
    'address',
    metadata,
    Column('id', Integer, primary_key=True),
    Column('street_name', String(250)),
    Column('street_number', String(250)),
    Column('post_code', String(250), nullable=False),
    Column('person_id', Integer, ForeignKey('person.id'))
)

engine: Engine = create_engine('sqlite://')
metadata.create_all(engine)
metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

def bench_sqlalchemy(loops: int, npeople: int) -> float:
    total_dt: float = 0.0
    for _ in range(loops):
        cur: Connection = engine.connect()
        delete_person: Delete = Person.delete()
        cur.execute(delete_person)
        delete_address: Delete = Address.delete()
        cur.execute(delete_address)
        cur.close()

        t0: float = pyperf.perf_counter()
        for i in range(npeople):
            conn: Connection = engine.connect()
            insert_person: Insert = Person.insert().values(name=f'name {i}')
            conn.execute(insert_person)
            insert_address: Insert = Address.insert().values(post_code=f'{i:05}')
            conn.execute(insert_address)
            conn.close()
        
        for _ in range(npeople):
            conn: Connection = engine.connect()
            select_person: Select = Person.select()
            conn.execute(select_person)
            conn.close()
        
        total_dt += (pyperf.perf_counter() - t0)
    return total_dt

def add_cmdline_args(cmd: List[str], args: argparse.Namespace) -> None:
    cmd.extend(('--rows', str(args.rows)))

if __name__ == '__main__':
    runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'SQLAlchemy Imperative benchmark using SQLite'
    runner.argparser.add_argument('--rows', type=int, default=100, help='Number of rows (default: 100)')
    args: argparse.Namespace = runner.parse_args()
    runner.bench_time_func('sqlalchemy_imperative', bench_sqlalchemy, args.rows)
