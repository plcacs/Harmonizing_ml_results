import pyperf
from sqlalchemy import Column, ForeignKey, Integer, String, Table, MetaData
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.sql import Select, Delete, Insert
from pyperf._runner import Runner
from argparse import Namespace

metadata: MetaData = MetaData()
Person: Table = Table('person', metadata, Column('id', Integer, primary_key=True), Column('name', String(250), nullable=False))
Address: Table = Table('address', metadata, Column('id', Integer, primary_key=True), Column('street_name', String(250)), Column('street_number', String(250)), Column('post_code', String(250), nullable=False), Column('person_id', Integer, ForeignKey('person.id')))
engine: Engine = create_engine('sqlite://')
metadata.create_all(engine)
metadata.bind = engine
DBSession = sessionmaker(bind=engine)
session = DBSession()

def bench_sqlalchemy(loops: int, npeople: int) -> float:
    total_dt: float = 0.0
    for loops in range(loops):
        cur: Delete = Person.delete()
        cur.execute()
        cur = Address.delete()
        cur.execute()
        t0: float = pyperf.perf_counter()
        for i in range(npeople):
            new_person: Insert = Person.insert()
            new_person.execute(name=('name %i' % i))
            new_address: Insert = Address.insert()
            new_address.execute(post_code=('%05i' % i))
        for i in range(npeople):
            cur: Select = Person.select()
            cur.execute()
        total_dt += (pyperf.perf_counter() - t0)
    return total_dt

def add_cmdline_args(cmd: list[str], args: Namespace) -> None:
    cmd.extend(('--rows', str(args.rows)))

if (__name__ == '__main__'):
    runner: Runner = pyperf.Runner(add_cmdline_args=add_cmdline_args)
    runner.metadata['description'] = 'SQLAlchemy Imperative benchmark using SQLite'
    runner.argparser.add_argument('--rows', type=int, default=100, help='Number of rows (default: 100)')
    args: Namespace = runner.parse_args()
    runner.bench_time_func('sqlalchemy_imperative', bench_sqlalchemy, args.rows)
