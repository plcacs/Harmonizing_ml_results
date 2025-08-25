import pyperf
from typing import Optional, List, Any, Dict

class OrderedCollection(list[Any]):
    pass

class Strength:
    REQUIRED: 'Strength'
    STRONG_PREFERRED: 'Strength'
    PREFERRED: 'Strength'
    STRONG_DEFAULT: 'Strength'
    NORMAL: 'Strength'
    WEAK_DEFAULT: 'Strength'
    WEAKEST: 'Strength'

    strength: int
    name: str

    def __init__(self, strength: int, name: str) -> None:
        super().__init__()
        self.strength = strength
        self.name = name

    @classmethod
    def stronger(cls, s1: 'Strength', s2: 'Strength') -> bool:
        return s1.strength < s2.strength

    @classmethod
    def weaker(cls, s1: 'Strength', s2: 'Strength') -> bool:
        return s1.strength > s2.strength

    @classmethod
    def weakest_of(cls, s1: 'Strength', s2: 'Strength') -> 'Strength':
        if cls.weaker(s1, s2):
            return s1
        return s2

    @classmethod
    def strongest(cls, s1: 'Strength', s2: 'Strength') -> 'Strength':
        if cls.stronger(s1, s2):
            return s1
        return s2

    def next_weaker(self) -> 'Strength':
        strengths: Dict[int, 'Strength'] = {
            0: self.__class__.WEAKEST,
            1: self.__class__.WEAK_DEFAULT,
            2: self.__class__.NORMAL,
            3: self.__class__.STRONG_DEFAULT,
            4: self.__class__.PREFERRED,
            5: self.__class__.REQUIRED
        }
        return strengths[self.strength]

Strength.REQUIRED = Strength(0, 'required')
Strength.STRONG_PREFERRED = Strength(1, 'strongPreferred')
Strength.PREFERRED = Strength(2, 'preferred')
Strength.STRONG_DEFAULT = Strength(3, 'strongDefault')
Strength.NORMAL = Strength(4, 'normal')
Strength.WEAK_DEFAULT = Strength(5, 'weakDefault')
Strength.WEAKEST = Strength(6, 'weakest')

class Constraint:

    strength: 'Strength'

    def __init__(self, strength: 'Strength') -> None:
        super().__init__()
        self.strength = strength

    def add_constraint(self) -> None:
        global planner
        self.add_to_graph()
        planner.incremental_add(self)

    def satisfy(self, mark: int) -> Optional['Constraint']:
        global planner
        self.choose_method(mark)
        if not self.is_satisfied():
            if self.strength == Strength.REQUIRED:
                print('Could not satisfy a required constraint!')
            return None
        self.mark_inputs(mark)
        out = self.output()
        overridden = out.determined_by
        if overridden is not None:
            overridden.mark_unsatisfied()
        out.determined_by = self
        if not planner.add_propagate(self, mark):
            print('Cycle encountered')
        out.mark = mark
        return overridden

    def destroy_constraint(self) -> None:
        global planner
        if self.is_satisfied():
            planner.incremental_remove(self)
        else:
            self.remove_from_graph()

    def is_input(self) -> bool:
        return False

    # The following methods are expected to be implemented by subclasses
    def add_to_graph(self) -> None:
        pass

    def choose_method(self, mark: int) -> None:
        pass

    def is_satisfied(self) -> bool:
        return False

    def mark_inputs(self, mark: int) -> None:
        pass

    def output(self) -> 'Variable':
        return Variable("", 0)

    def recalculate(self) -> None:
        pass

    def mark_unsatisfied(self) -> None:
        pass

    def inputs_known(self, mark: int) -> bool:
        return False

    def remove_from_graph(self) -> None:
        pass

class UrnaryConstraint(Constraint):

    my_output: 'Variable'
    satisfied: bool

    def __init__(self, v: 'Variable', strength: 'Strength') -> None:
        super().__init__(strength)
        self.my_output = v
        self.satisfied = False
        self.add_constraint()

    def add_to_graph(self) -> None:
        self.my_output.add_constraint(self)
        self.satisfied = False

    def choose_method(self, mark: int) -> None:
        if (self.my_output.mark != mark) and Strength.stronger(self.strength, self.my_output.walk_strength):
            self.satisfied = True
        else:
            self.satisfied = False

    def is_satisfied(self) -> bool:
        return self.satisfied

    def mark_inputs(self, mark: int) -> None:
        pass

    def output(self) -> 'Variable':
        return self.my_output

    def recalculate(self) -> None:
        self.my_output.walk_strength = self.strength
        self.my_output.stay = not self.is_input()
        if self.my_output.stay:
            self.execute()

    def mark_unsatisfied(self) -> None:
        self.satisfied = False

    def inputs_known(self, mark: int) -> bool:
        return True

    def remove_from_graph(self) -> None:
        if self.my_output is not None:
            self.my_output.remove_constraint(self)
            self.satisfied = False

class StayConstraint(UrnaryConstraint):

    def __init__(self, v: 'Variable', string: 'Strength') -> None:
        super().__init__(v, string)

    def execute(self) -> None:
        pass

class EditConstraint(UrnaryConstraint):

    def __init__(self, v: 'Variable', string: 'Strength') -> None:
        super().__init__(v, string)

    def is_input(self) -> bool:
        return True

    def execute(self) -> None:
        pass

class Direction:
    NONE: int = 0
    FORWARD: int = 1
    BACKWARD: int = -1

class BinaryConstraint(Constraint):

    v1: 'Variable'
    v2: 'Variable'
    direction: int

    def __init__(self, v1: 'Variable', v2: 'Variable', strength: 'Strength') -> None:
        super().__init__(strength)
        self.v1 = v1
        self.v2 = v2
        self.direction = Direction.NONE
        self.add_constraint()

    def choose_method(self, mark: int) -> None:
        if self.v1.mark == mark:
            if (self.v2.mark != mark) and Strength.stronger(self.strength, self.v2.walk_strength):
                self.direction = Direction.FORWARD
            else:
                self.direction = Direction.BACKWARD
        if self.v2.mark == mark:
            if (self.v1.mark != mark) and Strength.stronger(self.strength, self.v1.walk_strength):
                self.direction = Direction.BACKWARD
            else:
                self.direction = Direction.NONE
        if Strength.weaker(self.v1.walk_strength, self.v2.walk_strength):
            if Strength.stronger(self.strength, self.v1.walk_strength):
                self.direction = Direction.BACKWARD
            else:
                self.direction = Direction.NONE
        elif Strength.stronger(self.strength, self.v2.walk_strength):
            self.direction = Direction.FORWARD
        else:
            self.direction = Direction.BACKWARD

    def add_to_graph(self) -> None:
        self.v1.add_constraint(self)
        self.v2.add_constraint(self)
        self.direction = Direction.NONE

    def is_satisfied(self) -> bool:
        return self.direction != Direction.NONE

    def mark_inputs(self, mark: int) -> None:
        self.input().mark = mark

    def input(self) -> 'Variable':
        if self.direction == Direction.FORWARD:
            return self.v1
        return self.v2

    def output(self) -> 'Variable':
        if self.direction == Direction.FORWARD:
            return self.v2
        return self.v1

    def recalculate(self) -> None:
        ihn = self.input()
        out = self.output()
        out.walk_strength = Strength.weakest_of(self.strength, ihn.walk_strength)
        out.stay = ihn.stay
        if out.stay:
            self.execute()

    def mark_unsatisfied(self) -> None:
        self.direction = Direction.NONE

    def inputs_known(self, mark: int) -> bool:
        i = self.input()
        return (i.mark == mark) or i.stay or (i.determined_by is None)

    def remove_from_graph(self) -> None:
        if self.v1 is not None:
            self.v1.remove_constraint(self)
        if self.v2 is not None:
            self.v2.remove_constraint(self)
        self.direction = Direction.NONE

class ScaleConstraint(BinaryConstraint):

    scale: 'Variable'
    offset: 'Variable'

    def __init__(self, src: 'Variable', scale: 'Variable', offset: 'Variable', dest: 'Variable', strength: 'Strength') -> None:
        self.direction = Direction.NONE
        self.scale = scale
        self.offset = offset
        super().__init__(src, dest, strength)

    def add_to_graph(self) -> None:
        super().add_to_graph()
        self.scale.add_constraint(self)
        self.offset.add_constraint(self)

    def remove_from_graph(self) -> None:
        super().remove_from_graph()
        if self.scale is not None:
            self.scale.remove_constraint(self)
        if self.offset is not None:
            self.offset.remove_constraint(self)

    def mark_inputs(self, mark: int) -> None:
        super().mark_inputs(mark)
        self.scale.mark = mark
        self.offset.mark = mark

    def execute(self) -> None:
        if self.direction == Direction.FORWARD:
            self.v2.value = (self.v1.value * self.scale.value) + self.offset.value
        else:
            self.v1.value = (self.v2.value - self.offset.value) / self.scale.value

    def recalculate(self) -> None:
        ihn = self.input()
        out = self.output()
        out.walk_strength = Strength.weakest_of(self.strength, ihn.walk_strength)
        out.stay = ihn.stay and self.scale.stay and self.offset.stay
        if out.stay:
            self.execute()

class EqualityConstraint(BinaryConstraint):

    def execute(self) -> None:
        self.output().value = self.input().value

class Variable:
    name: str
    value: float
    constraints: OrderedCollection
    determined_by: Optional[Constraint]
    mark: int
    walk_strength: 'Strength'
    stay: bool

    def __init__(self, name: str, initial_value: float = 0) -> None:
        super().__init__()
        self.name = name
        self.value = initial_value
        self.constraints = OrderedCollection()
        self.determined_by = None
        self.mark = 0
        self.walk_strength = Strength.WEAKEST
        self.stay = True

    def __repr__(self) -> str:
        return f'<Variable: {self.name} - {self.value}>'

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def remove_constraint(self, constraint: Constraint) -> None:
        self.constraints.remove(constraint)
        if self.determined_by == constraint:
            self.determined_by = None

class Planner:

    current_mark: int

    def __init__(self) -> None:
        super().__init__()
        self.current_mark = 0

    def incremental_add(self, constraint: Constraint) -> None:
        mark = self.new_mark()
        overridden = constraint.satisfy(mark)
        while overridden is not None:
            overridden = overridden.satisfy(mark)

    def incremental_remove(self, constraint: Constraint) -> None:
        out = constraint.output()
        constraint.mark_unsatisfied()
        constraint.remove_from_graph()
        unsatisfied = self.remove_propagate_from(out)
        strength = Strength.REQUIRED
        repeat = True
        while repeat:
            for u in unsatisfied:
                if u.strength == strength:
                    self.incremental_add(u)
            strength = strength.next_weaker()
            repeat = strength != Strength.WEAKEST

    def new_mark(self) -> int:
        self.current_mark += 1
        return self.current_mark

    def make_plan(self, sources: OrderedCollection) -> 'Plan':
        mark = self.new_mark()
        plan = Plan()
        todo: List[Constraint] = list(sources)
        while todo:
            c = todo.pop(0)
            if c.output().mark != mark and c.inputs_known(mark):
                plan.add_constraint(c)
                c.output().mark = mark
                self.add_constraints_consuming_to(c.output(), todo)
        return plan

    def extract_plan_from_constraints(self, constraints: OrderedCollection) -> 'Plan':
        sources: OrderedCollection = OrderedCollection()
        for c in constraints:
            if c.is_input() and c.is_satisfied():
                sources.append(c)
        return self.make_plan(sources)

    def add_propagate(self, c: Constraint, mark: int) -> bool:
        todo: List[Constraint] = [c]
        while todo:
            d = todo.pop(0)
            if d.output().mark == mark:
                self.incremental_remove(c)
                return False
            d.recalculate()
            self.add_constraints_consuming_to(d.output(), todo)
        return True

    def remove_propagate_from(self, out: 'Variable') -> OrderedCollection:
        out.determined_by = None
        out.walk_strength = Strength.WEAKEST
        out.stay = True
        unsatisfied: OrderedCollection = OrderedCollection()
        todo: List['Variable'] = [out]
        while todo:
            v = todo.pop(0)
            for c in v.constraints:
                if not c.is_satisfied():
                    unsatisfied.append(c)
            determining = v.determined_by
            for c in v.constraints:
                if c != determining and c.is_satisfied():
                    c.recalculate()
                    todo.append(c.output())
        return unsatisfied

    def add_constraints_consuming_to(self, v: 'Variable', coll: List[Constraint]) -> None:
        determining = v.determined_by
        cc = v.constraints
        for c in cc:
            if c != determining and c.is_satisfied():
                coll.append(c)

class Plan:
    v: OrderedCollection

    def __init__(self) -> None:
        super().__init__()
        self.v = OrderedCollection()

    def add_constraint(self, c: Constraint) -> None:
        self.v.append(c)

    def __len__(self) -> int:
        return len(self.v)

    def __getitem__(self, index: int) -> Constraint:
        return self.v[index]

    def execute(self) -> None:
        for c in self.v:
            c.execute()

def chain_test(n: int) -> None:
    """
    This is the standard DeltaBlue benchmark. A long chain of equality
    constraints is constructed with a stay constraint on one end. An
    edit constraint is then added to the opposite end and the time is
    measured for adding and removing this constraint, and extracting
    and executing a constraint satisfaction plan. There are two cases.
    In case 1, the added constraint is stronger than the stay
    constraint and values must propagate down the entire length of the
    chain. In case 2, the added constraint is weaker than the stay
    constraint so it cannot be accomodated. The cost in this case is,
    of course, very low. Typical situations lie somewhere between these
    two extremes.
    """
    global planner
    planner = Planner()
    prev: Optional[Variable] = None
    first: Optional[Variable] = None
    last: Optional[Variable] = None
    for i in range(n + 1):
        name = f'v{i}'
        v = Variable(name)
        if prev is not None:
            EqualityConstraint(prev, v, Strength.REQUIRED)
        if i == 0:
            first = v
        if i == n:
            last = v
        prev = v
    if last is not None:
        StayConstraint(last, Strength.STRONG_DEFAULT)
    if first is not None:
        edit = EditConstraint(first, Strength.PREFERRED)
        edits: OrderedCollection = OrderedCollection()
        edits.append(edit)
        plan = planner.extract_plan_from_constraints(edits)
        for i in range(100):
            first.value = i
            plan.execute()
            if last is not None and last.value != i:
                print('Chain test failed.')

def projection_test(n: int) -> None:
    """
    This test constructs a two sets of variables related to each
    other by a simple linear transformation (scale and offset). The
    time is measured to change a variable on either side of the
    mapping and to change the scale and offset factors.
    """
    global planner
    planner = Planner()
    scale = Variable('scale', 10)
    offset = Variable('offset', 1000)
    src: Optional[Variable] = None
    dests: OrderedCollection = OrderedCollection()
    for i in range(n):
        src = Variable(f'src{i}', float(i))
        dst = Variable(f'dst{i}', float(i))
        dests.append(dst)
        StayConstraint(src, Strength.NORMAL)
        ScaleConstraint(src, scale, offset, dst, Strength.REQUIRED)
    if src is not None:
        change(src, 17)
        if dests and dests[0].value != 1170:
            print('Projection 1 failed')
    if dests:
        change(dests[0], 1050)
        if src is not None and src.value != 5:
            print('Projection 2 failed')
    change(scale, 5)
    for i in range(min(n, len(dests))):
        expected = (i * 5) + 1000
        if dests[i].value != expected:
            print('Projection 3 failed')
    change(offset, 2000)
    for i in range(min(n, len(dests))):
        expected = (i * 5) + 2000
        if dests[i].value != expected:
            print('Projection 4 failed')

def change(v: 'Variable', new_value: float) -> None:
    global planner
    edit = EditConstraint(v, Strength.PREFERRED)
    edits: OrderedCollection = OrderedCollection()
    edits.append(edit)
    plan = planner.extract_plan_from_constraints(edits)
    for _ in range(10):
        v.value = new_value
        plan.execute()
    edit.destroy_constraint()

planner: Optional[Planner] = None

def delta_blue(n: int) -> None:
    chain_test(n)
    projection_test(n)

if __name__ == '__main__':
    runner = pyperf.Runner()
    runner.metadata['description'] = 'DeltaBlue benchmark'
    n = 100
    runner.bench_func('deltablue', delta_blue, n)
