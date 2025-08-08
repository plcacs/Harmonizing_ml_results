import re
from collections import defaultdict
from typing import ClassVar
import pytest
from _pytest.outcomes import Failed, Skipped
from pytest import raises
from hypothesis import HealthCheck, Phase, __version__, reproduce_failure, seed, settings as Settings, strategies as st
from hypothesis.control import current_build_context
from hypothesis.core import encode_failure
from hypothesis.database import ExampleDatabase
from hypothesis.errors import DidNotReproduce, Flaky, InvalidArgument, InvalidDefinition
from hypothesis.internal.entropy import deterministic_PRNG
from hypothesis.stateful import Bundle, RuleBasedStateMachine, consumes, initialize, invariant, multiple, precondition, rule, run_state_machine_as_test
from hypothesis.strategies import binary, data, integers, just, lists
from tests.common.utils import capture_out, validate_deprecation
from tests.nocover.test_stateful import DepthMachine
NO_BLOB_SETTINGS: Settings = Settings(print_blob=False, phases=tuple(Phase)[:-1])

class MultipleRulesSameFuncMachine(RuleBasedStateMachine):

    def myfunc(self, data: str) -> None:
        print(data)
    rule1: ClassVar = rule(data=just('rule1data'))(myfunc)
    rule2: ClassVar = rule(data=just('rule2data'))(myfunc)

class PreconditionMachine(RuleBasedStateMachine):
    num: int = 0

    @rule()
    def add_one(self) -> None:
        self.num += 1

    @rule()
    def set_to_zero(self) -> None:
        self.num = 0

    @rule(num=integers())
    @precondition(lambda self: self.num != 0)
    def div_by_precondition_after(self, num: int) -> None:
        self.num = num / self.num

    @precondition(lambda self: self.num != 0)
    @rule(num=integers())
    def div_by_precondition_before(self, num: int) -> None:
        self.num = num / self.num
TestPrecondition: ClassVar = PreconditionMachine.TestCase
TestPrecondition.settings: Settings = Settings(TestPrecondition.settings, max_examples=10)

def test_picks_up_settings_at_first_use_of_testcase() -> None:
    assert TestPrecondition.settings.max_examples == 10

def test_multiple_rules_same_func() -> None:
    test_class: ClassVar = MultipleRulesSameFuncMachine.TestCase
    with capture_out() as o:
        test_class().runTest()
    output: str = o.getvalue()
    assert 'rule1data' in output
    assert 'rule2data' in output

def test_can_get_test_case_off_machine_instance() -> None:
    assert DepthMachine().TestCase is DepthMachine().TestCase
    assert DepthMachine().TestCase is not None

class FlakyDrawLessMachine(RuleBasedStateMachine):

    @rule(d=data())
    def action(self, d: str) -> None:
        if current_build_context().is_final:
            d.draw(binary(min_size=1, max_size=1))
        else:
            buffer: bytes = binary(min_size=1024, max_size=1024)
            assert 0 not in buffer

def test_flaky_draw_less_raises_flaky() -> None:
    with raises(Flaky):
        FlakyDrawLessMachine.TestCase().runTest()

def test_result_is_added_to_target() -> None:

    class TargetStateMachine(RuleBasedStateMachine):
        nodes: Bundle = Bundle('nodes')

        @rule(target=nodes, source=lists(nodes))
        def bunch(self, source: List[str]) -> List[str]:
            assert len(source) == 0
            return source
    test_class: ClassVar = TargetStateMachine.TestCase
    try:
        test_class().runTest()
        raise RuntimeError('Expected an assertion error')
    except AssertionError as err:
        notes: List[str] = err.__notes__
    regularized_notes: List[str] = [re.sub('[0-9]+', 'i', note) for note in notes]
    assert 'state.bunch(source=[nodes_i])' in regularized_notes

class FlakyStateMachine(RuleBasedStateMachine):

    @rule()
    def action(self) -> None:
        assert current_build_context().is_final

def test_flaky_raises_flaky() -> None:
    with raises(Flaky):
        FlakyStateMachine.TestCase().runTest()

class FlakyRatchettingMachine(RuleBasedStateMachine):
    ratchet: int = 0

    @rule(d=data())
    def action(self, d: str) -> None:
        FlakyRatchettingMachine.ratchet += 1
        n: int = FlakyRatchettingMachine.ratchet
        d.draw(lists(integers(), min_size=n, max_size=n))
        raise AssertionError

@Settings(stateful_step_count=10, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much])
class MachineWithConsumingRule(RuleBasedStateMachine):
    b1: Bundle = Bundle('b1')
    b2: Bundle = Bundle('b2')

    def __init__(self) -> None:
        self.created_counter: int = 0
        self.consumed_counter: int = 0
        super().__init__()

    @invariant()
    def bundle_length(self) -> None:
        assert len(self.bundle('b1')) == self.created_counter - self.consumed_counter

    @rule(target=b1)
    def populate_b1(self) -> int:
        self.created_counter += 1
        return self.created_counter

    @rule(target=b2, consumed=consumes(b1))
    def depopulate_b1(self, consumed: int) -> int:
        self.consumed_counter += 1
        return consumed

    @rule(consumed=lists(consumes(b1), max_size=3))
    def depopulate_b1_multiple(self, consumed: List[int]) -> None:
        self.consumed_counter += len(consumed)

    @rule(value1=b1, value2=b2)
    def check(self, value1: int, value2: int) -> None:
        assert value1 != value2
TestMachineWithConsumingRule: ClassVar = MachineWithConsumingRule.TestCase

def test_multiple() -> None:
    none: multiple = multiple()
    some: multiple = multiple(1, 2.01, '3', b'4', 5)
    assert len(none.values) == 0
    assert len(some.values) == 5
    assert set(some.values) == {1, 2.01, '3', b'4', 5}

class MachineUsingMultiple(RuleBasedStateMachine):
    b: Bundle = Bundle('b')

    def __init__(self) -> None:
        self.expected_bundle_length: int = 0
        super().__init__()

    @invariant()
    def bundle_length(self) -> None:
        assert len(self.bundle('b')) == self.expected_bundle_length

    @rule(target=b, items=lists(elements=integers(), max_size=10))
    def populate_bundle(self, items: List[int]) -> multiple:
        self.expected_bundle_length += len(items)
        return multiple(*items)

    @rule(target=b)
    def do_not_populate(self) -> multiple:
        return multiple()
TestMachineUsingMultiple: ClassVar = MachineUsingMultiple.TestCase

def test_multiple_variables_printed() -> None:

    class ProducesMultiple(RuleBasedStateMachine):
        b: Bundle = Bundle('b')

        @initialize(target=b)
        def populate_bundle(self) -> multiple:
            return multiple(1, 2)

        @rule()
        def fail_fast(self) -> None:
            raise AssertionError
    with raises(AssertionError) as err:
        run_state_machine_as_test(ProducesMultiple)
    assignment_line: str = err.value.__notes__[2]
    assert assignment_line == 'b_0, b_1 = state.populate_bundle()'
    state: ProducesMultiple = ProducesMultiple()
    b_0, b_1 = state.populate_bundle()
    with raises(AssertionError):
        state.fail_fast()

def test_multiple_variables_printed_single_element() -> None:

    class ProducesMultiple(RuleBasedStateMachine):
        b: Bundle = Bundle('b')

        @initialize(target=b)
        def populate_bundle(self) -> multiple:
            return multiple(1)

        @rule(b=b)
        def fail_fast(self, b: int) -> None:
            assert b != 1
    with raises(AssertionError) as err:
        run_state_machine_as_test(ProducesMultiple)
    assignment_line: str = err.value.__notes__[2]
    assert assignment_line == '(b_0,) = state.populate_bundle()'
    state: ProducesMultiple = ProducesMultiple()
    v1, = state.populate_bundle()
    state.fail_fast((v1,))
    with raises(AssertionError):
        state.fail_fast(v1)

def test_no_variables_printed() -> None:

    class ProducesNoVariables(RuleBasedStateMachine):
        b: Bundle = Bundle('b')

        @initialize(target=b)
        def populate_bundle(self) -> multiple:
            return multiple()

        @rule()
        def fail_fast(self) -> None:
            raise AssertionError
    with raises(AssertionError) as err:
        run_state_machine_as_test(ProducesNoVariables)
    assignment_line: str = err.value.__notes__[2]
    assert assignment_line == 'state.populate_bundle()'

def test_consumes_typecheck() -> None:
    with pytest.raises(TypeError):
        consumes(integers())

def test_ratchetting_raises_flaky() -> None:
    with raises(Flaky):
        FlakyRatchettingMachine.TestCase().runTest()

def test_empty_machine_is_invalid() -> None:

    class EmptyMachine(RuleBasedStateMachine):
        pass
    with raises(InvalidDefinition):
        EmptyMachine.TestCase().runTest()

def test_machine_with_no_terminals_is_invalid() -> None:

    class NonTerminalMachine(RuleBasedStateMachine):

        @rule(value=Bundle('hi'))
        def bye(self, hi: str) -> None:
            pass
    with raises(InvalidDefinition):
        NonTerminalMachine.TestCase().runTest()

def test_minimizes_errors_in_teardown() -> None:
    counter: int = 0

    class Foo(RuleBasedStateMachine):

        @initialize()
        def init(self) -> None:
            nonlocal counter
            counter = 0

        @rule()
        def increment(self) -> None:
            nonlocal counter
            counter += 1

        def teardown(self) -> None:
            nonlocal counter
            assert not counter
    with raises(AssertionError):
        run_state_machine_as_test(Foo)
    assert counter == 1

class RequiresInit(RuleBasedStateMachine):

    def __init__(self, threshold: int):
        super().__init__()
        self.threshold: int = threshold

    @rule(value=integers())
    def action(self, value: int) -> None:
        if value > self.threshold:
            raise ValueError(f'{value} is too high')

def test_can_use_factory_for_tests() -> None:
    with raises(ValueError):
        run_state_machine_as_test(lambda: RequiresInit(42), settings=Settings(max_examples=100))

class FailsEventually(RuleBasedStateMachine):

    def __init__(self) -> None:
        super().__init__()
        self.counter: int = 0

    @rule()
    def increment(self) -> None:
        self.counter += 1
        assert self.counter < 10
FailsEventually.TestCase.settings: Settings = Settings(stateful_step_count=5)

def test_can_explicitly_pass_settings() -> None:
    run_state_machine_as_test(FailsEventually)
    try:
        FailsEventually.TestCase.settings = Settings(FailsEventually.TestCase.settings, stateful_step_count=15)
        run_state_machine_as_test(FailsEventually, settings=Settings(stateful_step_count=2))
    finally:
        FailsEventually.TestCase.settings = Settings(FailsEventually.TestCase.settings, stateful_step_count=5)

def test_settings_argument_is_validated() -> None:
    with pytest.raises(InvalidArgument):
        run_state_machine_as_test(FailsEventually, settings=object())

def test_runner_that_checks_factory_produced_a_machine() -> None:
    with pytest.raises(InvalidArgument):
        run_state_machine_as_test(object)

def test_settings_attribute_is_validated() -> None:
    real_settings: Settings = FailsEventually.TestCase.settings
    try:
        FailsEventually.TestCase.settings = object()
        with pytest.raises(InvalidArgument):
            run_state_machine_as_test(FailsEventually)
    finally:
        FailsEventually.TestCase.settings = real_settings

def test_saves_failing_example_in_database() -> None:
    db: ExampleDatabase = ExampleDatabase(':memory:')
    ss: Settings = Settings(database=db, max_examples=1000, suppress_health_check=list(HealthCheck))
    with raises(AssertionError):
        run_state_machine_as_test(DepthMachine, settings=ss)
    assert any(list(db.data.values()))

def test_can_run_with_no_db() -> None:
    with deterministic_PRNG(), raises(AssertionError):
        run_state_machine_as_test(DepthMachine, settings=Settings(database=None, max_examples=10000))

def test_stateful_double_rule_is_forbidden(recwarn) -> None:
    with pytest.raises(InvalidDefinition):

        class DoubleRuleMachine(RuleBasedStateMachine):

            @rule(num=just(1))
            @rule(num=just(2))
            def whatevs(self, num: int) -> None:
                pass

def test_can_explicitly_call_functions_when_precondition_not_satisfied() -> None:

    class BadPrecondition(RuleBasedStateMachine):

        def __init__(self) -> None:
            super().__init__()

        @precondition(lambda self: False)
        @rule()
        def test_blah(self) -> None:
            raise ValueError

        @rule()
        def test_foo(self) -> None:
            self.test_blah()
    with pytest.raises(ValueError):
        run_state_machine_as_test(BadPrecondition)

def test_invariant() -> None:
    """If an invariant raise an exception, the exception is propagated."""

    class Invariant(RuleBasedStateMachine):

        def __init__(self) -> None:
            super().__init__()

        @invariant()
        def test_blah(self) -> None:
            raise ValueError

        @rule()
        def do_stuff(self) -> None:
            pass
    with pytest.raises(ValueError):
        run_state_machine_as_test(Invariant)

def test_no_double_invariant() -> None:
    """The invariant decorator can't be applied multiple times to a single
    function."""
    with raises(InvalidDefinition):

        class Invariant(RuleBasedStateMachine):

            def __init__(self) -> None:
                super().__init__()

            @invariant()
            @invariant()
            def test_blah(self) -> None:
                pass

            @rule()
            def do_stuff(self) -> None:
                pass

def test_invariant_precondition() -> None:
    """If an invariant precodition isn't met, the invariant isn't run.

    The precondition decorator can be applied in any order.
    """

    class Invariant(RuleBasedStateMachine):

        def __init__(self) -> None:
            super().__init__()

        @invariant()
        @precondition(lambda _: False)
        def an_invariant(self) -> None:
            raise ValueError

        @precondition(lambda _: False)
        @invariant()
        def another_invariant(self) -> None:
            raise ValueError

        @rule()
        def do_stuff(self) -> None:
            pass
    run_state_machine_as_test(Invariant)

@pytest.mark.parametrize('decorators', [(invariant(), rule()), (rule(), invariant()), (invariant(), initialize()), (initialize(), invariant()), (invariant(), precondition(lambda self: True), rule()), (rule(), precondition(lambda self: True), invariant()), (precondition(lambda self: True), invariant(), rule()), (precondition(lambda self: True), rule(), invariant())], ids=lambda x: '-'.join((f.__qualname__.split('.')[0] for f in x)))
def test_invariant_and_rule_are_incompatible(decorators) -> None:
    """It's an error to apply @invariant and @rule to the same method."""

    def method(self) -> None:
        pass
    for d in decorators[:-1]:
        method = d(method)
    with pytest.raises(InvalidDefinition):
        decorators[-1](method)

def test_invalid_rule_argument() -> None:
    """Rule kwargs that are not a Strategy are expected to raise an InvalidArgument error."""
    with pytest.raises(InvalidArgument):

        class InvalidRuleMachine(RuleBasedStateMachine):

            @rule(strategy=object())
            def do_stuff(self) -> None:
                pass

def test_invalid_initialize_argument() -> None:
    """Initialize kwargs that are not a Strategy are expected to raise an InvalidArgument error."""
    with pytest.raises(InvalidArgument):

        class InvalidInitialize(RuleBasedStateMachine):

            @initialize(strategy=object())
            def initialize(self) -> None:
                pass

def test_multiple_invariants() -> None:
    """If multiple invariants are present, they all get run."""

    class Invariant(RuleBasedStateMachine):

        def __init__(self) -> None:
            super().__init__()
            self.first_invariant_ran: bool = False

        @invariant()
        def invariant_1(self) -> None:
            self.first_invariant_ran = True

        @precondition(lambda self: self.first_invariant_ran)
        @invariant()
        def invariant_2(self) -> None:
            raise ValueError

        @rule()
        def do_stuff(self) -> None:
            pass
    with pytest.raises(ValueError):
        run_state_machine_as_test(Invariant)

def test_explicit_invariant_call_with_precondition() -> None:

    class BadPrecondition(RuleBasedStateMachine):

        def __init__(self) -> None:
            super().__init__()

        @precondition(lambda self: False)
        @invariant()
        def test_blah(self) -> None:
            raise ValueError

        @rule()
        def test_foo(self) -> None:
            self.test_blah()
    with pytest.raises(ValueError):
        run_state_machine_as_test(BadPrecondition)

def test_invariant() -> None:
    """If an invariant raise an exception, the exception is propagated."""

    class Invariant(RuleBasedStateMachine):

        def __init__(self) -> None:
            super().__init__()

        @invariant()
        def test_blah(self) -> None:
            raise ValueError

        @rule()
        def do_stuff(self) -> None:
            pass
    with pytest.raises(ValueError):
        run_state_machine_as_test(Invariant)

def test_no_double_invariant() -> None:
    """The invariant decorator can't be applied multiple times to a single
    function."""
    with raises(InvalidDefinition):

        class Invariant(RuleBasedStateMachine):

            def __init__(self) -> None:
                super().__init__()

            @invariant()
            @invariant()
            def test_blah(self) -> None:
                pass

            @rule()
            def do_stuff(self) -> None:
                pass

def test_invariant_precondition() -> None:
    """If an invariant precodition isn't met, the invariant isn't run.

    The precondition decorator can be applied in any order.
    """

    class Invariant(RuleBasedStateMachine):

        def __init__(self)