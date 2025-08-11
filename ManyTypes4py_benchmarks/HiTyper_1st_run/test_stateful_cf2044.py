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
NO_BLOB_SETTINGS = Settings(print_blob=False, phases=tuple(Phase)[:-1])

class MultipleRulesSameFuncMachine(RuleBasedStateMachine):

    def myfunc(self, data: Union[str, bytes, int]) -> None:
        print(data)
    rule1 = rule(data=just('rule1data'))(myfunc)
    rule2 = rule(data=just('rule2data'))(myfunc)

class PreconditionMachine(RuleBasedStateMachine):
    num = 0

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
TestPrecondition = PreconditionMachine.TestCase
TestPrecondition.settings = Settings(TestPrecondition.settings, max_examples=10)

def test_picks_up_settings_at_first_use_of_testcase() -> None:
    assert TestPrecondition.settings.max_examples == 10

def test_multiple_rules_same_func() -> None:
    test_class = MultipleRulesSameFuncMachine.TestCase
    with capture_out() as o:
        test_class().runTest()
    output = o.getvalue()
    assert 'rule1data' in output
    assert 'rule2data' in output

def test_can_get_test_case_off_machine_instance() -> None:
    assert DepthMachine().TestCase is DepthMachine().TestCase
    assert DepthMachine().TestCase is not None

class FlakyDrawLessMachine(RuleBasedStateMachine):

    @rule(d=data())
    def action(self, d) -> None:
        if current_build_context().is_final:
            d.draw(binary(min_size=1, max_size=1))
        else:
            buffer = binary(min_size=1024, max_size=1024)
            assert 0 not in buffer

def test_flaky_draw_less_raises_flaky() -> None:
    with raises(Flaky):
        FlakyDrawLessMachine.TestCase().runTest()

def test_result_is_added_to_target() -> None:

    class TargetStateMachine(RuleBasedStateMachine):
        nodes = Bundle('nodes')

        @rule(target=nodes, source=lists(nodes))
        def bunch(self, source: Any):
            assert len(source) == 0
            return source
    test_class = TargetStateMachine.TestCase
    try:
        test_class().runTest()
        raise RuntimeError('Expected an assertion error')
    except AssertionError as err:
        notes = err.__notes__
    regularized_notes = [re.sub('[0-9]+', 'i', note) for note in notes]
    assert 'state.bunch(source=[nodes_i])' in regularized_notes

class FlakyStateMachine(RuleBasedStateMachine):

    @rule()
    def action(self) -> None:
        assert current_build_context().is_final

def test_flaky_raises_flaky() -> None:
    with raises(Flaky):
        FlakyStateMachine.TestCase().runTest()

class FlakyRatchettingMachine(RuleBasedStateMachine):
    ratchet = 0

    @rule(d=data())
    def action(self, d) -> None:
        FlakyRatchettingMachine.ratchet += 1
        n = FlakyRatchettingMachine.ratchet
        d.draw(lists(integers(), min_size=n, max_size=n))
        raise AssertionError

@Settings(stateful_step_count=10, max_examples=30, suppress_health_check=[HealthCheck.filter_too_much])
class MachineWithConsumingRule(RuleBasedStateMachine):
    b1 = Bundle('b1')
    b2 = Bundle('b2')

    def __init__(self) -> None:
        self.created_counter = 0
        self.consumed_counter = 0
        super().__init__()

    @invariant()
    def bundle_length(self) -> None:
        assert len(self.bundle('b1')) == self.created_counter - self.consumed_counter

    @rule(target=b1)
    def populate_b1(self):
        self.created_counter += 1
        return self.created_counter

    @rule(target=b2, consumed=consumes(b1))
    def depopulate_b1(self, consumed: Union[int, str]) -> Union[int, str]:
        self.consumed_counter += 1
        return consumed

    @rule(consumed=lists(consumes(b1), max_size=3))
    def depopulate_b1_multiple(self, consumed: Union[list[int], list]) -> None:
        self.consumed_counter += len(consumed)

    @rule(value1=b1, value2=b2)
    def check(self, value1: Union[str, int], value2: Union[str, int]) -> None:
        assert value1 != value2
TestMachineWithConsumingRule = MachineWithConsumingRule.TestCase

def test_multiple() -> None:
    none = multiple()
    some = multiple(1, 2.01, '3', b'4', 5)
    assert len(none.values) == 0
    assert len(some.values) == 5
    assert set(some.values) == {1, 2.01, '3', b'4', 5}

class MachineUsingMultiple(RuleBasedStateMachine):
    b = Bundle('b')

    def __init__(self) -> None:
        self.expected_bundle_length = 0
        super().__init__()

    @invariant()
    def bundle_length(self) -> None:
        assert len(self.bundle('b')) == self.expected_bundle_length

    @rule(target=b, items=lists(elements=integers(), max_size=10))
    def populate_bundle(self, items):
        self.expected_bundle_length += len(items)
        return multiple(*items)

    @rule(target=b)
    def do_not_populate(self) -> Union[str, int, typing.Iterable[str]]:
        return multiple()
TestMachineUsingMultiple = MachineUsingMultiple.TestCase

def test_multiple_variables_printed() -> None:

    class ProducesMultiple(RuleBasedStateMachine):
        b = Bundle('b')

        @initialize(target=b)
        def populate_bundle(self):
            return multiple(1, 2)

        @rule()
        def fail_fast(self) -> None:
            raise AssertionError
    with raises(AssertionError) as err:
        run_state_machine_as_test(ProducesMultiple)
    assignment_line = err.value.__notes__[2]
    assert assignment_line == 'b_0, b_1 = state.populate_bundle()'
    state = ProducesMultiple()
    b_0, b_1 = state.populate_bundle()
    with raises(AssertionError):
        state.fail_fast()

def test_multiple_variables_printed_single_element() -> None:

    class ProducesMultiple(RuleBasedStateMachine):
        b = Bundle('b')

        @initialize(target=b)
        def populate_bundle(self):
            return multiple(1)

        @rule(b=b)
        def fail_fast(self, b) -> None:
            assert b != 1
    with raises(AssertionError) as err:
        run_state_machine_as_test(ProducesMultiple)
    assignment_line = err.value.__notes__[2]
    assert assignment_line == '(b_0,) = state.populate_bundle()'
    state = ProducesMultiple()
    v1, = state.populate_bundle()
    state.fail_fast((v1,))
    with raises(AssertionError):
        state.fail_fast(v1)

def test_no_variables_printed() -> None:

    class ProducesNoVariables(RuleBasedStateMachine):
        b = Bundle('b')

        @initialize(target=b)
        def populate_bundle(self):
            return multiple()

        @rule()
        def fail_fast(self) -> None:
            raise AssertionError
    with raises(AssertionError) as err:
        run_state_machine_as_test(ProducesNoVariables)
    assignment_line = err.value.__notes__[2]
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
        def bye(self, hi: Any) -> None:
            pass
    with raises(InvalidDefinition):
        NonTerminalMachine.TestCase().runTest()

def test_minimizes_errors_in_teardown() -> None:
    counter = 0

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

    def __init__(self, threshold: Union[int, float]) -> None:
        super().__init__()
        self.threshold = threshold

    @rule(value=integers())
    def action(self, value: Union[bytes, str, int]) -> None:
        if value > self.threshold:
            raise ValueError(f'{value} is too high')

def test_can_use_factory_for_tests() -> None:
    with raises(ValueError):
        run_state_machine_as_test(lambda: RequiresInit(42), settings=Settings(max_examples=100))

class FailsEventually(RuleBasedStateMachine):

    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    @rule()
    def increment(self) -> None:
        self.counter += 1
        assert self.counter < 10
FailsEventually.TestCase.settings = Settings(stateful_step_count=5)

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
    real_settings = FailsEventually.TestCase.settings
    try:
        FailsEventually.TestCase.settings = object()
        with pytest.raises(InvalidArgument):
            run_state_machine_as_test(FailsEventually)
    finally:
        FailsEventually.TestCase.settings = real_settings

def test_saves_failing_example_in_database() -> None:
    db = ExampleDatabase(':memory:')
    ss = Settings(database=db, max_examples=1000, suppress_health_check=list(HealthCheck))
    with raises(AssertionError):
        run_state_machine_as_test(DepthMachine, settings=ss)
    assert any(list(db.data.values()))

def test_can_run_with_no_db() -> None:
    with deterministic_PRNG(), raises(AssertionError):
        run_state_machine_as_test(DepthMachine, settings=Settings(database=None, max_examples=10000))

def test_stateful_double_rule_is_forbidden(recwarn: tests.basilisp.helpers.CompileFn) -> None:
    with pytest.raises(InvalidDefinition):

        class DoubleRuleMachine(RuleBasedStateMachine):

            @rule(num=just(1))
            @rule(num=just(2))
            def whatevs(self, num: Any) -> None:
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
def test_invariant_and_rule_are_incompatible(decorators: Union[tests.basilisp.helpers.CompileFn, tests.hints.CleavageCall]) -> None:
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
            self.first_invariant_ran = False

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
    """Invariants can be called explicitly even if their precondition is not
    satisfied."""

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

def test_invariant_checks_initial_state_if_no_initialize_rules() -> None:
    """Invariants are checked before any rules run."""

    class BadPrecondition(RuleBasedStateMachine):

        def __init__(self) -> None:
            super().__init__()
            self.num = 0

        @invariant()
        def test_blah(self) -> None:
            if self.num == 0:
                raise ValueError

        @rule()
        def test_foo(self) -> None:
            self.num += 1
    with pytest.raises(ValueError):
        run_state_machine_as_test(BadPrecondition)

def test_invariant_failling_present_in_falsifying_example() -> None:

    @Settings(print_blob=False)
    class BadInvariant(RuleBasedStateMachine):

        @initialize()
        def initialize_1(self) -> None:
            pass

        @invariant()
        def invariant_1(self) -> None:
            raise ValueError

        @rule()
        def rule_1(self) -> None:
            pass
    with pytest.raises(ValueError) as err:
        run_state_machine_as_test(BadInvariant)
    result = '\n'.join(err.value.__notes__)
    assert result == '\nFalsifying example:\nstate = BadInvariant()\nstate.initialize_1()\nstate.invariant_1()\nstate.teardown()\n'.strip()

def test_invariant_present_in_falsifying_example() -> None:

    @Settings(print_blob=False, phases=tuple(Phase)[:-1])
    class BadRuleWithGoodInvariants(RuleBasedStateMachine):

        def __init__(self) -> None:
            super().__init__()
            self.num = 0

        @initialize()
        def initialize_1(self) -> None:
            pass

        @invariant(check_during_init=True)
        def invariant_1(self) -> None:
            pass

        @invariant(check_during_init=False)
        def invariant_2(self) -> None:
            pass

        @precondition(lambda self: self.num > 0)
        @invariant()
        def invariant_3(self) -> None:
            pass

        @rule()
        def rule_1(self) -> None:
            self.num += 1
            if self.num == 2:
                raise ValueError
    with pytest.raises(ValueError) as err:
        run_state_machine_as_test(BadRuleWithGoodInvariants)
    expected = '\nFalsifying example:\nstate = BadRuleWithGoodInvariants()\nstate.invariant_1()\nstate.initialize_1()\nstate.invariant_1()\nstate.invariant_2()\nstate.rule_1()\nstate.invariant_1()\nstate.invariant_2()\nstate.invariant_3()\nstate.rule_1()\nstate.teardown()\n'.strip()
    result = '\n'.join(err.value.__notes__).strip()
    assert expected == result

def test_always_runs_at_least_one_step() -> None:

    class CountSteps(RuleBasedStateMachine):

        def __init__(self) -> None:
            super().__init__()
            self.count = 0

        @rule()
        def do_something(self) -> None:
            self.count += 1

        def teardown(self) -> None:
            assert self.count > 0
    run_state_machine_as_test(CountSteps)

def test_removes_needless_steps() -> None:
    """Regression test from an example based on
    tests/nocover/test_database_agreement.py, but without the expensive bits.
    Comparing two database implementations in which deletion is broken, so as
    soon as a key/value pair is successfully deleted the test will now fail if
    you ever check that key.

    The main interesting feature of this is that it has a lot of
    opportunities to generate keys and values before it actually fails,
    but will still fail with very high probability.
    """

    @Settings(derandomize=True, max_examples=1000, deadline=None)
    class IncorrectDeletion(RuleBasedStateMachine):

        def __init__(self) -> None:
            super().__init__()
            self.__saved = defaultdict(set)
            self.__deleted = defaultdict(set)
        keys = Bundle('keys')
        values = Bundle('values')

        @rule(target=keys, k=binary())
        def k(self, k: Any):
            return k

        @rule(target=values, v=binary())
        def v(self, v: Any):
            return v

        @rule(k=keys, v=values)
        def save(self, k: Any, v: Any) -> None:
            self.__saved[k].add(v)

        @rule(k=keys, v=values)
        def delete(self, k: Any, v: Any) -> None:
            if v in self.__saved[k]:
                self.__deleted[k].add(v)

        @rule(k=keys)
        def values_agree(self, k: Any) -> None:
            assert not self.__deleted[k]
    with pytest.raises(AssertionError) as err:
        run_state_machine_as_test(IncorrectDeletion)
    result = '\n'.join(err.value.__notes__)
    assert result.count(' = state.k(') == 1
    assert result.count(' = state.v(') == 1

def test_prints_equal_values_with_correct_variable_name() -> None:

    @Settings(max_examples=100, suppress_health_check=list(HealthCheck))
    class MovesBetweenBundles(RuleBasedStateMachine):
        b1 = Bundle('b1')
        b2 = Bundle('b2')

        @rule(target=b1)
        def create(self) -> list:
            return []

        @rule(target=b2, source=b1)
        def transfer(self, source: Any):
            return source

        @rule(source=b2)
        def fail(self, source: Any) -> None:
            raise AssertionError
    with pytest.raises(AssertionError) as err:
        run_state_machine_as_test(MovesBetweenBundles)
    result = '\n'.join(err.value.__notes__)
    for m in ['create', 'transfer', 'fail']:
        assert result.count('state.' + m) == 1
    assert 'b1_0 = state.create()' in result
    assert 'b2_0 = state.transfer(source=b1_0)' in result
    assert 'state.fail(source=b2_0)' in result

def test_initialize_rule() -> None:

    @Settings(max_examples=1000)
    class WithInitializeRules(RuleBasedStateMachine):
        initialized = []

        @initialize()
        def initialize_a(self) -> None:
            self.initialized.append('a')

        @initialize()
        def initialize_b(self) -> None:
            self.initialized.append('b')

        @initialize()
        def initialize_c(self) -> None:
            self.initialized.append('c')

        @rule()
        def fail_fast(self) -> None:
            raise AssertionError
    with pytest.raises(AssertionError) as err:
        run_state_machine_as_test(WithInitializeRules)
    assert set(WithInitializeRules.initialized[-3:]) == {'a', 'b', 'c'}
    result = err.value.__notes__[1:]
    assert result[0] == 'state = WithInitializeRules()'
    assert {result[1], result[2], result[3]} == {'state.initialize_a()', 'state.initialize_b()', 'state.initialize_c()'}
    assert result[4] == 'state.fail_fast()'
    assert result[5] == 'state.teardown()'

def test_initialize_rule_populate_bundle() -> None:

    class WithInitializeBundleRules(RuleBasedStateMachine):
        a = Bundle('a')

        @initialize(target=a, dep=just('dep'))
        def initialize_a(self, dep) -> None:
            return f'a a_0 with ({dep})'

        @rule(param=a)
        def fail_fast(self, param) -> None:
            raise AssertionError
    WithInitializeBundleRules.TestCase.settings = NO_BLOB_SETTINGS
    with pytest.raises(AssertionError) as err:
        run_state_machine_as_test(WithInitializeBundleRules)
    result = '\n'.join(err.value.__notes__)
    assert result == "\nFalsifying example:\nstate = WithInitializeBundleRules()\na_0 = state.initialize_a(dep='dep')\nstate.fail_fast(param=a_0)\nstate.teardown()\n".strip()

def test_initialize_rule_dont_mix_with_precondition() -> None:
    with pytest.raises(InvalidDefinition) as exc:

        class BadStateMachine(RuleBasedStateMachine):

            @precondition(lambda self: True)
            @initialize()
            def initialize(self) -> None:
                pass
    assert 'An initialization rule cannot have a precondition.' in str(exc.value)
    with pytest.raises(InvalidDefinition) as exc:

        class BadStateMachineReverseOrder(RuleBasedStateMachine):

            @initialize()
            @precondition(lambda self: True)
            def initialize(self) -> None:
                pass
    assert 'An initialization rule cannot have a precondition.' in str(exc.value)

def test_initialize_rule_dont_mix_with_regular_rule() -> None:
    with pytest.raises(InvalidDefinition) as exc:

        class BadStateMachine(RuleBasedStateMachine):

            @rule()
            @initialize()
            def initialize(self) -> None:
                pass
    assert 'A function cannot be used for two distinct rules.' in str(exc.value)

def test_initialize_rule_cannot_be_double_applied() -> None:
    with pytest.raises(InvalidDefinition) as exc:

        class BadStateMachine(RuleBasedStateMachine):

            @initialize()
            @initialize()
            def initialize(self) -> None:
                pass
    assert 'A function cannot be used for two distinct rules.' in str(exc.value)

def test_initialize_rule_in_state_machine_with_inheritance() -> None:

    class ParentStateMachine(RuleBasedStateMachine):
        initialized = []

        @initialize()
        def initialize_a(self) -> None:
            self.initialized.append('a')

    class ChildStateMachine(ParentStateMachine):

        @initialize()
        def initialize_b(self) -> None:
            self.initialized.append('b')

        @rule()
        def fail_fast(self) -> None:
            raise AssertionError
    with pytest.raises(AssertionError) as err:
        run_state_machine_as_test(ChildStateMachine)
    assert set(ChildStateMachine.initialized[-2:]) == {'a', 'b'}
    result = err.value.__notes__[1:]
    assert result[0] == 'state = ChildStateMachine()'
    assert {result[1], result[2]} == {'state.initialize_a()', 'state.initialize_b()'}
    assert result[3] == 'state.fail_fast()'
    assert result[4] == 'state.teardown()'

def test_can_manually_call_initialize_rule() -> None:

    class StateMachine(RuleBasedStateMachine):
        initialize_called_counter = 0

        @initialize()
        def initialize(self) -> None:
            self.initialize_called_counter += 1

        @rule()
        def fail_eventually(self) -> None:
            self.initialize()
            assert self.initialize_called_counter <= 2
    StateMachine.TestCase.settings = NO_BLOB_SETTINGS
    with pytest.raises(AssertionError) as err:
        run_state_machine_as_test(StateMachine)
    result = '\n'.join(err.value.__notes__)
    assert result == '\nFalsifying example:\nstate = StateMachine()\nstate.initialize()\nstate.fail_eventually()\nstate.fail_eventually()\nstate.teardown()\n'.strip()

def test_steps_printed_despite_pytest_fail() -> None:

    @Settings(print_blob=False)
    class RaisesProblem(RuleBasedStateMachine):

        @rule()
        def oops(self) -> None:
            pytest.fail('note that this raises a BaseException')
    with pytest.raises(Failed) as err:
        run_state_machine_as_test(RaisesProblem)
    assert '\n'.join(err.value.__notes__).strip() == '\nFalsifying example:\nstate = RaisesProblem()\nstate.oops()\nstate.teardown()'.strip()

def test_steps_not_printed_with_pytest_skip(capsys: Union[tests.basilisp.helpers.CompileFn, str]) -> None:

    class RaisesProblem(RuleBasedStateMachine):

        @rule()
        def skip_whole_test(self) -> None:
            pytest.skip()
    with pytest.raises(Skipped):
        run_state_machine_as_test(RaisesProblem)
    out, _ = capsys.readouterr()
    assert 'state' not in out

def test_rule_deprecation_targets_and_target() -> None:
    k, v = (Bundle('k'), Bundle('v'))
    with pytest.raises(InvalidArgument):
        rule(targets=(k,), target=v)

def test_rule_deprecation_bundle_by_name() -> None:
    Bundle('k')
    with pytest.raises(InvalidArgument):
        rule(target='k')

def test_rule_non_bundle_target() -> None:
    with pytest.raises(InvalidArgument):
        rule(target=integers())

def test_rule_non_bundle_target_oneof() -> None:
    k, v = (Bundle('k'), Bundle('v'))
    pattern = '.+ `one_of(a, b)` or `a | b` .+'
    with pytest.raises(InvalidArgument, match=pattern):
        rule(target=k | v)

def test_uses_seed(capsys: Union[str, tests.e2e.Helper]) -> None:

    @seed(0)
    class TrivialMachine(RuleBasedStateMachine):

        @rule()
        def oops(self) -> None:
            raise AssertionError
    with pytest.raises(AssertionError):
        run_state_machine_as_test(TrivialMachine)
    out, _ = capsys.readouterr()
    assert '@seed' not in out

def test_reproduce_failure_works() -> None:

    @reproduce_failure(__version__, encode_failure([False, 0, True]))
    class TrivialMachine(RuleBasedStateMachine):

        @rule()
        def oops(self) -> None:
            raise AssertionError
    with pytest.raises(AssertionError):
        run_state_machine_as_test(TrivialMachine, settings=Settings(print_blob=True))

def test_reproduce_failure_fails_if_no_error() -> None:

    @reproduce_failure(__version__, encode_failure([False, 0, True]))
    class TrivialMachine(RuleBasedStateMachine):

        @rule()
        def ok(self) -> None:
            pass
    with pytest.raises(DidNotReproduce):
        run_state_machine_as_test(TrivialMachine, settings=Settings(print_blob=True))

def test_cannot_have_zero_steps() -> None:
    with pytest.raises(InvalidArgument):
        Settings(stateful_step_count=0)

def test_arguments_do_not_use_names_of_return_values() -> None:

    class TrickyPrintingMachine(RuleBasedStateMachine):
        data = Bundle('data')

        @initialize(target=data, value=integers())
        def init_data(self, value: Any):
            return value

        @rule(d=data)
        def mostly_fails(self, d: Any) -> None:
            assert d == 42
    with pytest.raises(AssertionError) as err:
        run_state_machine_as_test(TrickyPrintingMachine)
    assert 'data_0 = state.init_data(value=0)' in err.value.__notes__
    assert 'data_0 = state.init_data(value=data_0)' not in err.value.__notes__

class TrickyInitMachine(RuleBasedStateMachine):

    @initialize()
    def init_a(self) -> None:
        self.a = 0

    @rule()
    def inc(self) -> None:
        self.a += 1

    @invariant()
    def check_a_positive(self) -> None:
        assert self.a >= 0

def test_invariants_are_checked_after_init_steps() -> None:
    run_state_machine_as_test(TrickyInitMachine)

def test_invariants_can_be_checked_during_init_steps() -> None:

    class UndefinedMachine(TrickyInitMachine):

        @invariant(check_during_init=True)
        def check_a_defined(self) -> None:
            self.a
    with pytest.raises(AttributeError):
        run_state_machine_as_test(UndefinedMachine)

def test_check_during_init_must_be_boolean() -> None:
    invariant(check_during_init=False)
    invariant(check_during_init=True)
    with pytest.raises(InvalidArgument):
        invariant(check_during_init='not a bool')

def test_deprecated_target_consumes_bundle() -> None:
    with validate_deprecation():
        rule(target=consumes(Bundle('b')))

@Settings(stateful_step_count=5)
class MinStepsMachine(RuleBasedStateMachine):

    @initialize()
    def init_a(self) -> None:
        self.a = 0

    @rule()
    def inc(self) -> None:
        self.a += 1

    @invariant()
    def not_too_many_steps(self) -> None:
        assert self.a < 10

    def teardown(self) -> None:
        assert self.a >= 2

def test_min_steps_argument() -> None:
    for n_steps in (-1, 'nan', 5.0):
        with pytest.raises(InvalidArgument):
            run_state_machine_as_test(MinStepsMachine, _min_steps=n_steps)
    run_state_machine_as_test(MinStepsMachine, _min_steps=3)
    run_state_machine_as_test(MinStepsMachine, _min_steps=20)

class ErrorsOnClassAttributeSettings(RuleBasedStateMachine):
    settings = Settings(derandomize=True)

    @rule()
    def step(self) -> None:
        pass

def test_fails_on_settings_class_attribute() -> None:
    with pytest.raises(InvalidDefinition, match='Assigning .+ as a class attribute does nothing'):
        run_state_machine_as_test(ErrorsOnClassAttributeSettings)

def test_single_target_multiple() -> None:

    class Machine(RuleBasedStateMachine):
        a = Bundle('a')

        @initialize(target=a)
        def initialize(self) -> None:
            return multiple('ret1', 'ret2', 'ret3')

        @rule(param=a)
        def fail_fast(self, param) -> None:
            raise AssertionError
    Machine.TestCase.settings = NO_BLOB_SETTINGS
    with pytest.raises(AssertionError) as err:
        run_state_machine_as_test(Machine)
    result = '\n'.join(err.value.__notes__)
    assert result == '\nFalsifying example:\nstate = Machine()\na_0, a_1, a_2 = state.initialize()\nstate.fail_fast(param=a_2)\nstate.teardown()\n'.strip()

def test_multiple_targets() -> None:

    class Machine(RuleBasedStateMachine):
        a = Bundle('a')
        b = Bundle('b')

        @initialize(targets=(a, b))
        def initialize(self) -> None:
            return multiple('ret1', 'ret2', 'ret3')

        @rule(a1=consumes(a), a2=consumes(a), a3=consumes(a), b1=consumes(b), b2=consumes(b), b3=consumes(b))
        def fail_fast(self, a1, a2, a3, b1, b2, b3) -> None:
            raise AssertionError
    Machine.TestCase.settings = NO_BLOB_SETTINGS
    with pytest.raises(AssertionError) as err:
        run_state_machine_as_test(Machine)
    result = '\n'.join(err.value.__notes__)
    assert result == '\nFalsifying example:\nstate = Machine()\na_0, b_0, a_1, b_1, a_2, b_2 = state.initialize()\nstate.fail_fast(a1=a_2, a2=a_1, a3=a_0, b1=b_2, b2=b_1, b3=b_0)\nstate.teardown()\n'.strip()

def test_multiple_common_targets() -> None:

    class Machine(RuleBasedStateMachine):
        a = Bundle('a')
        b = Bundle('b')

        @initialize(targets=(a, b, a))
        def initialize(self) -> None:
            return multiple('ret1', 'ret2', 'ret3')

        @rule(a1=consumes(a), a2=consumes(a), a3=consumes(a), a4=consumes(a), a5=consumes(a), a6=consumes(a), b1=consumes(b), b2=consumes(b), b3=consumes(b))
        def fail_fast(self, a1, a2, a3, a4, a5, a6, b1, b2, b3) -> None:
            raise AssertionError
    Machine.TestCase.settings = NO_BLOB_SETTINGS
    with pytest.raises(AssertionError) as err:
        run_state_machine_as_test(Machine)
    result = '\n'.join(err.value.__notes__)
    assert result == '\nFalsifying example:\nstate = Machine()\na_0, b_0, a_1, a_2, b_1, a_3, a_4, b_2, a_5 = state.initialize()\nstate.fail_fast(a1=a_5, a2=a_4, a3=a_3, a4=a_2, a5=a_1, a6=a_0, b1=b_2, b2=b_1, b3=b_0)\nstate.teardown()\n'.strip()

class LotsOfEntropyPerStepMachine(RuleBasedStateMachine):

    @rule(data=binary(min_size=512, max_size=512))
    def rule1(self, data: Union[list[str], list[list[typing.Any]], bytes]) -> None:
        assert data
TestLotsOfEntropyPerStepMachine = LotsOfEntropyPerStepMachine.TestCase

def test_flatmap() -> None:

    class Machine(RuleBasedStateMachine):
        buns = Bundle('buns')

        @initialize(target=buns)
        def create_bun(self) -> int:
            return 0

        @rule(target=buns, bun=buns.flatmap(lambda x: just(x + 1)))
        def use_flatmap(self, bun: Any):
            assert isinstance(bun, int)
            return bun

        @rule(bun=buns)
        def use_directly(self, bun: Any) -> None:
            assert isinstance(bun, int)
    Machine.TestCase.settings = Settings(stateful_step_count=5, max_examples=10)
    run_state_machine_as_test(Machine)

def test_use_bundle_within_other_strategies() -> None:

    class Class:

        def __init__(self, value) -> None:
            self.value = value

    class Machine(RuleBasedStateMachine):
        my_bundle = Bundle('my_bundle')

        @initialize(target=my_bundle)
        def set_initial(self, /) -> typing.Text:
            return 'sample text'

        @rule(instance=st.builds(Class, my_bundle))
        def check(self, instance) -> None:
            assert isinstance(instance, Class)
            assert isinstance(instance.value, str)
    Machine.TestCase.settings = Settings(stateful_step_count=5, max_examples=10)
    run_state_machine_as_test(Machine)