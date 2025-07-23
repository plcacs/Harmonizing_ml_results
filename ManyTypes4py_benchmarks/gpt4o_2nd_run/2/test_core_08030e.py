import unittest
import pytest
from _pytest.outcomes import Failed, Skipped
from hypothesis import Phase, example, find, given, reject, settings, strategies as s
from hypothesis.database import InMemoryExampleDatabase
from hypothesis.errors import InvalidArgument, NoSuchExample, Unsatisfiable
from typing import Callable, Optional

def test_stops_after_max_examples_if_satisfying() -> None:
    tracker: list[int] = []

    def track(x: int) -> bool:
        tracker.append(x)
        return False

    max_examples: int = 100
    with pytest.raises(NoSuchExample):
        find(s.integers(0, 10000), track, settings=settings(max_examples=max_examples))
    assert len(tracker) == max_examples

def test_stops_after_ten_times_max_examples_if_not_satisfying() -> None:
    count: int = 0

    def track(x: int) -> None:
        nonlocal count
        count += 1
        reject()

    max_examples: int = 100
    with pytest.raises(Unsatisfiable):
        find(s.integers(0, 10000), track, settings=settings(max_examples=max_examples))
    assert count <= 10 * max_examples

some_normal_settings: settings = settings()

def test_is_not_normally_default() -> None:
    assert settings.default is not some_normal_settings

@given(s.booleans())
@some_normal_settings
def test_settings_are_default_in_given(x: bool) -> None:
    assert settings.default is some_normal_settings

def test_given_shrinks_pytest_helper_errors() -> None:
    final_value: Optional[int] = None

    @settings(derandomize=True, max_examples=100)
    @given(s.integers())
    def inner(x: int) -> None:
        nonlocal final_value
        final_value = x
        if x > 100:
            pytest.fail(f'x={x!r} is too big!')

    with pytest.raises(Failed):
        inner()
    assert final_value == 101

def test_pytest_skip_skips_shrinking() -> None:
    seen_large: bool = False

    @settings(derandomize=True, max_examples=100)
    @given(s.integers())
    def inner(x: int) -> None:
        nonlocal seen_large
        if x > 100:
            if seen_large:
                raise Exception('Should never replay a skipped test!')
            seen_large = True
            pytest.skip(f'x={x!r} is too big!')

    with pytest.raises(Skipped):
        inner()

def test_can_find_with_db_eq_none() -> None:
    find(s.integers(), bool, settings=settings(database=None, max_examples=100))

def test_no_such_example() -> None:
    with pytest.raises(NoSuchExample):
        find(s.none(), bool, database_key=b'no such example')

def test_validates_strategies_for_test_method() -> None:
    invalid_strategy = s.lists(s.nothing(), min_size=1)

    class TestStrategyValidation:

        @given(invalid_strategy)
        def test_method_with_bad_strategy(self, x: list) -> None:
            pass

    instance = TestStrategyValidation()
    with pytest.raises(InvalidArgument):
        instance.test_method_with_bad_strategy()

@example(1)
@given(s.integers())
@settings(phases=[Phase.target, Phase.shrink, Phase.explain])
def no_phases(_: int) -> None:
    raise Exception

@given(s.integers())
@settings(phases=[Phase.explicit])
def no_explicit(_: int) -> None:
    raise Exception

@given(s.integers())
@settings(phases=[Phase.reuse], database=InMemoryExampleDatabase())
def empty_db(_: int) -> None:
    raise Exception

@pytest.mark.parametrize('test_fn', [no_phases, no_explicit, empty_db], ids=lambda t: t.__name__)
def test_non_executed_tests_raise_skipped(test_fn: Callable[[], None]) -> None:
    with pytest.raises(unittest.SkipTest):
        test_fn()

@pytest.mark.parametrize('codec, max_codepoint, exclude_categories, categories', [
    ('ascii', None, None, None),
    ('ascii', 128, None, None),
    ('ascii', 100, None, None),
    ('utf-8', None, None, None),
    ('utf-8', None, ['Cs'], None),
    ('utf-8', None, ['N'], None),
    ('utf-8', None, None, ['N'])
])
@given(s.data())
def test_characters_codec(codec: str, max_codepoint: Optional[int], exclude_categories: Optional[list[str]], categories: Optional[list[str]], data: s.DataObject) -> None:
    strategy = s.characters(codec=codec, max_codepoint=max_codepoint, exclude_categories=exclude_categories, categories=categories)
    example = data.draw(strategy)
    assert example.encode(encoding=codec).decode(encoding=codec) == example
