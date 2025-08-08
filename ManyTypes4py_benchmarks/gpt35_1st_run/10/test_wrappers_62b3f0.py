from typing import Any, Dict, List, Optional, Union

class User(faust.Record):
    pass

def same_items(a: Any, b: Any) -> bool:
    a_list = _maybe_items(a)
    b_list = _maybe_items(b)
    return same(a_list, b_list)

def _maybe_items(d: Any) -> Any:
    try:
        items = d.items
    except AttributeError:
        return d
    else:
        return items()

def same(a: List, b: List) -> bool:
    return sorted(a) == sorted(b)

def current_event(*, freeze_time: Any) -> Any:
    ...

class test_WindowSet:

    def wset(self, *, wtable: Any, event: Any) -> Any:
        ...

    def test_constructor(self, *, event: Any, table: Any, wset: Any, wtable: Any) -> None:
        ...

    def test_apply(self, *, wset: Any, event: Any) -> Any:
        ...

    def mock_get_timestamp(self, wset: Any) -> Any:
        ...

    def test_apply__custom_event(self, *, wset: Any, event: Any) -> Any:
        ...

    def test_value(self, *, event: Any, wset: Any) -> Any:
        ...

    def test_now(self, *, wset: Any) -> Any:
        ...

    def test_current(self, *, table: Any, wset: Any) -> Any:
        ...

    def test_current__default_event(self, *, table: Any, wset: Any) -> Any:
        ...

    def test_delta(self, *, table: Any, wset: Any) -> Any:
        ...

    def test_delta__default_event(self, *, table: Any, wset: Any) -> Any:
        ...

    def test_getitem(self, *, wset: Any) -> Any:
        ...

    def test_getitem__event(self, app: Any, wset: Any) -> Any:
        ...

    def test_setitem(self, *, wset: Any) -> Any:
        ...

    def test_setitem__event(self, app: Any, wset: Any) -> Any:
        ...

    def test_delitem(self, *, wset: Any) -> Any:
        ...

    def test_delitem__event(self, app: Any, wset: Any) -> Any:
        ...

    def test_operators(self, meth: str, expected_op: Any, *, wset: Any) -> Any:
        ...

    def test_repr(self, *, wset: Any) -> Any:
        ...

class test_WindowWrapper:

    def test_name(self, *, wtable: Any) -> Any:
        ...

    def test_relative_to(self, *, wtable: Any) -> Any:
        ...

    def test_relative_to_now(self, *, table: Any, wtable: Any) -> Any:
        ...

    def test_relative_to_field(self, *, table: Any, wtable: Any) -> Any:
        ...

    def test_relative_to_stream(self, *, table: Any, wtable: Any) -> Any:
        ...

    def test_get_timestamp(self, input: Any, expected: Any, *, event: Any, wtable: Any) -> Any:
        ...

    def test_get_timestamp__event_is_None(self, *, event: Any, wtable: Any) -> Any:
        ...

    def test_on_recover(self, *, wtable: Any, table: Any) -> Any:
        ...

    def test_contains(self, *, table: Any, wtable: Any) -> Any:
        ...

    def test_getitem(self, *, wtable: Any) -> Any:
        ...

    def test_setitem(self, *, table: Any, wtable: Any) -> Any:
        ...

    def test_setitem__key_is_WindowSet(self, *, wtable: Any) -> Any:
        ...

    def test_delitem(self, *, table: Any, wtable: Any) -> Any:
        ...

    def test_len__no_key_index_raises(self, *, wtable: Any) -> Any:
        ...

    def test_as_ansitable__raises(self, *, wtable: Any) -> Any:
        ...

    def test_keys_raises(self, *, wtable: Any) -> Any:
        ...

    def test_relative_handler(self, input: Any, *, wtable: Any) -> Any:
        ...

    def test_relative_handler__invalid_handler(self, *, wtable: Any) -> Any:
        ...

class test_WindowWrapper_using_key_index:

    def wset(self, *, iwtable: Any, event: Any) -> Any:
        ...

    def data(self, *, freeze_time: Any, iwtable: Any) -> Any:
        ...

    def data_with_30s_delta(self, *, freeze_time: Any, iwtable: Any, data: Any) -> Any:
        ...

    def remove_a_key(self, *, iwtable: Any, data: Any) -> Any:
        ...

    def test_len(self, *, iwtable: Any) -> Any:
        ...

    def test_as_ansitable(self, *, iwtable: Any, data: Any) -> Any:
        ...

    def test_items(self, *, iwtable: Any, data: Any) -> Any:
        ...

    def test_items_keys_in_index_not_in_table(self, *, iwtable: Any, remove_a_key: Any) -> Any:
        ...

    def test_items_now(self, *, iwtable: Any, data: Any) -> Any:
        ...

    def test_items_now_keys_in_index_not_in_table(self, *, iwtable: Any, remove_a_key: Any) -> Any:
        ...

    def test_items_current(self, *, iwtable: Any, data: Any, current_event: Any) -> Any:
        ...

    def test_items_current_keys_in_index_not_in_table(self, *, iwtable: Any, remove_a_key: Any, current_event: Any) -> Any:
        ...

    def test_items_delta(self, *, iwtable: Any, data_with_30s_delta: Any, current_event: Any) -> Any:
        ...

    def test_items_delta_key_not_in_table(self, *, iwtable: Any, data_with_30s_delta: Any, remove_a_key: Any, current_event: Any) -> Any:
        ...

    def test_keys(self, *, iwtable: Any, data: Any) -> Any:
        ...

    def test_keys__now(self, *, iwtable: Any, data: Any) -> Any:
        ...

    def test_keys__current(self, *, iwtable: Any, data: Any, current_event: Any) -> Any:
        ...

    def test_keys__delta(self, *, iwtable: Any, data: Any, current_event: Any) -> Any:
        ...

    def test_iter(self, *, iwtable: Any, data: Any) -> Any:
        ...

    def test_values(self, *, iwtable: Any, data: Any) -> Any:
        ...

    def test_values_keys_in_index_not_in_table(self, *, iwtable: Any, remove_a_key: Any) -> Any:
        ...

    def test_values_now(self, *, iwtable: Any, data: Any) -> Any:
        ...

    def test_values_now_keys_in_index_not_in_table(self, *, iwtable: Any, remove_a_key: Any) -> Any:
        ...

    def test_values_current(self, *, iwtable: Any, data: Any, current_event: Any) -> Any:
        ...

    def test_values_current_keys_in_index_not_in_table(self, *, iwtable: Any, remove_a_key: Any, current_event: Any) -> Any:
        ...

    def test_values_delta(self, *, iwtable: Any, data_with_30s_delta: Any, current_event: Any) -> Any:
        ...

    def test_values_delta_key_not_in_table(self, *, iwtable: Any, data_with_30s_delta: Any, remove_a_key: Any, current_event: Any) -> Any:
        ...

    def test_setitem(self, *, wset: Any) -> Any:
        ...

    def test_delitem(self, *, wset: Any) -> Any:
        ...
