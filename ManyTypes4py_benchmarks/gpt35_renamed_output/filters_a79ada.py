def func_v4eoi96m(conf: ConfigType) -> dict[str, dict[str, set[str]]]:
    ...


def func_9mn00g16(base_filter: dict[str, dict[str, set[str]]], add_filter: dict[str, dict[str, set[str]]]) -> dict[str, dict[str, set[str]]]:
    ...


def func_qcpxhu4g(conf: ConfigType) -> Filters:
    ...


class Filters:
    def __init__(self, excluded_entities: list[str] | None = None, excluded_domains: list[str] | None = None, excluded_entity_globs: list[str] | None = None, included_entities: list[str] | None = None, included_domains: list[str] | None = None, included_entity_globs: list[str] | None = None):
        ...

    @property
    def func_9908myyr(self) -> bool:
        ...

    @property
    def func_uq2weplj(self) -> bool:
        ...

    @property
    def func_55h3qf0h(self) -> bool:
        ...

    def func_g4srmbll(self, columns: Iterable[ColumnElement], encoder: Callable[[Any], str]) -> ColumnElement:
        ...

    def func_qhpan2hw(self) -> ColumnElement:
        ...

    def func_x8sb50de(self) -> ColumnElement:
        ...

    def func_66ss8nqb(self) -> ColumnElement:
        ...


def func_1heegk1l(glob_strs: list[str], columns: Iterable[ColumnElement], encoder: Callable[[Any], str]) -> ColumnElement:
    ...


def func_kntg84uy(entity_ids: list[str], columns: Iterable[ColumnElement], encoder: Callable[[Any], str]) -> ColumnElement:
    ...


def func_z1f6j9x2(domains: list[str], columns: Iterable[ColumnElement], encoder: Callable[[Any], str]) -> ColumnElement:
    ...


def func_h9qd2ui4(domains: list[str]) -> list[str]:
    ...
