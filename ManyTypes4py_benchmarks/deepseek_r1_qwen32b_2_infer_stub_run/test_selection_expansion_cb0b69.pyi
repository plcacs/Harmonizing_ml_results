import pytest
from dbt.tests.util import run_dbt

class TestSelectionExpansion:
    @pytest.fixture(scope='class')
    def project_config_update(self) -> dict[str, int | list[str]]:
        ...

    def list_tests_and_assert(
        self,
        include: str | None,
        exclude: str | None,
        expected_tests: list[str],
        indirect_selection: str = 'eager',
        selector_name: str | None = None
    ) -> None:
        ...

    def run_tests_and_assert(
        self,
        include: str | None,
        exclude: str | None,
        expected_tests: list[str],
        indirect_selection: str = 'eager',
        selector_name: str | None = None
    ) -> None:
        ...

    def test_all_tests_no_specifiers(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_alone(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_model_b(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_sources(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_exclude_model_b(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_exclude_specific_test(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_exclude_specific_test_cautious(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_exclude_specific_test_buildable(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_only_generic(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_only_singular_unset(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_only_singular_eager(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_only_singular_cautious(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_only_singular(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_only_singular(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_test_name_intersection(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_tag_test_name_intersection(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_select_column_level_tag(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_exclude_column_level_tag(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_test_level_tag(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_exclude_data_test_tag(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_indirect_selection(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_indirect_selection_eager(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_indirect_selection_cautious(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_indirect_selection_buildable(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_indirect_selection_exclude_unique_tests(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_model_a_indirect_selection_empty(self, project: pytest.FixtureDef[Any]) -> None:
        ...

class TestExpansionWithSelectors(TestSelectionExpansion):
    @pytest.fixture(scope='class')
    def selectors(self) -> str:
        ...

    def test_selector_model_a_unset_indirect_selection(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_selector_model_a_cautious_indirect_selection(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_selector_model_a_eager_indirect_selection(self, project: pytest.FixtureDef[Any]) -> None:
        ...

    def test_selector_model_a_buildable_indirect_selection(self, project: pytest.FixtureDef[Any]) -> None:
        ...