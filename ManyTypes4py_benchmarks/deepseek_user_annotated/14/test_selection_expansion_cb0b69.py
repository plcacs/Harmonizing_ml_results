import pytest
from typing import List, Optional, Dict, Any
from dbt.tests.util import run_dbt


class TestSelectionExpansion:
    @pytest.fixture(scope="class")
    def project_config_update(self) -> Dict[str, Any]:
        return {"config-version": 2, "test-paths": ["tests"]}

    def list_tests_and_assert(
        self,
        include: Optional[str],
        exclude: Optional[str],
        expected_tests: List[str],
        indirect_selection: str = "eager",
        selector_name: Optional[str] = None,
    ) -> None:
        list_args: List[str] = ["ls", "--resource-type", "test"]
        if include:
            list_args.extend(("--select", include))
        if exclude:
            list_args.extend(("--exclude", exclude))
        if indirect_selection:
            list_args.extend(("--indirect-selection", indirect_selection))
        if selector_name:
            list_args.extend(("--selector", selector_name))

        listed: List[str] = run_dbt(list_args)
        assert len(listed) == len(expected_tests)

        test_names: List[str] = [name.split(".")[-1] for name in listed]
        assert sorted(test_names) == sorted(expected_tests)

    def run_tests_and_assert(
        self,
        include: Optional[str],
        exclude: Optional[str],
        expected_tests: List[str],
        indirect_selection: str = "eager",
        selector_name: Optional[str] = None,
    ) -> None:
        results: List[Any] = run_dbt(["run"])
        assert len(results) == 2

        test_args: List[str] = ["test"]
        if include:
            test_args.extend(("--models", include))
        if exclude:
            test_args.extend(("--exclude", exclude))
        if indirect_selection:
            test_args.extend(("--indirect-selection", indirect_selection))
        if selector_name:
            test_args.extend(("--selector", selector_name))

        results = run_dbt(test_args)
        tests_run: List[str] = [r.node.name for r in results]
        assert len(tests_run) == len(expected_tests)

        assert sorted(tests_run) == sorted(expected_tests)

    def test_all_tests_no_specifiers(
        self,
        project: Any,
    ) -> None:
        select: None = None
        exclude: None = None
        expected: List[str] = [
            "cf_a_b",
            "cf_a_src",
            "just_a",
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
            "source_unique_my_src_my_tbl_fun",
            "unique_model_a_fun",
        ]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_model_a_alone(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a"
        exclude: None = None
        expected: List[str] = [
            "cf_a_b",
            "cf_a_src",
            "just_a",
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
            "unique_model_a_fun",
        ]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_model_a_model_b(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a model_b"
        exclude: None = None
        expected: List[str] = [
            "cf_a_b",
            "cf_a_src",
            "just_a",
            "unique_model_a_fun",
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
        ]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_model_a_sources(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a source:*"
        exclude: None = None
        expected: List[str] = [
            "cf_a_b",
            "cf_a_src",
            "just_a",
            "unique_model_a_fun",
            "source_unique_my_src_my_tbl_fun",
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
        ]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_exclude_model_b(
        self,
        project: Any,
    ) -> None:
        select: None = None
        exclude: str = "model_b"
        expected: List[str] = [
            "cf_a_src",
            "just_a",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
            "source_unique_my_src_my_tbl_fun",
            "unique_model_a_fun",
        ]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_model_a_exclude_specific_test(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a"
        exclude: str = "unique_model_a_fun"
        expected: List[str] = [
            "cf_a_b",
            "cf_a_src",
            "just_a",
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
        ]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_model_a_exclude_specific_test_cautious(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a"
        exclude: str = "unique_model_a_fun"
        expected: List[str] = ["just_a"]
        indirect_selection: str = "cautious"

        self.list_tests_and_assert(select, exclude, expected, indirect_selection)
        self.run_tests_and_assert(select, exclude, expected, indirect_selection)

    def test_model_a_exclude_specific_test_buildable(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a"
        exclude: str = "unique_model_a_fun"
        expected: List[str] = [
            "just_a",
            "cf_a_b",
            "cf_a_src",
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
        ]
        indirect_selection: str = "buildable"

        self.list_tests_and_assert(select, exclude, expected, indirect_selection)
        self.run_tests_and_assert(select, exclude, expected, indirect_selection)

    def test_only_generic(
        self,
        project: Any,
    ) -> None:
        select: str = "test_type:generic"
        exclude: None = None
        expected: List[str] = [
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
            "source_unique_my_src_my_tbl_fun",
            "unique_model_a_fun",
        ]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_model_a_only_singular_unset(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a,test_type:singular"
        exclude: None = None
        expected: List[str] = ["cf_a_b", "cf_a_src", "just_a"]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_model_a_only_singular_eager(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a,test_type:singular"
        exclude: None = None
        expected: List[str] = ["cf_a_b", "cf_a_src", "just_a"]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_model_a_only_singular_cautious(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a,test_type:singular"
        exclude: None = None
        expected: List[str] = ["just_a"]
        indirect_selection: str = "cautious"

        self.list_tests_and_assert(
            select, exclude, expected, indirect_selection=indirect_selection
        )
        self.run_tests_and_assert(select, exclude, expected, indirect_selection=indirect_selection)

    def test_only_singular(
        self,
        project: Any,
    ) -> None:
        select: str = "test_type:singular"
        exclude: None = None
        expected: List[str] = ["cf_a_b", "cf_a_src", "just_a"]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_model_a_only_singular(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a,test_type:singular"
        exclude: None = None
        expected: List[str] = ["cf_a_b", "cf_a_src", "just_a"]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_test_name_intersection(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a,test_name:unique"
        exclude: None = None
        expected: List[str] = ["unique_model_a_fun"]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_model_tag_test_name_intersection(
        self,
        project: Any,
    ) -> None:
        select: str = "tag:a_or_b,test_name:relationships"
        exclude: None = None
        expected: List[str] = [
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
        ]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_select_column_level_tag(
        self,
        project: Any,
    ) -> None:
        select: str = "tag:column_level_tag"
        exclude: None = None
        expected: List[str] = [
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
            "unique_model_a_fun",
        ]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_exclude_column_level_tag(
        self,
        project: Any,
    ) -> None:
        select: None = None
        exclude: str = "tag:column_level_tag"
        expected: List[str] = ["cf_a_b", "cf_a_src", "just_a", "source_unique_my_src_my_tbl_fun"]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_test_level_tag(
        self,
        project: Any,
    ) -> None:
        select: str = "tag:test_level_tag"
        exclude: None = None
        expected: List[str] = ["relationships_model_a_fun__fun__ref_model_b_"]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_exclude_data_test_tag(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a"
        exclude: str = "tag:data_test_tag"
        expected: List[str] = [
            "cf_a_b",
            "cf_a_src",
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
            "unique_model_a_fun",
        ]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_model_a_indirect_selection(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a"
        exclude: None = None
        expected: List[str] = [
            "cf_a_b",
            "cf_a_src",
            "just_a",
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
            "unique_model_a_fun",
        ]

        self.list_tests_and_assert(select, exclude, expected)
        self.run_tests_and_assert(select, exclude, expected)

    def test_model_a_indirect_selection_eager(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a"
        exclude: None = None
        expected: List[str] = [
            "cf_a_b",
            "cf_a_src",
            "just_a",
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
            "unique_model_a_fun",
        ]
        indirect_selection: str = "eager"

        self.list_tests_and_assert(select, exclude, expected, indirect_selection)
        self.run_tests_and_assert(select, exclude, expected, indirect_selection)

    def test_model_a_indirect_selection_cautious(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a"
        exclude: None = None
        expected: List[str] = [
            "just_a",
            "unique_model_a_fun",
        ]
        indirect_selection: str = "cautious"

        self.list_tests_and_assert(select, exclude, expected, indirect_selection)
        self.run_tests_and_assert(select, exclude, expected, indirect_selection)

    def test_model_a_indirect_selection_buildable(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a"
        exclude: None = None
        expected: List[str] = [
            "cf_a_b",
            "cf_a_src",
            "just_a",
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
            "unique_model_a_fun",
        ]
        indirect_selection: str = "buildable"

        self.list_tests_and_assert(select, exclude, expected, indirect_selection)
        self.run_tests_and_assert(select, exclude, expected, indirect_selection)

    def test_model_a_indirect_selection_exclude_unique_tests(
        self,
        project: Any,
    ) -> None:
        select: str = "model_a"
        exclude: str = "test_name:unique"
        indirect_selection: str = "eager"
        expected: List[str] = [
            "cf_a_b",
            "cf_a_src",
            "just_a",
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
        ]

        self.list_tests_and_assert(select, exclude, expected, indirect_selection)
        self.run_tests_and_assert(select, exclude, expected, indirect_selection=indirect_selection)

    def test_model_a_indirect_selection_empty(self, project: Any) -> None:
        results: List[str] = run_dbt(["ls", "--indirect-selection", "empty", "--select", "model_a"])
        assert len(results) == 1


class TestExpansionWithSelectors(TestSelectionExpansion):
    @pytest.fixture(scope="class")
    def selectors(self) -> str:
        return """
            selectors:
            - name: model_a_unset_indirect_selection
              definition:
                method: fqn
                value: model_a
            - name: model_a_cautious_indirect_selection
              definition:
                method: fqn
                value: model_a
                indirect_selection: "cautious"
            - name: model_a_eager_indirect_selection
              definition:
                method: fqn
                value: model_a
                indirect_selection: "eager"
            - name: model_a_buildable_indirect_selection
              definition:
                method: fqn
                value: model_a
                indirect_selection: "buildable"
        """

    def test_selector_model_a_unset_indirect_selection(
        self,
        project: Any,
    ) -> None:
        expected: List[str] = [
            "cf_a_b",
            "cf_a_src",
            "just_a",
            "relationships_model_a_fun__fun__ref_model_b_",
            "relationships_model_a_fun__fun__source_my_src_my_tbl_",
            "unique_model_a_fun",
        ]

        self.list_tests_and_assert(
            include=None,
            exclude=None,
            expected_tests=expected,
            selector_name="model_a_unset_indirect_selection",
        )
        self.run_tests_and_assert(
            include=None,
            exclude=None,
            expected_tests=expected,
            selector_name="