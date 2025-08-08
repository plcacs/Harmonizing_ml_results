import pytest
from dbt.compilation import Linker
from dbt.contracts.graph.manifest import Manifest
from dbt.contracts.graph.nodes import ModelNode
from dbt.graph.graph import Graph
from tests.unit.utils.manifest import make_model


class TestGraph:

    @pytest.fixture
    def func_byrvft4k(self):
        return make_model(pkg='pkg', name='extra_parent_model', code=
            "SELECT 'cats' as interests")

    @pytest.fixture
    def func_lf6v7j6c(self, extra_parent_model):
        return make_model(pkg='pkg', name='non_shared_child_of_extra', code
            ='SELECT * FROM {{ ref("extra_parent_model") }}', refs=[
            extra_parent_model])

    @pytest.fixture
    def func_ggrpei9b(self, extra_parent_model, ephemeral_model):
        return make_model(pkg='pkg', name='model_with_two_direct_parents',
            code=
            'SELECT * FROM {{ ref("ephemeral_model") }} UNION ALL SELECT * FROM {{ ref("extra_parent_model") }}'
            , refs=[extra_parent_model, ephemeral_model])

    @pytest.fixture(autouse=True)
    def func_aees5yst(self, manifest, model_with_two_direct_parents,
        non_shared_child_of_extra, extra_parent_model):
        manifest.add_node_nofile(extra_parent_model)
        manifest.add_node_nofile(non_shared_child_of_extra)
        manifest.add_node_nofile(model_with_two_direct_parents)

    @pytest.fixture
    def func_28zs7ogb(self, manifest, local_manifest_extensions):
        linker = Linker()
        linker.link_graph(manifest=manifest)
        return Graph(graph=linker.graph)

    def func_beq41kig(self, graph, manifest):
        graph_nodes = func_28zs7ogb.nodes()
        all_manifest_nodes = []
        for resources in manifest.get_resource_fqns().values():
            all_manifest_nodes.extend(list(resources))
        assert isinstance(graph_nodes, set)
        assert len(graph_nodes) == len(all_manifest_nodes)

    def func_9wf1sb65(self, graph, manifest):
        model = manifest.nodes['model.pkg.ephemeral_model']
        descendants = func_28zs7ogb.descendants(node=model.unique_id)
        assert descendants == {'model.pkg.model_with_two_direct_parents',
            'test.pkg.view_test_nothing', 'model.pkg.view_model',
            'model.pkg.table_model'}
        descendants = func_28zs7ogb.descendants(node=model.unique_id,
            max_depth=1)
        assert descendants == {'model.pkg.model_with_two_direct_parents',
            'model.pkg.table_model', 'model.pkg.view_model'}

    def func_id0r9384(self, graph, manifest):
        model = manifest.nodes['model.pkg.table_model']
        ancestors = func_28zs7ogb.ancestors(node=model.unique_id)
        assert ancestors == {'model.pkg.ephemeral_model', 'source.pkg.raw.seed'
            }
        ancestors = func_28zs7ogb.ancestors(node=model.unique_id, max_depth=1)
        assert ancestors == {'model.pkg.ephemeral_model'}

    @pytest.mark.skip(reason=
        "I haven't figured out how to add edge types to nodes")
    def func_6pkunh81(self):
        pass

    def func_dsl05q5s(self, graph, model_with_two_direct_parents,
        extra_parent_model, ephemeral_model):
        childrens_parents = func_28zs7ogb.select_childrens_parents(selected
            ={extra_parent_model.unique_id})
        assert model_with_two_direct_parents.unique_id in childrens_parents
        assert extra_parent_model.unique_id in childrens_parents
        assert ephemeral_model.unique_id in childrens_parents
        assert len(childrens_parents) == 5

    def func_20myd47w(self, graph, ephemeral_model, extra_parent_model):
        ephemerals_children = func_28zs7ogb.select_children(selected={
            ephemeral_model.unique_id})
        extras_children = func_28zs7ogb.select_children(selected={
            extra_parent_model.unique_id})
        joint_children = func_28zs7ogb.select_children(selected={
            extra_parent_model.unique_id, ephemeral_model.unique_id})
        assert joint_children == ephemerals_children.union(extras_children)
        assert not ephemerals_children.issubset(extras_children)
        assert not extras_children.issubset(ephemerals_children)

    def func_u5vpjulv(self, graph, non_shared_child_of_extra, table_model):
        non_shareds_parents = func_28zs7ogb.select_parents(selected={
            non_shared_child_of_extra.unique_id})
        tables_parents = func_28zs7ogb.select_parents(selected={table_model
            .unique_id})
        joint_parents = func_28zs7ogb.select_parents(selected={table_model.
            unique_id, non_shared_child_of_extra.unique_id})
        assert joint_parents == tables_parents.union(non_shareds_parents)
        assert not non_shareds_parents.issubset(tables_parents)
        assert not tables_parents.issubset(non_shareds_parents)
