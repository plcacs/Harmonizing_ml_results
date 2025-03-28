from __future__ import annotations
import json
from collections import Counter, defaultdict
from graphlib import CycleError, TopologicalSorter
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Union
import kedro
from kedro.pipeline.node import Node, _to_list
from .transcoding import _strip_transcoding
if TYPE_CHECKING:
    from collections.abc import Iterable

def __getattr__(name):
    if name == 'TRANSCODING_SEPARATOR':
        import warnings
        from kedro.pipeline.transcoding import TRANSCODING_SEPARATOR
        warnings.warn(f"{name!r} has been moved to 'kedro.pipeline.transcoding', and the alias will be removed in Kedro 0.20.0", kedro.KedroDeprecationWarning, stacklevel=2)
        return TRANSCODING_SEPARATOR
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

class OutputNotUniqueError(Exception):
    """Raised when two or more nodes that are part of the same pipeline
    produce outputs with the same name.
    """
    pass

class ConfirmNotUniqueError(Exception):
    """Raised when two or more nodes that are part of the same pipeline
    attempt to confirm the same dataset.
    """
    pass

class Pipeline:
    """A ``Pipeline`` defined as a collection of ``Node`` objects. This class
    treats nodes as part of a graph representation and provides inputs,
    outputs and execution order.
    """

    def __init__(self, nodes, *, tags: Optional[Union[str, Iterable[str]]]=None):
        if nodes is None:
            raise ValueError("'nodes' argument of 'Pipeline' is None. It must be an iterable of nodes and/or pipelines instead.")
        nodes_list = list(nodes)
        _validate_duplicate_nodes(nodes_list)
        nodes_chain = list(chain.from_iterable([[n] if isinstance(n, Node) else n.nodes for n in nodes_list]))
        _validate_transcoded_inputs_outputs(nodes_chain)
        _tags = set(_to_list(tags))
        if _tags:
            tagged_nodes = [n.tag(_tags) for n in nodes_chain]
        else:
            tagged_nodes = nodes_chain
        self._nodes_by_name: Dict[str, Node] = {node.name: node for node in tagged_nodes}
        _validate_unique_outputs(tagged_nodes)
        _validate_unique_confirms(tagged_nodes)
        self._nodes_by_input: Dict[str, Set[Node]] = defaultdict(set)
        for node in tagged_nodes:
            for input_ in node.inputs:
                self._nodes_by_input[_strip_transcoding(input_)].add(node)
        self._nodes_by_output: Dict[str, Node] = {}
        for node in tagged_nodes:
            for output in node.outputs:
                self._nodes_by_output[_strip_transcoding(output)] = node
        self._nodes: List[Node] = tagged_nodes
        self._toposorter: TopologicalSorter = TopologicalSorter(self.node_dependencies)
        try:
            self._toposorter.prepare()
        except CycleError as exc:
            loop = list(set(exc.args[1]))
            message = f'Circular dependencies exist among the following {len(loop)} item(s): {loop}'
            raise CircularDependencyError(message) from exc
        self._toposorted_nodes: List[Node] = []
        self._toposorted_groups: List[List[Node]] = []

    def __repr__(self):
        max_nodes_to_display = 10
        nodes_reprs = [repr(node) for node in self.nodes[:max_nodes_to_display]]
        if len(self.nodes) > max_nodes_to_display:
            nodes_reprs.append('...')
        sep = ',\n'
        nodes_reprs_str = f'[\n{sep.join(nodes_reprs)}\n]' if nodes_reprs else '[]'
        constructor_repr = f'({nodes_reprs_str})'
        return f'{self.__class__.__name__}{constructor_repr}'

    def __add__(self, other):
        if not isinstance(other, Pipeline):
            return NotImplemented
        return Pipeline(set(self._nodes + other._nodes))

    def __radd__(self, other):
        if isinstance(other, int) and other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Pipeline):
            return NotImplemented
        return Pipeline(set(self._nodes) - set(other._nodes))

    def __and__(self, other):
        if not isinstance(other, Pipeline):
            return NotImplemented
        return Pipeline(set(self._nodes) & set(other._nodes))

    def __or__(self, other):
        if not isinstance(other, Pipeline):
            return NotImplemented
        return Pipeline(set(self._nodes + other._nodes))

    def all_inputs(self):
        return set.union(set(), *(node.inputs for node in self._nodes))

    def all_outputs(self):
        return set.union(set(), *(node.outputs for node in self._nodes))

    def _remove_intermediates(self, datasets):
        intermediate = {_strip_transcoding(i) for i in self.all_inputs()} & {_strip_transcoding(o) for o in self.all_outputs()}
        return {d for d in datasets if _strip_transcoding(d) not in intermediate}

    def inputs(self):
        return self._remove_intermediates(self.all_inputs())

    def outputs(self):
        return self._remove_intermediates(self.all_outputs())

    def datasets(self):
        return self.all_outputs() | self.all_inputs()

    def _transcode_compatible_names(self):
        return {_strip_transcoding(ds) for ds in self.datasets()}

    def describe(self, names_only=True):

        def set_to_string(set_of_strings):
            return ', '.join(sorted(set_of_strings)) if set_of_strings else 'None'
        nodes_as_string = '\n'.join((node.name if names_only else str(node) for node in self.nodes))
        str_representation = '#### Pipeline execution order ####\nInputs: {0}\n\n{1}\n\nOutputs: {2}\n##################################'
        return str_representation.format(set_to_string(self.inputs()), nodes_as_string, set_to_string(self.outputs()))

    @property
    def node_dependencies(self):
        dependencies: Dict[Node, Set[Node]] = {node: set() for node in self._nodes}
        for parent in self._nodes:
            for output in parent.outputs:
                for child in self._nodes_by_input[_strip_transcoding(output)]:
                    dependencies[child].add(parent)
        return dependencies

    @property
    def nodes(self):
        if not self._toposorted_nodes:
            self._toposorted_nodes = [n for group in self.grouped_nodes for n in group]
        return list(self._toposorted_nodes)

    @property
    def grouped_nodes(self):
        if not self._toposorted_groups:
            while self._toposorter:
                group = sorted(self._toposorter.get_ready())
                self._toposorted_groups.append(group)
                self._toposorter.done(*group)
        return [list(group) for group in self._toposorted_groups]

    @property
    def grouped_nodes_by_namespace(self):
        grouped_nodes: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for node in self.nodes:
            key = node.namespace or node.name
            if key not in grouped_nodes:
                grouped_nodes[key] = {}
                grouped_nodes[key]['name'] = key
                grouped_nodes[key]['type'] = 'namespace' if node.namespace else 'node'
                grouped_nodes[key]['nodes'] = []
                grouped_nodes[key]['dependencies'] = set()
            grouped_nodes[key]['nodes'].append(node)
            dependencies = set()
            for parent in self.node_dependencies[node]:
                if parent.namespace and parent.namespace != key:
                    dependencies.add(parent.namespace)
                elif parent.namespace and parent.namespace == key:
                    continue
                else:
                    dependencies.add(parent.name)
            grouped_nodes[key]['dependencies'].update(dependencies)
        return grouped_nodes

    def only_nodes(self, *node_names: str):
        unregistered_nodes = set(node_names) - set(self._nodes_by_name.keys())
        if unregistered_nodes:
            namespaces = []
            for unregistered_node in unregistered_nodes:
                namespaces.extend([node_name for node_name in self._nodes_by_name.keys() if node_name.endswith(f'.{unregistered_node}')])
            if namespaces:
                raise ValueError(f'Pipeline does not contain nodes named {list(unregistered_nodes)}. Did you mean: {namespaces}?')
            raise ValueError(f'Pipeline does not contain nodes named {list(unregistered_nodes)}.')
        nodes = [self._nodes_by_name[name] for name in node_names]
        return Pipeline(nodes)

    def only_nodes_with_namespace(self, node_namespace):
        nodes = [n for n in self._nodes if n.namespace and n.namespace.startswith(node_namespace)]
        if not nodes:
            raise ValueError(f"Pipeline does not contain nodes with namespace '{node_namespace}'")
        return Pipeline(nodes)

    def _get_nodes_with_inputs_transcode_compatible(self, datasets):
        missing = sorted(datasets - self.datasets() - self._transcode_compatible_names())
        if missing:
            raise ValueError(f'Pipeline does not contain datasets named {missing}')
        relevant_nodes = set()
        for input_ in datasets:
            if _strip_transcoding(input_) == input_:
                relevant_nodes.update(self._nodes_by_input[_strip_transcoding(input_)])
            else:
                for node_ in self._nodes_by_input[_strip_transcoding(input_)]:
                    if input_ in node_.inputs:
                        relevant_nodes.add(node_)
        return relevant_nodes

    def _get_nodes_with_outputs_transcode_compatible(self, datasets):
        missing = sorted(datasets - self.datasets() - self._transcode_compatible_names())
        if missing:
            raise ValueError(f'Pipeline does not contain datasets named {missing}')
        relevant_nodes = set()
        for output in datasets:
            if _strip_transcoding(output) in self._nodes_by_output:
                node_with_output = self._nodes_by_output[_strip_transcoding(output)]
                if _strip_transcoding(output) == output or output in node_with_output.outputs:
                    relevant_nodes.add(node_with_output)
        return relevant_nodes

    def only_nodes_with_inputs(self, *inputs: str):
        starting = set(inputs)
        nodes = self._get_nodes_with_inputs_transcode_compatible(starting)
        return Pipeline(nodes)

    def from_inputs(self, *inputs: str):
        starting = set(inputs)
        result: Set[Node] = set()
        next_nodes = self._get_nodes_with_inputs_transcode_compatible(starting)
        while next_nodes:
            result |= next_nodes
            outputs = set(chain.from_iterable((node.outputs for node in next_nodes)))
            starting = outputs
            next_nodes = set(chain.from_iterable((self._nodes_by_input[_strip_transcoding(input_)] for input_ in starting)))
        return Pipeline(result)

    def only_nodes_with_outputs(self, *outputs: str):
        starting = set(outputs)
        nodes = self._get_nodes_with_outputs_transcode_compatible(starting)
        return Pipeline(nodes)

    def to_outputs(self, *outputs: str):
        starting = set(outputs)
        result: Set[Node] = set()
        next_nodes = self._get_nodes_with_outputs_transcode_compatible(starting)
        while next_nodes:
            result |= next_nodes
            inputs = set(chain.from_iterable((node.inputs for node in next_nodes)))
            starting = inputs
            next_nodes = {self._nodes_by_output[_strip_transcoding(output)] for output in starting if _strip_transcoding(output) in self._nodes_by_output}
        return Pipeline(result)

    def from_nodes(self, *node_names: str):
        res = self.only_nodes(*node_names)
        res += self.from_inputs(*map(_strip_transcoding, res.all_outputs()))
        return res

    def to_nodes(self, *node_names: str):
        res = self.only_nodes(*node_names)
        res += self.to_outputs(*map(_strip_transcoding, res.all_inputs()))
        return res

    def only_nodes_with_tags(self, *tags: str):
        unique_tags = set(tags)
        nodes = [node for node in self._nodes if unique_tags & node.tags]
        return Pipeline(nodes)

    def filter(self, tags=None, from_nodes=None, to_nodes=None, node_names=None, from_inputs=None, to_outputs=None, node_namespace=None):
        node_namespace_iterable = [node_namespace] if node_namespace else None
        filter_methods = {self.only_nodes_with_tags: tags, self.from_nodes: from_nodes, self.to_nodes: to_nodes, self.only_nodes: node_names, self.from_inputs: from_inputs, self.to_outputs: to_outputs, self.only_nodes_with_namespace: node_namespace_iterable}
        subset_pipelines = {filter_method(*filter_args) for filter_method, filter_args in filter_methods.items() if filter_args}
        filtered_pipeline = Pipeline(self._nodes)
        for subset_pipeline in subset_pipelines:
            filtered_pipeline &= subset_pipeline
        if not filtered_pipeline.nodes:
            raise ValueError('Pipeline contains no nodes after applying all provided filters. Please ensure that at least one pipeline with nodes has been defined.')
        return filtered_pipeline

    def tag(self, tags):
        nodes = [n.tag(tags) for n in self._nodes]
        return Pipeline(nodes)

    def to_json(self):
        transformed = [{'name': n.name, 'inputs': list(n.inputs), 'outputs': list(n.outputs), 'tags': list(n.tags)} for n in self._nodes]
        pipeline_versioned = {'kedro_version': kedro.__version__, 'pipeline': transformed}
        return json.dumps(pipeline_versioned)

def _validate_duplicate_nodes(nodes_or_pipes):
    seen_nodes: Set[str] = set()
    duplicates: Dict[Optional[Pipeline], Set[str]] = defaultdict(set)

    def _check_node(node_, pipeline_=None):
        name = node_.name
        if name in seen_nodes:
            duplicates[pipeline_].add(name)
        else:
            seen_nodes.add(name)
    for each in nodes_or_pipes:
        if isinstance(each, Node):
            _check_node(each)
        elif isinstance(each, Pipeline):
            for node in each.nodes:
                _check_node(node, pipeline_=each)
    if duplicates:
        duplicates_info = ''
        for pipeline, names in duplicates.items():
            pipe_repr = 'Free nodes' if pipeline is None else repr(pipeline).replace('\n', '')
            nodes_repr = '\n'.join((f'  - {name}' for name in sorted(names)))
            duplicates_info += f'{pipe_repr}:\n{nodes_repr}\n'
        raise ValueError(f"Pipeline nodes must have unique names. The following node names appear more than once:\n\n{duplicates_info}\nYou can name your nodes using the last argument of 'node()'.")

def _validate_unique_outputs(nodes):
    outputs_chain = chain.from_iterable((node.outputs for node in nodes))
    outputs = map(_strip_transcoding, outputs_chain)
    duplicates = [key for key, value in Counter(outputs).items() if value > 1]
    if duplicates:
        raise OutputNotUniqueError(f'Output(s) {sorted(duplicates)} are returned by more than one nodes. Node outputs must be unique.')

def _validate_unique_confirms(nodes):
    confirms_chain = chain.from_iterable((node.confirms for node in nodes))
    confirms = map(_strip_transcoding, confirms_chain)
    duplicates = [key for key, value in Counter(confirms).items() if value > 1]
    if duplicates:
        raise ConfirmNotUniqueError(f'{sorted(duplicates)} datasets are confirmed by more than one node. Node confirms must be unique.')

def _validate_transcoded_inputs_outputs(nodes):
    all_inputs_outputs = set(chain(chain.from_iterable((node.inputs for node in nodes)), chain.from_iterable((node.outputs for node in nodes))))
    invalid = set()
    for dataset_name in all_inputs_outputs:
        name = _strip_transcoding(dataset_name)
        if name != dataset_name and name in all_inputs_outputs:
            invalid.add(name)
    if invalid:
        raise ValueError(f'The following datasets are used with transcoding, but were referenced without the separator: {', '.join(invalid)}.\nPlease specify a transcoding option or rename the datasets.')

class CircularDependencyError(Exception):
    """Raised when it is not possible to provide a topological execution
    order for nodes, due to a circular dependency existing in the node
    definition.
    """
    pass