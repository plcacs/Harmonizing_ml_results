"""A ``Pipeline`` is a collection of ``Node`` objects which can be executed as
a Directed Acyclic Graph, sequentially or in parallel. The ``Pipeline`` class
offers quick access to input dependencies,
produced outputs and execution order.
"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from graphlib import CycleError, TopologicalSorter
from itertools import chain
from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple, Union, Optional, DefaultDict, Iterable, cast
import kedro
from kedro.pipeline.node import Node, _to_list
from .transcoding import _strip_transcoding

if TYPE_CHECKING:
    from collections.abc import Iterable as CollectionsIterable

def __getattr__(name: str) -> str:
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

    def __init__(self, nodes: Optional[Iterable[Union[Node, Pipeline]]], *, tags: Optional[Union[str, Iterable[str]]] = None) -> None:
        """Initialise ``Pipeline`` with a list of ``Node`` instances.

        Args:
            nodes: The iterable of nodes the ``Pipeline`` will be made of. If you
                provide pipelines among the list of nodes, those pipelines will
                be expanded and all their nodes will become part of this
                new pipeline.
            tags: Optional set of tags to be applied to all the pipeline nodes.

        Raises:
            ValueError:
                When an empty list of nodes is provided, or when not all
                nodes have unique names.
            CircularDependencyError:
                When visiting all the nodes is not
                possible due to the existence of a circular dependency.
            OutputNotUniqueError:
                When multiple ``Node`` instances produce the same output.
            ConfirmNotUniqueError:
                When multiple ``Node`` instances attempt to confirm the same
                dataset.
        Example:
        ::

            >>> from kedro.pipeline import Pipeline
            >>> from kedro.pipeline import node
            >>>
            >>> # In the following scenario first_ds and second_ds
            >>> # are datasets provided by io. Pipeline will pass these
            >>> # datasets to first_node function and provides the result
            >>> # to the second_node as input.
            >>>
            >>> def first_node(first_ds, second_ds):
            >>>     return dict(third_ds=first_ds+second_ds)
            >>>
            >>> def second_node(third_ds):
            >>>     return third_ds
            >>>
            >>> pipeline = Pipeline([
            >>>     node(first_node, ['first_ds', 'second_ds'], ['third_ds']),
            >>>     node(second_node, dict(third_ds='third_ds'), 'fourth_ds')])
            >>>
            >>> pipeline.describe()
            >>>

        """
        if nodes is None:
            raise ValueError("'nodes' argument of 'Pipeline' is None. It must be an iterable of nodes and/or pipelines instead.")
        nodes_list = list(nodes)
        _validate_duplicate_nodes(nodes_list)
        nodes_chain = list(chain.from_iterable([[n] if isinstance(n, Node) else n.nodes for n in nodes_list]))
        _validate_transcoded_inputs_outputs(nodes_chain)
        _tags = set(_to_list(tags)) if tags else set()
        if _tags:
            tagged_nodes = [n.tag(_tags) for n in nodes_chain]
        else:
            tagged_nodes = nodes_chain
        self._nodes_by_name: Dict[str, Node] = {node.name: node for node in tagged_nodes}
        _validate_unique_outputs(tagged_nodes)
        _validate_unique_confirms(tagged_nodes)
        self._nodes_by_input: DefaultDict[str, Set[Node]] = defaultdict(set)
        for node in tagged_nodes:
            for input_ in node.inputs:
                self._nodes_by_input[_strip_transcoding(input_)].add(node)
        self._nodes_by_output: Dict[str, Node] = {}
        for node in tagged_nodes:
            for output in node.outputs:
                self._nodes_by_output[_strip_transcoding(output)] = node
        self._nodes: List[Node] = tagged_nodes
        self._toposorter = TopologicalSorter(self.node_dependencies)
        try:
            self._toposorter.prepare()
        except CycleError as exc:
            loop = list(set(exc.args[1]))
            message = f'Circular dependencies exist among the following {len(loop)} item(s): {loop}'
            raise CircularDependencyError(message) from exc
        self._toposorted_nodes: List[Node] = []
        self._toposorted_groups: List[List[Node]] = []

    def __repr__(self) -> str:
        """Pipeline ([node1, ..., node10 ...], name='pipeline_name')"""
        max_nodes_to_display = 10
        nodes_reprs = [repr(node) for node in self.nodes[:max_nodes_to_display]]
        if len(self.nodes) > max_nodes_to_display:
            nodes_reprs.append('...')
        sep = ',\n'
        nodes_reprs_str = f'[\n{sep.join(nodes_reprs)}\n]' if nodes_reprs else '[]'
        constructor_repr = f'({nodes_reprs_str})'
        return f'{self.__class__.__name__}{constructor_repr}'

    def __add__(self, other: Any) -> Pipeline:
        if not isinstance(other, Pipeline):
            return NotImplemented
        return Pipeline(set(self._nodes + other._nodes))

    def __radd__(self, other: Any) -> Pipeline:
        if isinstance(other, int) and other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other: Any) -> Pipeline:
        if not isinstance(other, Pipeline):
            return NotImplemented
        return Pipeline(set(self._nodes) - set(other._nodes))

    def __and__(self, other: Any) -> Pipeline:
        if not isinstance(other, Pipeline):
            return NotImplemented
        return Pipeline(set(self._nodes) & set(other._nodes))

    def __or__(self, other: Any) -> Pipeline:
        if not isinstance(other, Pipeline):
            return NotImplemented
        return Pipeline(set(self._nodes + other._nodes))

    def all_inputs(self) -> Set[str]:
        """All inputs for all nodes in the pipeline.

        Returns:
            All node input names as a Set.

        """
        return set.union(set(), *(node.inputs for node in self._nodes))

    def all_outputs(self) -> Set[str]:
        """All outputs of all nodes in the pipeline.

        Returns:
            All node outputs.

        """
        return set.union(set(), *(node.outputs for node in self._nodes))

    def _remove_intermediates(self, datasets: Set[str]) -> Set[str]:
        intermediate = {_strip_transcoding(i) for i in self.all_inputs()} & {_strip_transcoding(o) for o in self.all_outputs()}
        return {d for d in datasets if _strip_transcoding(d) not in intermediate}

    def inputs(self) -> Set[str]:
        """The names of free inputs that must be provided at runtime so that
        the pipeline is runnable. Does not include intermediate inputs which
        are produced and consumed by the inner pipeline nodes. Resolves
        transcoded names where necessary.

        Returns:
            The set of free input names needed by the pipeline.

        """
        return self._remove_intermediates(self.all_inputs())

    def outputs(self) -> Set[str]:
        """The names of outputs produced when the whole pipeline is run.
        Does not include intermediate outputs that are consumed by
        other pipeline nodes. Resolves transcoded names where necessary.

        Returns:
            The set of final pipeline outputs.

        """
        return self._remove_intermediates(self.all_outputs())

    def datasets(self) -> Set[str]:
        """The names of all datasets used by the ``Pipeline``,
        including inputs and outputs.

        Returns:
            The set of all pipeline datasets.

        """
        return self.all_outputs() | self.all_inputs()

    def _transcode_compatible_names(self) -> Set[str]:
        return {_strip_transcoding(ds) for ds in self.datasets()}

    def describe(self, names_only: bool = True) -> str:
        """Obtain the order of execution and expected free input variables in
        a loggable pre-formatted string. The order of nodes matches the order
        of execution given by the topological sort.

        Args:
            names_only: The flag to describe names_only pipeline with just
                node names.

        Example:
        ::

            >>> pipeline = Pipeline([ ... ])
            >>>
            >>> logger = logging.getLogger(__name__)
            >>>
            >>> logger.info(pipeline.describe())

        After invocation the following will be printed as an info level log
        statement:
        ::

            #### Pipeline execution order ####
            Inputs: C, D

            func1([C]) -> [A]
            func2([D]) -> [B]
            func3([A, D]) -> [E]

            Outputs: B, E
            ##################################

        Returns:
            The pipeline description as a formatted string.

        """

        def set_to_string(set_of_strings: Set[str]) -> str:
            """Convert set to a string but return 'None' in case of an empty
            set.
            """
            return ', '.join(sorted(set_of_strings)) if set_of_strings else 'None'
        nodes_as_string = '\n'.join((node.name if names_only else str(node) for node in self.nodes))
        str_representation = '#### Pipeline execution order ####\nInputs: {0}\n\n{1}\n\nOutputs: {2}\n##################################'
        return str_representation.format(set_to_string(self.inputs()), nodes_as_string, set_to_string(self.outputs()))

    @property
    def node_dependencies(self) -> Dict[Node, Set[Node]]:
        """All dependencies of nodes where the first Node has a direct dependency on
        the second Node.

        Returns:
            Dictionary where keys are nodes and values are sets made up of
            their parent nodes. Independent nodes have this as empty sets.
        """
        dependencies = {node: set() for node in self._nodes}
        for parent in self._nodes:
            for output in parent.outputs:
                for child in self._nodes_by_input[_strip_transcoding(output)]:
                    dependencies[child].add(parent)
        return dependencies

    @property
    def nodes(self) -> List[Node]:
        """Return a list of the pipeline nodes in topological order, i.e. if
        node A needs to be run before node B, it will appear earlier in the
        list.

        Returns:
            The list of all pipeline nodes in topological order.

        """
        if not self._toposorted_nodes:
            self._toposorted_nodes = [n for group in self.grouped_nodes for n in group]
        return list(self._toposorted_nodes)

    @property
    def grouped_nodes(self) -> List[List[Node]]:
        """Return a list of the pipeline nodes in topologically ordered groups,
        i.e. if node A needs to be run before node B, it will appear in an
        earlier group.

        Returns:
            The pipeline nodes in topologically ordered groups.

        """
        if not self._toposorted_groups:
            while self._toposorter:
                group = sorted(self._toposorter.get_ready())
                self._toposorted_groups.append(group)
                self._toposorter.done(*group)
        return [list(group) for group in self._toposorted_groups]

    @property
    def grouped_nodes_by_namespace(self) -> Dict[str, Dict[str, Union[str, List[Node], Set[str]]]]:
        """Return a dictionary of the pipeline nodes grouped by namespace with
        information about the nodes, their type, and dependencies. The structure of the dictionary is:
        {'node_name/namespace_name' : {'name': 'node_name/namespace_name','type': 'namespace' or 'node','nodes': [list of nodes],'dependencies': [list of dependencies]}}
        This property is intended to be used by deployment plugins to group nodes by namespace.

        """
        grouped_nodes: Dict[str, Dict[str, Union[str, List[Node], Set[str]]] = defaultdict(dict)
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

    def only_nodes(self, *node_names: str) -> Pipeline:
        """Create a new ``Pipeline`` which will contain only the specified
        nodes by name.

        Args:
            *node_names: One or more node names. The returned ``Pipeline``
                will only contain these nodes.

        Raises:
            ValueError: When some invalid node name is given.

        Returns:
            A new ``Pipeline``, containing only ``nodes``.

        """
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

    def only_nodes_with_namespace(self, node_namespace: str) -> Pipeline:
        """Creates a new ``Pipeline`` containing only nodes with the specified
        namespace.

        Args:
            node_namespace: One node namespace.

        Raises:
            ValueError: When pipeline contains no nodes with the specified namespace.

        Returns:
            A new ``Pipeline`` containing nodes with the specified namespace.
        """
        nodes = [n for n in self._nodes if n.namespace and n.namespace.startswith(node_namespace)]
        if not nodes:
            raise ValueError(f"Pipeline does not contain nodes with namespace '{node_namespace}'")
        return Pipeline(nodes)

    def _get_nodes_with_inputs_transcode_compatible(self, datasets: Set[str]) -> Set[Node]:
        """Retrieves nodes that use the given `datasets` as inputs.
        If provided a name, but no format, for a transcoded dataset, it
        includes all nodes that use inputs with that name, otherwise it
        matches to the fully-qualified name only (i.e. name@format).

        Raises:
            ValueError: if any of the given datasets do not exist in the
                ``Pipeline`` object

        Returns:
            Set of ``Nodes`` that use the given datasets as inputs.
        """
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

    def _get_nodes_with_outputs_transcode_compatible(self, datasets: Set[str]) -> Set[Node]:
        """Retrieves nodes that output to the given `datasets`.
        If provided a name, but no format, for a transcoded dataset, it
        includes the node that outputs to that name, otherwise it matches
        to the fully-qualified name only (i.e. name@format).

        Raises:
            ValueError: if any of the given datasets do not exist in the
                ``Pipeline`` object

        Returns:
            Set of ``Nodes`` that output to the given datasets.
        """
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

    def only_nodes_with_inputs(self, *inputs: str) -> Pipeline:
        """Create a new ``Pipeline`` object with the nodes which depend
        directly on the provided inputs.
        If provided a name, but no format, for a transcoded input, it
        includes all the nodes that use inputs with that name, otherwise it
        matches to the fully-qualified name only (i.e. name@format).

        Args:
            *inputs: A list of inputs which should be used as